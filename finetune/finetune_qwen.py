#!/usr/bin/env python3
"""
Fine-tune Qwen 3.5 (4B / 9B) on the Guide-LLM tool-calling dataset.

Based on Unsloth's official guide:
  https://unsloth.ai/docs/models/qwen3.5/fine-tune
  https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)_Vision.ipynb

Usage:
    # Fine-tune 2B model (~5 GB VRAM)
    python3 finetune/finetune_qwen.py --model-size 2b --epochs 3

    # Fine-tune 4B model (~10 GB VRAM)
    python3 finetune/finetune_qwen.py --model-size 4b --epochs 3

    # Fine-tune 9B model (~22 GB VRAM)
    python3 finetune/finetune_qwen.py --model-size 9b --epochs 3

    # Quick test run (30 steps only)
    python3 finetune/finetune_qwen.py --model-size 2b --max-steps 30

Requirements:
    pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
    pip install transformers>=5.0 datasets trl peft

    NOTE: transformers v5 is REQUIRED for Qwen 3.5.
    NOTE: Do NOT use QLoRA (4-bit) on Qwen 3.5 — use 16-bit LoRA.
"""

import argparse
import json
import os
import sys
import torch


def check_dependencies():
    missing = []
    for pkg in ["unsloth", "transformers", "datasets", "trl", "peft"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with:")
        print("  pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo")
        print("  pip install 'transformers>=5.0' datasets trl peft")
        sys.exit(1)

    import transformers
    major = int(transformers.__version__.split(".")[0])
    if major < 5:
        print(f"ERROR: transformers v5+ required for Qwen 3.5, found v{transformers.__version__}")
        print("Install with: pip install 'transformers>=5.0'")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 3.5 for Guide-LLM tool calling")
    parser.add_argument("--model-size", choices=["2b", "4b", "9b"], default="4b",
                        help="Model size: 2b (~5GB VRAM), 4b (~10GB VRAM), or 9b (~22GB VRAM)")
    parser.add_argument("--train-file", default="finetune/data/train.jsonl")
    parser.add_argument("--val-file", default="finetune/data/val.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Override epochs with fixed step count (e.g. 30 for quick test)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--no-export-gguf", action="store_true", default=False)
    parser.add_argument("--gguf-quant", default="q4_k_m",
                        help="GGUF quantization: q4_k_m, q8_0, f16")
    parser.add_argument("--no-think", action="store_true", default=True,
                        help="Disable thinking mode in training data")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"finetune/output/qwen3.5-{args.model_size}-toolcall"

    check_dependencies()

    # =================================================================
    # 1. Load model + tokenizer
    #    Ref: https://unsloth.ai/docs/models/qwen3.5/fine-tune
    #    - load_in_4bit=False (QLoRA NOT recommended for Qwen 3.5)
    #    - load_in_16bit=True (use bf16 LoRA)
    # =================================================================
    from unsloth import FastLanguageModel

    model_name = f"Qwen/Qwen3.5-{args.model_size.upper()}"
    max_seq_length = args.max_seq_len

    print(f"\n{'='*60}")
    print(f"  Guide-LLM Fine-tuning")
    print(f"  Model:   {model_name}")
    print(f"  LoRA:    r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Epochs:  {args.epochs}" + (f" (overridden by --max-steps {args.max_steps})" if args.max_steps > 0 else ""))
    print(f"  Batch:   {args.batch_size} x {args.grad_accum} grad_accum")
    print(f"  LR:      {args.lr}")
    print(f"  Seq len: {max_seq_length}")
    print(f"{'='*60}\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )

    # =================================================================
    # 2. Apply LoRA adapters
    #    Matches Unsloth's official config exactly:
    #    r=16, lora_alpha=16, lora_dropout=0, bias="none"
    # =================================================================
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )

    # =================================================================
    # 3. Load dataset
    # =================================================================
    from datasets import load_dataset

    print(f"Loading training data from {args.train_file}")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")

    val_dataset = None
    if os.path.exists(args.val_file):
        print(f"Loading validation data from {args.val_file}")
        val_dataset = load_dataset("json", data_files=args.val_file, split="train")

    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")

    # =================================================================
    # 4. Format conversations using the tokenizer's chat template
    #    The tokenizer handles Qwen 3.5's XML tool-call format
    #    automatically. We just pass messages + tools.
    # =================================================================
    def format_conversation(example):
        messages = example["messages"]
        tools = example.get("tools", None)

        # Format with apply_chat_template, passing tools for proper tool-use formatting
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            print(f"Error formatting example: {e}")
            print(f"Messages count: {len(messages)}")
            print(f"Tools provided: {tools is not None}")
            raise

        return {"text": text}

    train_dataset = train_dataset.map(format_conversation)
    if val_dataset:
        val_dataset = val_dataset.map(format_conversation)

    # Print a sample to verify formatting
    print(f"\n{'='*60}")
    print("  Sample formatted conversation (first 2000 chars):")
    print(f"{'='*60}")
    print(train_dataset[0]["text"][:2000])
    print("...(truncated)")
    print(f"{'='*60}\n")

    # =================================================================
    # 5. Training with SFTTrainer
    #    Matches Unsloth's official SFTConfig:
    #    - optim="adamw_8bit"
    #    - seed=3407
    #    - dataset_num_proc=1
    #    - weight_decay=0.001
    #    - lr_scheduler_type="linear"
    # =================================================================
    from trl import SFTTrainer, SFTConfig

    os.makedirs(args.output_dir, exist_ok=True)

    sft_config_kwargs = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=10,
        learning_rate=args.lr,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        dataset_num_proc=1,
        packing=False,
        bf16=True,
        save_strategy="epoch",
    )

    if args.max_steps > 0:
        sft_config_kwargs["max_steps"] = args.max_steps
    else:
        sft_config_kwargs["num_train_epochs"] = args.epochs

    if val_dataset:
        sft_config_kwargs["eval_strategy"] = "epoch"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(**sft_config_kwargs),
    )

    # Memory check before training
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    print("\nStarting training...")
    trainer_stats = trainer.train()

    # Training stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    print(f"\n{trainer_stats.metrics['train_runtime']:.1f} seconds used for training.")
    print(f"{trainer_stats.metrics['train_runtime']/60:.1f} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage}%.")

    # =================================================================
    # 6. Save LoRA adapters
    # =================================================================
    lora_dir = os.path.join(args.output_dir, "lora")
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    print(f"\nLoRA adapter saved to {lora_dir}")

    # =================================================================
    # 7. Export to GGUF for Ollama
    # =================================================================
    if not args.no_export_gguf:
        print(f"\nExporting to GGUF ({args.gguf_quant})...")
        gguf_dir = os.path.join(args.output_dir, "gguf")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method=args.gguf_quant,
        )
        print(f"GGUF model saved to {gguf_dir}")

        # Create Ollama Modelfile
        gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
        if gguf_files:
            gguf_path = os.path.join(gguf_dir, gguf_files[0])
            modelfile_path = os.path.join(args.output_dir, "Modelfile")
            model_tag = f"guide-llm-{args.model_size}"
            with open(modelfile_path, "w") as f:
                f.write(f"FROM {os.path.abspath(gguf_path)}\n")
                f.write(f'PARAMETER temperature 0.7\n')
                f.write(f'PARAMETER num_ctx {max_seq_length}\n')
                f.write(f'PARAMETER stop "<|im_end|>"\n')
                f.write(f'PARAMETER stop "<|endoftext|>"\n')
            print(f"\nOllama Modelfile saved to {modelfile_path}")
            print(f"\nTo import into Ollama:")
            print(f"  ollama create {model_tag} -f {os.path.abspath(modelfile_path)}")
            print(f"\nThen use in the agent:")
            print(f"  python3 scene_graph_agent_local.py --model {model_tag}")

    # =================================================================
    # 8. Save merged 16-bit (for vLLM deployment)
    # =================================================================
    merged_dir = os.path.join(args.output_dir, "merged_16bit")
    print(f"\nSaving merged 16-bit model to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  LoRA adapter:  {lora_dir}")
    if not args.no_export_gguf:
        print(f"  GGUF:          {gguf_dir}")
    print(f"  Merged 16-bit: {merged_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
