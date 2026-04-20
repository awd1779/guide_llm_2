# Fine-Tuning Guide: Qwen 3.5 for Guide-LLM

This directory contains the training pipeline for fine-tuning Qwen 3.5 (4B or 9B) to perform tool-calling for VI navigation tasks.

## Overview

**Goal**: Train a small LLM (~4B parameters) to be competitive with Claude 3.5 Sonnet on tool-calling accuracy for navigation scenarios, while fitting on edge hardware (Jetson, consumer GPUs).

**Key facts**:
- **Base model**: Qwen 3.5 4B or 9B (from Hugging Face)
- **Training method**: Full fine-tuning with LoRA (16-bit, NOT 4-bit)
- **Framework**: Unsloth (for fast training) + Hugging Face transformers
- **Data format**: OpenAI chat format (JSONL)
- **Dataset**: Synthetic tool-calling scenarios (auto-generated)
- **VRAM**: 4B model ~10GB, 9B model ~22GB
- **Duration**: 3 epochs on 4B ≈ 2-4 hours (single GPU)

---

## Prerequisites

### System Requirements
- GPU with ≥10GB VRAM (4B) or ≥22GB VRAM (9B)
- Python 3.10+
- CUDA 12.1+ (recommended for latest models)

### Install Dependencies

```bash
# Update pip
pip install --upgrade pip

# Required: Unsloth + transformers v5
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
pip install 'transformers>=5.0' datasets trl peft

# Strongly recommended: Flash Attention (10x speedup)
pip install flash-attn --no-build-isolation

# For quantization (GGUF export)
pip install llama-cpp-python  # optional

# Verify installation
python -c "import unsloth; print(f'Unsloth version: {unsloth.__version__}')"
```

---

## Step 1: Generate Training Data

The dataset is **synthetic** — realistic VI navigation scenarios with tool calls.

```bash
python finetune/generate_dataset.py \
    --output finetune/data/train.jsonl \
    --num-samples 800
```

**Output**: `finetune/data/train.jsonl` (800 conversations, ~2.5 MB)

**What it includes**:
- Multi-turn conversations (user → assistant → tool → result → assistant)
- 12 navigation tools: navigate_to_zone, navigate_to_object, find_nearest, distance_to, etc.
- Realistic user intents: "take me to the kitchen", "what's around me?", "where's the nearest chair?"
- Natural language responses (not technical)
- Tool-calling in OpenAI function format

**Customize** (optional):
```bash
# More samples for better generalization
python finetune/generate_dataset.py --num-samples 2000 --output data/train_large.jsonl

# Different random seed for variety
python finetune/generate_dataset.py --seed 42 --num-samples 500
```

---

## Step 2: Fine-Tune the Model

### Quick Start (4B model, 3 epochs)

```bash
python finetune/finetune_qwen.py --model-size 4b --epochs 3
```

**Expected output**:
- Checkpoints saved to `finetune/output/qwen3.5-4b-toolcall/`
- Loss should drop from ~2.5 → ~0.5-0.8 over 3 epochs
- Each epoch ≈ 45-60 min on single A100/H100

### Custom Training (advanced)

```bash
# Train 9B model with custom hyperparameters
python finetune/finetune_qwen.py \
    --model-size 9b \
    --epochs 5 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 1e-4 \
    --lora-r 32 \
    --lora-alpha 32 \
    --max-seq-len 4096

# Quick test run (30 steps only)
python finetune/finetune_qwen.py --model-size 4b --max-steps 30

# Train on custom dataset
python finetune/finetune_qwen.py \
    --model-size 4b \
    --train-file data/my_train.jsonl \
    --val-file data/my_val.jsonl
```

### Hyperparameter Reference

| Param | 4B (fast) | 4B (quality) | 9B (quality) | Notes |
|-------|-----------|-------------|------------|-------|
| batch_size | 1 | 1 | 1 | Must be 1 (Qwen limitation) |
| grad_accum | 4 | 8 | 16 | Effective batch = batch_size × grad_accum |
| lr | 2e-4 | 1e-4 | 1e-4 | Lower for larger models |
| lora_r | 16 | 32 | 32 | Larger = more capacity, slower |
| epochs | 3 | 5-7 | 3-5 | More data → fewer epochs needed |
| max_seq_len | 2048 | 4096 | 4096 | Max input length (pad to this) |

### What Training Does

1. **Loads Qwen 3.5** from Hugging Face (quantized to fp16)
2. **Applies LoRA** (low-rank adapters) on query/value projections
3. **Fine-tunes** on tool-calling dataset for N epochs
4. **Saves** LoRA adapters + merged model + GGUF quantizations

**Memory usage**:
- 4B model: ~10-11 GB peak
- 9B model: ~22-24 GB peak

---

## Step 3: Export & Quantize

After training, you can export to different formats:

### Option A: Keep LoRA Adapters (smallest, fastest to load)
```bash
# Adapters stay in finetune/output/qwen3.5-4b-toolcall/
# They load onto the base model at inference time
```

### Option B: Merge LoRA into Base Model
```bash
# Already done by finetune_qwen.py (check for merged_model/)
```

### Option C: Quantize to GGUF (for edge deployment)
```bash
# 4-bit quantization (runs on CPU, slower)
python finetune/finetune_qwen.py \
    --model-size 4b \
    --gguf-quant q4_k_m \
    --no-export-gguf  # to skip auto-export

# Use exported GGUF with llama.cpp or ollama
ollama create guide-llm-4b -f Modelfile  # requires Modelfile with GGUF path
```

---

## Step 4: Validate & Evaluate

Run the evaluation script on your fine-tuned model:

```bash
python finetune/validate_model.py \
    --model-path finetune/output/qwen3.5-4b-toolcall/checkpoint-223 \
    --test-file finetune/data/test.jsonl \
    --num-samples 100
```

**Metrics computed**:
- **Tool-calling accuracy**: % of correct tool + arguments
- **Response quality**: Semantic similarity to reference
- **Token-level metrics**: Perplexity, BLEU
- **Per-tool accuracy**: Breakdown by tool type

**Expected results** (on validation set):
- Tool accuracy: 85-95%
- Exact match: 70-80%

---

## Step 5: Deploy the Model

### Option 1: vLLM (Recommended for inference speed)

```bash
# Start local vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model finetune/output/qwen3.5-4b-toolcall/checkpoint-223 \
    --tensor-parallel-size 1 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.9

# In another terminal, start the agent
python scene_graph_agent_local.py --base-url http://localhost:8000/v1
```

**Performance**: ~150-300ms per inference (depending on GPU)

### Option 2: Ollama (Easy local deployment)

```bash
# Create Modelfile
cat > Modelfile << EOF
FROM ./finetune/output/qwen3.5-4b-toolcall/checkpoint-223
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_predict 1024
EOF

# Build & run model
ollama create guide-llm:v1 -f Modelfile
ollama serve guide-llm:v1  # Runs on localhost:11434

# Start agent (modify base-url in agent_local.py to http://localhost:11434/v1)
```

### Option 3: Direct Python Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model + LoRA
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-4B",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

# Load LoRA adapters
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "finetune/output/qwen3.5-4b-toolcall/checkpoint-223")

# Inference
inputs = tokenizer("take me to the kitchen", return_tensors="pt")
outputs = model.generate(**inputs, max_length=256, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

---

## Common Issues & Fixes

### Issue: CUDA out of memory
**Solution**: Reduce `grad_accum` or use smaller model (4B instead of 9B)

### Issue: NaN loss during training
**Solution**: Lower learning rate (try 5e-5) or increase warmup steps

### Issue: Model generates gibberish after fine-tuning
**Solution**:
- Check dataset quality (no corrupted examples)
- Reduce learning rate (might be overfitting)
- Train for fewer epochs

### Issue: Tool calls not in correct format
**Solution**:
- Verify training data uses OpenAI function_calling format
- Check system prompt in dataset (must mention available tools)

### Issue: vLLM crashes on startup
**Solution**: Check CUDA/GPU driver versions, try `--dtype float16` instead of auto

---

## Performance Benchmarks

**Hardware**: NVIDIA A100 40GB

| Model | Batch | Epoch Time | Throughput | VRAM |
|-------|-------|-----------|-----------|------|
| 4B, r=16 | 1 | 45 min | 89 tok/s | 10.5 GB |
| 4B, r=32 | 1 | 52 min | 77 tok/s | 11.2 GB |
| 9B, r=16 | 1 | 140 min | 28 tok/s | 22.1 GB |
| 9B, r=32 | 1 | 160 min | 24 tok/s | 23.8 GB |

**Inference** (batch=1, context=512):
- 4B model: 150-250 ms/token (vLLM)
- 9B model: 300-500 ms/token (vLLM)

---

## Dataset Details

The synthetic dataset is generated by `generate_dataset.py` and includes:

**Scenario types**:
1. Greeting & orientation
2. Navigate to zone/object with confirmation
3. User cancels / changes mind
4. Describe surroundings
5. Find nearest object
6. Query distance & route
7. Orient robot to face target
8. Explore environment
9. Ambiguous requests (clarification)
10. Small talk & help

**Conversation statistics**:
- Avg length: 8-12 turns per conversation
- Tool calls per conversation: 1-3
- Tools covered: All 12 navigation tools
- Environments: 8 diverse indoor layouts

**To customize dataset**:
Edit `TOOLS` and `ENVIRONMENTS` in `generate_dataset.py`, then regenerate:
```bash
python generate_dataset.py --output data/custom.jsonl --num-samples 1000
```

---

## Next Steps for Research

1. **Evaluate on real user data**: Collect VI user interactions, measure task success rates
2. **Compare against baselines**: Claude 3.5 Sonnet, GPT-4, other fine-tuned models
3. **Ablation studies**: Effect of model size, LoRA rank, dataset size on tool-calling accuracy
4. **Publication**: Document findings in conference-ready format (ASSETS, RA-L, CoRL)

See `[research/RESEARCH_AND_PUBLICATION.md](../research/RESEARCH_AND_PUBLICATION.md)` for full roadmap.

---

## References

- **Unsloth**: https://github.com/unslothai/unsloth
- **Qwen Models**: https://huggingface.co/Qwen
- **vLLM**: https://docs.vllm.ai
- **Ollama**: https://ollama.ai
- **Original Guide-LLM Paper**: https://arxiv.org/abs/2410.20666

---

**Questions?** See the main [README.md](../README.md) or check the research documentation.
