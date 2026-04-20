# Guide-LLM Fine-tuning: Research Findings

## Model: Qwen 3.5 (4B / 9B)

### Key Discovery: Qwen 3.5 Has Native Tool Calling Support

Qwen 3.5's tokenizer chat template handles tool/function calling natively using an **XML-based format**, unlike older models (e.g., Gemma) where tool calls must be embedded as plain text.

**Qwen 3.5 tool call format:**
```xml
<tool_call>
<function=function_name>
<parameter=param_name>
value
</parameter>
</function>
</tool_call>
```

**Tool results are returned as:**
```xml
<tool_response>
{"key": "value"}
</tool_response>
```

This is handled automatically by `tokenizer.apply_chat_template(messages, tools=TOOLS)`.

---

## Critical Requirements

### 1. Transformers v5+ Required
Qwen 3.5 will NOT work with transformers v4. Must use v5+.
```bash
pip install "transformers>=5.0"
```

### 2. Do NOT Use QLoRA (4-bit) for Qwen 3.5
Unsloth docs explicitly warn: *"It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models due to higher than normal quantization differences."*

Use 16-bit LoRA instead:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    load_in_4bit=False,      # Do NOT use 4-bit
    load_in_16bit=True,      # Use 16-bit
    full_finetuning=False,
)
```

### 3. Tool Call Arguments Must Be Dicts, Not Strings
Qwen 3.5's Jinja chat template uses `|items` filter on `tool_call.arguments`, which expects a Python dict. Passing a JSON string causes a `TypeError: Can only get item pairs from a mapping` error.

```python
# WRONG — causes template error
{"arguments": '{"zone_name": "kitchen"}'}

# CORRECT
{"arguments": {"zone_name": "kitchen"}}
```

This is a known bug discussed in: https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4

### 4. Thinking Mode
Qwen 3.5 has a thinking/reasoning mode. When disabled via `enable_thinking=False`, empty `<think></think>` tags are still present in the output but contain no content. For our use case (real-time navigation), thinking is disabled to reduce latency.

In Ollama:
```bash
ollama run qwen3.5:9b --think=false
```

In OpenAI-compatible API:
```python
client.chat.completions.create(..., extra_body={"think": False})
```

---

## VRAM Requirements

| Model | bf16 LoRA | Notes |
|-------|-----------|-------|
| 4B    | ~10 GB    | Fits comfortably on A10G (23 GB) |
| 9B    | ~22 GB    | Tight fit on A10G, works with batch_size=1 |

---

## Training Configuration (Aligned with Unsloth Docs)

Source: https://unsloth.ai/docs/models/qwen3.5/fine-tune

```python
# Model loading
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3.5-4B",
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_16bit=True,
    full_finetuning=False,
)

# LoRA config
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,           # Same as rank
    lora_dropout=0,          # Recommended: 0
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=2048,
)

# Training config
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    learning_rate=2e-4,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.001,
    lr_scheduler_type="linear",
    seed=3407,
    dataset_num_proc=1,
    max_seq_length=2048,
    bf16=True,
)
```

---

## Dataset Format

Each training sample is a JSONL line with `messages` and `tools`:

```json
{
  "messages": [
    {"role": "system", "content": "You are a navigation assistant..."},
    {"role": "user", "content": "Where am I?"},
    {"role": "assistant", "content": "", "tool_calls": [
      {"id": "call_abc123", "type": "function",
       "function": {"name": "get_robot_pose", "arguments": {}}}
    ]},
    {"role": "tool", "tool_call_id": "call_abc123",
     "content": "{\"x\": 2.0, \"y\": 1.5, \"current_zone\": \"kitchen\"}"},
    {"role": "assistant", "content": "You're in the kitchen."}
  ],
  "tools": [...]
}
```

The `tokenizer.apply_chat_template()` converts this into Qwen 3.5's native format with `<tool_call>` XML tags, `<tool_response>` blocks, and `<think>` sections.

---

## Differences: Tool Calling vs Conversational Fine-tuning

| Aspect | Conversational | Tool Calling |
|--------|---------------|--------------|
| Message roles | user, assistant | user, assistant, tool |
| Assistant output | Text only | Text + `tool_calls` array |
| Chat template | `apply_chat_template(messages)` | `apply_chat_template(messages, tools=TOOLS)` |
| Special tokens | `<\|im_start\|>`, `<\|im_end\|>` | Same + `<tool_call>`, `<function=...>`, `<parameter=...>`, `<tool_response>` |
| What model learns | How to respond | Which tool + args + how to respond after results |
| System prompt | Instructions only | Instructions + tool definitions in `<tools>` XML |

---

## HuggingFace Agents Course Approach (for reference)

Source: https://huggingface.co/learn/agents-course/bonus-unit1/fine-tuning

The HF course fine-tunes Gemma-2-2B-it using a different approach:
- Tool calls are embedded as **plain text** in message content (not structured objects)
- Uses custom special tokens: `<tools>`, `<tool_call>`, `<tool_response>`, `<think>`
- Requires `model.resize_token_embeddings()` for new tokens
- Dataset: `Jofthomas/hermes-function-calling-thinking-V1` (3570 samples)

**This approach is NOT needed for Qwen 3.5** since it has native tool calling support in the chat template. Our approach uses structured `tool_calls` objects and lets the tokenizer handle formatting.

---

## Export and Deployment

### GGUF for Ollama
```python
model.save_pretrained_gguf("gguf_dir", tokenizer, quantization_method="q4_k_m")
```

```bash
# Modelfile
FROM /path/to/model.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Import
ollama create guide-llm-4b -f Modelfile
```

### Merged 16-bit for vLLM
```python
model.save_pretrained_merged("merged_dir", tokenizer, save_method="merged_16bit")
```

### LoRA adapter only
```python
model.save_pretrained("lora_dir")
tokenizer.save_pretrained("lora_dir")
```

---

## Our Dataset: 18 Conversation Types

| # | Type | Tools Practiced | What It Teaches |
|---|------|----------------|-----------------|
| 1 | Greeting & orient | get_robot_pose, describe_surroundings | Guide introduces itself, checks location |
| 2 | Navigate to zone (with confirm) | list_zones, navigate_to_zone, get_navigation_status | Always confirm before moving |
| 3 | Navigate to object (with confirm) | find_nearest, navigate_to_object | Find then confirm then navigate |
| 4 | User declines navigation | list_zones | Handle "never mind" gracefully |
| 5 | Explore surroundings | describe_surroundings, query_object | Describe nearby, follow-up on items |
| 6 | Distance and route | distance_to, describe_route, navigate_to_zone | Check distance, preview path, then go |
| 7 | Orient and explore | orient_me, query_zone | Face target, ask what's there |
| 8 | Query zones then navigate | list_zones, query_zone, navigate_to_zone | List rooms, pick one, go |
| 9 | Small talk and help | get_robot_pose | Handle thanks, "what can you do?" |
| 10 | Ambiguous request | list_zones | Ask clarification, don't guess |
| 11 | Find without navigating | find_nearest | User just wants info, not movement |
| 12 | Arrived then explore | get_navigation_status, describe_surroundings, orient_me | Post-arrival orientation |
| 13 | Safety: always confirm | list_zones, navigate_to_zone | User changes mind, guide adapts |
| 14 | Find, distance, navigate | find_nearest, distance_to, navigate_to_object | Multi-step before moving |
| 15 | List objects | list_objects | What things exist |
| 16 | Full journey | get_robot_pose, describe_route, navigate_to_zone, get_navigation_status, describe_surroundings | End-to-end: greet to thanks |
| 17 | Navigation failed | list_zones, navigate_to_zone, get_navigation_status | Handle failures, offer retry |
| 18 | Wrong name correction | list_zones, navigate_to_zone | Nonexistent room, list alternatives |

**Stats:** 486 conversations, 1134 tool-call turns, 1458 pure text turns, all 13 tools covered, 5 environment variations.

---

## References

- [Unsloth Qwen 3.5 Fine-tuning Guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Unsloth Qwen 3.5 Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- [Unsloth Tool Calling Guide](https://unsloth.ai/docs/basics/tool-calling-guide-for-local-llms)
- [HuggingFace Agents Course - Function Calling Fine-tuning](https://huggingface.co/learn/agents-course/bonus-unit1/fine-tuning)
- [Qwen 3.5 Tool Calling Template Bug](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4)
- [Disable Thinking in Qwen 3.5 (Ollama)](https://github.com/ollama/ollama/issues/14617)
- [Qwen Function Calling Docs](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
