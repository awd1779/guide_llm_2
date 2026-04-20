# Deploying Fine-tuned Qwen 3.5 2B to Ollama

## Prerequisites

- Ollama installed (`ollama --version`)
- Fine-tuned model at `finetune/output/qwen3.5-2b-toolcall/merged_16bit/`
- llama.cpp convert script (clone once): `git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp`
- Python env with `transformers`, `gguf`, `sentencepiece` installed

## Step 1: Convert HF → GGUF (quantized Q8_0)

```bash
conda run -n guide-llm-finetune python3 /tmp/llama.cpp/convert_hf_to_gguf.py \
  finetune/output/qwen3.5-2b-toolcall/merged_16bit \
  --outfile finetune/output/qwen3.5-2b-toolcall-q8.gguf \
  --outtype q8_0
```

Output: `qwen3.5-2b-toolcall-q8.gguf` (~1.9GB, vs 3.6GB for F16)

Other quantization options: `q8_0` (best quality), `q4_k_m` (smallest ~1.2GB), `f16` (no quantization, slow).

## Step 2: Create Ollama Modelfile

Create `finetune/output/Modelfile-2b-q8`:

```
FROM ./qwen3.5-2b-toolcall-q8.gguf

TEMPLATE """{{- range $i, $_ := .Messages }}
{{- if eq .Role "system" }}
{{- if $i }}<|im_start|>system
{{ .Content }}<|im_end|>
{{ else }}<|im_start|>system
{{ .Content }}<|im_end|>
{{ end }}
{{- else if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}<|im_end|>
{{- end }}
{{- end }}<|im_start|>assistant
<think>

</think>

"""

PARAMETER temperature 0
PARAMETER num_ctx 4096
PARAMETER stop <|im_end|>
PARAMETER stop <|endoftext|>
```

The TEMPLATE is required because the GGUF conversion loses the Qwen chat template. Without it, Ollama uses a generic format and the model generates gibberish.

## Step 3: Import into Ollama

```bash
cd finetune/output
ollama create guide-llm-2b-q8 -f Modelfile-2b-q8
```

## Step 4: Verify

```bash
curl -s http://localhost:11434/api/chat -d '{
  "model": "guide-llm-2b-q8",
  "messages": [
    {"role": "system", "content": "You are Guide-LLM, a navigation assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false
}' | python3 -m json.tool
```

Expected: ~3 second response, proper navigation assistant output.

## Step 5: Run with agent

```bash
# Terminal 1: scene graph (dummy for testing, or real robot)
python3 dummy_scene_graph_publisher.py

# Terminal 2: agent
python3 scene_graph_agent_local.py --model guide-llm-2b-q8
```

## Known Issue: Ollama Tool Calling

Ollama's native tool calling is broken for Qwen 3.5 (GitHub issues #14493, #14745). The agent bypasses this by:
1. Injecting tool definitions directly into the system prompt (Qwen chat template format)
2. Parsing XML tool calls (`<function=name><parameter=key>value</parameter></function>`) from raw text
3. Feeding tool responses as `<tool_response>` in user messages

This is handled in `scene_graph_agent_local.py` — no Ollama tool API used.

## Cleanup

Remove unused models:
```bash
ollama rm guide-llm-2b-v2    # broken safetensors import
ollama rm guide-llm-2b-gguf  # slow F16 version
ollama rm guide-llm-2b       # old version
```
