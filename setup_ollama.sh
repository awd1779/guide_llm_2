#!/bin/bash
# Setup Guide-LLM with Ollama — works on x86 (dev server) and ARM (Jetson/Go2)
#
# Usage:
#   bash setup_ollama.sh                         # default: 2b-q8
#   bash setup_ollama.sh all                     # all six variants
#   bash setup_ollama.sh 2b-q4 2b-q8 4b-q4       # specific variants
#
# Variants: 2b-q4, 2b-q8, 2b-f16, 4b-q4, 4b-q8, 4b-f16
#
# After setup:
#   ollama serve &
#   python3 scene_graph_agent_local.py --model guide-llm-2b-q8

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"

# Variant → "hf-repo|gguf-filename"
declare -A VARIANTS=(
    ["2b-q4"]="SMSong/guide-llm-2b-q4|qwen3.5-2b-toolcall-q4_k_m.gguf"
    ["2b-q8"]="SMSong/guide-llm-2b-q8|qwen3.5-2b-toolcall-q8.gguf"
    ["2b-f16"]="SMSong/guide-llm-2b-f16|qwen3.5-2b-toolcall-f16.gguf"
    ["4b-q4"]="SMSong/guide-llm-4b-q4|qwen3.5-4b-toolcall-q4_k_m.gguf"
    ["4b-q8"]="SMSong/guide-llm-4b-q8|qwen3.5-4b-toolcall-q8.gguf"
    ["4b-f16"]="SMSong/guide-llm-4b-f16|qwen3.5-4b-toolcall-f16.gguf"
)

# Parse args
if [ $# -eq 0 ]; then
    SELECTED=("2b-q8")
elif [ "$1" = "all" ]; then
    SELECTED=("2b-q4" "2b-q8" "2b-f16" "4b-q4")
else
    SELECTED=("$@")
fi

# Validate
for name in "${SELECTED[@]}"; do
    if [ -z "${VARIANTS[$name]}" ]; then
        echo "Unknown variant: $name"
        echo "Valid: ${!VARIANTS[@]}"
        exit 1
    fi
done

# 1. Install Ollama (if not present)
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed."
else
    echo "Ollama already installed: $(ollama --version)"
fi

mkdir -p "$MODEL_DIR"

# 2. Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# 3. Ensure HF client available (for downloads)
if ! command -v hf &>/dev/null && ! command -v huggingface-cli &>/dev/null; then
    pip install -q huggingface_hub 2>/dev/null
fi

# 4. Per-variant: download + modelfile + ollama create
write_modelfile() {
    local path="$1" gguf="$2"
    cat > "$path" <<EOF
FROM ./$gguf

TEMPLATE """{{- range \$i, \$_ := .Messages }}
{{- if eq .Role "system" }}
{{- if \$i }}<|im_start|>system
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
EOF
}

for name in "${SELECTED[@]}"; do
    entry="${VARIANTS[$name]}"
    repo="${entry%|*}"
    gguf_file="${entry#*|}"
    gguf_path="$MODEL_DIR/$gguf_file"
    modelfile="$MODEL_DIR/Modelfile-$name"
    ollama_name="guide-llm-$name"

    echo ""
    echo "=== $ollama_name ==="

    # Download GGUF from HF (skip if exists)
    if [ ! -f "$gguf_path" ]; then
        echo "Downloading $repo..."
        if command -v hf &>/dev/null; then
            hf download "$repo" "$gguf_file" --local-dir "$MODEL_DIR"
        else
            huggingface-cli download "$repo" "$gguf_file" --local-dir "$MODEL_DIR"
        fi
    else
        echo "GGUF exists: $gguf_path"
    fi

    # Write Modelfile if missing
    if [ ! -f "$modelfile" ]; then
        write_modelfile "$modelfile" "$gguf_file"
        echo "Wrote $modelfile"
    fi

    # Import into Ollama
    (cd "$MODEL_DIR" && ollama create "$ollama_name" -f "Modelfile-$name")
done

echo ""
echo "========================================="
echo "Setup complete — installed: ${SELECTED[*]}"
echo "========================================="
echo ""
echo "To run the agent:"
echo "  ollama serve &"
echo "  python3 scene_graph_agent_local.py --model guide-llm-2b-q8"
echo ""
echo "To benchmark installed variants:"
echo "  for m in ${SELECTED[@]}; do"
echo "    echo \"=== guide-llm-\$m ===\""
echo "    ollama run guide-llm-\$m --verbose 'Describe a navigation route in 50 words.'"
echo "  done"
echo ""
