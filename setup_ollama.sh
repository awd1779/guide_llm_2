#!/bin/bash
# Setup Guide-LLM with Ollama — works on x86 (dev server) and ARM (Jetson/Go2)
#
# Usage:
#   bash setup_ollama.sh
#
# After setup:
#   ollama serve &
#   python3 scene_graph_agent_local.py --model guide-llm-2b-q8

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
GGUF="$MODEL_DIR/qwen3.5-2b-toolcall-q8.gguf"
MODELFILE="$MODEL_DIR/Modelfile-2b-q8"
HF_REPO="SMSong/guide-llm-2b-q8"

# 1. Install Ollama (if not present)
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installed."
else
    echo "Ollama already installed: $(ollama --version)"
fi

# 2. Download model from HuggingFace (if not present)
if [ ! -f "$GGUF" ]; then
    echo "Downloading model from HuggingFace ($HF_REPO)..."
    pip install -q huggingface_hub 2>/dev/null

    # Try 'hf' first (newer), fall back to 'huggingface-cli'
    if command -v hf &>/dev/null; then
        hf download "$HF_REPO" qwen3.5-2b-toolcall-q8.gguf --local-dir "$MODEL_DIR"
    else
        huggingface-cli download "$HF_REPO" qwen3.5-2b-toolcall-q8.gguf --local-dir "$MODEL_DIR"
    fi
    echo "Model downloaded to $GGUF"
else
    echo "Model already exists: $GGUF"
fi

# 3. Check Modelfile exists
if [ ! -f "$MODELFILE" ]; then
    echo "ERROR: Modelfile not found at $MODELFILE"
    echo "Make sure you cloned the full repo."
    exit 1
fi

# 4. Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

# 5. Import model into Ollama
echo "Importing model into Ollama..."
cd "$MODEL_DIR" && ollama create guide-llm-2b-q8 -f Modelfile-2b-q8

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run:"
echo "  ollama serve &    # if not already running"
echo "  python3 scene_graph_agent_local.py --model guide-llm-2b-q8"
echo ""
