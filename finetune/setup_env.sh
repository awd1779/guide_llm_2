#!/bin/bash
# ============================================================
#  Guide-LLM Fine-tuning Environment Setup
#
#  Creates a conda env with Unsloth + Qwen 3.5 dependencies.
#  Tested on: NVIDIA A10G (23 GB), CUDA 12.4
#
#  Usage:
#    bash finetune/setup_env.sh
#
#  After setup:
#    conda activate guide-llm-finetune
#    python3 finetune/generate_dataset.py
#    python3 finetune/finetune_qwen.py --model-size 4b --epochs 3
# ============================================================

set -e

ENV_NAME="guide-llm-finetune"

echo "============================================================"
echo "  Guide-LLM Fine-tuning Environment Setup"
echo "============================================================"

# Check conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "WARNING: No GPU detected. Fine-tuning requires a GPU."
fi

# Remove existing env if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing '${ENV_NAME}' environment..."
    conda env remove -n ${ENV_NAME} -y
fi

# Create fresh conda env with Python 3.11
echo ""
echo "Creating conda environment '${ENV_NAME}' with Python 3.11..."
conda create -n ${ENV_NAME} python=3.11 -y

# Activate
echo ""
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

# Install Unsloth — handles PyTorch + CUDA automatically
echo ""
echo "Installing Unsloth (this will also install PyTorch + CUDA)..."
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo

# Install transformers v5+ (REQUIRED for Qwen 3.5)
echo ""
echo "Installing transformers v5+ and training dependencies..."
pip install "transformers>=5.0" trl peft datasets accelerate bitsandbytes

# Install additional utilities
echo ""
echo "Installing utilities..."
pip install sentencepiece protobuf openai numpy scipy

# Verify installation
echo ""
echo "============================================================"
echo "  Verifying installation..."
echo "============================================================"

python3 -c "
import torch
import transformers
import unsloth
import trl
import peft

print(f'  Python:       {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:      {torch.__version__}')
print(f'  CUDA:         {torch.version.cuda}')
print(f'  GPU:          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT AVAILABLE\"}')
print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB' if torch.cuda.is_available() else '')
print(f'  Transformers: {transformers.__version__}')
print(f'  Unsloth:      {unsloth.__version__}')
print(f'  TRL:          {trl.__version__}')
print(f'  PEFT:         {peft.__version__}')

# Check transformers version
major = int(transformers.__version__.split('.')[0])
if major < 5:
    print(f'  WARNING: transformers v5+ required for Qwen 3.5, got v{transformers.__version__}')
else:
    print(f'  Qwen 3.5:     COMPATIBLE')
"

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  To activate:   conda activate ${ENV_NAME}"
echo ""
echo "  Quick start:"
echo "    # 1. Generate dataset"
echo "    python3 finetune/generate_dataset.py"
echo ""
echo "    # 2. Fine-tune 4B (~10 GB VRAM, ~30 min)"
echo "    python3 finetune/finetune_qwen.py --model-size 4b --epochs 3"
echo ""
echo "    # 3. Fine-tune 9B (~22 GB VRAM, ~1-2 hrs)"
echo "    python3 finetune/finetune_qwen.py --model-size 9b --epochs 3"
echo ""
echo "    # 4. Import into Ollama"
echo "    ollama create guide-llm-4b -f finetune/output/qwen3.5-4b-toolcall/Modelfile"
echo ""
echo "    # 5. Test with your agent"
echo "    python3 scene_graph_agent_local.py --model guide-llm-4b"
echo "============================================================"
