#!/bin/bash
set -e

LOG_DIR="finetune/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/finetune_${TIMESTAMP}.log"

echo "Starting sequential fine-tuning..."
echo "Logs: $LOG_FILE"
echo ""

{
    echo "==============================================="
    echo "Sequential Fine-tuning: 2B → 4B"
    echo "Started: $(date)"
    echo "==============================================="
    echo ""

    # Fine-tune 2B (Unsloth-recommended parameters)
    echo ">>> STARTING 2B FINE-TUNING <<<"
    echo "Start time: $(date)"
    python3 finetune/finetune_qwen.py --model-size 2b --epochs 3 --lr 1e-4 2>&1

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 2B FINE-TUNING COMPLETE"
        echo "Completion time: $(date)"
        echo ""
    else
        echo "❌ 2B FINE-TUNING FAILED"
        exit 1
    fi

    # Fine-tune 4B (Unsloth-recommended parameters)
    echo ""
    echo ">>> STARTING 4B FINE-TUNING <<<"
    echo "Start time: $(date)"
    python3 finetune/finetune_qwen.py --model-size 4b --epochs 3 --lr 1e-4 2>&1

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ 4B FINE-TUNING COMPLETE"
        echo "Completion time: $(date)"
        echo ""
    else
        echo "❌ 4B FINE-TUNING FAILED"
        exit 1
    fi

    echo "==============================================="
    echo "✅ ALL FINE-TUNING COMPLETE!"
    echo "Finished: $(date)"
    echo "==============================================="
    echo ""
    echo "Models ready:"
    echo "  - 2B: finetune/output/qwen3.5-2b-toolcall"
    echo "  - 4B: finetune/output/qwen3.5-4b-toolcall"

} | tee -a "$LOG_FILE"

# Print summary
echo ""
echo "Log file: $LOG_FILE"
