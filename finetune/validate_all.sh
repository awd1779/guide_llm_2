#!/bin/bash
set -e

# Comprehensive validation script for both 2B and 4B models
# Tests fine-tuned models on multi-turn conversations

VALIDATION_DIR="finetune/validation"
mkdir -p "$VALIDATION_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$VALIDATION_DIR/validation_${TIMESTAMP}.log"

echo "=================================================="
echo "  Guide-LLM Model Validation (2B + 4B)"
echo "  Started: $(date)"
echo "=================================================="
echo ""

{
    echo "Validation Log: $(date)"
    echo "=================================================="
    echo ""

    # Check if models exist
    echo "Checking for trained models..."
    for model_size in 2b 4b; do
        model_path="finetune/output/qwen3.5-${model_size}-toolcall/merged_16bit"
        if [ -d "$model_path" ]; then
            echo "✓ Found ${model_size^} model at $model_path"
        else
            echo "✗ Missing ${model_size^} model at $model_path"
            echo "  Run: bash finetune/train_all.sh"
            exit 1
        fi
    done
    echo ""

    # Validate both models
    echo "Running validation on both models..."
    echo "========================================"
    echo ""

    python3 finetune/validate_multiturn.py --model both --save-results

    echo ""
    echo "========================================"
    echo "Fine-tuned model validation complete!"
    echo ""

    # Test vanilla baseline models
    echo "Testing vanilla (untrained) baseline models..."
    echo "========================================"
    echo ""

    python3 finetune/validate_baseline.py --model both --save-results

    echo ""
    echo "========================================"
    echo "Validation complete!"
    echo ""

    # Parse and compare results
    echo "COMPARISON: Fine-Tuned vs Vanilla Baseline"
    echo "========================================"
    echo ""

    python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

print(f"{'Model':<12} {'Avg Match':<12} {'High Match %':<15} {'Tool Accuracy %':<18}")
print("-" * 60)

# Fine-tuned results
for model in ['2b', '4b']:
    result_file = f"finetune/validation_results_{model}.jsonl"
    if Path(result_file).exists():
        with open(result_file) as f:
            examples = [json.loads(line) for line in f]

        # Aggregate scores
        all_scores = []
        tool_correct = 0
        tool_total = 0

        for ex in examples:
            for result in ex.get('results', []):
                if not result.get('error'):
                    all_scores.append(result.get('match', 0))
                    if result.get('expected_tools'):
                        tool_total += 1
                        if result.get('tool_match_correct'):
                            tool_correct += 1

        if all_scores:
            avg_match = sum(all_scores) / len(all_scores)
            high_match_pct = sum(1 for s in all_scores if s > 0.7) / len(all_scores) * 100
            tool_accuracy = (tool_correct / tool_total * 100) if tool_total > 0 else 0

            label = f"{model}(ft)"
            print(f"{label:<12} {avg_match:.3f}        {high_match_pct:>6.1f}%         {tool_accuracy:>6.1f}%")

# Vanilla baseline results
for model in ['2b', '4b']:
    result_file = f"finetune/validation_baseline_{model}.jsonl"
    if Path(result_file).exists():
        with open(result_file) as f:
            examples = [json.loads(line) for line in f]

        # Aggregate scores
        all_scores = []
        tool_correct = 0
        tool_total = 0

        for ex in examples:
            for result in ex.get('results', []):
                if not result.get('error'):
                    all_scores.append(result.get('match', 0))
                    if result.get('expected_tools'):
                        tool_total += 1
                        if result.get('tool_match_correct'):
                            tool_correct += 1

        if all_scores:
            avg_match = sum(all_scores) / len(all_scores)
            high_match_pct = sum(1 for s in all_scores if s > 0.7) / len(all_scores) * 100
            tool_accuracy = (tool_correct / tool_total * 100) if tool_total > 0 else 0

            label = f"{model}(vanilla)"
            print(f"{label:<12} {avg_match:.3f}        {high_match_pct:>6.1f}%         {tool_accuracy:>6.1f}%")

PYTHON_SCRIPT

    echo ""
    echo "========================================"
    echo "Results saved:"
    echo "  Fine-tuned models:"
    echo "    - finetune/validation_results_2b.jsonl"
    echo "    - finetune/validation_results_4b.jsonl"
    echo "  Vanilla baseline:"
    echo "    - finetune/validation_baseline_2b.jsonl"
    echo "    - finetune/validation_baseline_4b.jsonl"
    echo ""
    echo "Full log: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  - Review match scores (target: >0.80)"
    echo "  - Check tool accuracy (target: >90%)"
    echo "  - Compare fine-tuned vs vanilla baseline (should be significantly better)"
    echo ""
    echo "Completed: $(date)"

} | tee -a "$LOG_FILE"

echo ""
echo "✅ Validation complete!"
