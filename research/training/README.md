# Training: Model Fine-Tuning Documentation

Technical findings and validation results from fine-tuning Qwen 3.5 for tool calling.

---

## Documents

### [FINDINGS.md](FINDINGS.md)

Important technical discoveries for Qwen 3.5 fine-tuning:

- **Tool calling format**: Qwen 3.5 uses XML-based tool calls (native support)
- **Critical requirement**: Transformers v5+ (NOT v4)
- **Important warning**: Do NOT use QLoRA (4-bit) on Qwen 3.5
  - Use 16-bit LoRA instead due to quantization issues
- **LoRA configuration**: FastLanguageModel settings
- **Inference**: Running fine-tuned model with tool calling

**Read this if**: Setting up fine-tuning or troubleshooting training issues

---

### [VALIDATION_RESULTS.md](VALIDATION_RESULTS.md)

Validation metrics and performance results from trained model:

- Tool selection accuracy
- Parameter correctness
- Response quality metrics
- Comparison vs. baseline models

**Read this if**: Evaluating model performance or understanding training results

---

## Related Code

The actual training code lives in `finetune/` folder:
- `finetune/generate_dataset.py` - Dataset generation
- `finetune/finetune_qwen.py` - Training script
- `finetune/validate_model.py` - Validation utilities

---

## How to Use

**If you're fine-tuning the model**:
1. Read FINDINGS.md (critical for avoiding issues)
2. Setup training per FINDINGS.md requirements
3. Run finetune/finetune_qwen.py
4. Check VALIDATION_RESULTS.md for expected metrics

**If you're evaluating performance**:
1. Read VALIDATION_RESULTS.md
2. Compare your results against baseline metrics

**If you're debugging**:
1. Check FINDINGS.md for known issues (e.g., QLoRA warnings)
2. See VALIDATION_RESULTS.md for expected behavior

---

**Version**: 2026-03-23
**Status**: Documentation from completed fine-tuning work
