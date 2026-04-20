# Guide-LLM Dataset Generation & Training Pipeline

Complete pipeline with quality validation based on research from Amazon & EMNLP 2024.

## Full Pipeline (New)

```bash
# Step 1: Generate dataset with fixed "already in" patterns (30-45 min)
python3 finetune/generate_templated.py \
  --scenario-a 300 \
  --scenario-b 300 \
  --scenario-c 300 \
  --scenario-d 300 \
  --scenario-e 300 \
  --scenario-f 300 \
  --scenario-g 300 \
  --scenario-h 300 \
  --workers 10

# Step 2: Fix tool call format (5 min)
python3 finetune/fix_tool_calls.py

# Step 3: Add tools field for Qwen (5 min)
python3 finetune/restructure_for_qwen.py

# Step 4: VALIDATE AND FILTER (NEW!) (10-15 min)
python3 finetune/validate_dataset.py \
  --input finetune/data/train.jsonl \
  --output finetune/data/train_filtered.jsonl \
  --threshold 0.7

# Step 5: Replace original with filtered
mv finetune/data/train.jsonl finetune/data/train_unfiltered.jsonl
mv finetune/data/train_filtered.jsonl finetune/data/train.jsonl

# Step 6: Stratified split (val set uses filtered data) (5 min)
python3 finetune/stratified_split.py

# Step 7: Train both models (3-4 hours)
bash finetune/train_all.sh

# Step 8: Validate results (2-3 hours)
bash finetune/validate_all.sh
```

## What Each Step Does

### 1. Generate (generate_templated.py)
- Uses Claude API to fill template slots
- Creates 2400 examples (300 per scenario)
- ~200 API calls with batching (5 per call)
- **Output**: `finetune/data/train.jsonl`, `finetune/data/val.jsonl`

### 2. Fix Tool Calls (fix_tool_calls.py)
- Converts tool arguments from strings to dicts
- Ensures tool responses have proper format
- **Output**: Updated JSONL files

### 3. Restructure for Qwen (restructure_for_qwen.py)
- Adds `tools` field to each example
- Formats for Qwen chat template
- **Output**: Updated JSONL files with tools field

### 4. VALIDATE & FILTER (validate_dataset.py) ⭐ NEW
Applies research-backed quality metrics:

**Validity Checks:**
- ✓ Tool call format (id, type, function, arguments)
- ✓ Tool responses have required fields
- ✓ Tool sequences match expected patterns

**Semantic Checks:**
- ✓ No bad patterns (e.g., "already in")
- ✓ Response length reasonable (<100 words)
- ✓ Sufficient diversity (avoid repetition)

**Output:**
- Filtered dataset (keep only score ≥ 0.7)
- Quality report showing issues found
- Score distribution

### 5. Replace & Split (stratified_split.py)
- Uses **filtered** training data
- Stratified 90/10 split by scenario
- **Output**: train.jsonl, val.jsonl (clean split)

### 6. Train (finetune_qwen.py + train_all.sh)
- LoRA fine-tuning on both 2B and 4B
- 5 epochs (vs default 3)
- Learning rate: 1e-4
- Batch size: 1, gradient accumulation: 4

### 7. Validate (validate_multiturn.py + validate_all.sh)
- Tests on validation set
- Compares fine-tuned vs vanilla baseline
- Reports match scores and tool accuracy

## Expected Results

### Before Validation
- 2400 generated examples
- Some with quality issues (~30-40%)
- Expected: Lower tool accuracy

### After Validation
- ~1500-1800 high-quality examples (keep 60-75%)
- All bad patterns removed
- Expected: **85%+ average match, 90%+ tool accuracy for 4B**

## Quick Start (Single Command)

```bash
#!/bin/bash
# Run entire pipeline
set -e

echo "Step 1: Generate..."
python3 finetune/generate_templated.py --scenario-a 300 --scenario-b 300 --scenario-c 300 --scenario-d 300 --scenario-e 300 --scenario-f 300 --scenario-g 300 --scenario-h 300 --workers 10

echo "Step 2: Fix tool calls..."
python3 finetune/fix_tool_calls.py

echo "Step 3: Restructure..."
python3 finetune/restructure_for_qwen.py

echo "Step 4: Validate & Filter..."
python3 finetune/validate_dataset.py --input finetune/data/train.jsonl --output finetune/data/train_filtered.jsonl --threshold 0.7

echo "Step 5: Replace with filtered..."
mv finetune/data/train.jsonl finetune/data/train_unfiltered.jsonl
mv finetune/data/train_filtered.jsonl finetune/data/train.jsonl

echo "Step 6: Stratified split..."
python3 finetune/stratified_split.py

echo "Step 7: Train..."
bash finetune/train_all.sh

echo "Step 8: Validate..."
bash finetune/validate_all.sh

echo "✓ Complete!"
```

## Validation Script Options

```bash
# Default: threshold 0.7, keep ~70% of data
python3 finetune/validate_dataset.py

# Strict: threshold 0.8, keep only best ~50%
python3 finetune/validate_dataset.py --threshold 0.8

# Lenient: threshold 0.6, keep ~80%
python3 finetune/validate_dataset.py --threshold 0.6

# Custom input/output
python3 finetune/validate_dataset.py \
  --input custom_data.jsonl \
  --output custom_data_filtered.jsonl \
  --threshold 0.75
```

## Troubleshooting

### "Too many examples filtered out"
- Reduce threshold: `--threshold 0.6`
- Check if generation is producing bad examples
- Review the issue report to fix generation prompts

### "Still seeing bad patterns"
- Increase threshold: `--threshold 0.8`
- Check that generate_templated.py includes the constraint fixes
- Manually review filtered examples

### "Performance still low"
- Check validation report for most common issues
- Add more specific constraints to generation prompts
- Consider manually annotating a small subset for comparison

## References

Research papers this pipeline is based on:
- Amazon Science: "Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs" (EMNLP 2024)
- Survey: "The LLM Data Auditor" (OpenReview)
- Key finding: **Filtering 30-40% of data = 15-20% performance improvement**

