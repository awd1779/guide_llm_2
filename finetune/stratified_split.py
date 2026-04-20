#!/usr/bin/env python3
"""
Stratified train/val split to ensure balanced scenario distribution.

Instead of sequential split (first 90% = train, last 10% = val),
this ensures each scenario is split 90/10 independently.
"""

import json
from pathlib import Path
from collections import defaultdict
import random

def stratified_split(input_file, output_train, output_val, train_ratio=0.9, seed=42):
    """Split JSONL file with stratification by scenario_type."""

    random.seed(seed)

    # Load all examples
    with open(input_file) as f:
        all_examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_examples)} total examples")

    # Group by scenario
    by_scenario = defaultdict(list)
    for ex in all_examples:
        scenario = ex.get('scenario_type', 'unknown')
        by_scenario[scenario].append(ex)

    print(f"\nScenario distribution (before split):")
    for scenario in sorted(by_scenario.keys()):
        count = len(by_scenario[scenario])
        pct = 100 * count / len(all_examples)
        print(f"  {scenario}: {count:3d} ({pct:5.1f}%)")

    # Split each scenario independently
    train_examples = []
    val_examples = []

    print(f"\nStratified split (90/10):")
    for scenario in sorted(by_scenario.keys()):
        examples = by_scenario[scenario]

        # Shuffle and split
        random.shuffle(examples)
        split_point = int(len(examples) * train_ratio)

        train_part = examples[:split_point]
        val_part = examples[split_point:]

        train_examples.extend(train_part)
        val_examples.extend(val_part)

        print(f"  {scenario}: {len(train_part)} train, {len(val_part)} val")

    # Write stratified files
    with open(output_train, 'w') as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + '\n')

    with open(output_val, 'w') as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\nOutput files:")
    print(f"  {output_train}: {len(train_examples)} examples")
    print(f"  {output_val}: {len(val_examples)} examples")

    # Verify distribution
    print(f"\nValidation set distribution (should be similar to training):")
    val_by_scenario = defaultdict(int)
    for ex in val_examples:
        scenario = ex.get('scenario_type', 'unknown')
        val_by_scenario[scenario] += 1

    for scenario in sorted(val_by_scenario.keys()):
        count = val_by_scenario[scenario]
        pct = 100 * count / len(val_examples)
        print(f"  {scenario}: {count:3d} ({pct:5.1f}%)")

    print(f"\n✅ Stratified split complete!")
    return train_examples, val_examples

if __name__ == "__main__":
    input_file = "finetune/data/train.jsonl"  # This will be overwritten

    # First, backup the old file
    import shutil
    backup_file = "finetune/data/train_old.jsonl"

    if Path(input_file).exists():
        print(f"Backing up {input_file} to {backup_file}")
        shutil.copy(input_file, backup_file)

    # Combine train and val into one file for re-splitting
    combined_file = "finetune/data/all_examples.jsonl"

    all_examples = []

    # Load old train
    if Path(backup_file).exists():
        with open(backup_file) as f:
            all_examples.extend([json.loads(line) for line in f if line.strip()])

    # Load old val
    val_file = "finetune/data/val.jsonl"
    if Path(val_file).exists():
        with open(val_file) as f:
            all_examples.extend([json.loads(line) for line in f if line.strip()])

    print(f"Combined {len(all_examples)} examples for re-splitting\n")

    # Write combined
    with open(combined_file, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + '\n')

    # Do stratified split
    train_examples, val_examples = stratified_split(
        combined_file,
        "finetune/data/train.jsonl",
        "finetune/data/val.jsonl"
    )

    print(f"\n{'='*70}")
    print("Stratified split created successfully!")
    print(f"{'='*70}")
