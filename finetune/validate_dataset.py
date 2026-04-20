#!/usr/bin/env python3
"""
Validate and filter synthetic training data using research-backed quality metrics.

Research sources:
- Amazon: "Quality Matters: Evaluating Synthetic Data for Tool-Using LLMs"
- Survey: "The LLM Data Auditor"

Usage:
    python3 finetune/validate_dataset.py
    python3 finetune/validate_dataset.py --input finetune/data/train.jsonl --threshold 0.7
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List

# =============================================================================
# Quality Metrics
# =============================================================================

EXPECTED_TOOL_SEQUENCES = {
    "scenario_a": ["list_all", "distance_to", "navigate_to"],
    "scenario_a_plus": ["list_all", "distance_to", "navigate_to", "get_navigation_status"],
    "scenario_b": ["find_nearest", "distance_to", "navigate_to"],
    "scenario_c": ["get_robot_pose", "describe_surroundings"],
    "scenario_d": ["get_robot_pose", "orient_me"],
    "scenario_e": ["distance_to"],
    "scenario_f": ["list_all"],
    "scenario_g": [],  # No tools
    "scenario_h": ["get_robot_pose", "distance_to", "navigate_to"],
}

REQUIRED_TOOL_RESPONSE_FIELDS = {
    "list_all": {"zones", "objects"},
    "distance_to": {"destination", "distance_m", "direction"},
    "navigate_to": {"status", "destination"},
    "find_nearest": {"label", "distance_m", "direction"},
    "get_robot_pose": {"x", "y", "yaw_deg", "frame", "current_zone"},
    "describe_surroundings": {"current_zone", "radius_m", "nearby_objects"},
    "orient_me": {"target", "direction_from_current", "turn_degrees"},
    "get_navigation_status": {"status", "destination"},
    "cancel_navigation": {"status", "message"},
}

BAD_PATTERNS = [
    "already in",
    "already at",
    "no navigation needed",
    "no need to navigate",
    "don't need to",
    "don't have to",
]

# =============================================================================
# Validation Functions
# =============================================================================

def validate_tool_call_format(example: Dict) -> Tuple[bool, str]:
    """Check if tool calls have proper format."""
    messages = example.get("messages", [])

    for msg_idx, msg in enumerate(messages):
        if msg.get("tool_calls"):
            for tc_idx, tc in enumerate(msg["tool_calls"]):
                # Check required fields
                if "id" not in tc:
                    return False, f"msg[{msg_idx}].tool_call[{tc_idx}] missing 'id'"

                if "type" not in tc:
                    return False, f"msg[{msg_idx}].tool_call[{tc_idx}] missing 'type'"

                func = tc.get("function", {})
                if not func.get("name"):
                    return False, f"msg[{msg_idx}].tool_call[{tc_idx}] missing function name"

                # Check arguments exist and are not empty
                args = func.get("arguments")
                if args is None:
                    return False, f"msg[{msg_idx}].tool_call[{tc_idx}] missing arguments"

                if isinstance(args, str) and args.strip() in ["", "{}", "null"]:
                    return False, f"msg[{msg_idx}].tool_call[{tc_idx}] has empty arguments"

    return True, "OK"


def validate_tool_responses(example: Dict) -> Tuple[bool, str]:
    """Check if tool responses have realistic, complete fields."""
    messages = example.get("messages", [])

    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "tool":
            try:
                response = json.loads(msg.get("content", "{}"))
            except json.JSONDecodeError:
                return False, f"msg[{msg_idx}] tool response not valid JSON"

            # Find what tool this response is for (look backwards for tool_call_id)
            tool_call_id = msg.get("tool_call_id", "unknown")

            # Try to infer tool name from context
            tool_name = None
            for prev_msg in reversed(messages[:msg_idx]):
                if prev_msg.get("tool_calls"):
                    for tc in prev_msg["tool_calls"]:
                        if tc.get("id") == tool_call_id:
                            tool_name = tc.get("function", {}).get("name")
                            break

            if not tool_name:
                continue  # Can't validate without knowing tool name

            # Check required fields for this tool
            if tool_name in REQUIRED_TOOL_RESPONSE_FIELDS:
                required = REQUIRED_TOOL_RESPONSE_FIELDS[tool_name]
                missing = required - set(response.keys())
                if missing:
                    return False, f"msg[{msg_idx}] {tool_name} response missing fields: {missing}"

    return True, "OK"


def validate_tool_sequence(example: Dict) -> Tuple[bool, str]:
    """Check if tool calls follow expected sequence for scenario."""
    scenario = example.get("scenario_type", "unknown")
    messages = example.get("messages", [])

    # Extract actual tool sequence
    actual_sequence = []
    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_name = tc.get("function", {}).get("name")
                if tool_name:
                    actual_sequence.append(tool_name)

    # Get expected sequence
    if scenario not in EXPECTED_TOOL_SEQUENCES:
        return True, "OK"  # Unknown scenario, can't validate

    expected = EXPECTED_TOOL_SEQUENCES[scenario]

    # Check sequence matches (order matters, length must match)
    if actual_sequence != expected:
        return False, f"Sequence mismatch: expected {expected}, got {actual_sequence}"

    return True, "OK"


def validate_no_bad_patterns(example: Dict) -> Tuple[bool, str]:
    """Check for known bad response patterns."""
    messages = example.get("messages", [])

    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"].lower()

            for pattern in BAD_PATTERNS:
                if pattern in content:
                    return False, f"msg[{msg_idx}] contains bad pattern: '{pattern}'"

    return True, "OK"


def validate_response_length(example: Dict) -> Tuple[bool, str]:
    """Check responses aren't too long (avoid verbose generation)."""
    messages = example.get("messages", [])
    max_words = 100

    for msg_idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            word_count = len(msg["content"].split())
            if word_count > max_words:
                return False, f"msg[{msg_idx}] too long ({word_count} words, max {max_words})"

    return True, "OK"


def validate_diversity(example: Dict, all_examples: List[Dict]) -> Tuple[bool, str]:
    """Diversity check removed - scenarios with few variations are valid."""
    return True, "OK"


# =============================================================================
# Scoring System
# =============================================================================

def score_example(example: Dict, all_examples: List[Dict]) -> float:
    """Score example 0-1 based on quality metrics."""
    score = 1.0
    penalties = []

    # Validity checks (critical - fail fast)
    valid, msg = validate_tool_call_format(example)
    if not valid:
        return 0.0, f"FORMAT: {msg}"

    valid, msg = validate_tool_responses(example)
    if not valid:
        return 0.0, f"RESPONSES: {msg}"

    valid, msg = validate_tool_sequence(example)
    if not valid:
        score *= 0.7  # Penalize but don't fail
        penalties.append(f"SEQUENCE: {msg}")

    # Semantic checks
    valid, msg = validate_no_bad_patterns(example)
    if not valid:
        score *= 0.5
        penalties.append(f"PATTERNS: {msg}")

    valid, msg = validate_response_length(example)
    if not valid:
        score *= 0.7
        penalties.append(f"LENGTH: {msg}")


    return max(0.0, score), "; ".join(penalties) if penalties else "OK"


# =============================================================================
# Main Validation Pipeline
# =============================================================================

def validate_and_filter(input_file: str, output_file: str, threshold: float = 0.7):
    """Load, validate, score, and filter dataset."""

    print("="*80)
    print("SYNTHETIC DATA QUALITY VALIDATION")
    print("="*80)
    print(f"\nInput: {input_file}")
    print(f"Threshold: {threshold}")
    print(f"Output: {output_file}\n")

    # Load examples
    print("Loading examples...")
    with open(input_file) as f:
        all_examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(all_examples)} examples\n")

    # Score all examples
    print("Scoring examples...")
    scored = []
    issue_counts = defaultdict(int)

    for idx, ex in enumerate(all_examples):
        score, issues = score_example(ex, all_examples)
        scored.append((ex, score, issues))

        if ":" in issues:
            issue_type = issues.split(":")[0]
            issue_counts[issue_type] += 1

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(all_examples)}")

    print(f"  {len(all_examples)}/{len(all_examples)} ✓\n")

    # Filter by threshold
    print(f"Filtering (threshold: {threshold})...")
    filtered = [ex for ex, score, _ in scored if score >= threshold]

    kept_pct = 100 * len(filtered) / len(all_examples)
    print(f"Kept: {len(filtered)}/{len(all_examples)} ({kept_pct:.1f}%)\n")

    # Statistics
    print("Issues Found:")
    print("-" * 80)
    for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(all_examples)
        print(f"  {issue_type:<20s}: {count:4d} ({pct:5.1f}%)")

    print("\nScore Distribution:")
    print("-" * 80)
    score_ranges = [
        (0.9, 1.0, "Excellent"),
        (0.7, 0.9, "Good"),
        (0.5, 0.7, "Fair"),
        (0.0, 0.5, "Poor"),
    ]
    for low, high, label in score_ranges:
        count = sum(1 for _, score, _ in scored if low <= score < high)
        pct = 100 * count / len(all_examples)
        print(f"  {label:<12s} ({low:.1f}-{high:.1f}): {count:4d} ({pct:5.1f}%)")

    # Save filtered
    print(f"\nSaving to {output_file}...")
    with open(output_file, "w") as f:
        for ex in filtered:
            f.write(json.dumps(ex) + "\n")

    print(f"✓ Saved {len(filtered)} examples\n")

    print("="*80)
    print(f"SUMMARY: Filtered to {kept_pct:.1f}% high-quality examples")
    print("="*80)

    return len(filtered), len(all_examples)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and filter synthetic dataset")
    parser.add_argument(
        "--input",
        default="finetune/data/train.jsonl",
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        default="finetune/data/train_filtered.jsonl",
        help="Output filtered JSONL file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Quality score threshold (0-1)",
    )
    args = parser.parse_args()

    validate_and_filter(args.input, args.output, args.threshold)
