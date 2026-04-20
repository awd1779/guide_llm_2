#!/usr/bin/env python3
"""
Validate models on multi-turn conversations from validation dataset.
Tests if models can continue realistic navigation conversations.

Usage:
    python3 finetune/validate_multiturn.py
    python3 finetune/validate_multiturn.py --model 2b
    python3 finetune/validate_multiturn.py --num-examples 10
"""

import json
import argparse
import re
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import sys

# ============================================================================
# Config
# ============================================================================

MODELS = {
    "2b": "finetune/output/qwen3.5-2b-toolcall/merged_16bit",
    "4b": "finetune/output/qwen3.5-4b-toolcall/merged_16bit",
}

VALID_TOOLS = {
    "get_robot_pose", "describe_surroundings", "list_all", "query",
    "find_nearest", "distance_to", "describe_route", "navigate_to",
    "orient_me", "cancel_navigation", "get_navigation_status", "replan_route",
}

# ============================================================================
# Validation Logic
# ============================================================================

def similarity_ratio(a, b):
    """Calculate similarity between two strings (0-1)."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def extract_tool_calls_from_output(text):
    """Extract tool call names from model output (XML format)."""
    tool_calls = []

    # Pattern: <function=tool_name> or function="tool_name"
    patterns = [
        r'<function=(\w+)>',
        r'<function\s*=\s*["\']?(\w+)["\']?',
        r'"function":\s*"(\w+)"',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        tool_calls.extend(matches)

    return list(set(tool_calls))

def check_tool_validity(tool_names):
    """Check if extracted tool names are valid."""
    invalid = [t for t in tool_names if t not in VALID_TOOLS]
    return len(invalid) == 0

def has_distance_info(messages_history):
    """Check if conversation history already contains distance information."""
    history_text = "\n".join([
        msg.get("content", "") or ""
        for msg in messages_history
    ])

    # Check for explicit distance in tool results or responses
    distance_keywords = ["metre", "meter", "metres", "meters", "away", "nearby", "right here", "short walk"]
    return any(keyword in history_text.lower() for keyword in distance_keywords)

def is_valid_alternative_tool(expected_tools, generated_tools, messages_history):
    """Check if generated tools are acceptable alternatives to expected tools."""
    if not expected_tools or not generated_tools:
        return False

    expected = set(expected_tools)
    generated = set(generated_tools)

    # If exact match or partial match, not an "alternative"
    if expected == generated or (expected & generated):
        return False

    # Special case: expected distance_to but got navigate_to
    # This is acceptable if distance info is already in conversation
    if expected == {"distance_to"} and generated == {"navigate_to"}:
        return has_distance_info(messages_history)

    # Special case: expected distance_to but got find_nearest
    # find_nearest returns distance info, so it's acceptable
    if expected == {"distance_to"} and generated == {"find_nearest"}:
        return True

    return False

def extract_response_type(text):
    """Determine if response is tool call, natural language, or mixed."""
    has_tool = '<tool_call>' in text or '<function=' in text
    has_language = len(re.sub(r'<[^>]+>', '', text).strip()) > 20

    if has_tool and has_language:
        return "mixed"
    elif has_tool:
        return "tool_call"
    elif has_language:
        return "language"
    else:
        return "empty"

def collect_validation_turns(examples, tools):
    """Collect all turns to validate from all examples.

    Returns list of (example_idx, turn_idx, context_text, expected_response, expected_tools)
    """
    turns = []

    for ex_idx, example in enumerate(examples):
        messages = example["messages"]

        for i in range(len(messages)):
            if messages[i]["role"] != "assistant":
                continue

            # Skip if this is the last message
            if i == len(messages) - 1:
                continue

            # Build context
            context_messages = messages[:i]
            expected_response = messages[i].get("content")
            expected_tools = messages[i].get("tool_calls")

            turns.append((ex_idx, i, context_messages, expected_response, expected_tools, tools))

    return turns


def validate_batch(model, tokenizer, turns_batch):
    """Validate a batch of turns using single batch generation.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer (should have padding_side='left')
        turns_batch: list of (example_idx, turn_idx, context_messages, expected_response, expected_tools, tools)

    Returns:
        list of result dicts
    """
    results_batch = []

    # Prepare all texts for this batch
    texts = []
    metadata = []

    for ex_idx, turn_idx, context_messages, expected_response, expected_tools, tools in turns_batch:
        try:
            text = tokenizer.apply_chat_template(
                context_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)
            metadata.append({
                "ex_idx": ex_idx,
                "turn_idx": turn_idx,
                "expected_response": expected_response,
                "expected_tools": expected_tools,
                "context_messages": context_messages,
            })
        except Exception as e:
            results_batch.append({
                "ex_idx": ex_idx,
                "turn_idx": turn_idx,
                "error": "template_error",
                "match": 0,
                "response_type": "error",
            })

    if not texts:
        return results_batch

    # Batch tokenize with LEFT padding (required for decoder-only models)
    try:
        # Ensure left padding for proper generation
        tokenizer.padding_side = 'left'

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        # Batch generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        # Decode each output - skip only input tokens (not padding)
        # With left-padding, input_ids are [padded_tokens, input_tokens, eos]
        # We need to skip only the actual input tokens
        for batch_idx, output in enumerate(outputs):
            # Get the actual input length for this sequence (excluding padding)
            input_ids = inputs['input_ids'][batch_idx]
            # Count non-padding tokens at the end
            actual_input_len = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
            if len(actual_input_len) > 0:
                # Last position is where input ends
                actual_input_len = actual_input_len[-1].item() + 1
            else:
                actual_input_len = inputs['input_ids'].shape[1]

            generated = tokenizer.decode(
                output[actual_input_len:],
                skip_special_tokens=True
            )

            meta = metadata[batch_idx]
            response_type = extract_response_type(generated)
            generated_tools = extract_tool_calls_from_output(generated)

            # Extract expected tool names
            expected_tool_names = set()
            if meta["expected_tools"]:
                for tc in meta["expected_tools"]:
                    if isinstance(tc, dict):
                        func_name = tc.get("function", {}).get("name")
                        if func_name:
                            expected_tool_names.add(func_name)

            # Calculate match score
            if meta["expected_response"]:
                gen_clean = re.sub(r'<[^>]+>', '', generated).strip()
                exp_clean = str(meta["expected_response"]).strip()
                match = similarity_ratio(gen_clean, exp_clean)
                tool_match_correct = False
            else:
                if expected_tool_names:
                    if set(generated_tools) == expected_tool_names:
                        match = 1.0
                        tool_match_correct = True
                    elif set(generated_tools) & expected_tool_names:
                        match = 0.5
                        tool_match_correct = False
                    elif is_valid_alternative_tool(list(expected_tool_names), generated_tools, meta["context_messages"]):
                        match = 0.7
                        tool_match_correct = False
                    else:
                        match = 0.0
                        tool_match_correct = False
                else:
                    match = 0.5
                    tool_match_correct = False

            results_batch.append({
                "ex_idx": meta["ex_idx"],
                "turn_idx": meta["turn_idx"],
                "error": None,
                "match": match,
                "response_type": response_type,
                "expected_language": bool(meta["expected_response"]),
                "expected_tools": list(expected_tool_names),
                "generated_tools": generated_tools,
                "tool_match_correct": tool_match_correct,
                "generated": generated[:100],
            })

    except Exception as e:
        # If batch generation fails, mark all as errors
        for meta in metadata:
            results_batch.append({
                "ex_idx": meta["ex_idx"],
                "turn_idx": meta["turn_idx"],
                "error": "generation_error",
                "match": 0,
                "response_type": "error",
            })

    return results_batch


def validate_on_example(model, tokenizer, example):
    """Validate model on a multi-turn conversation example."""
    messages = example["messages"]
    tools = example.get("tools", [])

    results = []

    # Test at different conversation points (after every assistant message)
    for i in range(len(messages)):
        if messages[i]["role"] != "assistant":
            continue

        # Skip if this is the last message
        if i == len(messages) - 1:
            continue

        # Get expected response
        expected_response = messages[i].get("content")
        expected_tools = messages[i].get("tool_calls")

        # Build context up to this point (exclude this assistant message)
        context_messages = messages[:i]

        # Format with chat template (thinking disabled for clean tool-call output)
        try:
            text = tokenizer.apply_chat_template(
                context_messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception as e:
            results.append({
                "turn": i,
                "error": f"template_error",
                "match": 0,
                "response_type": "error",
            })
            continue

        # Generate
        try:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,
                )
            generated = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
        except Exception as e:
            results.append({
                "turn": i,
                "error": f"generation_error",
                "match": 0,
                "response_type": "error",
            })
            continue

        # Compare with expected
        response_type = extract_response_type(generated)
        generated_tools = extract_tool_calls_from_output(generated)

        # Extract expected tool names from answer sheet
        expected_tool_names = set()
        if expected_tools:
            for tc in expected_tools:
                if isinstance(tc, dict):
                    func_name = tc.get("function", {}).get("name")
                    if func_name:
                        expected_tool_names.add(func_name)

        # Calculate match score
        if expected_response:
            # Expecting natural language response
            gen_clean = re.sub(r'<[^>]+>', '', generated).strip()
            exp_clean = str(expected_response).strip()
            match = similarity_ratio(gen_clean, exp_clean)
            tool_match_correct = False
        else:
            # Expecting tool call(s)
            if expected_tool_names:
                # Check if generated tools match expected
                if set(generated_tools) == expected_tool_names:
                    match = 1.0  # Correct tool(s)
                    tool_match_correct = True
                elif set(generated_tools) & expected_tool_names:
                    match = 0.5  # Partial match (some correct tools)
                    tool_match_correct = False
                elif is_valid_alternative_tool(list(expected_tool_names), generated_tools, context_messages):
                    match = 0.7  # Valid alternative tool choice
                    tool_match_correct = False
                else:
                    match = 0.0  # Wrong tools or no tools
                    tool_match_correct = False
            else:
                # No tools expected
                match = 0.5

        results.append({
            "turn": i,
            "error": None,
            "match": match,
            "response_type": response_type,
            "expected_language": bool(expected_response),
            "expected_tools": list(expected_tool_names),
            "generated_tools": generated_tools,
            "tool_match_correct": set(generated_tools) == expected_tool_names if expected_tool_names else None,
            "generated": generated[:100],
        })

    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate models on multi-turn conversations")
    parser.add_argument("--model", choices=["2b", "4b", "both"], default="both")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-results", action="store_true", help="Save detailed results to file")
    args = parser.parse_args()

    # Load validation data
    val_file = "finetune/data/val.jsonl"
    with open(val_file) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if args.num_examples:
        examples = examples[:args.num_examples]

    print(f"Loaded {len(examples)} validation examples\n")

    models_to_test = ["2b", "4b"] if args.model == "both" else [args.model]
    results = {}

    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"Testing {model_name.upper()} Model on Multi-Turn Conversations")
        print(f"{'='*70}\n")

        model_path = MODELS[model_name]

        if not Path(model_path).exists():
            print(f"⚠️  Model not found at {model_path}")
            continue

        print(f"Loading {model_name} model...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                device_map=args.device,
            )
            model.eval()
            print(f"✓ Model loaded\n")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
            continue

        # Validate with batch processing
        print(f"Testing on {len(examples)} examples...")
        model_results = {
            "total_turns": 0,
            "total_errors": 0,
            "match_scores": [],
            "response_types": defaultdict(int),
            "issues": defaultdict(int),
            "high_match": 0,  # > 0.7 match
            "medium_match": 0,  # 0.4-0.7
            "low_match": 0,  # < 0.4
            "by_scenario": defaultdict(lambda: {  # Per-scenario breakdown
                "total_turns": 0,
                "match_scores": [],
                "high_match": 0,
                "medium_match": 0,
                "low_match": 0,
                "tool_mismatch": 0,
            }),
        }

        all_validations = []  # Store for file saving

        # Collect all turns to validate
        tools = examples[0].get("tools", []) if examples else []
        all_turns = collect_validation_turns(examples, tools)

        print(f"  Found {len(all_turns)} turns to validate")

        # Process in batches with progress bar
        batch_size = 8
        num_batches = (len(all_turns) + batch_size - 1) // batch_size

        with tqdm(total=len(all_turns), desc=f"  {model_name.upper()}", unit="turn", file=sys.stdout) as pbar:
            for batch_start in range(0, len(all_turns), batch_size):
                batch_end = min(batch_start + batch_size, len(all_turns))
                turns_batch = all_turns[batch_start:batch_end]

                # Validate batch
                batch_results = validate_batch(model, tokenizer, turns_batch)

                # Process results
                for result in batch_results:
                    ex_idx = result["ex_idx"]
                    model_results["total_turns"] += 1

                    # Get scenario from example
                    scenario = examples[ex_idx].get("scenario_type", "unknown")

                    if result["error"]:
                        model_results["total_errors"] += 1
                    else:
                        match = result["match"]
                        model_results["match_scores"].append(match)
                        model_results["response_types"][result["response_type"]] += 1

                        # Track per-scenario metrics
                        model_results["by_scenario"][scenario]["total_turns"] += 1
                        model_results["by_scenario"][scenario]["match_scores"].append(match)

                        # Track tool mismatches
                        if result.get("expected_tools") and not result.get("tool_match_correct"):
                            model_results["issues"]["tool_mismatch"] += 1
                            model_results["by_scenario"][scenario]["tool_mismatch"] += 1

                        if match > 0.7:
                            model_results["high_match"] += 1
                            model_results["by_scenario"][scenario]["high_match"] += 1
                        elif match > 0.4:
                            model_results["medium_match"] += 1
                            model_results["by_scenario"][scenario]["medium_match"] += 1
                        else:
                            model_results["low_match"] += 1
                            model_results["by_scenario"][scenario]["low_match"] += 1

                # Save for later
                if args.save_results:
                    # Group results by example for saving
                    for turn_idx, result in enumerate(batch_results):
                        ex_idx = result["ex_idx"]
                        # Find or create validation entry for this example
                        existing = next((v for v in all_validations if v["example_idx"] == ex_idx), None)
                        if not existing:
                            existing = {
                                "example_idx": ex_idx,
                                "messages": examples[ex_idx]["messages"],
                                "results": [],
                            }
                            all_validations.append(existing)
                        existing["results"].append(result)

                # Update progress bar
                pbar.update(len(batch_results))

        print(f"\n{'='*70}")
        print(f"Results for {model_name.upper()}")
        print(f"{'='*70}\n")

        if model_results["match_scores"]:
            avg_match = sum(model_results["match_scores"]) / len(model_results["match_scores"])
            print(f"Total turns tested: {model_results['total_turns']}")
            print(f"Average match score: {avg_match:.2f} (0-1 scale)")
            print(f"\nMatch distribution:")
            print(f"  ✓ High match (>0.7): {model_results['high_match']} ({model_results['high_match']/model_results['total_turns']*100:.1f}%)")
            print(f"  ~ Medium match (0.4-0.7): {model_results['medium_match']} ({model_results['medium_match']/model_results['total_turns']*100:.1f}%)")
            print(f"  ✗ Low match (<0.4): {model_results['low_match']} ({model_results['low_match']/model_results['total_turns']*100:.1f}%)")

            print(f"\nResponse types generated:")
            for response_type, count in sorted(model_results["response_types"].items(), key=lambda x: -x[1]):
                pct = count / model_results["total_turns"] * 100
                print(f"  - {response_type}: {count} ({pct:.1f}%)")

            if model_results["issues"]:
                print(f"\nIssues found:")
                for issue, count in sorted(model_results["issues"].items(), key=lambda x: -x[1]):
                    pct = count / model_results["total_turns"] * 100
                    print(f"  ✗ {issue}: {count} ({pct:.1f}%)")

            # Per-scenario breakdown
            print(f"\n{'='*70}")
            print(f"Per-Scenario Breakdown for {model_name.upper()}")
            print(f"{'='*70}")
            print(f"{'Scenario':<15} {'Turns':<8} {'Avg Match':<12} {'High %':<10} {'Tool Mismatch':<15}")
            print("-" * 60)
            for scenario in sorted(model_results["by_scenario"].keys()):
                scenario_stats = model_results["by_scenario"][scenario]
                if scenario_stats["total_turns"] == 0:
                    continue
                avg_match = sum(scenario_stats["match_scores"]) / len(scenario_stats["match_scores"]) if scenario_stats["match_scores"] else 0
                high_pct = scenario_stats["high_match"] / scenario_stats["total_turns"] * 100
                mismatch = scenario_stats["tool_mismatch"]
                mismatch_pct = mismatch / scenario_stats["total_turns"] * 100 if scenario_stats["total_turns"] > 0 else 0
                print(f"{scenario:<15} {scenario_stats['total_turns']:<8} {avg_match:.3f}       {high_pct:>6.1f}%    {mismatch} ({mismatch_pct:.1f}%)")

        results[model_name] = model_results

        # Save detailed results
        if args.save_results:
            output_file = f"finetune/validation_results_{model_name}.jsonl"
            with open(output_file, 'w') as f:
                for validation in all_validations:
                    f.write(json.dumps(validation) + '\n')
            print(f"\n✓ Saved detailed results to {output_file}")

        # Cleanup
        del model
        del tokenizer
        torch.cuda.empty_cache()

    # Comparison
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("Comparison: 2B vs 4B")
        print(f"{'='*70}\n")

        print(f"{'Model':<10} {'Avg Match':<15} {'High Match %':<15}")
        print("-" * 45)
        for model_name in ["2b", "4b"]:
            if model_name not in results:
                continue
            r = results[model_name]
            if r["match_scores"]:
                avg = sum(r["match_scores"]) / len(r["match_scores"])
                high_pct = r["high_match"] / r["total_turns"] * 100
                print(f"{model_name:<10} {avg:.3f}         {high_pct:.1f}%")

if __name__ == "__main__":
    main()
