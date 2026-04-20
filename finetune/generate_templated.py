#!/usr/bin/env python3
"""Generate 1000 high-quality training examples using scenario templates.

Usage:
    python3 finetune/generate_templated.py --scenario-a 250 --workers 5
    python3 finetune/generate_templated.py --all 1000 --workers 10
"""

import json
import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import anthropic

from templates import (
    scenario_a,
    scenario_b,
    scenario_c,
    scenario_d,
    scenario_e,
    scenario_f,
    scenario_g,
    scenario_h,
    constants,
    assembler,
)


def call_claude_with_retry(client, prompt, max_tokens=500, max_retries=3):
    """Call Claude API with exponential backoff for rate limits.

    Args:
        client: anthropic.Anthropic client
        prompt: str, the user message
        max_tokens: int, max tokens in response
        max_retries: int, number of retries on rate limit

    Returns:
        str or None: response text, or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.RateLimitError as e:
            if attempt < max_retries:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...", file=sys.stderr)
                time.sleep(wait_time)
            else:
                print(f"Rate limit persisted after {max_retries} retries. Giving up.", file=sys.stderr)
                return None
        except anthropic.APIError as e:
            print(f"API error: {e}", file=sys.stderr)
            return None

    return None


def parse_slot_fills(response_text):
    """Parse Claude's JSON response to extract filled slots.

    Args:
        response_text: str, response from Claude API

    Returns:
        dict or list: parsed slot fills (dict for single, list for batch), or None if parse failed
    """
    try:
        # Try to extract JSON from response (handle markdown code blocks)
        text = response_text.strip()
        if text.startswith("```"):
            # Remove markdown code block markers
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        return json.loads(text)
    except (json.JSONDecodeError, IndexError) as e:
        return None


def generate_examples_batch(scenario_name, template_vars_list, scenario_module, client):
    """Call Claude API to fill slots for multiple examples (batch mode).

    Args:
        scenario_name: str (e.g., "scenario_a")
        template_vars_list: list of up to 5 dicts with context
        scenario_module: module with build_fill_prompt_batch, build_skeleton, get_expected_tools
        client: anthropic.Anthropic client

    Returns:
        list: list of complete examples (may be shorter if some failed validation)
    """
    try:
        # Use batch function if available, otherwise fall back to single mode
        if not hasattr(scenario_module, 'build_fill_prompt_batch'):
            # Fallback: generate individually
            results = []
            for tv in template_vars_list:
                result = generate_example_with_api(scenario_name, tv, scenario_module, client)
                if result:
                    results.append(result)
            return results

        # Build the fill prompt for batch
        fill_prompt = scenario_module.build_fill_prompt_batch(template_vars_list)

        # Call Claude with retry logic
        response_text = call_claude_with_retry(client, fill_prompt, max_tokens=2000, max_retries=3)
        if not response_text:
            return []

        # Parse response (should be array)
        slot_fills_list = parse_slot_fills(response_text)
        if not slot_fills_list or not isinstance(slot_fills_list, list):
            return []

        # Ensure we got exactly the right number of results
        if len(slot_fills_list) != len(template_vars_list):
            return []

        # Build examples
        examples = []
        expected_tools = scenario_module.get_expected_tools()

        for template_vars, slot_fills in zip(template_vars_list, slot_fills_list):
            try:
                # Build skeleton
                messages = scenario_module.build_skeleton(template_vars, slot_fills)

                # Validate sequence
                if not assembler.validate_sequence(messages, expected_tools):
                    continue

                # Assemble example
                example = assembler.assemble_example(
                    scenario_name,
                    template_vars,
                    slot_fills,
                    scenario_module.build_skeleton,
                )

                # Sanitize
                example = assembler.sanitize_example(example)
                examples.append(example)

            except Exception:
                continue

        return examples

    except Exception as e:
        print(f"Error generating batch: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []


def generate_example_with_api(scenario_name, template_vars, scenario_module, client):
    """Call Claude API to fill slots for one example.

    Args:
        scenario_name: str (e.g., "scenario_a")
        template_vars: dict with context
        scenario_module: module with build_fill_prompt, build_skeleton, get_expected_tools
        client: anthropic.Anthropic client

    Returns:
        dict: complete example, or None if generation failed
    """
    try:
        # Build the fill prompt
        fill_prompt = scenario_module.build_fill_prompt(template_vars)

        # Call Claude with retry logic
        response_text = call_claude_with_retry(client, fill_prompt, max_tokens=500, max_retries=3)
        if not response_text:
            return None

        # Parse response
        slot_fills = parse_slot_fills(response_text)
        if not slot_fills:
            return None

        # Build skeleton
        messages = scenario_module.build_skeleton(template_vars, slot_fills)

        # Validate sequence
        expected_tools = scenario_module.get_expected_tools()
        if not assembler.validate_sequence(messages, expected_tools):
            return None

        # Assemble example
        example = assembler.assemble_example(
            scenario_name,
            template_vars,
            slot_fills,
            scenario_module.build_skeleton,
        )

        # Sanitize
        example = assembler.sanitize_example(example)

        return example

    except Exception as e:
        print(f"Error generating example: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate templated training dataset"
    )
    parser.add_argument(
        "--scenario-a",
        type=int,
        default=350,
        help="Number of Scenario A examples (navigate to known location)",
    )
    parser.add_argument(
        "--scenario-b",
        type=int,
        default=150,
        help="Number of Scenario B examples (navigate to nearest)",
    )
    parser.add_argument(
        "--scenario-c",
        type=int,
        default=125,
        help="Number of Scenario C examples (describe surroundings)",
    )
    parser.add_argument(
        "--scenario-d",
        type=int,
        default=100,
        help="Number of Scenario D examples (reorientation)",
    )
    parser.add_argument(
        "--scenario-e",
        type=int,
        default=100,
        help="Number of Scenario E examples (distance query only)",
    )
    parser.add_argument(
        "--scenario-f",
        type=int,
        default=100,
        help="Number of Scenario F examples (error handling)",
    )
    parser.add_argument(
        "--scenario-g",
        type=int,
        default=50,
        help="Number of Scenario G examples (greetings)",
    )
    parser.add_argument(
        "--scenario-h",
        type=int,
        default=25,
        help="Number of Scenario H examples (return navigation)",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Number of parallel API workers (use 2-3 to avoid rate limits)"
    )
    parser.add_argument(
        "--output-dir",
        default="finetune/data",
        help="Output directory for train.jsonl and val.jsonl",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize API client
    client = anthropic.Anthropic()

    print(f"\n{'='*70}")
    print("Template-Based Dataset Generation")
    print(f"{'='*70}\n")

    # Define all scenarios
    scenarios = [
        ("A", args.scenario_a, scenario_a, "Navigate to known location"),
        ("B", args.scenario_b, scenario_b, "Navigate to nearest instance"),
        ("C", args.scenario_c, scenario_c, "Describe surroundings"),
        ("D", args.scenario_d, scenario_d, "Reorientation / where am I"),
        ("E", args.scenario_e, scenario_e, "Distance query only"),
        ("F", args.scenario_f, scenario_f, "Error handling"),
        ("G", args.scenario_g, scenario_g, "Greetings / small talk"),
        ("H", args.scenario_h, scenario_h, "Return navigation"),
    ]

    all_examples = []
    total_success = 0
    total_fail = 0

    for letter, count, module, description in scenarios:
        if count == 0:
            continue

        print(f"Generating Scenario {letter}: {description} ({count} examples)...")
        print("  Collecting variables...", end="", flush=True)

        scenario_vars = list(module.generate_vars())
        print(f" {len(scenario_vars)} total variations")

        # Repeat variations to reach target count
        if len(scenario_vars) > 0:
            repeat_factor = (count + len(scenario_vars) - 1) // len(scenario_vars)
            scenario_vars = (scenario_vars * repeat_factor)[:count]

        # Batch into groups of 5
        batch_size = 5
        batches = [
            scenario_vars[i : i + batch_size]
            for i in range(0, len(scenario_vars), batch_size)
        ]
        print(f"  Generating with API ({len(batches)} batch calls, 5 examples per batch)...")

        success_count = 0
        fail_count = 0
        examples_generated = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    generate_examples_batch,
                    f"scenario_{letter.lower()}",
                    batch,
                    module,
                    client,
                ): (batch_idx, batch)
                for batch_idx, batch in enumerate(batches)
            }

            for completed_idx, future in enumerate(as_completed(futures)):
                batch_idx, batch = futures[future]
                batch_results = future.result()
                for example in batch_results:
                    all_examples.append(example)
                examples_generated += len(batch_results)
                fail_count += len(batch) - len(batch_results)

                if (completed_idx + 1) % 10 == 0:
                    print(f"    {completed_idx+1}/{len(batches)} batches", end="\r", flush=True)

        print(f"    {len(batches)}/{len(batches)} batches complete")
        print(f"  Generated: {examples_generated}/{len(scenario_vars)}\n")

        total_success += examples_generated
        total_fail += len(scenario_vars) - examples_generated

    # Split into train/val (90/10)
    total_examples = len(all_examples)
    val_count = max(1, total_examples // 10)
    train_count = total_examples - val_count

    train_examples = all_examples[:train_count]
    val_examples = all_examples[train_count:]

    # Write files
    print(f"Writing to {output_dir}...")

    with open(output_dir / "train.jsonl", "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(output_dir / "val.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"  train.jsonl: {len(train_examples)} examples")
    print(f"  val.jsonl: {len(val_examples)} examples")

    print(f"\n{'='*70}")
    print("Dataset generation complete!")
    print(f"Total: {total_success} successful, {total_fail} failed")
    print(f"Next steps:")
    print(f"  1. python3 finetune/fix_tool_calls.py")
    print(f"  2. python3 finetune/restructure_for_qwen.py")
    print(f"  3. bash finetune/train_all.sh")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
