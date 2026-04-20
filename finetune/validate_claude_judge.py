#!/usr/bin/env python3
"""
Validate fine-tuned models using Claude as judge.

For each validation example:
1. Run fine-tuned model to get its response at each assistant turn
2. Send context + model response + expected response to Claude
3. Claude scores tool correctness, language quality, and overall helpfulness

Usage:
    python3 finetune/validate_claude_judge.py
    python3 finetune/validate_claude_judge.py --model 2b
    python3 finetune/validate_claude_judge.py --num-examples 10
"""

import json
import argparse
import re
import time
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import anthropic

# ============================================================================
# Config
# ============================================================================

FINETUNED_MODELS = {
    "2b": "finetune/output/qwen3.5-2b-toolcall/merged_16bit",
    "4b": "finetune/output/qwen3.5-4b-toolcall/merged_16bit",
}

EXPECTED_TOOL_SEQUENCES = {
    "scenario_a": ["list_all", "distance_to", "navigate_to"],
    "scenario_b": ["find_nearest", "distance_to", "navigate_to"],
    "scenario_c": ["get_robot_pose", "describe_surroundings"],
    "scenario_d": ["get_robot_pose", "orient_me"],
    "scenario_e": ["distance_to"],
    "scenario_f": ["list_all"],
    "scenario_g": [],
    "scenario_h": ["get_robot_pose", "distance_to", "navigate_to"],
}

SCENARIO_DESCRIPTIONS = {
    "scenario_a": "Navigate to known location (zone or object)",
    "scenario_b": "Navigate to nearest instance of an object",
    "scenario_c": "Describe surroundings",
    "scenario_d": "Reorientation / where am I",
    "scenario_e": "Distance query only (no navigation)",
    "scenario_f": "Error handling (unavailable destination)",
    "scenario_g": "Greeting / small talk (no tools)",
    "scenario_h": "Return navigation (go back)",
}

# ============================================================================
# Claude Judge
# ============================================================================

JUDGE_PROMPT = """You are evaluating a navigation assistant for vision-impaired users.

SYSTEM RULES THE ASSISTANT MUST FOLLOW:
- Tool ordering: Known location → list_all → distance_to → [confirm] → navigate_to
- Find nearest → find_nearest → distance_to → [confirm] → navigate_to
- Describe surroundings → get_robot_pose → describe_surroundings
- Reorientation → get_robot_pose → orient_me
- 1-2 sentences max, spoken English only
- Distance vocab: <1m "right here", 1-3m "nearby", 3-6m "a short walk away", >6m "a bit further away"
- Never say zone codes, coordinates, or bearings

SCENARIO: {scenario_type} — {scenario_description}
EXPECTED TOOL SEQUENCE: {expected_tools}

CONVERSATION CONTEXT (messages before this turn):
{context}

EXPECTED ASSISTANT RESPONSE:
{expected_response}

ACTUAL MODEL RESPONSE:
{model_response}

Score the model's response on these dimensions (0.0 to 1.0 each):

1. tool_correctness: Did the model call the right tool(s) with correct arguments? If no tool was expected and none were called, score 1.0. If tools were expected but wrong/missing, score 0.0.
2. language_quality: Is the natural language clear, concise (1-2 sentences), and appropriate for a vision-impaired user? Does it use correct distance vocabulary?
3. helpfulness: Would this response actually help the user accomplish their navigation goal?

Respond with ONLY a JSON object, no explanation:
{{"tool_correctness": 0.0, "language_quality": 0.0, "helpfulness": 0.0, "issues": "brief note on any problems or empty string"}}"""


def call_claude_judge(client, scenario_type, context_msgs, expected_response, model_response):
    """Send a single turn to Claude for judgment.

    Returns dict with scores or None on failure.
    """
    # Format context messages for readability
    context_lines = []
    for msg in context_msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        if tool_calls:
            tool_names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
            context_lines.append(f"[{role}] tool_calls: {tool_names}")
        elif content:
            context_lines.append(f"[{role}] {content[:200]}")

    context_text = "\n".join(context_lines[-10:])  # Last 10 messages for context

    expected_tools = EXPECTED_TOOL_SEQUENCES.get(scenario_type, [])
    scenario_desc = SCENARIO_DESCRIPTIONS.get(scenario_type, "Unknown scenario")

    prompt = JUDGE_PROMPT.format(
        scenario_type=scenario_type,
        scenario_description=scenario_desc,
        expected_tools=expected_tools or "(no tools expected)",
        context=context_text or "(start of conversation)",
        expected_response=expected_response or "(tool call only, no text)",
        model_response=model_response or "(empty response)",
    )

    for attempt in range(3):
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = message.content[0].text.strip()

            # Parse JSON response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            scores = json.loads(text)
            return scores

        except anthropic.RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
        except (json.JSONDecodeError, Exception) as e:
            return None

    return None


# ============================================================================
# Model Inference
# ============================================================================

def generate_model_response(model, tokenizer, context_messages, tools):
    """Generate a response from the fine-tuned model given context."""
    try:
        text = tokenizer.apply_chat_template(
            context_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return generated.strip()

    except Exception as e:
        return f"[ERROR: {e}]"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Validate with Claude as judge")
    parser.add_argument("--model", choices=["2b", "4b", "both"], default="both")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=3, help="Parallel Claude API workers")
    parser.add_argument("--save-results", action="store_true", help="Save detailed results")
    args = parser.parse_args()

    # Load validation data
    val_file = "finetune/data/val.jsonl"
    with open(val_file) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if args.num_examples:
        examples = examples[:args.num_examples]

    print(f"Loaded {len(examples)} validation examples")

    # Check scenario coverage
    scenario_counts = defaultdict(int)
    for ex in examples:
        scenario_counts[ex.get("scenario_type", "unknown")] += 1
    print("Scenario coverage:")
    for s in sorted(scenario_counts):
        print(f"  {s}: {scenario_counts[s]}")

    # Init Claude client
    client = anthropic.Anthropic()

    models_to_test = ["2b", "4b"] if args.model == "both" else [args.model]
    all_model_results = {}

    for model_name in models_to_test:
        model_path = FINETUNED_MODELS[model_name]

        print(f"\n{'='*70}")
        print(f"Evaluating {model_name.upper()} with Claude Judge")
        print(f"{'='*70}")

        if not Path(model_path).exists():
            print(f"Model not found at {model_path}, skipping.")
            continue

        # Load model
        print(f"Loading {model_name} model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            device_map=args.device,
        )
        model.eval()
        print(f"Model loaded.\n")

        # Collect all assistant turns
        turns_to_judge = []
        for ex_idx, example in enumerate(examples):
            messages = example["messages"]
            tools = example.get("tools", [])
            scenario = example.get("scenario_type", "unknown")

            for i in range(len(messages)):
                if messages[i]["role"] != "assistant":
                    continue

                context = messages[:i]
                expected = messages[i]

                # Format expected response
                expected_text = expected.get("content") or ""
                if expected.get("tool_calls"):
                    tool_names = [tc.get("function", {}).get("name", "?") for tc in expected["tool_calls"]]
                    expected_text += f" [tool_calls: {tool_names}]"

                turns_to_judge.append({
                    "ex_idx": ex_idx,
                    "turn_idx": i,
                    "scenario": scenario,
                    "context": context,
                    "tools": tools,
                    "expected_text": expected_text.strip(),
                })

        print(f"Found {len(turns_to_judge)} assistant turns to evaluate")

        # Generate model responses
        print("Generating model responses...")
        for turn in tqdm(turns_to_judge, desc="  Inference", unit="turn", file=sys.stdout):
            turn["model_response"] = generate_model_response(
                model, tokenizer, turn["context"], turn["tools"]
            )

        # Free GPU memory before Claude calls
        del model
        torch.cuda.empty_cache()

        # Judge with Claude
        print("Sending to Claude for judgment...")
        results_by_scenario = defaultdict(lambda: {
            "scores": [],
            "tool_scores": [],
            "language_scores": [],
            "helpfulness_scores": [],
            "issues": [],
        })
        total_scores = []
        failed_judgments = 0

        with tqdm(total=len(turns_to_judge), desc="  Judging", unit="turn", file=sys.stdout) as pbar:
            # Process in small parallel batches to respect rate limits
            batch_size = args.workers
            for batch_start in range(0, len(turns_to_judge), batch_size):
                batch = turns_to_judge[batch_start:batch_start + batch_size]

                with ThreadPoolExecutor(max_workers=batch_size) as executor:
                    futures = {}
                    for turn in batch:
                        future = executor.submit(
                            call_claude_judge,
                            client,
                            turn["scenario"],
                            turn["context"],
                            turn["expected_text"],
                            turn["model_response"],
                        )
                        futures[future] = turn

                    for future in as_completed(futures):
                        turn = futures[future]
                        scores = future.result()

                        if scores:
                            tc = scores.get("tool_correctness", 0)
                            lq = scores.get("language_quality", 0)
                            hp = scores.get("helpfulness", 0)
                            overall = tc * 0.4 + lq * 0.3 + hp * 0.3
                            issues = scores.get("issues", "")

                            total_scores.append(overall)
                            scenario = turn["scenario"]
                            results_by_scenario[scenario]["scores"].append(overall)
                            results_by_scenario[scenario]["tool_scores"].append(tc)
                            results_by_scenario[scenario]["language_scores"].append(lq)
                            results_by_scenario[scenario]["helpfulness_scores"].append(hp)
                            if issues:
                                results_by_scenario[scenario]["issues"].append(issues)

                            turn["claude_scores"] = scores
                            turn["overall_score"] = overall
                        else:
                            failed_judgments += 1

                        pbar.update(1)

        # ================================================================
        # Report
        # ================================================================
        print(f"\n{'='*70}")
        print(f"CLAUDE JUDGE RESULTS: {model_name.upper()}")
        print(f"{'='*70}\n")

        if total_scores:
            avg = sum(total_scores) / len(total_scores)
            print(f"Overall Score: {avg:.3f} ({avg*100:.1f}%)")
            print(f"Turns evaluated: {len(total_scores)}")
            print(f"Failed judgments: {failed_judgments}\n")

            # Per-scenario table
            print(f"{'Scenario':<15} {'Overall':<10} {'Tools':<10} {'Language':<10} {'Helpful':<10} {'Turns':<8}")
            print("-" * 63)

            for scenario in sorted(results_by_scenario.keys()):
                r = results_by_scenario[scenario]
                n = len(r["scores"])
                if n == 0:
                    continue
                avg_s = sum(r["scores"]) / n
                avg_t = sum(r["tool_scores"]) / n
                avg_l = sum(r["language_scores"]) / n
                avg_h = sum(r["helpfulness_scores"]) / n
                print(f"{scenario:<15} {avg_s:.3f}     {avg_t:.3f}     {avg_l:.3f}     {avg_h:.3f}     {n}")

            # Overall averages
            all_tool = [s for r in results_by_scenario.values() for s in r["tool_scores"]]
            all_lang = [s for r in results_by_scenario.values() for s in r["language_scores"]]
            all_help = [s for r in results_by_scenario.values() for s in r["helpfulness_scores"]]
            print("-" * 63)
            print(f"{'AVERAGE':<15} {avg:.3f}     {sum(all_tool)/len(all_tool):.3f}     {sum(all_lang)/len(all_lang):.3f}     {sum(all_help)/len(all_help):.3f}     {len(total_scores)}")

            # Score distribution
            print(f"\nScore Distribution:")
            brackets = [(0.9, 1.01, "Excellent"), (0.7, 0.9, "Good"), (0.5, 0.7, "Fair"), (0.0, 0.5, "Poor")]
            for low, high, label in brackets:
                count = sum(1 for s in total_scores if low <= s < high)
                pct = count / len(total_scores) * 100
                print(f"  {label:<12} ({low:.1f}-{high:.1f}): {count:4d} ({pct:.1f}%)")

            # Top issues
            all_issues = [i for r in results_by_scenario.values() for i in r["issues"] if i]
            if all_issues:
                print(f"\nTop Issues ({len(all_issues)} total):")
                # Count unique issues
                issue_counts = defaultdict(int)
                for issue in all_issues:
                    issue_counts[issue[:80]] += 1
                for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
                    print(f"  [{count}x] {issue}")

        all_model_results[model_name] = {
            "overall": sum(total_scores) / len(total_scores) if total_scores else 0,
            "total_turns": len(total_scores),
            "by_scenario": {k: {
                "overall": sum(v["scores"]) / len(v["scores"]) if v["scores"] else 0,
                "tools": sum(v["tool_scores"]) / len(v["tool_scores"]) if v["tool_scores"] else 0,
                "language": sum(v["language_scores"]) / len(v["language_scores"]) if v["language_scores"] else 0,
                "helpfulness": sum(v["helpfulness_scores"]) / len(v["helpfulness_scores"]) if v["helpfulness_scores"] else 0,
            } for k, v in results_by_scenario.items()},
        }

        # Save detailed results
        if args.save_results:
            output_file = f"finetune/claude_judge_results_{model_name}.jsonl"
            with open(output_file, "w") as f:
                for turn in turns_to_judge:
                    f.write(json.dumps({
                        "ex_idx": turn["ex_idx"],
                        "scenario": turn["scenario"],
                        "expected": turn["expected_text"],
                        "model_response": turn["model_response"],
                        "claude_scores": turn.get("claude_scores"),
                        "overall_score": turn.get("overall_score"),
                    }) + "\n")
            print(f"\nSaved detailed results to {output_file}")

        del tokenizer
        torch.cuda.empty_cache()

    # Comparison
    if len(all_model_results) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON (Claude Judge)")
        print(f"{'='*70}\n")

        print(f"{'Model':<10} {'Overall':<12} {'Tools':<10} {'Language':<10} {'Helpful':<10}")
        print("-" * 52)
        for mn in ["2b", "4b"]:
            if mn not in all_model_results:
                continue
            r = all_model_results[mn]
            by_s = r["by_scenario"]
            avg_t = sum(v["tools"] for v in by_s.values()) / len(by_s) if by_s else 0
            avg_l = sum(v["language"] for v in by_s.values()) / len(by_s) if by_s else 0
            avg_h = sum(v["helpfulness"] for v in by_s.values()) / len(by_s) if by_s else 0
            print(f"{mn:<10} {r['overall']:.3f}       {avg_t:.3f}     {avg_l:.3f}     {avg_h:.3f}")


if __name__ == "__main__":
    main()
