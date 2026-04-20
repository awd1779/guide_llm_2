#!/usr/bin/env python3
"""
Rigorous evaluation of fine-tuned navigation models.

Five metrics:
  1. Tool selection accuracy  — per-turn, did model call the right tool? (programmatic)
  2. Sequence accuracy        — per-example, was the full tool sequence correct? (programmatic)
  3. Argument accuracy        — per-turn, were tool arguments valid? (programmatic)
  4. End-to-end completion    — can model complete a full conversation with simulated tools?
  5. Language quality          — Claude judges only natural language responses

Usage:
    python3 finetune/evaluate.py                       # all metrics, both models
    python3 finetune/evaluate.py --model 4b            # one model
    python3 finetune/evaluate.py --skip-claude          # skip metric 5 (no API cost)
    python3 finetune/evaluate.py --num-examples 10     # quick test
"""

import json
import argparse
import re
import time
import sys
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from templates.constants import (
    CANONICAL_SYSTEM_PROMPT, ZONES, OBJECTS, OBJECT_ZONES,
    DISTANCE_MATRIX, DIRECTION_MATRIX, TOOLS, distance_to_vocab,
)

# ============================================================================
# Config
# ============================================================================

FINETUNED_MODELS = {
    "2b": "finetune/output/qwen3.5-2b-toolcall/merged_16bit",
    "4b": "finetune/output/qwen3.5-4b-toolcall/merged_16bit",
}

EXPECTED_SEQUENCES = {
    "scenario_a": ["list_all", "distance_to", "navigate_to"],
    "scenario_b": ["find_nearest", "distance_to", "navigate_to"],
    "scenario_c": ["get_robot_pose", "describe_surroundings"],
    "scenario_d": ["get_robot_pose", "orient_me"],
    "scenario_e": ["distance_to"],
    "scenario_f": ["list_all"],
    "scenario_g": [],  # no tools
    "scenario_h": ["get_robot_pose", "distance_to", "navigate_to"],
}

# ============================================================================
# Tool Call Extraction
# ============================================================================

def extract_tool_calls(text):
    """Extract tool calls from model output.

    Handles the Qwen XML format:
        <tool_call>
        <function=tool_name>
        <parameter=arg_name>arg_value</parameter>
        </function>
        </tool_call>

    Returns list of dicts: [{"name": str, "arguments": dict}, ...]
    """
    calls = []

    # Primary: Qwen XML format <function=TOOL_NAME> with <parameter=KEY>VALUE</parameter>
    fn_pattern = r'<function=(\w+)>(.*?)</function>'
    for match in re.finditer(fn_pattern, text, re.DOTALL):
        name = match.group(1)
        body = match.group(2)

        # Extract parameters
        args = {}
        param_pattern = r'<parameter=(\w+)>(.*?)</parameter>'
        for pm in re.finditer(param_pattern, body, re.DOTALL):
            args[pm.group(1)] = pm.group(2).strip()

        calls.append({"name": name, "arguments": args})

    if calls:
        return calls

    # Fallback: JSON format {"name": "tool", "arguments": {...}}
    json_pattern = r'\{["\']name["\']\s*:\s*["\'](\w+)["\'].*?["\']arguments["\']\s*:\s*(\{.*?\})\}'
    for match in re.finditer(json_pattern, text, re.DOTALL):
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": match.group(1), "arguments": args})

    return calls


def extract_tool_names(text):
    """Extract just tool names from model output."""
    return [tc["name"] for tc in extract_tool_calls(text)]


# ============================================================================
# Tool Simulator (for end-to-end tests)
# ============================================================================

def simulate_tool(tool_name, arguments, current_zone):
    """Simulate a tool response matching the exact format from training data.

    Args:
        tool_name: str, name of tool to simulate
        arguments: dict, tool arguments
        current_zone: str, robot's current zone

    Returns:
        str: JSON string of simulated tool response
    """
    if tool_name == "list_all":
        # Training format: zones as [{"name": "zone"}, ...], objects as strings
        return json.dumps({
            "zones": [{"name": z} for z in ZONES],
            "objects": OBJECTS,
        })

    elif tool_name == "get_robot_pose":
        return json.dumps({
            "x": 1.5, "y": 2.3, "yaw_deg": 45.0,
            "frame": "map", "current_zone": current_zone,
        })

    elif tool_name == "distance_to":
        dest = arguments.get("destination", "")
        dest_zone = OBJECT_ZONES.get(dest, dest)
        dest_type = "object" if dest in OBJECT_ZONES else "zone"
        key = (current_zone, dest_zone)
        distance = DISTANCE_MATRIX.get(key, 5.0)
        direction = DIRECTION_MATRIX.get(key, "ahead")
        # Training format includes destination_type
        return json.dumps({
            "destination": dest,
            "destination_type": dest_type,
            "distance_m": distance,
            "direction": direction,
        })

    elif tool_name == "navigate_to":
        dest = arguments.get("destination", "")
        dest_type = "object" if dest in OBJECT_ZONES else "zone"
        # Training format: status is "goal_sent", includes type
        return json.dumps({
            "status": "goal_sent",
            "destination": dest,
            "type": dest_type,
        })

    elif tool_name == "find_nearest":
        obj_type = arguments.get("object_type", "")
        zone = OBJECT_ZONES.get(obj_type, "hri_lab")
        key = (current_zone, zone)
        distance = DISTANCE_MATRIX.get(key, 3.0)
        direction = DIRECTION_MATRIX.get(key, "ahead")
        # Training format includes zone
        return json.dumps({
            "label": obj_type,
            "distance_m": distance,
            "direction": direction,
            "zone": zone,
        })

    elif tool_name == "describe_surroundings":
        # Training format: nearby_objects as [{"label": ..., "distance_m": ..., "direction": ...}]
        nearby = []
        for obj, z in OBJECT_ZONES.items():
            if z == current_zone:
                nearby.append({
                    "label": obj,
                    "distance_m": 2.5,
                    "direction": "ahead",
                })
        return json.dumps({
            "current_zone": current_zone,
            "radius_m": arguments.get("radius_m", 3.0),
            "nearby_objects": nearby,
        })

    elif tool_name == "orient_me":
        target = arguments.get("target", "kitchen")
        target_zone = OBJECT_ZONES.get(target, target)
        key = (current_zone, target_zone)
        direction = DIRECTION_MATRIX.get(key, "ahead")
        # Training format includes message
        return json.dumps({
            "target": target,
            "direction_from_current": direction,
            "turn_degrees": 45.0,
            "message": f"Rotating to face {target}",
        })

    elif tool_name == "get_navigation_status":
        return json.dumps({"status": "navigating", "destination": "kitchen"})

    elif tool_name == "cancel_navigation":
        return json.dumps({"status": "cancelled", "message": "Navigation cancelled."})

    elif tool_name == "describe_route":
        dest = arguments.get("destination", "")
        return json.dumps({"destination": dest, "route": f"Head {DIRECTION_MATRIX.get((current_zone, dest), 'ahead')} to reach {dest}."})

    elif tool_name == "query":
        name = arguments.get("name", "")
        return json.dumps({"name": name, "type": "zone" if name in ZONES else "object"})

    elif tool_name == "replan_route":
        return json.dumps({"status": "replanned", "message": "Route replanned."})

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ============================================================================
# Model Inference
# ============================================================================

def generate_response(model, tokenizer, messages, tools):
    """Generate a single model response (used for E2E where turns are sequential)."""
    try:
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
            )

        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    except Exception as e:
        return f"[ERROR: {e}]"


def generate_batch(model, tokenizer, prompts, max_new_tokens=128):
    """Generate responses for a batch of prompts using left-padding.

    Args:
        model: the loaded model
        tokenizer: tokenizer with padding_side="left"
        prompts: list of formatted prompt strings
        max_new_tokens: max tokens to generate

    Returns:
        list of generated strings
    """
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )

    # Decode each sequence, accounting for variable left-padding
    results = []
    for batch_idx, output in enumerate(outputs):
        input_ids = inputs["input_ids"][batch_idx]
        # Find actual input length (skip left-padding)
        non_pad = (input_ids != tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(non_pad) > 0:
            actual_input_len = non_pad[-1].item() + 1
        else:
            actual_input_len = len(input_ids)
        generated = tokenizer.decode(output[actual_input_len:], skip_special_tokens=True).strip()
        results.append(generated)

    return results


# ============================================================================
# Metric 1-3: Turn-by-Turn Programmatic Evaluation (Batched)
# ============================================================================

def evaluate_turn_by_turn(model, tokenizer, examples, batch_size=8):
    """Evaluate tool selection, sequence, and argument accuracy.

    Collects all turns first, then processes in batches for speed.

    Returns dict with per-scenario and overall metrics.
    """
    results = {
        "tool_selection": {"correct": 0, "total": 0},
        "sequence": {"correct": 0, "total": 0},
        "argument": {"valid": 0, "total": 0},
        "by_scenario": defaultdict(lambda: {
            "tool_selection": {"correct": 0, "total": 0},
            "sequence_correct": 0, "sequence_total": 0,
            "argument": {"valid": 0, "total": 0},
        }),
    }

    # Phase 1: Collect all turns to evaluate
    all_turns = []
    for ex_idx, ex in enumerate(examples):
        messages = ex["messages"]
        tools = ex.get("tools", TOOLS)
        scenario = ex.get("scenario_type", "unknown")

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                continue

            # Expected tools for this turn
            expected_tools_this_turn = []
            if msg.get("tool_calls"):
                expected_tools_this_turn = [
                    tc.get("function", {}).get("name", "")
                    for tc in msg["tool_calls"]
                ]

            # Build prompt
            context = messages[:i]
            try:
                prompt = tokenizer.apply_chat_template(
                    context, tools=tools, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False,
                )
            except Exception:
                continue

            all_turns.append({
                "ex_idx": ex_idx,
                "scenario": scenario,
                "expected_tools": expected_tools_this_turn,
                "prompt": prompt,
            })

    print(f"  Collected {len(all_turns)} turns across {len(examples)} examples")

    # Phase 2: Batch generate
    generated_texts = []
    for batch_start in tqdm(range(0, len(all_turns), batch_size),
                            desc="  Batch inference", unit="batch", file=sys.stdout):
        batch_end = min(batch_start + batch_size, len(all_turns))
        batch_prompts = [t["prompt"] for t in all_turns[batch_start:batch_end]]

        try:
            batch_results = generate_batch(model, tokenizer, batch_prompts)
            generated_texts.extend(batch_results)
        except Exception as e:
            # Fallback: fill with errors
            generated_texts.extend([f"[ERROR: {e}]"] * len(batch_prompts))

    # Phase 3: Score results
    # Group turns by example to compute sequence accuracy
    example_sequences = defaultdict(list)  # ex_idx -> list of generated tool names

    for turn, generated in zip(all_turns, generated_texts):
        ex_idx = turn["ex_idx"]
        scenario = turn["scenario"]
        expected_tools_this_turn = turn["expected_tools"]

        generated_tools = extract_tool_names(generated)
        example_sequences[ex_idx].extend(generated_tools)

        # Metric 1: Tool selection (per-turn)
        if expected_tools_this_turn:
            results["tool_selection"]["total"] += 1
            results["by_scenario"][scenario]["tool_selection"]["total"] += 1

            if set(generated_tools) == set(expected_tools_this_turn):
                results["tool_selection"]["correct"] += 1
                results["by_scenario"][scenario]["tool_selection"]["correct"] += 1

        # Metric 3: Argument validity (per tool call)
        for tc in extract_tool_calls(generated):
            results["argument"]["total"] += 1
            results["by_scenario"][scenario]["argument"]["total"] += 1

            name = tc["name"]
            args = tc["arguments"]
            valid = True

            if name == "distance_to" and "destination" not in args:
                valid = False
            elif name == "navigate_to" and "destination" not in args:
                valid = False
            elif name == "find_nearest" and "object_type" not in args:
                valid = False
            elif name == "orient_me" and "target" not in args:
                valid = False
            elif name == "query" and "name" not in args:
                valid = False

            if valid:
                results["argument"]["valid"] += 1
                results["by_scenario"][scenario]["argument"]["valid"] += 1

    # Metric 2: Full sequence accuracy (per-example)
    for ex_idx, ex in enumerate(examples):
        scenario = ex.get("scenario_type", "unknown")
        expected_seq = EXPECTED_SEQUENCES.get(scenario, [])
        generated_seq = example_sequences.get(ex_idx, [])

        results["sequence"]["total"] += 1
        results["by_scenario"][scenario]["sequence_total"] += 1

        if generated_seq == expected_seq:
            results["sequence"]["correct"] += 1
            results["by_scenario"][scenario]["sequence_correct"] += 1

    return results


# ============================================================================
# Metric 4: End-to-End Simulation
# ============================================================================

# Test cases for end-to-end evaluation
E2E_TEST_CASES = [
    # Scenario A: Navigate to known location
    {"scenario": "scenario_a", "current_zone": "hri_lab",
     "user_msg": "Take me to the kitchen",
     "expected_tools": ["list_all", "distance_to", "navigate_to"],
     "success_criteria": "navigate_to called with kitchen"},
    {"scenario": "scenario_a", "current_zone": "kitchen",
     "user_msg": "I want to go to the lab",
     "expected_tools": ["list_all", "distance_to", "navigate_to"],
     "success_criteria": "navigate_to called with hri_lab"},
    {"scenario": "scenario_a", "current_zone": "hallway_1",
     "user_msg": "Bring me to hallway 0",
     "expected_tools": ["list_all", "distance_to", "navigate_to"],
     "success_criteria": "navigate_to called with hallway_0"},

    # Scenario B: Navigate to nearest
    {"scenario": "scenario_b", "current_zone": "hallway_0",
     "user_msg": "Where is the nearest chair?",
     "expected_tools": ["find_nearest", "distance_to", "navigate_to"],
     "success_criteria": "find_nearest called with chair"},
    {"scenario": "scenario_b", "current_zone": "hri_lab",
     "user_msg": "Find me the closest microwave",
     "expected_tools": ["find_nearest", "distance_to", "navigate_to"],
     "success_criteria": "find_nearest called with microwave"},

    # Scenario C: Describe surroundings
    {"scenario": "scenario_c", "current_zone": "hri_lab",
     "user_msg": "What's around me?",
     "expected_tools": ["get_robot_pose", "describe_surroundings"],
     "success_criteria": "describe_surroundings called"},
    {"scenario": "scenario_c", "current_zone": "kitchen",
     "user_msg": "Describe what you see",
     "expected_tools": ["get_robot_pose", "describe_surroundings"],
     "success_criteria": "describe_surroundings called"},

    # Scenario D: Reorientation
    {"scenario": "scenario_d", "current_zone": "hallway_2",
     "user_msg": "Where am I?",
     "expected_tools": ["get_robot_pose", "orient_me"],
     "success_criteria": "orient_me called"},
    {"scenario": "scenario_d", "current_zone": "hri_lab",
     "user_msg": "I'm lost, help me orient myself",
     "expected_tools": ["get_robot_pose", "orient_me"],
     "success_criteria": "orient_me called"},

    # Scenario E: Distance query
    {"scenario": "scenario_e", "current_zone": "hri_lab",
     "user_msg": "How far is the kitchen?",
     "expected_tools": ["distance_to"],
     "success_criteria": "distance_to called with kitchen"},
    {"scenario": "scenario_e", "current_zone": "kitchen",
     "user_msg": "What's the distance to hallway 1?",
     "expected_tools": ["distance_to"],
     "success_criteria": "distance_to called with hallway_1"},

    # Scenario F: Error handling (unknown destination)
    {"scenario": "scenario_f", "current_zone": "hri_lab",
     "user_msg": "Take me to the swimming pool",
     "expected_tools": ["list_all"],
     "success_criteria": "list_all called, no navigate_to"},

    # Scenario G: Greeting (no tools)
    {"scenario": "scenario_g", "current_zone": "hri_lab",
     "user_msg": "Hello!",
     "expected_tools": [],
     "success_criteria": "no tools called, language response given"},
    {"scenario": "scenario_g", "current_zone": "kitchen",
     "user_msg": "Thank you for your help",
     "expected_tools": [],
     "success_criteria": "no tools called, language response given"},

    # Scenario H: Return navigation
    {"scenario": "scenario_h", "current_zone": "kitchen",
     "user_msg": "Take me back to where I started",
     "expected_tools": ["get_robot_pose", "distance_to", "navigate_to"],
     "success_criteria": "navigate_to called"},
]


def evaluate_end_to_end(model, tokenizer):
    """Run full conversations with simulated tool responses.

    Returns dict with per-scenario and overall completion rates.
    """
    results = {
        "completed": 0,
        "total": len(E2E_TEST_CASES),
        "by_scenario": defaultdict(lambda: {"completed": 0, "total": 0}),
        "failures": [],
    }

    for tc in tqdm(E2E_TEST_CASES, desc="  End-to-end", unit="test", file=sys.stdout):
        scenario = tc["scenario"]
        current_zone = tc["current_zone"]
        results["by_scenario"][scenario]["total"] += 1

        # Build initial messages
        messages = [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": tc["user_msg"]},
        ]

        tools_called = []
        max_turns = 10  # safety limit
        success = False
        language_responses = 0  # track consecutive language responses

        for turn in range(max_turns):
            # Generate model response
            generated = generate_response(model, tokenizer, messages, TOOLS)

            # Extract tool calls
            tool_calls = extract_tool_calls(generated)

            if tool_calls:
                language_responses = 0  # reset counter
                # Model wants to call tools
                for tc_call in tool_calls:
                    tool_name = tc_call["name"]
                    tool_args = tc_call["arguments"]
                    tools_called.append(tool_name)

                    # Simulate tool response
                    tool_response = simulate_tool(tool_name, tool_args, current_zone)

                    # Add assistant tool call + tool response to messages
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": f"call_{turn}", "type": "function",
                                        "function": {"name": tool_name, "arguments": tool_args}}],
                    })
                    messages.append({
                        "role": "tool",
                        "content": tool_response,
                        "name": tool_name,
                    })
            else:
                # Model gave a language response (e.g., confirmation ask)
                messages.append({"role": "assistant", "content": generated})
                language_responses += 1

                # If this is the first language response and we haven't completed
                # the expected sequence, inject a user confirmation to continue
                if language_responses == 1 and not all(t in tools_called for t in tc["expected_tools"]):
                    messages.append({"role": "user", "content": "Yes, please go ahead."})
                else:
                    # Second language response or sequence complete — done
                    break

        # Check success criteria
        expected_tools = tc["expected_tools"]
        criteria = tc["success_criteria"]

        if not expected_tools:
            # Scenario G: no tools expected
            success = len(tools_called) == 0 and len(generated) > 5
        elif "no navigate_to" in criteria:
            # Scenario F: should NOT navigate
            success = "list_all" in tools_called and "navigate_to" not in tools_called
        else:
            # Check all expected tools were called
            success = all(t in tools_called for t in expected_tools)

        if success:
            results["completed"] += 1
            results["by_scenario"][scenario]["completed"] += 1
        else:
            results["failures"].append({
                "scenario": scenario,
                "user_msg": tc["user_msg"],
                "expected_tools": expected_tools,
                "actual_tools": tools_called,
                "criteria": criteria,
            })

    return results


# ============================================================================
# Metric 5: Claude Language Judge (optional)
# ============================================================================

LANGUAGE_JUDGE_PROMPT = """You are evaluating the natural language quality of a navigation assistant for vision-impaired users.

Rules the assistant must follow:
- 1-2 sentences max, spoken English only
- Distance vocab: <1m "right here", 1-3m "nearby", 3-6m "short walk away", >6m "a bit further away"
- Never expose zone codes, coordinates, or bearings
- Use relative directions: ahead, left, right, behind

SCENARIO: {scenario}
USER SAID: {user_msg}
ASSISTANT RESPONDED: {response}

Score ONLY the language quality (ignore tool calls):
1. clarity: Is it clear and easy to understand? (0.0-1.0)
2. conciseness: Is it 1-2 sentences, not verbose? (0.0-1.0)
3. safety: No exposed coordinates/jargon/confusing info? (0.0-1.0)

Respond with ONLY JSON:
{{"clarity": 0.0, "conciseness": 0.0, "safety": 0.0}}"""


def evaluate_language_with_claude(model, tokenizer, examples, client, workers=3):
    """Use Claude to judge only the natural language responses.

    Returns dict with per-scenario language scores.
    """
    import anthropic
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {
        "clarity": [], "conciseness": [], "safety": [],
        "by_scenario": defaultdict(lambda: {"clarity": [], "conciseness": [], "safety": []}),
    }

    # Collect language turns (assistant messages with text content, no tool calls)
    turns = []
    for ex in examples:
        messages = ex["messages"]
        scenario = ex.get("scenario_type", "unknown")

        for i, msg in enumerate(messages):
            if msg["role"] != "assistant" or not msg.get("content"):
                continue
            if msg.get("tool_calls"):
                continue  # skip tool-call-only turns

            # Get preceding user message
            user_msg = ""
            for j in range(i - 1, -1, -1):
                if messages[j]["role"] == "user":
                    user_msg = messages[j].get("content", "")
                    break

            # Generate model response for this turn
            context = messages[:i]
            turns.append({
                "scenario": scenario,
                "user_msg": user_msg,
                "context": context,
                "tools": ex.get("tools", TOOLS),
            })

    print(f"  {len(turns)} language turns to judge")

    # Generate model responses
    for turn in tqdm(turns, desc="  Generating", unit="turn", file=sys.stdout):
        turn["response"] = generate_response(
            model, tokenizer, turn["context"], turn["tools"]
        )

    # Judge with Claude
    def judge_one(turn):
        prompt = LANGUAGE_JUDGE_PROMPT.format(
            scenario=turn["scenario"],
            user_msg=turn["user_msg"],
            response=turn["response"],
        )
        for attempt in range(3):
            try:
                msg = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = msg.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()
                return json.loads(text)
            except anthropic.RateLimitError:
                time.sleep(2 ** attempt)
            except Exception:
                return None
        return None

    failed = 0
    with tqdm(total=len(turns), desc="  Claude judging", unit="turn", file=sys.stdout) as pbar:
        for batch_start in range(0, len(turns), workers):
            batch = turns[batch_start:batch_start + workers]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(judge_one, t): t for t in batch}
                for future in as_completed(futures):
                    turn = futures[future]
                    scores = future.result()
                    if scores:
                        for key in ["clarity", "conciseness", "safety"]:
                            val = scores.get(key, 0)
                            results[key].append(val)
                            results["by_scenario"][turn["scenario"]][key].append(val)
                    else:
                        failed += 1
                    pbar.update(1)

    if failed:
        print(f"  ({failed} Claude judgments failed)")

    return results


# ============================================================================
# Reporting
# ============================================================================

def print_report(model_name, turn_results, e2e_results, lang_results=None):
    """Print formatted evaluation report."""
    print(f"\n{'='*70}")
    print(f"EVALUATION REPORT: {model_name.upper()}")
    print(f"{'='*70}")

    # Metric 1: Tool Selection
    ts = turn_results["tool_selection"]
    ts_pct = ts["correct"] / ts["total"] * 100 if ts["total"] > 0 else 0
    print(f"\n1. TOOL SELECTION ACCURACY: {ts['correct']}/{ts['total']} ({ts_pct:.1f}%)")

    # Metric 2: Sequence Accuracy
    sq = turn_results["sequence"]
    sq_pct = sq["correct"] / sq["total"] * 100 if sq["total"] > 0 else 0
    print(f"2. SEQUENCE ACCURACY:       {sq['correct']}/{sq['total']} ({sq_pct:.1f}%)")

    # Metric 3: Argument Accuracy
    ag = turn_results["argument"]
    ag_pct = ag["valid"] / ag["total"] * 100 if ag["total"] > 0 else 0
    print(f"3. ARGUMENT ACCURACY:       {ag['valid']}/{ag['total']} ({ag_pct:.1f}%)")

    # Metric 4: End-to-End
    e2e_pct = e2e_results["completed"] / e2e_results["total"] * 100
    print(f"4. END-TO-END COMPLETION:   {e2e_results['completed']}/{e2e_results['total']} ({e2e_pct:.1f}%)")

    # Metric 5: Language (if available)
    if lang_results and lang_results["clarity"]:
        avg_c = sum(lang_results["clarity"]) / len(lang_results["clarity"])
        avg_n = sum(lang_results["conciseness"]) / len(lang_results["conciseness"])
        avg_s = sum(lang_results["safety"]) / len(lang_results["safety"])
        avg_all = (avg_c + avg_n + avg_s) / 3
        print(f"5. LANGUAGE QUALITY:        {avg_all:.3f} (clarity={avg_c:.2f}, concise={avg_n:.2f}, safety={avg_s:.2f})")

    # Per-scenario breakdown
    print(f"\n{'='*70}")
    print("PER-SCENARIO BREAKDOWN")
    print(f"{'='*70}")

    scenarios = sorted(set(
        list(turn_results["by_scenario"].keys()) +
        list(e2e_results["by_scenario"].keys())
    ))

    header = f"{'Scenario':<15} {'Tool Sel':<10} {'Sequence':<10} {'Args':<10} {'E2E':<10}"
    if lang_results:
        header += f" {'Language':<10}"
    print(header)
    print("-" * len(header))

    for scenario in scenarios:
        # Turn-by-turn metrics
        ts_s = turn_results["by_scenario"].get(scenario, {})
        ts_sel = ts_s.get("tool_selection", {"correct": 0, "total": 0})
        ts_pct = ts_sel["correct"] / ts_sel["total"] * 100 if ts_sel["total"] > 0 else 0

        sq_c = ts_s.get("sequence_correct", 0)
        sq_t = ts_s.get("sequence_total", 0)
        sq_pct = sq_c / sq_t * 100 if sq_t > 0 else 0

        ag_s = ts_s.get("argument", {"valid": 0, "total": 0})
        ag_pct = ag_s["valid"] / ag_s["total"] * 100 if ag_s["total"] > 0 else 0

        # E2E metrics
        e2e_s = e2e_results["by_scenario"].get(scenario, {"completed": 0, "total": 0})
        e2e_pct = e2e_s["completed"] / e2e_s["total"] * 100 if e2e_s["total"] > 0 else 0

        row = f"{scenario:<15} {ts_pct:>6.1f}%   {sq_pct:>6.1f}%   {ag_pct:>6.1f}%   {e2e_pct:>6.1f}%  "

        if lang_results:
            lang_s = lang_results["by_scenario"].get(scenario, {"clarity": [], "conciseness": [], "safety": []})
            if lang_s["clarity"]:
                avg = (sum(lang_s["clarity"]) + sum(lang_s["conciseness"]) + sum(lang_s["safety"])) / (3 * len(lang_s["clarity"]))
                row += f" {avg:>6.3f}   "
            else:
                row += f"    N/A   "

        print(row)

    # E2E Failures
    if e2e_results["failures"]:
        print(f"\nEND-TO-END FAILURES ({len(e2e_results['failures'])}):")
        for f in e2e_results["failures"]:
            print(f"  [{f['scenario']}] \"{f['user_msg']}\"")
            print(f"    Expected: {f['expected_tools']}")
            print(f"    Got:      {f['actual_tools']}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Rigorous model evaluation")
    parser.add_argument("--model", choices=["2b", "4b", "both"], default="both")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--skip-claude", action="store_true", help="Skip language quality evaluation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=3, help="Claude API workers")
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    # Load validation data
    val_file = "finetune/data/val.jsonl"
    with open(val_file) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if args.num_examples:
        examples = examples[:args.num_examples]

    print(f"Loaded {len(examples)} validation examples")
    scenario_counts = defaultdict(int)
    for ex in examples:
        scenario_counts[ex.get("scenario_type", "unknown")] += 1
    for s in sorted(scenario_counts):
        print(f"  {s}: {scenario_counts[s]}")

    # Claude client (if needed)
    claude_client = None
    if not args.skip_claude:
        try:
            import anthropic
            claude_client = anthropic.Anthropic()
            print("Claude API available for language evaluation")
        except Exception:
            print("Claude API not available, skipping language evaluation")

    models_to_test = ["2b", "4b"] if args.model == "both" else [args.model]
    all_results = {}

    for model_name in models_to_test:
        model_path = FINETUNED_MODELS[model_name]

        if not Path(model_path).exists():
            print(f"\nModel not found at {model_path}, skipping.")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating {model_name.upper()}")
        print(f"{'='*70}")

        # Load model
        print("Loading model...")
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

        # Metrics 1-3: Turn-by-turn (batched)
        batch_size = 16 if model_name == "2b" else 8
        print(f"\nMetrics 1-3: Turn-by-turn evaluation (batch_size={batch_size})...")
        turn_results = evaluate_turn_by_turn(model, tokenizer, examples, batch_size=batch_size)

        # Metric 4: End-to-end
        print("\nMetric 4: End-to-end simulation...")
        e2e_results = evaluate_end_to_end(model, tokenizer)

        # Metric 5: Language quality (Claude)
        lang_results = None
        if claude_client:
            print("\nMetric 5: Language quality (Claude judge)...")
            lang_results = evaluate_language_with_claude(
                model, tokenizer, examples, claude_client, workers=args.workers,
            )

        # Report
        print_report(model_name, turn_results, e2e_results, lang_results)

        # Save
        if args.save_results:
            output = {
                "model": model_name,
                "turn_by_turn": {
                    "tool_selection": turn_results["tool_selection"],
                    "sequence": turn_results["sequence"],
                    "argument": turn_results["argument"],
                },
                "end_to_end": {
                    "completed": e2e_results["completed"],
                    "total": e2e_results["total"],
                    "failures": e2e_results["failures"],
                },
            }
            if lang_results:
                output["language"] = {
                    "clarity": sum(lang_results["clarity"]) / len(lang_results["clarity"]) if lang_results["clarity"] else 0,
                    "conciseness": sum(lang_results["conciseness"]) / len(lang_results["conciseness"]) if lang_results["conciseness"] else 0,
                    "safety": sum(lang_results["safety"]) / len(lang_results["safety"]) if lang_results["safety"] else 0,
                }
            with open(f"finetune/eval_results_{model_name}.json", "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved to finetune/eval_results_{model_name}.json")

        all_results[model_name] = (turn_results, e2e_results, lang_results)

        # Cleanup
        del model, tokenizer
        torch.cuda.empty_cache()

    # Comparison
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON")
        print(f"{'='*70}\n")

        print(f"{'Model':<10} {'Tool Sel':<12} {'Sequence':<12} {'Args':<12} {'E2E':<12}")
        print("-" * 58)
        for mn in ["2b", "4b"]:
            if mn not in all_results:
                continue
            tr, e2e, lang = all_results[mn]
            ts = tr["tool_selection"]
            sq = tr["sequence"]
            ag = tr["argument"]
            ts_p = ts["correct"] / ts["total"] * 100 if ts["total"] > 0 else 0
            sq_p = sq["correct"] / sq["total"] * 100 if sq["total"] > 0 else 0
            ag_p = ag["valid"] / ag["total"] * 100 if ag["total"] > 0 else 0
            e2e_p = e2e["completed"] / e2e["total"] * 100
            print(f"{mn:<10} {ts_p:>6.1f}%     {sq_p:>6.1f}%     {ag_p:>6.1f}%     {e2e_p:>6.1f}%")


if __name__ == "__main__":
    main()
