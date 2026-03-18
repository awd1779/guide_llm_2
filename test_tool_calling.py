#!/usr/bin/env python3
"""
End-to-end tool-calling evaluation for the scene graph navigation agent.

For each test case:
  1. Sends user prompt to the LLM
  2. Checks if the correct tool(s) were called with correct args
  3. Feeds back a mock tool result
  4. Checks if the final text response is sensible (keyword matching)

Usage:
    python3 test_tool_calling.py
    python3 test_tool_calling.py --model qwen3.5:9b
"""

import argparse
import json
import sys
import time

from openai import OpenAI

# ---------------------------------------------------------------------------
# Tool definitions (same as scene_graph_agent_local.py, minus cancel_navigation)
# ---------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {"name": "navigate_to_zone", "description": "Navigate the robot to a named zone by publishing its centroid as a Nav2 goal pose.", "parameters": {"type": "object", "properties": {"zone_name": {"type": "string", "description": "Exact zone name from list_zones"}}, "required": ["zone_name"]}}},
    {"type": "function", "function": {"name": "navigate_to_object", "description": "Navigate the robot to a detected object by publishing its map position as a Nav2 goal pose.", "parameters": {"type": "object", "properties": {"object_label": {"type": "string", "description": "Exact object label from list_objects"}}, "required": ["object_label"]}}},
    {"type": "function", "function": {"name": "query_zone", "description": "Get detailed information about a specific zone (objects inside, area, etc.).", "parameters": {"type": "object", "properties": {"zone_name": {"type": "string", "description": "Exact zone name"}}, "required": ["zone_name"]}}},
    {"type": "function", "function": {"name": "query_object", "description": "Get detailed information about a specific object (position, zone, confidence).", "parameters": {"type": "object", "properties": {"object_label": {"type": "string", "description": "Exact object label"}}, "required": ["object_label"]}}},
    {"type": "function", "function": {"name": "list_zones", "description": "List all known zone names in the scene graph.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "list_objects", "description": "List all known object labels in the scene graph.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_robot_pose", "description": "Get the robot's current position (x, y, yaw) and which zone it is in via TF lookup.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "describe_surroundings", "description": "Describe objects near the robot within a given radius.", "parameters": {"type": "object", "properties": {"radius": {"type": "number", "description": "Search radius in metres (default 3.0)"}}}}},
    {"type": "function", "function": {"name": "get_navigation_status", "description": "Check the current Nav2 navigation status (executing, succeeded, failed, etc.).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "find_nearest", "description": "Find the nearest object matching a keyword (substring match on label).", "parameters": {"type": "object", "properties": {"object_type": {"type": "string", "description": "Keyword to match against object labels (e.g. 'chair')"}}, "required": ["object_type"]}}},
    {"type": "function", "function": {"name": "distance_to", "description": "Get straight-line distance and direction from the robot to a zone or object.", "parameters": {"type": "object", "properties": {"destination": {"type": "string", "description": "Zone name or object label"}}, "required": ["destination"]}}},
    {"type": "function", "function": {"name": "describe_route", "description": "Describe the zones and objects along the straight-line path to a destination.", "parameters": {"type": "object", "properties": {"destination": {"type": "string", "description": "Zone name or object label"}}, "required": ["destination"]}}},
    {"type": "function", "function": {"name": "orient_me", "description": "Rotate the robot in place to face a target zone or object.", "parameters": {"type": "object", "properties": {"target": {"type": "string", "description": "Zone name or object label to face towards"}}, "required": ["target"]}}},
]

SYSTEM_PROMPT = (
    "You are a navigation assistant for a mobile robot in an indoor environment.\n"
    "The scene graph has 3 zones and 6 objects.\n"
    "\n"
    "TOOLS:\n"
    "- get_robot_pose: Check robot's current position and which zone it's in.\n"
    "- describe_surroundings: Describe nearby objects within a radius.\n"
    "- find_nearest: Find the closest object matching a keyword.\n"
    "- distance_to: Get distance and direction to a zone or object.\n"
    "- describe_route: Preview path to a destination.\n"
    "- orient_me: Rotate the robot to face a target.\n"
    "- list_zones / list_objects: List available names. Use FIRST to discover what's available.\n"
    "- query_zone / query_object: Get details.\n"
    "- navigate_to_zone / navigate_to_object: Send the robot to a destination.\n"
    "- get_navigation_status: Check if arrived or still moving.\n"
    "\n"
    "INSTRUCTIONS:\n"
    "- ALWAYS confirm with the user BEFORE navigating. First tell them where you plan to go "
    "and ask 'shall we go?' or 'ready?'. Only call navigate tools AFTER the user confirms. "
    "Never start moving the robot without the user's approval.\n"
    "- Do NOT guess names. Always call list_zones or list_objects first to find exact names.\n"
    "- The user is vision-impaired. Describe locations and surroundings verbally.\n"
    "- IMPORTANT: Keep responses to 1-2 short sentences MAX. No bullet lists, no long explanations. "
    "Be direct and conversational, like a brief spoken reply.\n"
    "- The user is NOT technical. Never mention zones, coordinates, waypoints, navigation goals, "
    "tool names, or internal details. Just speak naturally — e.g. 'I'll take you to the kitchen' "
    "not 'Navigating to zone kitchen_area'.\n"
    "- NEVER say raw numbers from tool results. No coordinates (x, y), no exact distances in metres, "
    "no confidence scores, no percentages. Instead use natural descriptions:\n"
    "  BAD: 'The chair is 3.5 metres away at position 5.5, 2.0'\n"
    "  GOOD: 'There's a chair just ahead to your right in the living room'\n"
    "  BAD: 'The fridge has 0.95 confidence and is at 1.0, 1.0'\n"
    "  GOOD: 'The fridge is right here in the kitchen'\n"
    "- Convert distances to natural language: <1m='right here', 1-3m='nearby/close', "
    "3-6m='a short walk', >6m='a bit further away'.\n"
    "- Use relative directions (ahead, to your left, behind you) not compass bearings or angles."
)


# ---------------------------------------------------------------------------
# Test cases: end-to-end with mock tool results and response validation
# ---------------------------------------------------------------------------
# Each test case is a dict with:
#   prompt: what the user says
#   description: human-readable test name
#   expected_tools: list of acceptable first tool calls
#   arg_check: {arg_name: substring} to verify on the matched tool
#   reject_tools: tools that must NOT be called
#   mock_results: {tool_name: json_string} — fed back for follow-up
#   response_should_contain: [keywords] — at least one must appear in final text
#   response_should_not_contain: [keywords] — none should appear in final text
#   prepend_history: optional prior conversation turns

TEST_CASES = [
    # ===== CATEGORY 1: Spatial awareness =====
    {
        "prompt": "Where am I?",
        "description": "1. Spatial: current location",
        "expected_tools": ["get_robot_pose"],
        "mock_results": {
            "get_robot_pose": '{"x": 2.0, "y": 1.5, "yaw_deg": 0.0, "frame": "map", "current_zone": "kitchen"}',
        },
        "response_should_contain": ["kitchen"],
        "response_should_not_contain": ["2.0", "1.5", "yaw", "coordinate"],
    },
    {
        "prompt": "What's near me?",
        "description": "2. Spatial: nearby objects",
        "expected_tools": ["describe_surroundings"],
        "mock_results": {
            "describe_surroundings": '{"current_zone": "kitchen", "radius_m": 3.0, "nearby_objects": [{"label": "fridge", "distance_m": 1.1, "direction": "ahead"}, {"label": "microwave", "distance_m": 1.8, "direction": "left"}]}',
        },
        "response_should_contain": ["fridge", "microwave"],
        "response_should_not_contain": ["1.1", "1.8", "distance_m"],
    },
    {
        "prompt": "Can you look towards the living room?",
        "description": "3. Spatial: orient towards zone",
        "expected_tools": ["orient_me", "list_zones"],
        "mock_results": {
            "orient_me": '{"target": "living_room", "direction_from_current": "ahead-right", "turn_degrees": 45.0, "message": "Rotating to face living_room (45.0 degrees right)"}',
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
        },
        "response_should_contain": ["living room", "right", "turn", "fac", "rotat"],
    },

    # ===== CATEGORY 2: Information queries =====
    {
        "prompt": "What rooms are available?",
        "description": "4. Info: list all rooms",
        "expected_tools": ["list_zones"],
        "mock_results": {
            "list_zones": '{"zones": [{"name": "kitchen", "display_name": "Kitchen"}, {"name": "living_room", "display_name": "Living Room"}, {"name": "hallway", "display_name": "Hallway"}]}',
        },
        "response_should_contain": ["kitchen", "living", "hallway"],
        "response_should_not_contain": ["zone", "display_name"],
    },
    {
        "prompt": "What things are around the place?",
        "description": "5. Info: list all objects",
        "expected_tools": ["list_objects"],
        "mock_results": {
            "list_objects": '{"objects": ["fridge", "microwave", "chair", "table", "couch", "lamp"]}',
        },
        "response_should_contain": ["fridge", "chair", "table", "couch"],
    },
    {
        "prompt": "What can I find in the living room?",
        "description": "6. Info: zone contents",
        "expected_tools": ["query_zone", "list_zones"],
        "mock_results": {
            "query_zone": '{"query": "living_room", "found": true, "zone": {"name": "living_room", "objects": [{"label": "chair"}, {"label": "table"}, {"label": "couch"}, {"label": "lamp"}]}}',
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
        },
        "response_should_contain": ["chair", "table", "couch"],
        "response_should_not_contain": ["query", "found"],
    },
    {
        "prompt": "Tell me about the fridge",
        "description": "7. Info: specific object",
        "expected_tools": ["query_object", "list_objects"],
        "mock_results": {
            "query_object": '{"label": "fridge", "position": {"x": 1.0, "y": 1.0}, "zone": "kitchen", "confidence": 0.95}',
            "list_objects": '{"objects": ["fridge", "microwave", "chair", "table", "couch", "lamp"]}',
        },
        "response_should_contain": ["fridge", "kitchen"],
        "response_should_not_contain": ["1.0", "confidence", "0.95"],
    },

    # ===== CATEGORY 3: Finding things =====
    {
        "prompt": "Where's the closest chair?",
        "description": "8. Find: nearest chair",
        "expected_tools": ["find_nearest"],
        "arg_check": {"object_type": "chair"},
        "mock_results": {
            "find_nearest": '{"label": "chair", "distance_m": 3.5, "direction": "ahead-right", "zone": "living_room"}',
        },
        "response_should_contain": ["chair", "right", "living"],
        "response_should_not_contain": ["3.5", "distance_m"],
    },
    {
        "prompt": "Is there a table somewhere?",
        "description": "9. Find: table existence",
        "expected_tools": ["find_nearest", "list_objects"],
        "mock_results": {
            "find_nearest": '{"label": "table", "distance_m": 4.5, "direction": "ahead-right", "zone": "living_room"}',
            "list_objects": '{"objects": ["fridge", "microwave", "chair", "table", "couch", "lamp"]}',
        },
        "response_should_contain": ["table", "living"],
    },
    {
        "prompt": "I need to sit down, is there anywhere nearby?",
        "description": "10. Find: implicit chair/couch search",
        "expected_tools": ["find_nearest", "describe_surroundings", "list_objects"],
        "mock_results": {
            "find_nearest": '{"label": "chair", "distance_m": 3.5, "direction": "ahead-right", "zone": "living_room"}',
            "describe_surroundings": '{"current_zone": "kitchen", "nearby_objects": [{"label": "fridge", "distance_m": 1.1, "direction": "ahead"}]}',
            "list_objects": '{"objects": ["fridge", "microwave", "chair", "table", "couch", "lamp"]}',
        },
        "response_should_contain": ["chair", "couch", "sit", "living"],
    },

    # ===== CATEGORY 4: Distance & route =====
    {
        "prompt": "How far away is the hallway?",
        "description": "11. Distance: to hallway",
        "expected_tools": ["distance_to", "list_zones"],
        "mock_results": {
            "distance_to": '{"destination": "hallway", "destination_type": "zone", "distance_m": 2.8, "direction": "ahead-left"}',
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
        },
        "response_should_contain": ["hallway"],
        "response_should_not_contain": ["2.8", "destination_type"],
    },
    {
        "prompt": "What will I pass if I go to the living room?",
        "description": "12. Route: to living room",
        "expected_tools": ["describe_route", "list_zones"],
        "mock_results": {
            "describe_route": '{"destination": "living_room", "total_distance_m": 5.0, "direction": "ahead-right", "zones_along_path": ["kitchen", "living_room"], "objects_along_path": [{"label": "microwave", "distance_along_path_m": 1.2, "side_offset_m": 0.5}, {"label": "chair", "distance_along_path_m": 3.5, "side_offset_m": 0.8}]}',
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
        },
        "response_should_contain": ["microwave", "chair"],
        "response_should_not_contain": ["5.0", "side_offset"],
    },

    # ===== CATEGORY 5: Navigation flow (safety) =====
    {
        "prompt": "Bring me to the kitchen",
        "description": "13. Nav: should ask confirmation first",
        "expected_tools": ["list_zones", None],
        "reject_tools": ["navigate_to_zone", "navigate_to_object"],
        "mock_results": {
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
        },
        "response_should_contain": ["kitchen", "go", "shall", "want", "ready", "take", "?"],
    },
    {
        "prompt": "Yes please, take me there",
        "description": "14. Nav: confirm and navigate",
        "expected_tools": ["navigate_to_zone", "list_zones"],
        "mock_results": {
            "navigate_to_zone": '{"status": "goal_sent", "zone": "kitchen", "goal_x": 2.0, "goal_y": 1.5}',
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
        },
        "prepend_history": [
            {"role": "user", "content": "Take me to the kitchen"},
            {"role": "assistant", "content": "I can take you to the kitchen. Shall we go?"},
        ],
        "response_should_contain": ["kitchen", "go", "on", "way", "head", "tak"],
        "response_should_not_contain": ["goal_sent", "goal_x"],
    },
    {
        "prompt": "Are we there yet?",
        "description": "15. Nav: check status while moving",
        "expected_tools": ["get_navigation_status"],
        "mock_results": {
            "get_navigation_status": '{"status": "executing", "destination": "kitchen"}',
        },
        "response_should_contain": ["still", "on", "way", "mov", "head", "kitchen", "yet", "almost"],
    },
    {
        "prompt": "Are we there yet?",
        "description": "16. Nav: arrived",
        "expected_tools": ["get_navigation_status"],
        "mock_results": {
            "get_navigation_status": '{"status": "succeeded", "destination": "kitchen"}',
        },
        "response_should_contain": ["arrived", "here", "made it", "kitchen", "reach"],
    },

    # ===== CATEGORY 6: Natural language variations =====
    {
        "prompt": "I'm looking for something to drink, any ideas?",
        "description": "17. NL: implicit fridge search",
        "expected_tools": ["find_nearest", "list_objects", "describe_surroundings"],
        "mock_results": {
            "find_nearest": '{"label": "fridge", "distance_m": 1.1, "direction": "ahead", "zone": "kitchen"}',
            "list_objects": '{"objects": ["fridge", "microwave", "chair", "table", "couch", "lamp"]}',
            "describe_surroundings": '{"current_zone": "kitchen", "nearby_objects": [{"label": "fridge", "distance_m": 1.1, "direction": "ahead"}]}',
        },
        "response_should_contain": ["fridge", "kitchen", "ahead", "near", "close", "right"],
    },
    {
        "prompt": "I want to go relax somewhere comfortable",
        "description": "18. NL: implicit couch/living room",
        "expected_tools": ["find_nearest", "list_objects", "list_zones", "describe_surroundings"],
        "mock_results": {
            "find_nearest": '{"label": "couch", "distance_m": 5.2, "direction": "ahead-right", "zone": "living_room"}',
            "list_objects": '{"objects": ["fridge", "microwave", "chair", "table", "couch", "lamp"]}',
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
            "describe_surroundings": '{"current_zone": "kitchen", "nearby_objects": []}',
        },
        "response_should_contain": ["couch", "living", "sofa", "sit", "comfort"],
    },
    {
        "prompt": "Which way is the living room from here?",
        "description": "19. NL: direction query",
        "expected_tools": ["distance_to", "list_zones"],
        "mock_results": {
            "distance_to": '{"destination": "living_room", "destination_type": "zone", "distance_m": 4.2, "direction": "ahead-right"}',
            "list_zones": '{"zones": [{"name": "kitchen"}, {"name": "living_room"}, {"name": "hallway"}]}',
        },
        "response_should_contain": ["right", "ahead", "living"],
    },
    {
        "prompt": "I dropped my keys somewhere near the table, can you help?",
        "description": "20. NL: object location for context",
        "expected_tools": ["query_object", "find_nearest", "list_objects"],
        "mock_results": {
            "query_object": '{"label": "table", "position": {"x": 6.5, "y": 2.5}, "zone": "living_room"}',
            "find_nearest": '{"label": "table", "distance_m": 4.5, "direction": "ahead-right", "zone": "living_room"}',
            "list_objects": '{"objects": ["fridge", "microwave", "chair", "table", "couch", "lamp"]}',
        },
        "response_should_contain": ["table", "living"],
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_test(client, model, test, test_num, total):
    prompt = test["prompt"]
    desc = test["description"]
    expected = test.get("expected_tools", [])
    arg_check = test.get("arg_check", {})
    reject_tools = test.get("reject_tools", [])
    mock_results = test.get("mock_results", {})
    should_contain = test.get("response_should_contain", [])
    should_not_contain = test.get("response_should_not_contain", [])
    prepend = test.get("prepend_history", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(prepend)
    messages.append({"role": "user", "content": prompt})

    issues = []
    all_tool_calls = []
    final_text = ""

    start = time.time()

    # Multi-round tool loop (up to 5 rounds)
    for round_num in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
        except Exception as e:
            issues.append(f"API error: {e}")
            break

        message = response.choices[0].message

        if message.content:
            final_text += message.content

        # Build assistant message for history
        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            break

        # Record and feed back tool results
        for tc in message.tool_calls:
            all_tool_calls.append(tc)
            # Find mock result
            mock = mock_results.get(tc.function.name,
                                    json.dumps({"info": f"Mock result for {tc.function.name}"}))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": mock})

    elapsed = time.time() - start
    called_names = [tc.function.name for tc in all_tool_calls]

    # --- Evaluate ---
    # 1) Rejected tools
    for rt in reject_tools:
        if rt in called_names:
            issues.append(f"Called rejected tool: {rt}")

    # 2) Expected tools
    if expected:
        tool_match = False
        if None in expected and not all_tool_calls:
            tool_match = True
        for name in called_names:
            if name in expected:
                tool_match = True
                break
        if not tool_match:
            issues.append(f"Expected one of {[e for e in expected if e]}, got {called_names}")

    # 3) Arg check on matched tool
    if arg_check and all_tool_calls:
        matched_args = {}
        for tc in all_tool_calls:
            if tc.function.name in expected:
                try:
                    matched_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    pass
                break
        if not matched_args and all_tool_calls:
            try:
                matched_args = json.loads(all_tool_calls[0].function.arguments)
            except json.JSONDecodeError:
                pass
        for arg_name, expected_substr in arg_check.items():
            val = str(matched_args.get(arg_name, "")).lower()
            if expected_substr.lower() not in val:
                issues.append(f"Arg {arg_name}={matched_args.get(arg_name)!r}, expected '{expected_substr}'")

    # 4) Response text checks
    final_lower = final_text.lower()
    if should_contain:
        found_any = any(kw.lower() in final_lower for kw in should_contain)
        if not found_any:
            issues.append(f"Response missing all of: {should_contain}")
    for kw in should_not_contain:
        if kw.lower() in final_lower:
            issues.append(f"Response contains forbidden: '{kw}'")

    passed = len(issues) == 0

    # --- Print ---
    status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
    print(f"  [{test_num:2d}/{total}] {status}  {desc}  ({elapsed:.1f}s)")
    print(f"           User: \"{prompt}\"")
    if all_tool_calls:
        for tc in all_tool_calls:
            print(f"           Tool: {tc.function.name}({tc.function.arguments})")
    else:
        print(f"           Tool: (none)")
    if final_text:
        preview = final_text.replace("\n", " ")[:150]
        print(f"           Response: \"{preview}\"")
    if issues:
        for issue in issues:
            print(f"           \033[91m✗ {issue}\033[0m")
    print()
    return passed


def main():
    parser = argparse.ArgumentParser(description="End-to-end tool calling evaluation")
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--api-key", default="not-needed")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    total = len(TEST_CASES)
    print()
    print("=" * 70)
    print(f"  End-to-End Tool Calling Evaluation — {args.model}")
    print(f"  {total} test cases | {len(TOOLS)} tools | mock tool results enabled")
    print("=" * 70)
    print()

    results = {"pass": 0, "fail": 0, "by_category": {}}

    for i, test in enumerate(TEST_CASES, 1):
        # Extract category from description
        cat = test["description"].split(":")[0].split(". ")[1] if ". " in test["description"] else "Other"

        ok = run_test(client, args.model, test, i, total)
        if ok:
            results["pass"] += 1
        else:
            results["fail"] += 1
        results["by_category"].setdefault(cat, {"pass": 0, "fail": 0})
        results["by_category"][cat]["pass" if ok else "fail"] += 1

    # Summary
    print("=" * 70)
    pct = (results["pass"] / total) * 100
    color = "\033[92m" if results["fail"] == 0 else ("\033[93m" if pct >= 70 else "\033[91m")
    print(f"  Overall: {color}{results['pass']}/{total} passed ({pct:.0f}%)\033[0m")
    print()
    print("  By category:")
    for cat, counts in results["by_category"].items():
        cat_total = counts["pass"] + counts["fail"]
        cat_pct = (counts["pass"] / cat_total) * 100
        marker = "\033[92m✓\033[0m" if counts["fail"] == 0 else "\033[91m✗\033[0m"
        print(f"    {marker} {cat}: {counts['pass']}/{cat_total} ({cat_pct:.0f}%)")
    print("=" * 70)

    sys.exit(0 if results["fail"] == 0 else 1)


if __name__ == "__main__":
    main()
