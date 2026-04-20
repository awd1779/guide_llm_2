"""Scenario D: Reorientation / Where am I."""

import json
from .constants import (
    CANONICAL_SYSTEM_PROMPT,
    ZONES,
    OBJECTS,
    OBJECT_ZONES,
    DIRECTION_MATRIX,
)


def get_robot_pose_response(current_zone):
    """Generate get_robot_pose tool response."""
    return {
        "x": 1.5,
        "y": 2.3,
        "yaw_deg": 45.0,
        "frame": "map",
        "current_zone": current_zone,
    }


def get_orient_me_response(target, current_zone):
    """Generate orient_me tool response."""
    direction = DIRECTION_MATRIX.get((current_zone, target), "ahead")
    turn_deg = {
        "ahead": 0,
        "ahead-left": -45,
        "left": -90,
        "behind-left": -135,
        "behind": 180,
        "behind-right": 135,
        "right": 90,
        "ahead-right": 45,
    }.get(direction, 0)

    return {
        "target": target,
        "direction_from_current": direction,
        "turn_degrees": turn_deg,
        "message": f"Rotating to face {target}",
    }


def generate_vars():
    """Generate all variations of Scenario D."""
    for zone in ZONES:
        # Pick a random object in or near this zone as target
        for obj in OBJECTS[:3]:  # Just use first 3 objects to reduce combinations
            if OBJECT_ZONES[obj] != zone:
                yield {
                    "current_zone": zone,
                    "target": obj,
                }


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario D slots."""
    return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User asks robot where it is and gets reoriented.

CONTEXT:
- Robot is in: {template_vars['current_zone']}
- Robot will be oriented to face: {template_vars['target']}

FILL THESE SLOTS (JSON only):

slot_user_request: User asks where they are or for reorientation.
  Vary: "Where am I?", "What's my location?", "Can you orient me?", "Which way am I facing?", "Tell me where we are"

slot_response: Assistant's 1-2 sentences describing location and new orientation.
  Example: "You're in the lab. I've rotated you to face the kitchen."

{{
  "slot_user_request": "...",
  "slot_response": "..."
}}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario D examples (batch)."""
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        examples_text.append(f"""
EXAMPLE {i}:
- Robot is in: {tv['current_zone']}
- Robot will be oriented to face: {tv['target']}
""")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot navigation training examples.
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

SCENARIO: User asks robot where it is and gets reoriented.

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

slot_user_request: User asks where they are or for reorientation.
  Vary: "Where am I?", "What's my location?", "Can you orient me?", "Which way am I facing?", "Tell me where we are"

slot_response: Assistant's 1-2 sentences describing location and new orientation.

[
  {{"slot_user_request": "...", "slot_response": "..."}},
  {{"slot_user_request": "...", "slot_response": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton."""
    zone = template_vars["current_zone"]
    target = template_vars["target"]

    messages = [
        {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
        {"role": "user", "content": slot_fills["slot_user_request"]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_robot_pose",
                        "arguments": json.dumps({}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(get_robot_pose_response(zone)),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "orient_me",
                        "arguments": json.dumps({"target": target}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": json.dumps(get_orient_me_response(target, zone)),
        },
        {
            "role": "assistant",
            "content": slot_fills["slot_response"],
        },
    ]

    return messages


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return ["get_robot_pose", "orient_me"]
