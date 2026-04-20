"""Scenario C: Describe surroundings."""

import json
from .constants import (
    CANONICAL_SYSTEM_PROMPT,
    ZONES,
    OBJECTS,
    OBJECT_ZONES,
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


def get_describe_surroundings_response(current_zone):
    """Generate describe_surroundings tool response."""
    # Find objects in this zone
    nearby = [
        {
            "label": obj,
            "distance_m": 1.2 + (OBJECTS.index(obj) * 0.5),
            "direction": ["ahead", "left", "right", "behind"][
                OBJECTS.index(obj) % 4
            ],
        }
        for obj in OBJECTS
        if OBJECT_ZONES[obj] == current_zone
    ]

    return {
        "current_zone": current_zone,
        "radius_m": 3.0,
        "nearby_objects": nearby,
    }


def generate_vars():
    """Generate all variations of Scenario C."""
    for zone in ZONES:
        yield {
            "current_zone": zone,
        }


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario C slots."""
    return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User asks robot to describe surroundings.

CONTEXT:
- Robot is in: {template_vars['current_zone']}

FILL THESE SLOTS (JSON only):

slot_user_request: User asks what's around them.
  Vary phrasing: "What's around me?", "Describe my surroundings", "What do you see?", "What's nearby?", "What's in this area?"

slot_response: Assistant's 1-2 sentence description of surroundings.
  Should describe the zone and nearby objects in natural language.
  Example: "You're in the lab. There's a chair to your left, a TV ahead, and a table to your right."

{{
  "slot_user_request": "...",
  "slot_response": "..."
}}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario C examples (batch)."""
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        examples_text.append(f"""
EXAMPLE {i}:
- Robot is in: {tv['current_zone']}
""")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot navigation training examples.
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

SCENARIO: User asks robot to describe surroundings.

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

slot_user_request: User asks what's around them.
  Vary phrasing: "What's around me?", "Describe my surroundings", "What do you see?", "What's nearby?", "What's in this area?"

slot_response: Assistant's 1-2 sentence description of surroundings describing the zone and nearby objects in natural language.

[
  {{"slot_user_request": "...", "slot_response": "..."}},
  {{"slot_user_request": "...", "slot_response": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton."""
    zone = template_vars["current_zone"]

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
                        "name": "describe_surroundings",
                        "arguments": json.dumps({}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": json.dumps(get_describe_surroundings_response(zone)),
        },
        {
            "role": "assistant",
            "content": slot_fills["slot_response"],
        },
    ]

    return messages


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return ["get_robot_pose", "describe_surroundings"]
