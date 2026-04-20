"""Scenario B: Navigate to nearest instance of something."""

import json
from .constants import (
    CANONICAL_SYSTEM_PROMPT,
    OBJECTS,
    ZONES,
    OBJECT_ZONES,
    DISTANCE_MATRIX,
    DIRECTION_MATRIX,
    distance_to_vocab,
)


def get_find_nearest_response(obj_type):
    """Generate find_nearest tool response."""
    zone = OBJECT_ZONES[obj_type]
    # Use average distance from all zones to this object's zone
    distances = [DISTANCE_MATRIX.get((z, zone), 5.0) for z in ZONES if z != zone]
    distance_m = sum(distances) / len(distances) if distances else 5.0

    direction = "ahead-left"  # Default
    return {
        "label": obj_type,
        "distance_m": distance_m,
        "direction": direction,
        "zone": zone,
    }


def get_distance_to_response(obj_type, current_zone):
    """Generate distance_to tool response."""
    obj_zone = OBJECT_ZONES[obj_type]
    distance_m = DISTANCE_MATRIX.get((current_zone, obj_zone), 5.0)
    direction = DIRECTION_MATRIX.get((current_zone, obj_zone), "ahead")

    return {
        "destination": obj_type,
        "destination_type": "object",
        "distance_m": distance_m,
        "direction": direction,
    }


def get_navigate_to_response(obj_type):
    """Generate navigate_to tool response."""
    return {
        "status": "goal_sent",
        "destination": obj_type,
        "type": "object",
    }


def generate_vars():
    """Generate all variations of Scenario B.

    Yields:
        dict with keys: object_type, current_zone, distance_m, direction, dest_vocab
    """
    for current_zone in ZONES:
        for obj_type in OBJECTS:
            obj_zone = OBJECT_ZONES[obj_type]
            if obj_zone == current_zone:
                continue  # Skip objects in current zone

            distance_m = DISTANCE_MATRIX.get((current_zone, obj_zone), 5.0)
            direction = DIRECTION_MATRIX.get((current_zone, obj_zone), "ahead")

            yield {
                "object_type": obj_type,
                "current_zone": current_zone,
                "distance_m": distance_m,
                "direction": direction,
                "dest_vocab": distance_to_vocab(distance_m),
            }


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario B slots."""
    obj_type = template_vars["object_type"]
    return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User wants robot to find and navigate to nearest instance of an object type.

CONTEXT:
- Robot is currently in: {template_vars['current_zone']}
- Object type to find: {obj_type}
- Distance: {template_vars['distance_m']:.1f} metres, direction: {template_vars['direction']}
- Distance vocabulary: "{template_vars['dest_vocab']}"

FILL THESE SLOTS (JSON only):

slot_user_request: User asks robot to find nearest {obj_type}.
  Vary phrasing: "Find me a {obj_type}", "Is there a {obj_type} nearby?", "Where's the nearest {obj_type}?", "Can you find a {obj_type}?"

slot_confirmation_ask: Assistant (NEVER say "you are already in" or "no navigation needed")'s response. CONSTRAINT: Never say "you are already in" or "no navigation needed".  Assistant's 1-2 sentences. Must:
  1. Acknowledge finding the {obj_type}
  2. State distance using vocabulary: "{template_vars['dest_vocab']}"
  3. End with a confirmation question
  Example: "Found a {obj_type} that's {template_vars['dest_vocab']}. Ready to head there?"

slot_user_confirms: User's short affirmative. Options: "yes", "sure", "let's go", "okay", "go ahead"

slot_departure_confirm: Assistant's 1 sentence confirming navigation started.
  Example: "Heading to the {obj_type} now."

{{
  "slot_user_request": "...",
  "slot_confirmation_ask": "...",
  "slot_user_confirms": "...",
  "slot_departure_confirm": "..."
}}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario B examples (batch)."""
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        examples_text.append(f"""
EXAMPLE {i}:
- Robot is currently in: {tv['current_zone']}
- Object type to find: {tv['object_type']}
- Distance: {tv['distance_m']:.1f} metres, direction: {tv['direction']}
- Distance vocabulary: "{tv['dest_vocab']}"
""")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot navigation training examples.
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

SCENARIO: User wants robot to find and navigate to nearest instance of an object type.

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

slot_user_request: User asks robot to find nearest object.
  Vary phrasing: "Find me a...", "Is there a... nearby?", "Where's the nearest...?", "Can you find a...?"

slot_confirmation_ask: Assistant (NEVER say "you are already in" or "no navigation needed")'s response. CONSTRAINT: Never say "you are already in" or "no navigation needed".  Assistant's 1-2 sentences acknowledging the object and stating distance using provided vocabulary, ending with confirmation question.

slot_user_confirms: User's short affirmative ("yes", "sure", "let's go", "okay", "go ahead").

slot_departure_confirm: Assistant's 1 sentence confirming navigation started.

[
  {{"slot_user_request": "...", "slot_confirmation_ask": "...", "slot_user_confirms": "...", "slot_departure_confirm": "..."}},
  {{"slot_user_request": "...", "slot_confirmation_ask": "...", "slot_user_confirms": "...", "slot_departure_confirm": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton."""
    obj_type = template_vars["object_type"]

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
                        "name": "find_nearest",
                        "arguments": json.dumps({"object_type": obj_type}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(get_find_nearest_response(obj_type)),
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "distance_to",
                        "arguments": json.dumps({"destination": obj_type}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": json.dumps(
                get_distance_to_response(obj_type, template_vars["current_zone"])
            ),
        },
        {
            "role": "assistant",
            "content": slot_fills["slot_confirmation_ask"],
        },
        {
            "role": "user",
            "content": slot_fills["slot_user_confirms"],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "navigate_to",
                        "arguments": json.dumps({"destination": obj_type}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_3",
            "content": json.dumps(get_navigate_to_response(obj_type)),
        },
        {
            "role": "assistant",
            "content": slot_fills["slot_departure_confirm"],
        },
    ]

    return messages


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return ["find_nearest", "distance_to", "navigate_to"]
