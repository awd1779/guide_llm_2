"""Scenario E: Distance/route query only (no navigation)."""

import json
from .constants import (
    CANONICAL_SYSTEM_PROMPT,
    ZONES,
    OBJECTS,
    OBJECT_ZONES,
    DISTANCE_MATRIX,
    DIRECTION_MATRIX,
    distance_to_vocab,
)


def get_distance_to_response(destination, current_zone):
    """Generate distance_to tool response."""
    distance_m = DISTANCE_MATRIX.get((current_zone, destination), 5.0)
    direction = DIRECTION_MATRIX.get((current_zone, destination), "ahead")

    dest_type = "zone" if destination in ZONES else "object"

    return {
        "destination": destination,
        "destination_type": dest_type,
        "distance_m": distance_m,
        "direction": direction,
    }


def generate_vars():
    """Generate all variations of Scenario E."""
    for current_zone in ZONES:
        for dest_zone in ZONES:
            if dest_zone == current_zone:
                continue

            distance_m = DISTANCE_MATRIX[(current_zone, dest_zone)]
            direction = DIRECTION_MATRIX[(current_zone, dest_zone)]

            yield {
                "destination": dest_zone,
                "current_zone": current_zone,
                "distance_m": distance_m,
                "direction": direction,
                "dest_vocab": distance_to_vocab(distance_m),
                "is_object": False,
            }

        for obj in OBJECTS:
            if OBJECT_ZONES[obj] == current_zone:
                continue

            distance_m = DISTANCE_MATRIX.get(
                (current_zone, OBJECT_ZONES[obj]), 5.0
            )
            direction = DIRECTION_MATRIX.get(
                (current_zone, OBJECT_ZONES[obj]), "ahead"
            )

            yield {
                "destination": obj,
                "current_zone": current_zone,
                "distance_m": distance_m,
                "direction": direction,
                "dest_vocab": distance_to_vocab(distance_m),
                "is_object": True,
            }


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario E slots."""
    return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User asks for distance to a location WITHOUT wanting to navigate there.

CONTEXT:
- Robot is in: {template_vars['current_zone']}
- Destination: {template_vars['destination']}
- Distance: {template_vars['distance_m']:.1f} metres
- Distance vocabulary: "{template_vars['dest_vocab']}"

FILL THESE SLOTS (JSON only):

slot_user_request: User asks "how far" but NOT "take me there".
  Vary: "How far is the {template_vars['destination']}?", "What's the distance to {template_vars['destination']}?", "How long to {template_vars['destination']}?"

slot_response: Assistant's 1 sentence stating distance. NO offer to navigate.
  Example: "The {template_vars['destination']} is {template_vars['dest_vocab']}."

{{
  "slot_user_request": "...",
  "slot_response": "..."
}}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario E examples (batch)."""
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        examples_text.append(f"""
EXAMPLE {i}:
- Robot is in: {tv['current_zone']}
- Destination: {tv['destination']}
- Distance: {tv['distance_m']:.1f} metres
- Distance vocabulary: "{tv['dest_vocab']}"
""")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot navigation training examples.
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

SCENARIO: User asks for distance to a location WITHOUT wanting to navigate there.

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

slot_user_request: User asks "how far" but NOT "take me there".
  Vary: "How far is the...?", "What's the distance to...?", "How long to...?"

slot_response: Assistant's 1 sentence stating distance. NO offer to navigate.

[
  {{"slot_user_request": "...", "slot_response": "..."}},
  {{"slot_user_request": "...", "slot_response": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton."""
    dest = template_vars["destination"]

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
                        "name": "distance_to",
                        "arguments": json.dumps({"destination": dest}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(
                get_distance_to_response(dest, template_vars["current_zone"])
            ),
        },
        {
            "role": "assistant",
            "content": slot_fills["slot_response"],
        },
    ]

    return messages


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return ["distance_to"]
