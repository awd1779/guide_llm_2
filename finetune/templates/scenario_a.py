"""Scenario A: Navigate to known location (zone or object)."""

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

# =============================================================================
# SCENARIO A - Navigate to Known Location (zone or object)
# Tool sequence: list_all → distance_to → [ask confirmation] → navigate_to
# =============================================================================

def get_list_all_response():
    """Generate list_all tool response."""
    return {
        "zones": [{"name": zone} for zone in ZONES],
        "objects": OBJECTS,
    }


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


def get_navigate_to_response(destination):
    """Generate navigate_to tool response."""
    dest_type = "zone" if destination in ZONES else "object"
    return {
        "status": "goal_sent",
        "destination": destination,
        "type": dest_type,
    }


def generate_vars():
    """Generate all variations of Scenario A.

    Yields:
        dict with keys: destination, current_zone, distance_m, direction, dest_vocab
    """
    # All (current_zone, destination) pairs where destination exists
    for current_zone in ZONES:
        # Navigate to other zones
        for dest_zone in ZONES:
            if dest_zone == current_zone:
                continue  # Skip same zone

            distance_m = DISTANCE_MATRIX[(current_zone, dest_zone)]
            direction = DIRECTION_MATRIX[(current_zone, dest_zone)]

            yield {
                "destination": dest_zone,
                "current_zone": current_zone,
                "distance_m": distance_m,
                "direction": direction,
                "dest_vocab": distance_to_vocab(distance_m),
            }

        # Navigate to objects
        for obj in OBJECTS:
            if OBJECT_ZONES[obj] == current_zone:
                continue  # Skip objects in current zone

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
            }


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario A slots.

    Args:
        template_vars: dict with destination, current_zone, distance_m, direction, dest_vocab

    Returns:
        str: the prompt to send to Claude
    """
    return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User wants to navigate to a known location (zone or object).

CONTEXT:
- Robot is currently in: {template_vars['current_zone']}
- Destination: {template_vars['destination']}
- Distance: {template_vars['distance_m']:.1f} metres, direction: {template_vars['direction']}
- Distance vocabulary: "{template_vars['dest_vocab']}"

FILL THESE SLOTS (JSON only):

slot_user_request: User asks robot to navigate to {template_vars['destination']}.
  Vary phrasing across examples. Options:
  - "Take me to the {template_vars['destination']}"
  - "Can we head to the {template_vars['destination']}"
  - "I want to go to the {template_vars['destination']}"
  - "Let's go to the {template_vars['destination']}"
  - "Bring me to the {template_vars['destination']}"
  - "Could you take me to the {template_vars['destination']}"

slot_confirmation_ask: Assistant's 1-2 sentence response. Must:
  1. Acknowledge the destination
  2. State distance using vocabulary: "{template_vars['dest_vocab']}"
  3. End with a confirmation question
  CONSTRAINT: Never say "you are already in X" or "no navigation needed"
  Example: "The {template_vars['destination']} is {template_vars['dest_vocab']}. Ready to head there?"

slot_user_confirms: User gives short affirmative response. Options:
  - "Yes"
  - "Sure"
  - "Let's go"
  - "Okay"
  - "Yeah, let's go"
  - "Go ahead"

slot_departure_confirm: Assistant's 1 sentence confirming navigation started.
  Example: "On my way to the {template_vars['destination']} now."

{{
  "slot_user_request": "...",
  "slot_confirmation_ask": "...",
  "slot_user_confirms": "...",
  "slot_departure_confirm": "..."
}}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario A examples (batch).

    Args:
        template_vars_list: list of up to 5 dicts with destination, current_zone, etc.

    Returns:
        str: the prompt to send to Claude
    """
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        examples_text.append(f"""
EXAMPLE {i}:
- Robot is currently in: {tv['current_zone']}
- Destination: {tv['destination']}
- Distance: {tv['distance_m']:.1f} metres, direction: {tv['direction']}
- Distance vocabulary: "{tv['dest_vocab']}"
""")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot navigation training examples.
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

SCENARIO: User wants to navigate to a known location (zone or object).

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

slot_user_request: User asks robot to navigate to the destination.
  Vary phrasing. Options: "Take me to the...", "Can we head to...", "I want to go to...", "Let's go to...", "Bring me to...", "Could you take me to..."

slot_confirmation_ask: Assistant's 1-2 sentence response acknowledging destination and stating distance using provided vocabulary, ending with confirmation question. NEVER say "you are already in" or "no navigation needed".

slot_user_confirms: User gives short affirmative response ("Yes", "Sure", "Let's go", "Okay", "Yeah, let's go", "Go ahead").

slot_departure_confirm: Assistant's 1 sentence confirming navigation started.

[
  {{"slot_user_request": "...", "slot_confirmation_ask": "...", "slot_user_confirms": "...", "slot_departure_confirm": "..."}},
  {{"slot_user_request": "...", "slot_confirmation_ask": "...", "slot_user_confirms": "...", "slot_departure_confirm": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton with filled slots.

    Args:
        template_vars: dict with template variables
        slot_fills: dict with filled slots from Claude

    Returns:
        list: list of message dicts
    """
    dest = template_vars["destination"]

    messages = [
        {
            "role": "system",
            "content": CANONICAL_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": slot_fills["slot_user_request"],
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "list_all",
                        "arguments": json.dumps({}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": json.dumps(get_list_all_response()),
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
                        "arguments": json.dumps({"destination": dest}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": json.dumps(
                get_distance_to_response(dest, template_vars["current_zone"])
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
                        "arguments": json.dumps({"destination": dest}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_3",
            "content": json.dumps(get_navigate_to_response(dest)),
        },
        {
            "role": "assistant",
            "content": slot_fills["slot_departure_confirm"],
        },
    ]

    return messages


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return ["list_all", "distance_to", "navigate_to"]
