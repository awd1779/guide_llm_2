"""Scenario H: Return navigation (go back where we came from)."""

import json
from .constants import (
    CANONICAL_SYSTEM_PROMPT,
    ZONES,
    DISTANCE_MATRIX,
    DIRECTION_MATRIX,
    distance_to_vocab,
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


def get_distance_to_response(destination, current_zone):
    """Generate distance_to tool response."""
    distance_m = DISTANCE_MATRIX.get((current_zone, destination), 5.0)
    direction = DIRECTION_MATRIX.get((current_zone, destination), "ahead")

    return {
        "destination": destination,
        "destination_type": "zone",
        "distance_m": distance_m,
        "direction": direction,
    }


def get_navigate_to_response(destination):
    """Generate navigate_to tool response."""
    return {
        "status": "goal_sent",
        "destination": destination,
        "type": "zone",
    }


def generate_vars():
    """Generate all return navigation variations."""
    for current_zone in ZONES:
        for home_zone in ZONES:
            if home_zone == current_zone:
                continue

            distance_m = DISTANCE_MATRIX.get((current_zone, home_zone), 5.0)
            direction = DIRECTION_MATRIX.get((current_zone, home_zone), "ahead")

            yield {
                "current_zone": current_zone,
                "home_zone": home_zone,
                "distance_m": distance_m,
                "direction": direction,
                "dest_vocab": distance_to_vocab(distance_m),
            }


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario H slots."""
    return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User asks to return to where they started (after reaching a destination).

CONTEXT:
- Robot is currently in: {template_vars['current_zone']}
- Home/starting location: {template_vars['home_zone']}
- Distance back: {template_vars['distance_m']:.1f} metres
- Distance vocabulary: "{template_vars['dest_vocab']}"

FILL THESE SLOTS (JSON only):

slot_user_request: User asks to return home.
  Vary: "Take me back", "Let's go back", "Return to the {template_vars['home_zone']}", "Go back home", "Back to where we started"

slot_confirmation_ask: Assistant's 1-2 sentences confirming return destination.
  Example: "The {template_vars['home_zone']} is {template_vars['dest_vocab']} back. Ready to return?"

slot_user_confirms: User's short affirmative: "yes", "sure", "let's go", "okay"

slot_departure_confirm: Assistant confirms navigation back. 1 sentence.
  Example: "Heading back to the {template_vars['home_zone']} now."

{{
  "slot_user_request": "...",
  "slot_confirmation_ask": "...",
  "slot_user_confirms": "...",
  "slot_departure_confirm": "..."
}}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario H examples (batch)."""
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        examples_text.append(f"""
EXAMPLE {i}:
- Robot is currently in: {tv['current_zone']}
- Home/starting location: {tv['home_zone']}
- Distance back: {tv['distance_m']:.1f} metres
- Distance vocabulary: "{tv['dest_vocab']}"
""")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot navigation training examples.
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

SCENARIO: User asks to return to where they started (after reaching a destination).

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

slot_user_request: User asks to return home.
  Vary: "Take me back", "Let's go back", "Return to the...", "Go back home", "Back to where we started"

slot_confirmation_ask: Assistant's 1-2 sentences confirming return destination and stating distance using provided vocabulary.

slot_user_confirms: User's short affirmative ("yes", "sure", "let's go", "okay").

slot_departure_confirm: Assistant's 1 sentence confirming navigation back started.

[
  {{"slot_user_request": "...", "slot_confirmation_ask": "...", "slot_user_confirms": "...", "slot_departure_confirm": "..."}},
  {{"slot_user_request": "...", "slot_confirmation_ask": "...", "slot_user_confirms": "...", "slot_departure_confirm": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton."""
    home_zone = template_vars["home_zone"]

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
            "content": json.dumps(get_robot_pose_response(template_vars["current_zone"])),
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
                        "arguments": json.dumps({"destination": home_zone}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": json.dumps(
                get_distance_to_response(home_zone, template_vars["current_zone"])
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
                        "arguments": json.dumps({"destination": home_zone}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_3",
            "content": json.dumps(get_navigate_to_response(home_zone)),
        },
        {
            "role": "assistant",
            "content": slot_fills["slot_departure_confirm"],
        },
    ]

    return messages


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return ["get_robot_pose", "distance_to", "navigate_to"]
