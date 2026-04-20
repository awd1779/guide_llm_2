"""Scenario F: Error handling (location not found, already there)."""

import json
from .constants import (
    CANONICAL_SYSTEM_PROMPT,
    ZONES,
    OBJECTS,
    DISTANCE_MATRIX,
)


def get_list_all_response():
    """Generate list_all tool response."""
    return {
        "zones": [{"name": zone} for zone in ZONES],
        "objects": OBJECTS,
    }


def generate_vars():
    """Generate error scenarios."""
    # Case 1: Ask for something that doesn't exist
    yield {
        "error_type": "not_found",
        "requested": "bathroom",
        "current_zone": "hri_lab",
    }

    yield {
        "error_type": "not_found",
        "requested": "stairs",
        "current_zone": "kitchen",
    }

    # Case 2: Already at destination
    for zone in ZONES:
        yield {
            "error_type": "already_there",
            "requested": zone,
            "current_zone": zone,
        }


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario F slots."""
    error_type = template_vars["error_type"]

    if error_type == "not_found":
        return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User asks to navigate to something that doesn't exist.

CONTEXT:
- Robot is in: {template_vars['current_zone']}
- User requested: {template_vars['requested']} (NOT in environment)
- Available zones: {', '.join(ZONES)}
- Available objects: {', '.join(OBJECTS)}

FILL THESE SLOTS (JSON only):

slot_user_request: User asks to go to {template_vars['requested']}.
  Example: "Take me to the {template_vars['requested']}"

slot_error_response: Assistant's response explaining {template_vars['requested']} not found.
  Should suggest what IS available. 1-2 sentences.
  Example: "I don't see a {template_vars['requested']}. I can take you to the kitchen, lab, or hallways."

{{
  "slot_user_request": "...",
  "slot_error_response": "..."
}}"""

    else:  # already_there
        return f"""You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User asks to navigate to a location they're already at.

CONTEXT:
- Robot is in: {template_vars['current_zone']}
- User requested: {template_vars['requested']} (same as current location)

FILL THESE SLOTS (JSON only):

slot_user_request: User asks to go to {template_vars['requested']}.
  Example: "Take me to the {template_vars['requested']}"

slot_already_there: Assistant's response noting they're already there. 1 sentence.
  Example: "You're already right here in the {template_vars['requested']}!"

{{
  "slot_user_request": "...",
  "slot_already_there": "..."
}}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario F examples (batch)."""
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        error_type = tv["error_type"]
        if error_type == "not_found":
            examples_text.append(f"""
EXAMPLE {i} (NOT_FOUND):
- Robot is in: {tv['current_zone']}
- User requested: {tv['requested']} (NOT in environment)
- Available zones: {', '.join(ZONES)}
- Available objects: {', '.join(OBJECTS)}
Slots: slot_user_request, slot_error_response
""")
        else:  # already_there
            examples_text.append(f"""
EXAMPLE {i} (ALREADY_THERE):
- Robot is in: {tv['current_zone']}
- User requested: {tv['requested']} (same as current location)
Slots: slot_user_request, slot_already_there
""")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot navigation training examples.
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

SCENARIO: User asks to navigate but error occurs (location not found or already there).

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

For NOT_FOUND examples:
  slot_user_request: User asks to go to a destination
  slot_error_response: Assistant's response explaining destination not found and suggesting available locations

For ALREADY_THERE examples:
  slot_user_request: User asks to go to their current location
  slot_already_there: Assistant's response noting they're already there

[
  {{"slot_user_request": "...", "slot_error_response": "..."}},
  {{"slot_user_request": "...", "slot_already_there": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton."""
    error_type = template_vars["error_type"]
    requested = template_vars["requested"]

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
    ]

    if error_type == "not_found":
        messages.append(
            {
                "role": "assistant",
                "content": slot_fills["slot_error_response"],
            }
        )
    else:  # already_there
        messages.append(
            {
                "role": "assistant",
                "content": slot_fills["slot_already_there"],
            }
        )

    return messages


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return ["list_all"]
