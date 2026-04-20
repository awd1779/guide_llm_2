"""Scenario G: Greeting and small talk (no tools)."""

import json
from .constants import CANONICAL_SYSTEM_PROMPT


def generate_vars():
    """Generate greeting scenarios."""
    scenarios = [
        {"type": "greeting"},
        {"type": "greeting"},
        {"type": "capabilities"},
        {"type": "thanks"},
        {"type": "greeting"},
    ]
    for scenario in scenarios:
        yield scenario


def build_fill_prompt(template_vars):
    """Build the API prompt for Claude to fill Scenario G slots."""
    scenario_type = template_vars["type"]

    if scenario_type == "greeting":
        return """You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User greets the robot.

FILL THESE SLOTS (JSON only):

slot_user_greeting: User says hello.
  Vary: "Hello", "Hi there", "Hey", "Good morning", "Hi!"

slot_assistant_greeting: Assistant's brief friendly response. 1 sentence.
  Example: "Hello! I can help you navigate. Where would you like to go?"

{
  "slot_user_greeting": "...",
  "slot_assistant_greeting": "..."
}"""

    elif scenario_type == "capabilities":
        return """You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User asks what the robot can do.

FILL THESE SLOTS (JSON only):

slot_user_question: User asks about capabilities.
  Vary: "What can you do?", "What are you capable of?", "What can you help me with?", "How can you help?"

slot_assistant_answer: Assistant briefly describes capabilities. 1-2 sentences.
  Example: "I can guide you to rooms and objects, describe your surroundings, and help you navigate the space."

{
  "slot_user_question": "...",
  "slot_assistant_answer": "..."
}"""

    else:  # thanks
        return """You are filling natural language slots for a VI robot navigation training example.
Output ONLY a JSON object with slot values. No explanation, no markdown.

SCENARIO: User thanks the robot.

FILL THESE SLOTS (JSON only):

slot_user_thanks: User thanks assistant.
  Vary: "Thanks", "Thank you", "Thanks for the help", "Much appreciated", "Thanks a lot"

slot_assistant_reply: Assistant's response. 1 sentence.
  Example: "Happy to help! Let me know if you need anything else."

{
  "slot_user_thanks": "...",
  "slot_assistant_reply": "..."
}"""


def build_fill_prompt_batch(template_vars_list):
    """Build the API prompt for Claude to fill multiple Scenario G examples (batch)."""
    examples_text = []
    for i, tv in enumerate(template_vars_list, 1):
        scenario_type = tv["type"]
        if scenario_type == "greeting":
            examples_text.append(f"EXAMPLE {i}: User greets the robot\nSlots: slot_user_greeting, slot_assistant_greeting\n")
        elif scenario_type == "capabilities":
            examples_text.append(f"EXAMPLE {i}: User asks what the robot can do\nSlots: slot_user_question, slot_assistant_answer\n")
        else:  # thanks
            examples_text.append(f"EXAMPLE {i}: User thanks the robot\nSlots: slot_user_thanks, slot_assistant_reply\n")

    return f"""You are filling natural language slots for {len(template_vars_list)} VI robot training examples (greetings and small talk).
Output ONLY a JSON array with {len(template_vars_list)} objects. No explanation, no markdown.

{"".join(examples_text)}

FILL THESE SLOTS FOR EACH EXAMPLE (JSON array only):

For greeting examples:
  slot_user_greeting: User says hello (vary: "Hello", "Hi there", "Hey", "Good morning", "Hi!")
  slot_assistant_greeting: Assistant's brief friendly response (1 sentence)

For capabilities examples:
  slot_user_question: User asks about capabilities (vary: "What can you do?", "What are you capable of?", "What can you help me with?", "How can you help?")
  slot_assistant_answer: Assistant briefly describes capabilities (1-2 sentences)

For thanks examples:
  slot_user_thanks: User thanks assistant (vary: "Thanks", "Thank you", "Thanks for the help", "Much appreciated", "Thanks a lot")
  slot_assistant_reply: Assistant's response (1 sentence)

[
  {{"slot_user_greeting": "...", "slot_assistant_greeting": "..."}},
  {{"slot_user_question": "...", "slot_assistant_answer": "..."}},
  ...
]"""


def build_skeleton(template_vars, slot_fills):
    """Build the complete conversation skeleton."""
    scenario_type = template_vars["type"]

    if scenario_type == "greeting":
        return [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": slot_fills["slot_user_greeting"]},
            {
                "role": "assistant",
                "content": slot_fills["slot_assistant_greeting"],
            },
        ]
    elif scenario_type == "capabilities":
        return [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": slot_fills["slot_user_question"]},
            {
                "role": "assistant",
                "content": slot_fills["slot_assistant_answer"],
            },
        ]
    else:  # thanks
        return [
            {"role": "system", "content": CANONICAL_SYSTEM_PROMPT},
            {"role": "user", "content": slot_fills["slot_user_thanks"]},
            {
                "role": "assistant",
                "content": slot_fills["slot_assistant_reply"],
            },
        ]


def get_expected_tools():
    """Return the expected tool sequence for validation."""
    return []  # No tools for greetings
