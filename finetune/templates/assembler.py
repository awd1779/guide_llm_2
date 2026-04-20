"""Assembler: merge filled slots into complete JSONL training examples."""

import json
from .constants import TOOLS


def assemble_example(scenario_name, template_vars, slot_fills, build_skeleton_fn):
    """Assemble a complete training example from filled slots.

    Args:
        scenario_name: str, name of scenario (e.g., "scenario_a")
        template_vars: dict, template variables
        slot_fills: dict, filled slots from Claude
        build_skeleton_fn: function that builds message list from vars + fills

    Returns:
        dict: complete JSONL-ready example with messages and tools
    """
    messages = build_skeleton_fn(template_vars, slot_fills)

    example = {
        "scenario_type": scenario_name,
        "messages": messages,
        "tools": TOOLS,
    }

    return example


def validate_sequence(messages, expected_tools):
    """Validate that tool sequence in messages matches expected.

    Args:
        messages: list of message dicts
        expected_tools: list of expected tool names in order

    Returns:
        bool: True if sequence matches, False otherwise
    """
    actual_tools = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_name = tc.get("function", {}).get("name")
                if tool_name:
                    actual_tools.append(tool_name)

    return actual_tools == expected_tools


def sanitize_example(example):
    """Ensure example is JSONL-safe (all values JSON-serializable).

    Args:
        example: dict to sanitize

    Returns:
        dict: sanitized example
    """
    # Ensure all strings, recursively
    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(item) for item in obj]
        elif isinstance(obj, str):
            return obj
        elif obj is None:
            return None
        else:
            return str(obj)

    return make_json_safe(example)
