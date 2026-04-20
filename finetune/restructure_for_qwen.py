#!/usr/bin/env python3
"""
Restructure training data for Qwen 3.5 tool-use format.
Adds tools definition to each example so apply_chat_template works properly.
"""

import json
import sys
from pathlib import Path

# Import tools from template constants to ensure consistency
try:
    from finetune.templates.constants import TOOLS
except ImportError:
    # Fallback to local definition if templates not available
    TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_robot_pose",
            "description": "Get the robot's current position and which zone it is in.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_surroundings",
            "description": "Describe objects near the robot within a given radius.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "Search radius in metres (default 3.0)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_all",
            "description": "List all known zones (rooms/areas) and objects in the environment.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Get detailed information about a specific zone or object by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Zone name or object label to look up"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearest",
            "description": "Find the nearest object matching a keyword (substring match on label).",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_type": {
                        "type": "string",
                        "description": "Keyword to match against object labels (e.g. 'chair')"
                    }
                },
                "required": ["object_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "distance_to",
            "description": "Get straight-line distance and direction from the robot to a zone or object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Zone name or object label"}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_route",
            "description": "Describe the zones and objects along the straight-line path to a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Zone name or object label"}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "orient_me",
            "description": "Rotate the robot in place to face a target zone or object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Zone name or object label to face towards"}
                },
                "required": ["target"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Navigate the robot to a destination (zone or object) by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Zone name or object label to navigate to"}
                },
                "required": ["destination"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_navigation_status",
            "description": "Check the current navigation status (executing, succeeded, failed, etc.).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_navigation",
            "description": "Cancel the current navigation goal.",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "replan_route",
            "description": "Replan route to a new destination or re-route around obstacles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "New destination zone or object (optional)"
                    },
                    "avoid_obstacles": {
                        "type": "boolean",
                        "description": "Set to true to re-route around detected obstacles (optional)"
                    }
                }
            }
        }
    },
    ]

def restructure_example(obj):
    """Add tools field to example."""
    obj["tools"] = TOOLS
    return obj

def main():
    print("Restructuring training data for Qwen 3.5 tool-use format...\n")

    for dataset_name in ["train", "val"]:
        input_file = f"finetune/data/{dataset_name}.jsonl"
        output_file = f"finetune/data/{dataset_name}.jsonl"

        print(f"Processing {dataset_name}.jsonl...")

        # Load all examples
        examples = []
        with open(input_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))

        # Add tools to each
        examples = [restructure_example(ex) for ex in examples]

        # Write back
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        print(f"  ✓ Added tools field to {len(examples)} examples")

    print("\n✅ Data restructuring complete!")
    print("\nAll examples now have:")
    print("  - 'messages': conversation history with tool calls")
    print("  - 'tools': list of 12 available tools (for apply_chat_template)")

if __name__ == "__main__":
    main()
