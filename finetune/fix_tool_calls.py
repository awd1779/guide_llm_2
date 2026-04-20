#!/usr/bin/env python3
"""
Fix tool_calls format for Qwen: parse arguments strings to dicts.
"""

import json

def fix_tool_calls(obj):
    """Parse arguments strings to dicts in tool_calls."""
    messages = obj.get("messages", [])

    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args_str = func.get("arguments")

                # If arguments is a string, parse it
                if isinstance(args_str, str):
                    try:
                        func["arguments"] = json.loads(args_str)
                    except json.JSONDecodeError:
                        # If parsing fails, leave as empty dict
                        func["arguments"] = {}

    return obj

def main():
    print("Fixing tool_calls format...\n")

    for dataset_name in ["train", "val"]:
        input_file = f"finetune/data/{dataset_name}.jsonl"

        # Load, fix, and write back
        examples = []
        with open(input_file) as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    obj = fix_tool_calls(obj)
                    examples.append(obj)

        with open(input_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')

        print(f"✓ Fixed {dataset_name}.jsonl ({len(examples)} examples)")

    print("\n✅ Tool calls format fixed!")

if __name__ == "__main__":
    main()
