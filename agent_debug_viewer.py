#!/usr/bin/env python3
"""
Debug viewer for scene_graph_agent.

Subscribes to /scene_graph/agent_debug and prints all tool calls,
navigation events, pose lookups, and other internal logs.

Usage:
    python3 agent_debug_viewer.py
"""

import sys
import signal

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ANSI colors for log categories
COLORS = {
    "tool":   "\033[96m",   # cyan
    "nav":    "\033[93m",   # yellow
    "pose":   "\033[92m",   # green
    "voice":  "\033[95m",   # magenta
    "speech": "\033[94m",   # blue
    "init":   "\033[90m",   # gray
    "error":  "\033[91m",   # red
}
RESET = "\033[0m"


class AgentDebugViewer(Node):
    def __init__(self):
        super().__init__("agent_debug_viewer")
        self.create_subscription(
            String, "/scene_graph/agent_debug", self._on_debug, 50
        )
        print("Listening on /scene_graph/agent_debug ...\n", flush=True)

    def _on_debug(self, msg):
        text = msg.data
        # Colorize based on tag
        color = ""
        for tag, c in COLORS.items():
            if text.startswith(f"[{tag}]"):
                color = c
                break
        print(f"{color}{text}{RESET}", flush=True)


def main():
    rclpy.init()
    node = AgentDebugViewer()
    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
