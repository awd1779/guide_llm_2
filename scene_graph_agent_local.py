#!/usr/bin/env python3
"""
ROS2 node: LLM-driven navigation agent using scene graph data + local LLM (Nemotron 3 Nano) + Nav2.

Subscribes to scene_graph_publisher topics for object/zone data, exposes 14 tools
to a local LLM via OpenAI-compatible API (navigate, query, list, localize, spatial),
and publishes Nav2 goal poses.
Uses TF lookup (map → base_link) for robot pose from any SLAM system (LiDAR, etc.).

Usage:
    # Start your local LLM server (vLLM, Ollama, NIM, etc.) first, then:
    python3 scene_graph_agent_local.py --base-url http://localhost:8000/v1

    # Then type natural language commands:
    # > take me to the kitchen
    # > what objects are in the hri lab?
    # > list all zones
"""

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time

import math

from openai import OpenAI
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from action_msgs.msg import GoalStatusArray, GoalStatus
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, LookupException, ExtrapolationException


# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_to_zone",
            "description": "Navigate the robot to a named zone by publishing its centroid as a Nav2 goal pose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {"type": "string", "description": "Exact zone name from list_zones"},
                },
                "required": ["zone_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to_object",
            "description": "Navigate the robot to a detected object by publishing its map position as a Nav2 goal pose.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_label": {"type": "string", "description": "Exact object label from list_objects"},
                },
                "required": ["object_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_zone",
            "description": "Get detailed information about a specific zone (objects inside, area, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "zone_name": {"type": "string", "description": "Exact zone name"},
                },
                "required": ["zone_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_object",
            "description": "Get detailed information about a specific object (position, zone, confidence).",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_label": {"type": "string", "description": "Exact object label"},
                },
                "required": ["object_label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_zones",
            "description": "List all known zone names in the scene graph.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_objects",
            "description": "List all known object labels in the scene graph.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_pose",
            "description": "Get the robot's current position (x, y, yaw) and which zone it is in via TF lookup.",
            "parameters": {"type": "object", "properties": {}},
        },
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
                        "description": "Search radius in metres (default 3.0)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_navigation_status",
            "description": "Check the current Nav2 navigation status (executing, succeeded, failed, etc.).",
            "parameters": {"type": "object", "properties": {}},
        },
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
                        "description": "Keyword to match against object labels (e.g. 'chair')",
                    },
                },
                "required": ["object_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "distance_to",
            "description": "Get straight-line distance and direction from the robot to a zone or object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Zone name or object label"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_route",
            "description": "Describe the zones and objects along the straight-line path to a destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {"type": "string", "description": "Zone name or object label"},
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "orient_me",
            "description": "Rotate the robot in place to face a target zone or object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Zone name or object label to face towards"},
                },
                "required": ["target"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class SceneGraphAgent(Node):
    def __init__(self, model, base_url, api_key, max_tokens, frame_id, base_frame):
        super().__init__("scene_graph_agent")

        self._model = model
        self._max_tokens = max_tokens
        self._frame_id = frame_id
        self._base_frame = base_frame
        self._client = OpenAI(base_url=base_url, api_key=api_key)

        # TF for robot pose lookup
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Scene graph state — only names/labels stored, not full data
        self._zone_names = None     # list of zone name strings
        self._object_labels = None  # list of unique object label strings
        self._objects_data = None   # full data kept only for navigate_to_object position lookup
        self._zones_data = None     # full data kept only for navigate_to_zone centroid computation
        self._system_prompt = "You are a navigation assistant. Waiting for scene graph data..."
        self._conversation_history = []

        # Query round-trip synchronization
        self._pending_zone_result = None
        self._zone_result_event = threading.Event()
        self._pending_query_result = None
        self._query_result_event = threading.Event()

        # QoS matching scene_graph_publisher's latched topics
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Subscriptions (destroyed after first receipt — data is static)
        self._objects_sub = self.create_subscription(
            String, "/scene_graph/objects", self._on_objects_msg, latched_qos
        )
        self._zones_sub = self.create_subscription(
            String, "/scene_graph/zones", self._on_zones_msg, latched_qos
        )
        self.create_subscription(
            String, "/scene_graph/zone_result", self._on_zone_result, latched_qos
        )
        self.create_subscription(
            String, "/scene_graph/query_result", self._on_query_result, latched_qos
        )

        # Voice interface: subscribe to transcribed utterances, publish speech feedback
        self.create_subscription(
            String, "/transcribed_utterances_guide_llm", self._on_voice_input, 10
        )
        self._speech_feedback_pub = self.create_publisher(String, "/speech_feedback", 10)

        # Debug log topic — tool calls, results, internal state
        self._debug_pub = self.create_publisher(String, "/scene_graph/agent_debug", 10)

        # Publishers
        self._goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # Nav2 status tracking
        self._nav_status = None  # latest GoalStatus code
        self._nav_goal_zone = None  # name of zone/object we're navigating to
        self.create_subscription(
            GoalStatusArray,
            "/navigate_to_pose/_action/status",
            self._on_nav_status,
            10,
        )
        self._zone_query_pub = self.create_publisher(
            String, "/scene_graph/zone_query", 10
        )
        self._object_query_pub = self.create_publisher(
            String, "/scene_graph/query", 10
        )

        # Stdin polling timer
        self.create_timer(0.1, self._timer_callback)

        self.get_logger().info(f"Scene graph agent started (model: {model}, base_url: {base_url})")
        self.get_logger().info("Waiting for scene graph data on /scene_graph/objects and /scene_graph/zones...")
        print("\n[Agent] Waiting for scene graph data...", flush=True)

    # -------------------------------------------------------------------
    # Subscription callbacks
    # -------------------------------------------------------------------

    def _on_objects_msg(self, msg):
        if self._object_labels is not None:
            return
        try:
            data = json.loads(msg.data)
            self._objects_data = data
            self._object_labels = sorted({o["label"] for o in data.get("objects", [])})
            self._debug_log(f"[init] Received {len(self._object_labels)} unique object labels")
            self.destroy_subscription(self._objects_sub)
            self._objects_sub = None
            self._check_ready()
        except json.JSONDecodeError as e:
            self._debug_log(f"[error] Failed to parse objects JSON: {e}")

    def _on_zones_msg(self, msg):
        if self._zone_names is not None:
            return
        try:
            data = json.loads(msg.data)
            self._zones_data = data
            self._zone_names = sorted(z["name"] for z in data.get("zones", []))
            self._debug_log(f"[init] Received {len(self._zone_names)} zone names")
            self.destroy_subscription(self._zones_sub)
            self._zones_sub = None
            self._check_ready()
        except json.JSONDecodeError as e:
            self._debug_log(f"[error] Failed to parse zones JSON: {e}")

    def _on_nav_status(self, msg):
        if msg.status_list:
            status = msg.status_list[-1].status
            prev_status = self._nav_status
            self._nav_status = status

    def _on_voice_input(self, msg):
        """Handle transcribed utterances from the voice interface."""
        text = msg.data.strip()
        if text:
            self._debug_log(f"[voice] {text}")
            print(f"\n[Voice] {text}", flush=True)
            self._handle_user_input(text)

    def _publish_speech_feedback(self, text):
        """Publish text to /speech_feedback for TTS output."""
        msg = String()
        msg.data = text
        self._speech_feedback_pub.publish(msg)
        self._debug_log(f"[speech] {text}")

    def _debug_log(self, text):
        """Publish debug info to /scene_graph/agent_debug and ROS logger."""
        self.get_logger().info(text)
        msg = String()
        msg.data = text
        self._debug_pub.publish(msg)

    def _on_zone_result(self, msg):
        self._pending_zone_result = msg.data
        self._zone_result_event.set()

    def _on_query_result(self, msg):
        self._pending_query_result = msg.data
        self._query_result_event.set()

    # -------------------------------------------------------------------
    # System prompt
    # -------------------------------------------------------------------

    def _check_ready(self):
        """Called after each subscription receives data. Build prompt when both are in."""
        if self._zone_names is not None and self._object_labels is not None:
            self._rebuild_system_prompt()
            print("\n[Agent] Scene graph data received. Ready for commands.", flush=True)
            print("You: ", end="", flush=True)

    def _rebuild_system_prompt(self):
        self._system_prompt = (
            "You are a navigation assistant for a mobile robot in an indoor environment.\n"
            f"The scene graph has {len(self._zone_names)} zones and {len(self._object_labels)} objects.\n"
            "\n"
            "TOOLS:\n"
            "- get_robot_pose: Check robot's current position and which zone it's in.\n"
            "- describe_surroundings: Describe nearby objects within a radius. Use when user asks what's around them.\n"
            "- find_nearest: Find the closest object matching a keyword (substring match). Use for 'where is the nearest chair?'.\n"
            "- distance_to: Get straight-line distance and direction to a zone or object without navigating.\n"
            "- describe_route: Preview what zones and objects are along the path to a destination.\n"
            "- orient_me: Rotate the robot in place to face a target zone or object. Use for 'face towards the kitchen'.\n"
            "- list_zones / list_objects: Returns only names. Use FIRST to discover what's available.\n"
            "- query_zone / query_object: Get details for a specific zone or object.\n"
            "- navigate_to_zone / navigate_to_object: Send the robot to a destination. The robot's path planner handles the route.\n"
            "- get_navigation_status: Check if the robot has arrived, is still moving, or if navigation failed.\n"
            "\n"
            "INSTRUCTIONS:\n"
            "- ALWAYS confirm with the user BEFORE navigating. First tell them where you plan to go "
            "and ask 'shall we go?' or 'ready?'. Only call navigate tools AFTER the user confirms. "
            "Never start moving the robot without the user's approval.\n"
            "- Do NOT guess names. Always call list_zones or list_objects first to find exact names.\n"
            "- The user is vision-impaired. Describe locations and surroundings verbally.\n"
            "- IMPORTANT: Keep responses to 1-2 short sentences MAX. No bullet lists, no long explanations. "
            "Be direct and conversational, like a brief spoken reply.\n"
            "- The user is NOT technical. Never mention zones, coordinates, waypoints, navigation goals, "
            "tool names, or internal details. Just speak naturally — e.g. 'I'll take you to the kitchen' "
            "not 'Navigating to zone kitchen_area'.\n"
            "- NEVER say raw numbers from tool results. No coordinates (x, y), no exact distances in metres, "
            "no confidence scores, no percentages. Instead use natural descriptions:\n"
            "  BAD: 'The chair is 3.5 metres away at position 5.5, 2.0'\n"
            "  GOOD: 'There's a chair just ahead to your right in the living room'\n"
            "  BAD: 'The fridge has 0.95 confidence and is at 1.0, 1.0'\n"
            "  GOOD: 'The fridge is right here in the kitchen'\n"
            "- Convert distances to natural language: <1m='right here', 1-3m='nearby/close', "
            "3-6m='a short walk', >6m='a bit further away'.\n"
            "- Use relative directions (ahead, to your left, behind you) not compass bearings or angles."
        )

    # -------------------------------------------------------------------
    # Stdin polling
    # -------------------------------------------------------------------

    def _timer_callback(self):
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            line = sys.stdin.readline().strip()
            if line:
                self._handle_user_input(line)

    # -------------------------------------------------------------------
    # Conversation handling
    # -------------------------------------------------------------------

    def _handle_user_input(self, text):
        self._conversation_history.append({"role": "user", "content": text})

        try:
            response = self._call_llm()
            self._process_response(response)
        except Exception as e:
            self.get_logger().error(f"LLM API error: {e}")
            print(f"\n[Error] {e}", flush=True)

        print("You: ", end="", flush=True)

    def _call_llm(self):
        messages = [{"role": "system", "content": self._system_prompt}] + self._conversation_history
        kwargs = dict(
            model=self._model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        if self._max_tokens > 0:
            kwargs["max_tokens"] = self._max_tokens
        return self._client.chat.completions.create(**kwargs)

    def _process_response(self, response):
        message = response.choices[0].message

        # Extract text content
        if message.content:
            print(f"\n[Assistant] {message.content}", flush=True)
            self._publish_speech_feedback(message.content)

        # Build assistant message for conversation history
        assistant_msg = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        self._conversation_history.append(assistant_msg)

        if not message.tool_calls:
            return

        # Execute each tool and add results to conversation
        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            try:
                func_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                func_args = {}

            result = self._execute_tool(func_name, func_args)
            self._debug_log(
                f"[tool] {func_name}({json.dumps(func_args)}) -> {result[:300]}"
            )
            self._conversation_history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Follow-up call so the LLM can respond with the tool results
        follow_up = self._call_llm()
        self._process_response(follow_up)

    # -------------------------------------------------------------------
    # Tool dispatch
    # -------------------------------------------------------------------

    def _execute_tool(self, name, tool_input):
        try:
            if name == "navigate_to_zone":
                return self._nav_to_zone(tool_input["zone_name"])
            elif name == "navigate_to_object":
                return self._nav_to_object(tool_input["object_label"])
            elif name == "query_zone":
                return self._query_zone(tool_input["zone_name"])
            elif name == "query_object":
                return self._query_object(tool_input["object_label"])
            elif name == "list_zones":
                return self._list_zones()
            elif name == "list_objects":
                return self._list_objects()
            elif name == "get_robot_pose":
                return self._get_robot_pose()
            elif name == "describe_surroundings":
                radius = tool_input.get("radius", 3.0)
                return self._describe_surroundings(radius)
            elif name == "get_navigation_status":
                return self._get_navigation_status()
            elif name == "find_nearest":
                return self._find_nearest(tool_input["object_type"])
            elif name == "distance_to":
                return self._distance_to(tool_input["destination"])
            elif name == "describe_route":
                return self._describe_route(tool_input["destination"])
            elif name == "orient_me":
                return self._orient_me(tool_input["target"])
            else:
                return json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    # -------------------------------------------------------------------
    # Navigation tools
    # -------------------------------------------------------------------

    def _publish_goal(self, x, y, yaw=0.0):
        goal = PoseStamped()
        goal.header.frame_id = self._frame_id
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = float(x)
        goal.pose.position.y = float(y)
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_pub.publish(goal)

    @staticmethod
    def _relative_direction(robot_yaw, rx, ry, tx, ty):
        """Return human-friendly relative direction string."""
        angle_to_target = math.atan2(ty - ry, tx - rx)
        relative = angle_to_target - robot_yaw
        relative = (relative + math.pi) % (2 * math.pi) - math.pi
        deg = math.degrees(relative)

        if -22.5 <= deg < 22.5:
            return "ahead"
        elif 22.5 <= deg < 67.5:
            return "ahead-left"
        elif 67.5 <= deg < 112.5:
            return "left"
        elif 112.5 <= deg < 157.5:
            return "behind-left"
        elif -67.5 <= deg < -22.5:
            return "ahead-right"
        elif -112.5 <= deg < -67.5:
            return "right"
        elif -157.5 <= deg < -112.5:
            return "behind-right"
        else:
            return "behind"

    def _nav_to_zone(self, zone_name):
        if not self._zones_data:
            return json.dumps({"error": "No zone data available"})

        zone_lower = zone_name.strip().lower()
        for zone in self._zones_data.get("zones", []):
            if zone["name"].lower() == zone_lower:
                verts = zone["vertices_world"]
                cx = sum(v[0] for v in verts) / len(verts)
                cy = sum(v[1] for v in verts) / len(verts)
                self._publish_goal(cx, cy)
                self._nav_goal_zone = zone["name"]
                self._nav_status = GoalStatus.STATUS_EXECUTING
                return json.dumps({
                    "status": "goal_sent",
                    "zone": zone["name"],
                    "goal_x": round(cx, 3),
                    "goal_y": round(cy, 3),
                })

        return json.dumps({
            "error": f"Zone '{zone_name}' not found.",
            "hint": "Use list_zones to see available zone names.",
        })

    def _nav_to_object(self, object_label):
        if not self._objects_data:
            return json.dumps({"error": "No object data available"})

        label_lower = object_label.strip().lower()
        for obj in self._objects_data.get("objects", []):
            if obj["label"].lower() == label_lower:
                pos = obj["map_position"]
                self._publish_goal(pos["x"], pos["y"])
                self._nav_goal_zone = obj["label"]
                self._nav_status = GoalStatus.STATUS_EXECUTING
                return json.dumps({
                    "status": "goal_sent",
                    "object": obj["label"],
                    "goal_x": round(pos["x"], 3),
                    "goal_y": round(pos["y"], 3),
                })

        return json.dumps({
            "error": f"Object '{object_label}' not found.",
            "hint": "Use list_objects to see available object labels.",
        })

    def _get_navigation_status(self):
        status_map = {
            GoalStatus.STATUS_UNKNOWN: "unknown",
            GoalStatus.STATUS_ACCEPTED: "accepted",
            GoalStatus.STATUS_EXECUTING: "executing",
            GoalStatus.STATUS_CANCELING: "canceling",
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_CANCELED: "canceled",
            GoalStatus.STATUS_ABORTED: "aborted",
        }
        return json.dumps({
            "status": status_map.get(self._nav_status, "no_goal"),
            "destination": self._nav_goal_zone,
        })

    def _describe_surroundings(self, radius=3.0):
        try:
            rx, ry, yaw = self._get_robot_pose_raw()
        except (LookupException, ExtrapolationException) as e:
            return json.dumps({"error": f"Cannot get robot pose: {e}"})

        if not self._objects_data:
            return json.dumps({"error": "No object data available"})

        nearby = []
        for obj in self._objects_data.get("objects", []):
            pos = obj["map_position"]
            ox, oy = pos["x"], pos["y"]
            dist = math.sqrt((rx - ox) ** 2 + (ry - oy) ** 2)
            if dist <= radius:
                direction = self._relative_direction(yaw, rx, ry, ox, oy)
                nearby.append({
                    "label": obj["label"],
                    "distance_m": round(dist, 2),
                    "direction": direction,
                })

        nearby.sort(key=lambda o: o["distance_m"])

        # Current zone
        current_zone = None
        if self._zones_data:
            for zone in self._zones_data.get("zones", []):
                if self._point_in_polygon(rx, ry, zone["vertices_world"]):
                    current_zone = zone["name"]
                    break

        return json.dumps({
            "current_zone": current_zone,
            "radius_m": radius,
            "nearby_objects": nearby,
        })

    # -------------------------------------------------------------------
    # Query tools (via scene_graph_publisher round-trip)
    # -------------------------------------------------------------------

    def _query_zone(self, zone_name):
        self._zone_result_event.clear()
        self._pending_zone_result = None

        msg = String()
        msg.data = zone_name
        self._zone_query_pub.publish(msg)

        if self._zone_result_event.wait(timeout=3.0):
            return self._pending_zone_result
        return json.dumps({"error": f"Timeout waiting for zone query result for '{zone_name}'"})

    def _query_object(self, object_label):
        self._query_result_event.clear()
        self._pending_query_result = None

        msg = String()
        msg.data = object_label
        self._object_query_pub.publish(msg)

        if self._query_result_event.wait(timeout=3.0):
            return self._pending_query_result
        return json.dumps({"error": f"Timeout waiting for object query result for '{object_label}'"})

    # -------------------------------------------------------------------
    # List tools (local data)
    # -------------------------------------------------------------------

    def _list_zones(self):
        if not self._zone_names:
            return json.dumps({"error": "No zone data available"})
        # Include hierarchy info so the LLM understands zone structure
        zone_list = []
        for zone in self._zones_data.get("zones", []):
            entry = {"name": zone["name"]}
            if zone.get("display_name"):
                entry["display_name"] = zone["display_name"]
            if zone.get("parent"):
                entry["parent"] = zone["parent"]
            if zone.get("children"):
                entry["children"] = zone["children"]
            zone_list.append(entry)
        return json.dumps({"zones": zone_list})

    def _list_objects(self):
        if not self._object_labels:
            return json.dumps({"error": "No object data available"})
        return json.dumps({"objects": self._object_labels})

    # -------------------------------------------------------------------
    # Localization tool (TF lookup)
    # -------------------------------------------------------------------

    def _get_robot_pose(self):
        try:
            t = self._tf_buffer.lookup_transform(
                self._frame_id, self._base_frame, rclpy.time.Time()
            )
        except (LookupException, ExtrapolationException) as e:
            return json.dumps({
                "error": f"Cannot get robot pose: {e}",
                "hint": f"Is SLAM running? Expecting TF: {self._frame_id} -> {self._base_frame}",
            })

        pos = t.transform.translation
        q = t.transform.rotation
        # Yaw from quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw_rad = math.atan2(siny_cosp, cosy_cosp)
        yaw_deg = math.degrees(yaw_rad)

        result = {
            "x": round(pos.x, 3),
            "y": round(pos.y, 3),
            "yaw_deg": round(yaw_deg, 1),
            "frame": self._frame_id,
            "current_zone": None,
        }

        # Determine which zone the robot is in
        if self._zones_data:
            for zone in self._zones_data.get("zones", []):
                if self._point_in_polygon(pos.x, pos.y, zone["vertices_world"]):
                    result["current_zone"] = zone["name"]
                    break

        self._debug_log(
            f"[pose] Robot at ({pos.x:.3f}, {pos.y:.3f}), yaw={yaw_deg:.1f}deg, zone={result['current_zone']}"
        )
        return json.dumps(result)

    @staticmethod
    def _point_in_polygon(x, y, vertices):
        """Ray-casting point-in-polygon test."""
        n = len(vertices)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # -------------------------------------------------------------------
    # Shared helpers for spatial tools
    # -------------------------------------------------------------------

    def _get_robot_pose_raw(self):
        """Return (rx, ry, yaw_rad) tuple from TF lookup. Raises on failure."""
        t = self._tf_buffer.lookup_transform(
            self._frame_id, self._base_frame, rclpy.time.Time()
        )
        rx = t.transform.translation.x
        ry = t.transform.translation.y
        q = t.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return rx, ry, yaw

    def _resolve_destination(self, destination):
        """Resolve a string to (x, y, canonical_name, kind).

        Tries zone name first (case-insensitive exact match via centroid),
        then object label (case-insensitive exact match).
        Returns None if not found.
        """
        dest_lower = destination.strip().lower()

        # Try zone match
        if self._zones_data:
            for zone in self._zones_data.get("zones", []):
                if zone["name"].lower() == dest_lower:
                    verts = zone["vertices_world"]
                    cx = sum(v[0] for v in verts) / len(verts)
                    cy = sum(v[1] for v in verts) / len(verts)
                    return cx, cy, zone["name"], "zone"

        # Try object match
        if self._objects_data:
            for obj in self._objects_data.get("objects", []):
                if obj["label"].lower() == dest_lower:
                    pos = obj["map_position"]
                    return pos["x"], pos["y"], obj["label"], "object"

        return None

    @staticmethod
    def _segment_intersects_polygon(ax, ay, bx, by, vertices):
        """Test if line segment (a->b) intersects any edge of a polygon."""
        n = len(vertices)
        for i in range(n):
            cx, cy = vertices[i]
            dx, dy = vertices[(i + 1) % n]
            if SceneGraphAgent._segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                return True
        return False

    @staticmethod
    def _segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
        """Parametric line-segment intersection test."""
        denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
        if abs(denom) < 1e-12:
            return False
        t = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / denom
        u = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / denom
        return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0

    # -------------------------------------------------------------------
    # Spatial tools
    # -------------------------------------------------------------------

    def _find_nearest(self, object_type):
        if not self._objects_data:
            return json.dumps({"error": "No object data available"})

        try:
            rx, ry, yaw = self._get_robot_pose_raw()
        except (LookupException, ExtrapolationException) as e:
            return json.dumps({"error": f"Cannot get robot pose: {e}"})

        keyword = object_type.strip().lower()
        best = None
        best_dist = float("inf")

        for obj in self._objects_data.get("objects", []):
            if keyword not in obj["label"].lower():
                continue
            pos = obj["map_position"]
            ox, oy = pos["x"], pos["y"]
            dist = math.sqrt((rx - ox) ** 2 + (ry - oy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best = obj

        if best is None:
            return json.dumps({
                "error": f"No objects matching '{object_type}' found.",
                "hint": "Use list_objects to see available labels.",
            })

        pos = best["map_position"]
        direction = self._relative_direction(yaw, rx, ry, pos["x"], pos["y"])

        # Find which zone the object is in
        obj_zone = None
        if self._zones_data:
            for zone in self._zones_data.get("zones", []):
                if self._point_in_polygon(pos["x"], pos["y"], zone["vertices_world"]):
                    obj_zone = zone["name"]
                    break

        return json.dumps({
            "label": best["label"],
            "distance_m": round(best_dist, 2),
            "direction": direction,
            "zone": obj_zone,
        })

    def _distance_to(self, destination):
        try:
            rx, ry, yaw = self._get_robot_pose_raw()
        except (LookupException, ExtrapolationException) as e:
            return json.dumps({"error": f"Cannot get robot pose: {e}"})

        resolved = self._resolve_destination(destination)
        if resolved is None:
            return json.dumps({
                "error": f"Destination '{destination}' not found.",
                "hint": "Use list_zones or list_objects to see available names.",
            })

        tx, ty, name, kind = resolved
        dist = math.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
        direction = self._relative_direction(yaw, rx, ry, tx, ty)

        return json.dumps({
            "destination": name,
            "destination_type": kind,
            "distance_m": round(dist, 2),
            "direction": direction,
        })

    def _describe_route(self, destination):
        try:
            rx, ry, yaw = self._get_robot_pose_raw()
        except (LookupException, ExtrapolationException) as e:
            return json.dumps({"error": f"Cannot get robot pose: {e}"})

        resolved = self._resolve_destination(destination)
        if resolved is None:
            return json.dumps({
                "error": f"Destination '{destination}' not found.",
                "hint": "Use list_zones or list_objects to see available names.",
            })

        tx, ty, name, kind = resolved
        total_dist = math.sqrt((rx - tx) ** 2 + (ry - ty) ** 2)
        direction = self._relative_direction(yaw, rx, ry, tx, ty)

        # Path vector
        dx, dy = tx - rx, ty - ry
        path_len = math.sqrt(dx * dx + dy * dy)
        if path_len < 1e-6:
            return json.dumps({
                "destination": name,
                "total_distance_m": round(total_dist, 2),
                "direction": direction,
                "zones_along_path": [],
                "objects_along_path": [],
            })

        # Unit vectors along and perpendicular to path
        ux, uy = dx / path_len, dy / path_len

        # Zones crossed by the path
        zones_along = []
        if self._zones_data:
            for zone in self._zones_data.get("zones", []):
                verts = zone["vertices_world"]
                start_inside = self._point_in_polygon(rx, ry, verts)
                end_inside = self._point_in_polygon(tx, ty, verts)
                crosses = self._segment_intersects_polygon(rx, ry, tx, ty, verts)
                if start_inside or end_inside or crosses:
                    zones_along.append(zone["name"])

        # Objects within 2m corridor of the path
        corridor_width = 2.0
        objects_along = []
        if self._objects_data:
            for obj in self._objects_data.get("objects", []):
                pos = obj["map_position"]
                ox, oy = pos["x"], pos["y"]
                # Project object onto path vector
                vox, voy = ox - rx, oy - ry
                proj = vox * ux + voy * uy  # scalar projection along path
                if proj < 0 or proj > path_len:
                    continue
                # Perpendicular distance
                perp = abs(vox * (-uy) + voy * ux)
                if perp <= corridor_width:
                    objects_along.append({
                        "label": obj["label"],
                        "distance_along_path_m": round(proj, 2),
                        "side_offset_m": round(perp, 2),
                    })

        objects_along.sort(key=lambda o: o["distance_along_path_m"])

        return json.dumps({
            "destination": name,
            "total_distance_m": round(total_dist, 2),
            "direction": direction,
            "zones_along_path": zones_along,
            "objects_along_path": objects_along,
        })

    def _orient_me(self, target):
        try:
            rx, ry, yaw = self._get_robot_pose_raw()
        except (LookupException, ExtrapolationException) as e:
            return json.dumps({"error": f"Cannot get robot pose: {e}"})

        resolved = self._resolve_destination(target)
        if resolved is None:
            return json.dumps({
                "error": f"Target '{target}' not found.",
                "hint": "Use list_zones or list_objects to see available names.",
            })

        tx, ty, name, kind = resolved
        target_yaw = math.atan2(ty - ry, tx - rx)

        # Current relative direction before turning
        direction = self._relative_direction(yaw, rx, ry, tx, ty)

        # Compute turn angle
        turn = target_yaw - yaw
        turn = (turn + math.pi) % (2 * math.pi) - math.pi
        turn_deg = math.degrees(turn)

        # Publish goal at current position with target yaw
        self._publish_goal(rx, ry, yaw=target_yaw)
        self._nav_goal_zone = f"orient:{name}"
        self._nav_status = GoalStatus.STATUS_EXECUTING

        return json.dumps({
            "target": name,
            "direction_from_current": direction,
            "turn_degrees": round(turn_deg, 1),
            "message": f"Rotating to face {name} ({round(abs(turn_deg), 1)} degrees {'left' if turn_deg > 0 else 'right'})",
        })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scene graph navigation agent using local LLM (Nemotron 3 Nano) + Nav2",
    )
    parser.add_argument(
        "--model", default="qwen3.5:9b",
        help="Model name served by the local LLM server (default: qwen3.5:9b)",
    )
    parser.add_argument(
        "--base-url", default="http://localhost:11434/v1",
        help="Base URL of the OpenAI-compatible LLM server (default: http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--api-key", default="not-needed",
        help="API key for the LLM server (default: 'not-needed' for local servers)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=-1,
        help="Max tokens per LLM response (default: -1, unlimited)",
    )
    parser.add_argument(
        "--frame-id", default="map",
        help="TF frame for goal poses (default: map)",
    )
    parser.add_argument(
        "--base-frame", default="base_link",
        help="Robot base TF frame for pose lookup (default: base_link)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rclpy.init()
    node = SceneGraphAgent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        frame_id=args.frame_id,
        base_frame=args.base_frame,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    def _shutdown_handler(*_):
        rclpy.try_shutdown()
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _shutdown_handler)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        # Unload model from VRAM
        print(f"\n[Agent] Unloading model {args.model} from VRAM...", flush=True)
        try:
            subprocess.run(["ollama", "stop", args.model], timeout=10,
                           capture_output=True)
            print("[Agent] Model unloaded.", flush=True)
        except Exception:
            pass
        node.destroy_node()


if __name__ == "__main__":
    main()
