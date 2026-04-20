#!/usr/bin/env python3
"""
Dummy scene graph publisher for testing scene_graph_agent_local.py without a robot.

Publishes fake objects and zones on the same ROS2 topics as the real
scene_graph_publisher.py, plus a dynamic TF (map → base_link) that
simulates robot movement towards nav goals.

Also handles query/zone_query round-trips so the agent's query tools work.

Usage:
    # Terminal 1: start the dummy publisher
    python3 dummy_scene_graph_publisher.py

    # Terminal 2: start the agent
    python3 scene_graph_agent_local.py
"""

import json
import math
import signal
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster


# ---------------------------------------------------------------------------
# Dummy data
# ---------------------------------------------------------------------------

DUMMY_ZONES = {
    "zones": [
        {
            "name": "kitchen",
            "display_name": "Kitchen",
            "parent": None,
            "depth": 0,
            "children": [],
            "vertices_world": [[0, 0], [4, 0], [4, 3], [0, 3]],
            "num_objects": 2,
            "num_direct_objects": 2,
        },
        {
            "name": "living_room",
            "display_name": "Living Room",
            "parent": None,
            "depth": 0,
            "children": [],
            "vertices_world": [[4, 0], [10, 0], [10, 5], [4, 5]],
            "num_objects": 3,
            "num_direct_objects": 3,
        },
        {
            "name": "hallway",
            "display_name": "Hallway",
            "parent": None,
            "depth": 0,
            "children": [],
            "vertices_world": [[0, 3], [4, 3], [4, 5], [0, 5]],
            "num_objects": 0,
            "num_direct_objects": 0,
        },
    ]
}

DUMMY_OBJECTS = {
    "objects": [
        {
            "id": 1, "label": "fridge", "confidence": 0.95, "num_points": 500,
            "centroid": {"x": 1.0, "y": 1.0, "z": 0.0},
            "map_position": {"x": 1.0, "y": 1.0},
            "location": {"zone": "kitchen", "parent_zone": None, "full_path": "kitchen"},
        },
        {
            "id": 2, "label": "microwave", "confidence": 0.88, "num_points": 200,
            "centroid": {"x": 3.0, "y": 1.5, "z": 0.0},
            "map_position": {"x": 3.0, "y": 1.5},
            "location": {"zone": "kitchen", "parent_zone": None, "full_path": "kitchen"},
        },
        {
            "id": 3, "label": "chair", "confidence": 0.92, "num_points": 350,
            "centroid": {"x": 5.5, "y": 2.0, "z": 0.0},
            "map_position": {"x": 5.5, "y": 2.0},
            "location": {"zone": "living_room", "parent_zone": None, "full_path": "living_room"},
        },
        {
            "id": 4, "label": "table", "confidence": 0.90, "num_points": 600,
            "centroid": {"x": 6.5, "y": 2.5, "z": 0.0},
            "map_position": {"x": 6.5, "y": 2.5},
            "location": {"zone": "living_room", "parent_zone": None, "full_path": "living_room"},
        },
        {
            "id": 5, "label": "couch", "confidence": 0.93, "num_points": 800,
            "centroid": {"x": 7.0, "y": 4.0, "z": 0.0},
            "map_position": {"x": 7.0, "y": 4.0},
            "location": {"zone": "living_room", "parent_zone": None, "full_path": "living_room"},
        },
        {
            "id": 6, "label": "lamp", "confidence": 0.85, "num_points": 150,
            "centroid": {"x": 8.5, "y": 1.0, "z": 0.0},
            "map_position": {"x": 8.5, "y": 1.0},
            "location": {"zone": "living_room", "parent_zone": None, "full_path": "living_room"},
        },
    ]
}

# Robot starting pose: in the kitchen at (2.0, 1.5), facing right (yaw=0)
ROBOT_START_X = 2.0
ROBOT_START_Y = 1.5
ROBOT_START_YAW = 0.0

# Simulated movement speed
ROBOT_SPEED = 0.5        # metres per second
ROBOT_TURN_SPEED = 1.0   # radians per second
ARRIVAL_THRESHOLD = 0.15 # metres — close enough to goal


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class DummySceneGraphPublisher(Node):
    def __init__(self):
        super().__init__("dummy_scene_graph_publisher")

        # Robot state
        self._robot_x = ROBOT_START_X
        self._robot_y = ROBOT_START_Y
        self._robot_yaw = math.radians(ROBOT_START_YAW)

        # Navigation goal
        self._goal_x = None
        self._goal_y = None
        self._goal_yaw = None
        self._nav_status = GoalStatus.STATUS_UNKNOWN

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # Publishers (same topics as real scene_graph_publisher)
        self._objects_pub = self.create_publisher(String, "/scene_graph/objects", latched_qos)
        self._zones_pub = self.create_publisher(String, "/scene_graph/zones", latched_qos)
        self._query_result_pub = self.create_publisher(String, "/scene_graph/query_result", latched_qos)
        self._zone_result_pub = self.create_publisher(String, "/scene_graph/zone_result", latched_qos)

        # Nav2 status publisher (so the agent can check navigation progress)
        self._nav_status_pub = self.create_publisher(
            GoalStatusArray, "/navigate_to_pose/_action/status", 10
        )

        # Subscriptions
        self.create_subscription(String, "/scene_graph/query", self._on_query, 10)
        self.create_subscription(String, "/scene_graph/zone_query", self._on_zone_query, 10)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_pose, 10)

        # Dynamic TF broadcaster (not static — we update it as robot moves)
        self._tf_broadcaster = TransformBroadcaster(self)

        # Publish objects and zones
        obj_msg = String()
        obj_msg.data = json.dumps(DUMMY_OBJECTS)
        self._objects_pub.publish(obj_msg)

        zones_msg = String()
        zones_msg.data = json.dumps(DUMMY_ZONES)
        self._zones_pub.publish(zones_msg)

        # Re-publish scene data periodically
        self._obj_msg = obj_msg
        self._zones_msg = zones_msg
        self.create_timer(2.0, self._republish_scene)

        # Movement + TF update at 20 Hz
        self._dt = 0.05
        self.create_timer(self._dt, self._tick)

        self.get_logger().info("Dummy scene graph publisher started (with simulated movement)")
        self.get_logger().info(f"  Zones: {[z['name'] for z in DUMMY_ZONES['zones']]}")
        self.get_logger().info(f"  Objects: {[o['label'] for o in DUMMY_OBJECTS['objects']]}")
        self.get_logger().info(f"  Robot at ({self._robot_x}, {self._robot_y}), speed={ROBOT_SPEED} m/s")

    # -------------------------------------------------------------------
    # Navigation goal
    # -------------------------------------------------------------------

    def _on_goal_pose(self, msg):
        """Receive a nav goal and start moving towards it."""
        self._goal_x = msg.pose.position.x
        self._goal_y = msg.pose.position.y
        # Extract goal yaw from quaternion
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._goal_yaw = math.atan2(siny_cosp, cosy_cosp)
        self._nav_status = GoalStatus.STATUS_EXECUTING

        self.get_logger().info(
            f"Goal received: ({self._goal_x:.2f}, {self._goal_y:.2f}), "
            f"yaw={math.degrees(self._goal_yaw):.1f}deg — moving..."
        )

    # -------------------------------------------------------------------
    # Simulation tick (20 Hz)
    # -------------------------------------------------------------------

    def _tick(self):
        """Move robot towards goal, publish TF and nav status."""

        if self._goal_x is not None and self._nav_status == GoalStatus.STATUS_EXECUTING:
            dx = self._goal_x - self._robot_x
            dy = self._goal_y - self._robot_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < ARRIVAL_THRESHOLD:
                # Arrived — snap to goal and handle final yaw
                self._robot_x = self._goal_x
                self._robot_y = self._goal_y
                if self._goal_yaw is not None:
                    self._robot_yaw = self._goal_yaw
                self._nav_status = GoalStatus.STATUS_SUCCEEDED
                self._goal_x = None
                self._goal_y = None
                self._goal_yaw = None
                self.get_logger().info(
                    f"Arrived at ({self._robot_x:.2f}, {self._robot_y:.2f})"
                )
            else:
                # Turn towards goal
                target_yaw = math.atan2(dy, dx)
                yaw_diff = target_yaw - self._robot_yaw
                # Normalize to [-pi, pi]
                yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi

                max_turn = ROBOT_TURN_SPEED * self._dt
                if abs(yaw_diff) > max_turn:
                    self._robot_yaw += max_turn if yaw_diff > 0 else -max_turn
                else:
                    self._robot_yaw = target_yaw

                # Move forward (only if roughly facing the goal)
                if abs(yaw_diff) < math.radians(30):
                    step = min(ROBOT_SPEED * self._dt, dist)
                    self._robot_x += step * math.cos(self._robot_yaw)
                    self._robot_y += step * math.sin(self._robot_yaw)

        # Publish TF
        self._publish_tf()

        # Publish nav status
        self._publish_nav_status()

    def _publish_tf(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self._robot_x
        t.transform.translation.y = self._robot_y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(self._robot_yaw / 2.0)
        t.transform.rotation.w = math.cos(self._robot_yaw / 2.0)
        self._tf_broadcaster.sendTransform(t)

    def _publish_nav_status(self):
        msg = GoalStatusArray()
        if self._nav_status != GoalStatus.STATUS_UNKNOWN:
            status = GoalStatus()
            status.status = self._nav_status
            msg.status_list.append(status)
        self._nav_status_pub.publish(msg)

    # -------------------------------------------------------------------
    # Scene data republish
    # -------------------------------------------------------------------

    def _republish_scene(self):
        self._objects_pub.publish(self._obj_msg)
        self._zones_pub.publish(self._zones_msg)

    # -------------------------------------------------------------------
    # Query handlers
    # -------------------------------------------------------------------

    def _on_query(self, msg):
        """Handle object query — match by label."""
        query_label = msg.data.strip().lower()
        matches = [o for o in DUMMY_OBJECTS["objects"] if o["label"].lower() == query_label]

        result = String()
        if matches:
            result.data = json.dumps({
                "query": msg.data,
                "found": True,
                "count": len(matches),
                "objects": matches,
            })
        else:
            result.data = json.dumps({
                "query": msg.data,
                "found": False,
                "objects": [],
            })
        self._query_result_pub.publish(result)
        self.get_logger().info(f"Object query '{msg.data}': {len(matches)} match(es)")

    def _on_zone_query(self, msg):
        """Handle zone query — match by name."""
        query_name = msg.data.strip().lower()
        zone = None
        for z in DUMMY_ZONES["zones"]:
            if z["name"].lower() == query_name:
                zone = z
                break

        result = String()
        if zone:
            objects_inside = [
                o for o in DUMMY_OBJECTS["objects"]
                if o.get("location", {}).get("zone") == zone["name"]
            ]
            result.data = json.dumps({
                "query": msg.data,
                "found": True,
                "zone": {
                    "name": zone["name"],
                    "display_name": zone.get("display_name", zone["name"]),
                    "parent": zone.get("parent"),
                    "children": zone.get("children", []),
                    "vertices_world": zone["vertices_world"],
                    "objects": objects_inside,
                    "direct_objects": objects_inside,
                },
            })
        else:
            result.data = json.dumps({
                "query": msg.data,
                "found": False,
                "zone": None,
            })
        self._zone_result_pub.publish(result)
        self.get_logger().info(f"Zone query '{msg.data}': {'found' if zone else 'not found'}")


def main():
    rclpy.init()
    node = DummySceneGraphPublisher()

    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
