#!/usr/bin/env python3
"""
Standalone ROS2 node: publish scene_graph.json as RViz2 MarkerArray.

Reads scene_graph.json + map_offset.json, projects 3D centroids to 2D,
applies the ZED→AMCL offset, and publishes markers on the "map" frame.

Supports live alignment via RViz2 "2D Pose Estimate" tool:
  - Click "2D Pose Estimate" on the map where the SLAM recording started
  - The scene graph markers will snap to that position + orientation
  - Each new click updates the alignment in real time
  - Use --save-offset to persist the final alignment

Dependencies: rclpy, std_msgs, visualization_msgs, geometry_msgs (standard ROS2)

Usage:
    # With live alignment from RViz2:
    python3 scene_graph_publisher.py \
        --scene-graph logs/lab/scene_graph.json \
        --frame-id map \
        --world-axes xz \
        --align-from-pose

    # With pre-saved offset:
    python3 scene_graph_publisher.py \
        --scene-graph logs/lab/scene_graph.json \
        --load-offset datasets/zed_lab/map_offset.json \
        --frame-id map
"""

import argparse
import colorsys
import hashlib
import json
import math
import signal
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


# ---------------------------------------------------------------------------
# Helpers (inlined — no project imports needed)
# ---------------------------------------------------------------------------

def generate_color(label):
    """Deterministic RGB color for a label. Returns (r, g, b) in [0, 255]."""
    h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
    hue = (h * 137.508) % 360 / 360.0
    sat = 0.7 + (h % 4) * 0.075
    val = 0.8 + (h % 3) * 0.067
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (
        max(30, min(255, int(r * 255))),
        max(30, min(255, int(g * 255))),
        max(30, min(255, int(b * 255))),
    )


def load_scene_graph(path):
    """Load scene_graph.json."""
    with open(path, "r") as f:
        return json.load(f)


def filter_objects(scene_graph, min_confidence, min_points, excluded_labels):
    """Return objects passing all filters."""
    excl = {l.lower() for l in excluded_labels}
    out = []
    for obj in scene_graph["objects"]:
        if obj["confidence"] < min_confidence:
            continue
        if obj["num_points"] < min_points:
            continue
        if obj["label"].lower() in excl:
            continue
        out.append(obj)
    return out


def project_centroids_2d(objects, world_axes):
    """Extract 2D positions from 3D centroids.

    world_axes='xz': Y is height (Replica) -> use X, Z
    world_axes='xy': Z is height (ROS)     -> use X, Y
    """
    coords = []
    for obj in objects:
        c = obj["centroid"]
        if world_axes == "xz":
            coords.append((c["x"], c["z"]))
        else:
            coords.append((c["x"], c["y"]))
    return coords


def apply_2d_offset(coords_2d, tx, ty, rotate_deg, scale=1.0):
    """Apply 2D translate + rotate to list of (x, y) tuples."""
    if not coords_2d:
        return coords_2d
    theta = math.radians(rotate_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    out = []
    for x, y in coords_2d:
        xs, ys = x * scale, y * scale
        out.append((xs * cos_t - ys * sin_t + tx,
                     xs * sin_t + ys * cos_t + ty))
    return out


def compute_zone_hierarchy(zones):
    """Compute parent/depth/children for a list of zone dicts.

    A zone's parent is the smallest zone that fully contains it.
    """
    n = len(zones)
    polys = []
    areas = []
    for z in zones:
        verts = z.get("vertices_world", [])
        if len(verts) >= 3:
            # Build simple polygon area via shoelace
            area = 0.0
            for i in range(len(verts)):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % len(verts)]
                area += x1 * y2 - x2 * y1
            area = abs(area) / 2.0
            polys.append(verts)
            areas.append(area)
        else:
            polys.append(None)
            areas.append(0)

    for z in zones:
        z["parent"] = z.get("parent", None)
        z["depth"] = z.get("depth", 0)
        z["children"] = z.get("children", [])

    # Compute parent as smallest containing zone
    for i in range(n):
        if polys[i] is None:
            continue
        best_parent = None
        best_area = float("inf")
        for j in range(n):
            if i == j or polys[j] is None:
                continue
            if _polygon_contains_polygon(polys[j], polys[i]) and areas[j] < best_area:
                best_parent = j
                best_area = areas[j]
        zones[i]["parent"] = zones[best_parent]["name"] if best_parent is not None else None

    # Compute depths
    name_to_idx = {z["name"]: idx for idx, z in enumerate(zones)}
    for i in range(n):
        depth = 0
        cur = i
        visited = set()
        while zones[cur]["parent"] is not None and zones[cur]["parent"] in name_to_idx:
            parent_idx = name_to_idx[zones[cur]["parent"]]
            if parent_idx in visited:
                break
            visited.add(parent_idx)
            depth += 1
            cur = parent_idx
        zones[i]["depth"] = depth

    # Compute children
    children_map = {z["name"]: [] for z in zones}
    for z in zones:
        if z["parent"] and z["parent"] in children_map:
            children_map[z["parent"]].append(z["name"])
    for z in zones:
        z["children"] = children_map.get(z["name"], [])


def _polygon_contains_polygon(outer_verts, inner_verts):
    """Check if all vertices of inner polygon are inside outer polygon."""
    for x, y in inner_verts:
        if not _point_in_polygon_static(x, y, outer_verts):
            return False
    return True


def _point_in_polygon_static(x, y, vertices):
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


def get_zone_display_name(zone, zones):
    """Build hierarchical display name like 'Parent > Child'."""
    if zone.get("parent") is None:
        return zone["name"]
    name_to_zone = {z["name"]: z for z in zones}
    parts = [zone["name"]]
    cur = zone
    visited = set()
    while cur.get("parent") is not None and cur["parent"] in name_to_zone:
        if cur["parent"] in visited:
            break
        visited.add(cur["parent"])
        cur = name_to_zone[cur["parent"]]
        parts.append(cur["name"])
    parts.reverse()
    return " > ".join(parts)


def compute_object_locations(objects, coords_2d, zones):
    """For each object, find its deepest (most nested) containing zone."""
    locations = {}
    if not zones:
        return locations
    for i, obj in enumerate(objects):
        wx, wy = coords_2d[i]
        best_zone = None
        best_depth = -1
        for z in zones:
            verts = z.get("vertices_world", [])
            if len(verts) < 3:
                continue
            if _point_in_polygon_static(wx, wy, verts):
                d = z.get("depth", 0)
                if d > best_depth:
                    best_depth = d
                    best_zone = z
        if best_zone is not None:
            locations[obj["id"]] = {
                "zone": best_zone["name"],
                "parent_zone": best_zone.get("parent", None),
                "full_path": get_zone_display_name(best_zone, zones),
            }
    return locations


def reverse_2d_offset(coords_2d, tx, ty, rotate_deg, scale=1.0):
    """Reverse a 2D translate + rotate to recover raw coordinates."""
    if not coords_2d:
        return coords_2d
    theta = math.radians(-rotate_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    inv_scale = 1.0 / scale if scale != 0 else 1.0
    out = []
    for x, y in coords_2d:
        # Undo translation, then undo rotation, then undo scale
        dx, dy = x - tx, y - ty
        rx, ry = dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t
        out.append((rx * inv_scale, ry * inv_scale))
    return out


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------

class SceneGraphPublisher(Node):
    def __init__(self, objects, coords_2d_raw, coords_2d_aligned, frame_id,
                 rate, align_from_pose=False, save_offset_path=None,
                 zones_data=None):
        super().__init__("scene_graph_publisher")

        self._objects = objects
        self._coords_2d_raw = coords_2d_raw        # un-transformed 2D coords
        self._coords_2d = coords_2d_aligned         # currently aligned coords
        self._frame_id = frame_id
        self._save_offset_path = save_offset_path
        self._zones_data = zones_data               # loaded location_zones.json
        self._zones_raw = None                      # raw (un-aligned) zone vertices
        self._obj_locations = {}                    # obj_id -> location dict

        # Current offset state (for saving)
        self._tx = 0.0
        self._ty = 0.0
        self._rot_deg = 0.0
        self._scale = 1.0

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # --- Always-publishing topics ---
        self._marker_pub = self.create_publisher(
            MarkerArray, "/scene_graph/markers", latched_qos
        )
        self._json_pub = self.create_publisher(
            String, "/scene_graph/objects", latched_qos
        )

        # --- Query interface ---
        self._query_marker_pub = self.create_publisher(
            MarkerArray, "/scene_graph/query_markers", latched_qos
        )
        self._query_result_pub = self.create_publisher(
            String, "/scene_graph/query_result", latched_qos
        )
        self.create_subscription(
            String, "/scene_graph/query", self._on_query, 10
        )

        # --- Zone query interface ---
        if self._zones_data:
            self._zone_index = {}
            for zone in self._zones_data.get("zones", []):
                self._zone_index[zone["name"].lower()] = zone

            # Store raw (un-aligned) zone vertices for re-alignment
            meta = self._zones_data.get("metadata", {}).get("alignment", {})
            orig_tx = meta.get("translate_x", 0.0)
            orig_ty = meta.get("translate_y", 0.0)
            orig_rot = meta.get("rotate_deg", 0.0)
            orig_scale = meta.get("scale", 1.0)
            self._zones_raw = []
            for zone in self._zones_data.get("zones", []):
                raw_verts = reverse_2d_offset(
                    zone["vertices_world"], orig_tx, orig_ty, orig_rot, orig_scale
                )
                self._zones_raw.append({
                    "name": zone["name"],
                    "vertices_raw": raw_verts,
                })

            self._zone_marker_pub = self.create_publisher(
                MarkerArray, "/scene_graph/zone_markers", latched_qos
            )
            self._zone_result_pub = self.create_publisher(
                String, "/scene_graph/zone_result", latched_qos
            )
            self._zones_pub = self.create_publisher(
                String, "/scene_graph/zones", latched_qos
            )
            self._list_zones_pub = self.create_publisher(
                String, "/scene_graph/list_zones_result", latched_qos
            )
            self.create_subscription(
                String, "/scene_graph/zone_query", self._on_zone_query, 10
            )
            self.create_subscription(
                String, "/scene_graph/list_zones", self._on_list_zones, 10
            )

        # --- Live alignment via 2D Pose Estimate ---
        if align_from_pose:
            self.create_subscription(
                PoseWithCovarianceStamped,
                "/initialpose",
                self._on_initial_pose,
                10,
            )
            self.get_logger().info(
                "Listening on /initialpose — click '2D Pose Estimate' in "
                "RViz2 to set the SLAM origin on the map"
            )

        # Build and publish initial state
        self._rebuild_and_publish()

        # Re-publish on timer
        period = 1.0 / rate if rate > 0 else 1.0
        self.create_timer(period, self._publish)

        self.get_logger().info(
            f"Publishing {len(objects)} objects "
            f"on /scene_graph/markers at {rate} Hz"
        )
        self.get_logger().info(
            f"Query ready on /scene_graph/query — "
            f"available labels: {sorted({o['label'].lower() for o in objects})}"
        )
        if self._zones_data:
            zone_names = sorted(self._zone_index.keys())
            self.get_logger().info(
                f"Zone query ready on /scene_graph/zone_query — "
                f"available zones: {zone_names}"
            )

    def _on_initial_pose(self, msg):
        """Handle 2D Pose Estimate from RViz2.

        The clicked pose represents where the SLAM origin (0,0) is on the map.
        Extract (x, y, yaw) and use as the alignment transform.
        """
        try:
            pose = msg.pose.pose
            tx = pose.position.x
            ty = pose.position.y

            # Extract yaw from quaternion
            q = pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw_rad = math.atan2(siny_cosp, cosy_cosp)
            rot_deg = math.degrees(yaw_rad)

            self._tx = tx
            self._ty = ty
            self._rot_deg = rot_deg

            self.get_logger().info(
                f"Pose estimate received: tx={tx:.3f}, ty={ty:.3f}, "
                f"yaw={rot_deg:.1f}deg — re-aligning scene graph"
            )

            # Re-align all objects
            self._coords_2d = apply_2d_offset(
                self._coords_2d_raw, tx, ty, rot_deg, self._scale
            )

            # Re-align zone vertices
            if self._zones_raw and self._zones_data:
                for zone, raw in zip(
                    self._zones_data.get("zones", []), self._zones_raw
                ):
                    zone["vertices_world"] = apply_2d_offset(
                        raw["vertices_raw"], tx, ty, rot_deg, self._scale
                    )
                # Rebuild zone index
                self._zone_index = {}
                for zone in self._zones_data.get("zones", []):
                    self._zone_index[zone["name"].lower()] = zone
                # Update published zones JSON
                self._zones_msg = String()
                self._zones_msg.data = json.dumps(self._zones_data)

            self._rebuild_and_publish()

            # Save offset if path provided
            if self._save_offset_path:
                import os
                os.makedirs(os.path.dirname(self._save_offset_path), exist_ok=True)
                offset_data = {
                    "translate_x": round(tx, 4),
                    "translate_y": round(ty, 4),
                    "rotate_deg": round(rot_deg, 2),
                    "scale": self._scale,
                }
                with open(self._save_offset_path, "w") as f:
                    json.dump(offset_data, f, indent=2)
                self.get_logger().info(f"Saved offset: {self._save_offset_path}")
        except Exception as e:
            self.get_logger().error(f"Error handling pose estimate: {e}")
            import traceback
            traceback.print_exc()

    def _rebuild_and_publish(self):
        """Rebuild markers and JSON from current coords, then publish."""
        # Rebuild label index
        self._label_index = {}
        for i, obj in enumerate(self._objects):
            key = obj["label"].lower()
            self._label_index.setdefault(key, []).append(
                (obj, self._coords_2d[i])
            )

        # Recompute zone hierarchy and object locations
        if self._zones_data:
            zone_list = self._zones_data.get("zones", [])
            compute_zone_hierarchy(zone_list)
            self._obj_locations = compute_object_locations(
                self._objects, self._coords_2d, zone_list
            )
            # Rebuild zone index with updated hierarchy
            self._zone_index = {}
            for zone in zone_list:
                self._zone_index[zone["name"].lower()] = zone
        else:
            self._obj_locations = {}

        self._marker_array = self._build_markers(
            self._objects, self._coords_2d, self._frame_id
        )
        self._json_msg = self._build_json(
            self._objects, self._coords_2d, self._obj_locations
        )

        # Build all-zones markers and zones JSON if zones are loaded
        if self._zones_data:
            zone_list = self._zones_data.get("zones", [])
            self._all_zones_marker_array = self._build_all_zone_markers(
                zone_list, self._frame_id
            )
            # Build zones JSON with hierarchy for /scene_graph/zones
            self._zones_msg = self._build_zones_json(
                zone_list, self._obj_locations
            )

        self._publish()

    def _publish(self):
        self._marker_pub.publish(self._marker_array)
        self._json_pub.publish(self._json_msg)
        if self._zones_data and hasattr(self, '_zones_msg'):
            self._zones_pub.publish(self._zones_msg)
        if self._zones_data and hasattr(self, '_all_zones_marker_array'):
            self._zone_marker_pub.publish(self._all_zones_marker_array)

    def _on_query(self, msg):
        """Handle query: find objects matching the requested label."""
        try:
            self._on_query_impl(msg)
        except Exception as e:
            self.get_logger().error(f"Error handling query '{msg.data}': {e}")

    def _on_query_impl(self, msg):
        query_label = msg.data.strip().lower()
        matches = self._label_index.get(query_label, [])

        if not matches:
            self.get_logger().warn(
                f"Query '{msg.data}': no objects found. "
                f"Available: {sorted(self._label_index.keys())}"
            )
            result = String()
            result.data = json.dumps({
                "query": msg.data,
                "found": False,
                "objects": [],
            })
            self._query_result_pub.publish(result)

            clear = MarkerArray()
            d = Marker()
            d.action = Marker.DELETEALL
            clear.markers.append(d)
            self._query_marker_pub.publish(clear)
            return

        result_objects = []
        query_objs = []
        query_coords = []
        for obj, (wx, wy) in matches:
            entry = {
                "id": obj["id"],
                "label": obj["label"],
                "confidence": obj["confidence"],
                "num_points": obj["num_points"],
                "centroid_3d": obj["centroid"],
                "map_position": {"x": float(wx), "y": float(wy)},
            }
            loc = self._obj_locations.get(obj["id"])
            if loc:
                entry["location"] = loc
            else:
                entry["location"] = {"zone": None, "parent_zone": None, "full_path": None}
            result_objects.append(entry)
            query_objs.append(obj)
            query_coords.append((wx, wy))

        result = String()
        result.data = json.dumps({
            "query": msg.data,
            "found": True,
            "count": len(matches),
            "objects": result_objects,
        })
        self._query_result_pub.publish(result)

        self._query_marker_pub.publish(
            self._build_markers(query_objs, query_coords, self._frame_id)
        )

        self.get_logger().info(
            f"Query '{msg.data}': {len(matches)} match(es) → "
            f"/scene_graph/query_markers + /scene_graph/query_result"
        )

    def _on_zone_query(self, msg):
        """Handle zone query: find zone by name and return polygon + objects."""
        try:
            self._on_zone_query_impl(msg)
        except Exception as e:
            self.get_logger().error(f"Error handling zone query '{msg.data}': {e}")

    def _on_zone_query_impl(self, msg):
        query_name = msg.data.strip().lower()
        zone = self._zone_index.get(query_name)

        if not zone:
            self.get_logger().warn(
                f"Zone query '{msg.data}': not found. "
                f"Available: {sorted(self._zone_index.keys())}"
            )
            result = String()
            result.data = json.dumps({
                "query": msg.data,
                "found": False,
                "zone": None,
            })
            self._zone_result_pub.publish(result)

            clear = MarkerArray()
            d = Marker()
            d.action = Marker.DELETEALL
            clear.markers.append(d)
            self._zone_marker_pub.publish(clear)
            return

        # Find ALL objects geometrically inside this zone
        all_objects = []
        direct_objects = []
        for i, obj in enumerate(self._objects):
            wx, wy = self._coords_2d[i]
            if self._point_in_polygon(wx, wy, zone["vertices_world"]):
                entry = {
                    "id": obj["id"],
                    "label": obj["label"],
                    "map_position": {"x": float(wx), "y": float(wy)},
                }
                loc = self._obj_locations.get(obj["id"])
                if loc:
                    entry["location"] = loc
                all_objects.append(entry)
                # Direct = object's deepest zone is this zone (not in a child)
                if loc and loc["zone"] == zone["name"]:
                    direct_objects.append(entry)

        # Build zone hierarchy display name
        zone_list = self._zones_data.get("zones", [])
        display_name = get_zone_display_name(zone, zone_list)

        result = String()
        result.data = json.dumps({
            "query": msg.data,
            "found": True,
            "zone": {
                "name": zone["name"],
                "display_name": display_name,
                "parent": zone.get("parent", None),
                "depth": zone.get("depth", 0),
                "children": zone.get("children", []),
                "vertices_world": zone["vertices_world"],
                "objects": all_objects,
                "direct_objects": direct_objects,
            },
        })
        self._zone_result_pub.publish(result)

        self._zone_marker_pub.publish(
            self._build_zone_markers(zone, self._frame_id)
        )

        self.get_logger().info(
            f"Zone query '{msg.data}': found with {len(all_objects)} "
            f"object(s) → /scene_graph/zone_markers + /scene_graph/zone_result"
        )

    def _on_list_zones(self, msg):
        """Handle list_zones query: return all zones with hierarchy info."""
        try:
            self._on_list_zones_impl(msg)
        except Exception as e:
            self.get_logger().error(f"Error handling list_zones: {e}")

    def _on_list_zones_impl(self, msg):
        if not self._zones_data:
            result = String()
            result.data = json.dumps({"zones": []})
            self._list_zones_pub.publish(result)
            return

        zone_list = self._zones_data.get("zones", [])
        zones_summary = []
        for z in zone_list:
            verts = z.get("vertices_world", [])
            # Count ALL objects geometrically inside this zone (including child zones)
            all_objects_count = 0
            # Count only objects whose deepest zone is this one (not in a child)
            direct_objects_count = 0
            if len(verts) >= 3:
                for i, obj in enumerate(self._objects):
                    wx, wy = self._coords_2d[i]
                    if self._point_in_polygon(wx, wy, verts):
                        all_objects_count += 1
                        loc = self._obj_locations.get(obj["id"])
                        if loc and loc["zone"] == z["name"]:
                            direct_objects_count += 1
            zones_summary.append({
                "name": z["name"],
                "display_name": get_zone_display_name(z, zone_list),
                "parent": z.get("parent", None),
                "depth": z.get("depth", 0),
                "children": z.get("children", []),
                "num_objects": all_objects_count,
                "num_direct_objects": direct_objects_count,
                "vertices_world": verts,
            })

        result = String()
        result.data = json.dumps({"zones": zones_summary})
        self._list_zones_pub.publish(result)

        self.get_logger().info(
            f"list_zones: {len(zones_summary)} zones → "
            f"/scene_graph/list_zones_result"
        )

    def _build_zones_json(self, zones, obj_locations):
        """Build zones JSON with full hierarchy for /scene_graph/zones."""
        zones_out = []
        for z in zones:
            verts = z.get("vertices_world", [])
            all_objects_count = 0
            direct_objects_count = 0
            if len(verts) >= 3:
                for i, obj in enumerate(self._objects):
                    wx, wy = self._coords_2d[i]
                    if self._point_in_polygon(wx, wy, verts):
                        all_objects_count += 1
                        loc = obj_locations.get(obj["id"])
                        if loc and loc["zone"] == z["name"]:
                            direct_objects_count += 1
            zones_out.append({
                "name": z["name"],
                "display_name": get_zone_display_name(z, zones),
                "parent": z.get("parent", None),
                "depth": z.get("depth", 0),
                "children": z.get("children", []),
                "vertices_world": verts,
                "num_objects": all_objects_count,
                "num_direct_objects": direct_objects_count,
            })
        msg = String()
        msg.data = json.dumps({"zones": zones_out})
        return msg

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

    @staticmethod
    def _build_all_zone_markers(zones, frame_id):
        """Build LINE_STRIP + label markers for ALL zones at once."""
        ma = MarkerArray()
        for zi, zone in enumerate(zones):
            verts = zone.get("vertices_world", [])
            if len(verts) < 3:
                continue

            r, g, b = generate_color(zone["name"])
            depth = zone.get("depth", 0)

            # Polygon boundary — nested zones rendered slightly higher
            line = Marker()
            line.header.frame_id = frame_id
            line.ns = "all_zone_polygons"
            line.id = zi * 2
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.03 if depth > 0 else 0.05
            line.color.r = r / 255.0
            line.color.g = g / 255.0
            line.color.b = b / 255.0
            line.color.a = 0.9

            z_height = 0.05 + depth * 0.05
            for vx, vy in verts:
                p = Point()
                p.x = float(vx)
                p.y = float(vy)
                p.z = z_height
                line.points.append(p)

            # Close polygon
            v0 = verts[0]
            vn = verts[-1]
            if v0[0] != vn[0] or v0[1] != vn[1]:
                p = Point()
                p.x = float(v0[0])
                p.y = float(v0[1])
                p.z = z_height
                line.points.append(p)

            ma.markers.append(line)

            # Zone label — show hierarchy path for nested zones
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)

            display_name = get_zone_display_name(zone, zones)

            text = Marker()
            text.header.frame_id = frame_id
            text.ns = "all_zone_labels"
            text.id = zi * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(cx)
            text.pose.position.y = float(cy)
            text.pose.position.z = 0.5 + depth * 0.15
            text.pose.orientation.w = 1.0
            text.scale.z = max(0.2, 0.4 - depth * 0.05)
            text.text = display_name
            text.color.r = 0.0
            text.color.g = 0.0
            text.color.b = 0.0
            text.color.a = 1.0
            ma.markers.append(text)

        return ma

    @staticmethod
    def _build_zone_markers(zone, frame_id):
        """Build a LINE_STRIP MarkerArray for a zone polygon boundary."""
        ma = MarkerArray()
        verts = zone.get("vertices_world", [])
        if len(verts) < 3:
            return ma

        r, g, b = generate_color(zone["name"])

        line = Marker()
        line.header.frame_id = frame_id
        line.ns = "zone_polygon"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.05  # line width
        line.color.r = r / 255.0
        line.color.g = g / 255.0
        line.color.b = b / 255.0
        line.color.a = 0.9

        for vx, vy in verts:
            p = Point()
            p.x = float(vx)
            p.y = float(vy)
            p.z = 0.05
            line.points.append(p)

        # Close the polygon if not already closed
        v0 = verts[0]
        vn = verts[-1]
        if v0[0] != vn[0] or v0[1] != vn[1]:
            p = Point()
            p.x = float(v0[0])
            p.y = float(v0[1])
            p.z = 0.05
            line.points.append(p)

        ma.markers.append(line)

        # Zone name label at centroid
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)

        text = Marker()
        text.header.frame_id = frame_id
        text.ns = "zone_label"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(cx)
        text.pose.position.y = float(cy)
        text.pose.position.z = 0.5
        text.pose.orientation.w = 1.0
        text.scale.z = 0.4
        text.text = zone["name"]
        text.color.r = 0.0
        text.color.g = 0.0
        text.color.b = 0.0
        text.color.a = 1.0
        ma.markers.append(text)

        return ma

    @staticmethod
    def _build_markers(objects, coords_2d, frame_id):
        ma = MarkerArray()
        for i, obj in enumerate(objects):
            wx, wy = coords_2d[i]
            label = obj["label"]
            obj_id = obj["id"]
            r, g, b = generate_color(label)

            # Sphere
            sphere = Marker()
            sphere.header.frame_id = frame_id
            sphere.ns = "centroids"
            sphere.id = obj_id * 2
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position.x = float(wx)
            sphere.pose.position.y = float(wy)
            sphere.pose.position.z = 0.3
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.4
            sphere.scale.y = 0.4
            sphere.scale.z = 0.4
            sphere.color.r = r / 255.0
            sphere.color.g = g / 255.0
            sphere.color.b = b / 255.0
            sphere.color.a = 0.8
            ma.markers.append(sphere)

            # Text label
            text = Marker()
            text.header.frame_id = frame_id
            text.ns = "labels"
            text.id = obj_id * 2 + 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(wx)
            text.pose.position.y = float(wy)
            text.pose.position.z = 0.6
            text.pose.orientation.w = 1.0
            text.scale.z = 0.3
            text.text = label
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            ma.markers.append(text)

        return ma

    @staticmethod
    def _build_json(objects, coords_2d, obj_locations=None):
        if obj_locations is None:
            obj_locations = {}
        entries = []
        for i, obj in enumerate(objects):
            wx, wy = coords_2d[i]
            entry = {
                "id": obj["id"],
                "label": obj["label"],
                "confidence": obj["confidence"],
                "num_points": obj["num_points"],
                "centroid_3d": obj["centroid"],
                "map_position": {"x": float(wx), "y": float(wy)},
            }
            loc = obj_locations.get(obj["id"])
            if loc:
                entry["location"] = loc
            else:
                entry["location"] = {"zone": None, "parent_zone": None, "full_path": None}
            entries.append(entry)
        msg = String()
        msg.data = json.dumps({"objects": entries})
        return msg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Publish scene graph as RViz2 MarkerArray on AMCL map frame",
    )
    parser.add_argument("--scene-graph", default=None, help="scene_graph.json path")
    parser.add_argument(
        "--load-combined", default=None,
        help="scene_with_zones.json — single file with objects + zones + locations. "
             "Replaces --scene-graph and --load-zones.",
    )
    parser.add_argument(
        "--load-offset", default=None,
        help="map_offset.json (overrides --translate-x/y/rotate-deg)",
    )
    parser.add_argument("--translate-x", type=float, default=0.0)
    parser.add_argument("--translate-y", type=float, default=0.0)
    parser.add_argument("--rotate-deg", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--frame-id", default="map", help="TF frame (default: map)")
    parser.add_argument(
        "--world-axes", choices=["xz", "xy"], default="xy",
        help="3D→2D projection: xz=Y-up, xy=Z-up (default: xy)",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Publish rate in Hz")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--excluded-labels", nargs="*", default=[])

    # Zone data
    parser.add_argument(
        "--load-zones", default=None,
        help="location_zones.json path — enables zone query interface",
    )

    # Live alignment from RViz2
    parser.add_argument(
        "--align-from-pose", action="store_true",
        help="Subscribe to /initialpose (RViz2 '2D Pose Estimate') for live "
             "alignment. Click where the SLAM recording started on the map.",
    )
    parser.add_argument(
        "--save-offset", default=None,
        help="Path to save map_offset.json when alignment is received "
             "(e.g. output/map_offset.json). Used with --align-from-pose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load from combined file or separate files ---
    if args.load_combined:
        with open(args.load_combined, "r") as f:
            combined = json.load(f)

        # Extract objects (with location info already embedded)
        all_objects = {"objects": combined["objects"]}
        objects = filter_objects(
            all_objects, args.min_confidence, args.min_points, args.excluded_labels
        )

        # Extract offset from metadata
        meta = combined.get("metadata", {})
        alignment = meta.get("alignment", {})
        args.translate_x = alignment.get("translate_x", args.translate_x)
        args.translate_y = alignment.get("translate_y", args.translate_y)
        args.rotate_deg = alignment.get("rotate_deg", args.rotate_deg)
        args.scale = alignment.get("scale", args.scale)
        args.world_axes = meta.get("world_axes", args.world_axes)

        # Extract zones
        zones_data = {
            "metadata": meta,
            "zones": combined.get("zones", []),
        }
        if not combined.get("zones"):
            zones_data = None

        source_name = args.load_combined
        print(f"Loaded combined file: {args.load_combined}")
        print(f"  {len(objects)} objects, "
              f"{len(combined.get('zones', []))} zones")

    else:
        if not args.scene_graph:
            print("Error: either --scene-graph or --load-combined is required.")
            sys.exit(1)

        # Load and filter
        scene_graph = load_scene_graph(args.scene_graph)
        objects = filter_objects(
            scene_graph, args.min_confidence, args.min_points, args.excluded_labels
        )
        source_name = args.scene_graph

        # Load offset
        if args.load_offset:
            with open(args.load_offset, "r") as f:
                offset = json.load(f)
            args.translate_x = offset["translate_x"]
            args.translate_y = offset["translate_y"]
            args.rotate_deg = offset["rotate_deg"]
            args.scale = offset.get("scale", 1.0)

        # Load zones if provided
        zones_data = None
        if args.load_zones:
            with open(args.load_zones, "r") as f:
                zones_data = json.load(f)
            zone_names = [z["name"] for z in zones_data.get("zones", [])]
            print(f"Zones: {len(zone_names)} zones from {args.load_zones}: {zone_names}")

    if not objects:
        print("No objects after filtering — nothing to publish.")
        return

    # Project 3D → 2D (raw, un-transformed)
    coords_2d_raw = project_centroids_2d(objects, args.world_axes)

    # Apply initial 2D offset (identity if --align-from-pose with no preset)
    aligned = apply_2d_offset(
        coords_2d_raw, args.translate_x, args.translate_y,
        args.rotate_deg, args.scale,
    )

    print(f"Scene graph: {len(objects)} objects from {source_name}")
    print(f"  Offset: tx={args.translate_x}, ty={args.translate_y}, "
          f"rot={args.rotate_deg}deg, scale={args.scale}")
    print(f"  Frame: {args.frame_id}, Rate: {args.rate} Hz")
    if args.align_from_pose:
        print(f"  Live alignment: listening on /initialpose")
        print(f"  → Use '2D Pose Estimate' in RViz2 to set SLAM origin")

    # Start ROS2 node
    rclpy.init()
    node = SceneGraphPublisher(
        objects,
        coords_2d_raw,
        aligned,
        args.frame_id,
        args.rate,
        align_from_pose=args.align_from_pose,
        save_offset_path=args.save_offset,
        zones_data=zones_data,
    )

    signal.signal(signal.SIGINT, lambda *_: rclpy.shutdown())
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
