#!/usr/bin/env python3
"""
Combined scene graph editor: adjust markers + draw zone boundaries on a 2D map.

Modes:
    ALIGN (default)   Move/rotate/delete object markers
    ZONE (press N)    Draw zone boundary polygons

Align Controls:
    Arrow keys      Translate all (Shift=fine, Ctrl=coarse)
    R / E           Rotate CW / CCW (Shift=1°, default 5°)
    Click           Select nearest marker (yellow ring)
    Drag            Move selected marker individually
    D / Delete      Delete selected marker
    U               Undo last delete
    A               Add new object at click location

Zone Controls:
    N               Start drawing a new zone polygon
    Click           Add vertex while drawing
    Enter/DblClick  Close polygon and name it
    Backspace       Remove last vertex while drawing
    Click zone      Select zone (when not drawing)
    Drag vertex     Move zone vertex
    D / Delete      Delete selected zone
    Z               Undo last zone deletion
    T               Toggle object marker visibility
    L               Toggle zone labels

General:
    Scroll          Zoom in/out (centered on cursor)
    Middle/Right-drag  Pan the map view
    S               Save all (scene graph + offset + zones)
    Q               Save and quit
    Escape          Cancel draw / deselect / quit without saving

Usage:
    # Load from combined file (recommended):
    python scene_graph_editor.py \
        --load-combined scene_with_zones.json \
        --map-pgm song_map_lidar_.pgm \
        --map-yaml song_map_lidar_.yaml

    # Legacy (separate files):
    python scene_graph_editor.py \
        --scene-graph scene_graph.json \
        --map-pgm song_map_lidar_.pgm \
        --map-yaml song_map_lidar_.yaml \
        --world-axes xz \
        --load-offset map_offset.json
"""

import argparse
import colorsys
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
from PIL import Image
from shapely.geometry import Point, Polygon as ShapelyPolygon


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def load_occupancy_grid(pgm_path, yaml_path):
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img = np.array(Image.open(pgm_path))
    return img, meta["resolution"], meta["origin"]


def load_scene_graph(path):
    with open(path) as f:
        return json.load(f)


def project_centroids_2d(objects, world_axes):
    coords = []
    for obj in objects:
        c = obj["centroid"]
        if world_axes == "xz":
            coords.append([c["x"], c["z"]])
        else:
            coords.append([c["x"], c["y"]])
    return np.array(coords) if coords else np.zeros((0, 2))


def apply_offset(coords, tx, ty, rotate_deg):
    if len(coords) == 0:
        return coords.copy()
    theta = np.radians(rotate_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rotated = coords.copy()
    rotated[:, 0] = coords[:, 0] * cos_t - coords[:, 1] * sin_t
    rotated[:, 1] = coords[:, 0] * sin_t + coords[:, 1] * cos_t
    rotated[:, 0] += tx
    rotated[:, 1] += ty
    return rotated


def world_to_pixel(coords, resolution, origin, img_height):
    if len(coords) == 0:
        return np.zeros((0, 2))
    coords = np.atleast_2d(coords)
    px = (coords[:, 0] - origin[0]) / resolution
    py = img_height - (coords[:, 1] - origin[1]) / resolution
    return np.column_stack([px, py])


def pixel_to_world(px, py, resolution, origin, img_height):
    wx = px * resolution + origin[0]
    wy = (img_height - py) * resolution + origin[1]
    return wx, wy


def generate_color(label):
    h = hash(label) & 0xFFFFFFFF
    r = ((h * 2654435761) >> 16) & 0xFF
    g = ((h * 2246822519) >> 8) & 0xFF
    b = ((h * 3266489917) >> 0) & 0xFF
    max_c = max(r, g, b, 1)
    scale = 220 / max_c
    return (min(int(r * scale), 255) / 255,
            min(int(g * scale), 255) / 255,
            min(int(b * scale), 255) / 255)


def generate_zone_color(index):
    golden = 0.618033988749895
    hue = (index * golden) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
    return (r, g, b)


def filter_objects(objects, min_confidence, min_points, excluded_labels):
    excl = set(l.lower() for l in excluded_labels)
    return [o for o in objects if
            o["confidence"] >= min_confidence and
            o["num_points"] >= min_points and
            o["label"].lower() not in excl]


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------

def compute_objects_in_zones(zones, objects, coords_2d_world):
    for zone in zones:
        verts = zone["vertices_world"]
        if len(verts) < 3:
            zone["object_ids"] = []
            zone["objects"] = []
            continue
        poly = ShapelyPolygon(verts)
        obj_ids = []
        obj_entries = []
        for i, obj in enumerate(objects):
            if i < len(coords_2d_world):
                pt = Point(coords_2d_world[i])
                if poly.contains(pt):
                    obj_ids.append(obj["id"])
                    obj_entries.append({"id": obj["id"], "label": obj["label"]})
        zone["object_ids"] = obj_ids
        zone["objects"] = obj_entries


def compute_zone_hierarchy(zones):
    """Determine parent-child relationships between zones.

    A zone's parent is the *smallest* zone that fully contains it.
    Sets zone["parent"] to the parent zone's name (or None for top-level)
    and zone["depth"] to the nesting depth (0 = top-level).
    """
    n = len(zones)
    polys = []
    areas = []
    for z in zones:
        verts = z["vertices_world"]
        if len(verts) >= 3:
            p = ShapelyPolygon(verts)
            polys.append(p)
            areas.append(p.area)
        else:
            polys.append(None)
            areas.append(0)

    # For each zone, find the smallest containing zone (by area)
    for i in range(n):
        zones[i]["parent"] = None
        zones[i]["depth"] = 0

    for i in range(n):
        if polys[i] is None:
            continue
        best_parent = None
        best_area = float("inf")
        for j in range(n):
            if i == j or polys[j] is None:
                continue
            if polys[j].contains(polys[i]) and areas[j] < best_area:
                best_parent = j
                best_area = areas[j]
        if best_parent is not None:
            zones[i]["parent"] = zones[best_parent]["name"]

    # Compute depths by walking parent chains
    name_to_idx = {z["name"]: idx for idx, z in enumerate(zones)}
    for i in range(n):
        depth = 0
        cur = i
        visited = set()
        while zones[cur]["parent"] is not None and zones[cur]["parent"] in name_to_idx:
            parent_idx = name_to_idx[zones[cur]["parent"]]
            if parent_idx in visited:
                break  # avoid cycles
            visited.add(parent_idx)
            depth += 1
            cur = parent_idx
        zones[i]["depth"] = depth


def get_zone_display_name(zone, zones):
    """Build a hierarchical display name like 'Parent > Child'."""
    if zone["parent"] is None:
        return zone["name"]
    # Walk up the chain to build full path
    name_to_zone = {z["name"]: z for z in zones}
    parts = [zone["name"]]
    cur = zone
    visited = set()
    while cur["parent"] is not None and cur["parent"] in name_to_zone:
        if cur["parent"] in visited:
            break
        visited.add(cur["parent"])
        cur = name_to_zone[cur["parent"]]
        parts.append(cur["name"])
    parts.reverse()
    return " > ".join(parts)


def load_zones(path):
    with open(path) as f:
        data = json.load(f)
    zones = []
    for z in data.get("zones", []):
        zones.append({
            "name": z["name"],
            "vertices_world": [list(v) for v in z["vertices_world"]],
            "object_ids": z.get("object_ids", []),
            "objects": z.get("objects", []),
            "parent": z.get("parent", None),
            "depth": z.get("depth", 0),
        })
    return zones


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scene graph editor: adjust markers + draw zone boundaries")
    parser.add_argument("--scene-graph", default=None,
                        help="scene_graph.json (legacy, use --load-combined instead)")
    parser.add_argument("--load-combined", default=None,
                        help="scene_with_zones.json — single file with objects + zones + offset")
    parser.add_argument("--map-pgm", required=True)
    parser.add_argument("--map-yaml", required=True)
    parser.add_argument("--world-axes", choices=["xz", "xy"], default="xz")
    parser.add_argument("--load-offset", default=None,
                        help="(legacy) Load alignment offset from map_offset.json")
    parser.add_argument("--load-zones", default=None,
                        help="(legacy) Load existing location_zones.json")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: next to input file)")
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--excluded-labels", nargs="*", default=[])
    parser.add_argument("--translate-x", type=float, default=0.0)
    parser.add_argument("--translate-y", type=float, default=0.0)
    parser.add_argument("--rotate-deg", type=float, default=0.0)
    args = parser.parse_args()

    if not args.scene_graph and not args.load_combined:
        print("Error: either --scene-graph or --load-combined is required.")
        sys.exit(1)

    # --- Load data ---
    grid_img, resolution, origin = load_occupancy_grid(args.map_pgm,
                                                       args.map_yaml)
    img_h = grid_img.shape[0]

    zones = []
    tx, ty, rot = args.translate_x, args.translate_y, args.rotate_deg

    if args.load_combined:
        # Load everything from scene_with_zones.json
        with open(args.load_combined) as f:
            combined_data = json.load(f)
        meta = combined_data.get("metadata", {})
        alignment = meta.get("alignment", {})
        tx = alignment.get("translate_x", tx)
        ty = alignment.get("translate_y", ty)
        rot = alignment.get("rotate_deg", rot)
        args.world_axes = meta.get("world_axes", args.world_axes)

        all_objects = combined_data["objects"]
        objects = filter_objects(all_objects, args.min_confidence,
                                args.min_points, args.excluded_labels)

        # Load zones from combined file
        for z in combined_data.get("zones", []):
            zones.append({
                "name": z["name"],
                "vertices_world": [list(v) for v in z["vertices_world"]],
                "object_ids": z.get("object_ids", []),
                "objects": z.get("objects", []),
                "parent": z.get("parent", None),
                "depth": z.get("depth", 0),
            })

        input_path = args.load_combined
        print(f"  Loaded combined: {len(objects)} objects, {len(zones)} zones")
    else:
        # Legacy: load from separate files
        sg_data = load_scene_graph(args.scene_graph)
        all_objects = sg_data["objects"]
        objects = filter_objects(all_objects, args.min_confidence,
                                args.min_points, args.excluded_labels)

        if args.load_offset:
            with open(args.load_offset) as f:
                off = json.load(f)
            tx = off.get("translate_x", tx)
            ty = off.get("translate_y", ty)
            rot = off.get("rotate_deg", rot)

        if args.load_zones:
            zones = load_zones(args.load_zones)
            print(f"  Loaded {len(zones)} zones from {args.load_zones}")

        input_path = args.scene_graph

    coords_2d = project_centroids_2d(objects, args.world_axes)
    active = [True] * len(objects)
    individual_offsets = np.zeros((len(objects), 2))
    colors = [generate_color(obj["label"]) for obj in objects]

    # Output path — always save as scene_with_zones.json
    out_dir = Path(args.output_dir) if args.output_dir else Path(input_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = str(out_dir / "scene_with_zones.json")

    # --- State ---
    state = {
        # Mode: "align" (move/delete objects), "zone_draw", "zone_select", "add_object"
        "mode": "align",
        "tx": tx, "ty": ty, "rot": rot,
        # Object alignment
        "selected": None,
        "dragging": False,
        "drag_start_px": None,
        "drag_start_offset": None,
        "obj_undo_stack": [],
        # Zone drawing
        "draw_verts_px": [],
        "draw_verts_world": [],
        "selected_zone": None,
        "dragging_vertex": None,
        "zone_undo_stack": [],
        # Display
        "show_objects": True,
        "show_zone_labels": True,
        # Panning
        "panning": False,
        "pan_start_px": None,
        "pan_start_xlim": None,
        "pan_start_ylim": None,
    }

    # --- Setup figure ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(grid_img, cmap="gray", origin="upper")
    ax.set_aspect("equal")

    # Object scatter
    scatter = ax.scatter([], [], s=120, edgecolors="black", linewidths=1,
                         zorder=10)

    # Object labels
    obj_annotations = []
    for i, obj in enumerate(objects):
        ann = ax.annotate(obj["label"], (0, 0),
                          fontsize=7, color="yellow", fontweight="bold",
                          xytext=(5, 5), textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.2",
                                    facecolor="black", alpha=0.75),
                          zorder=11, visible=False)
        obj_annotations.append(ann)

    # Selection ring
    sel_ring, = ax.plot([], [], 'o', markersize=22, markeredgewidth=3,
                        markeredgecolor='yellow', markerfacecolor='none',
                        zorder=12)

    # Zone draw preview
    draw_line, = ax.plot([], [], '--', color='cyan', linewidth=2, zorder=20)
    draw_markers, = ax.plot([], [], 'o', color='cyan', markersize=7,
                            markeredgecolor='black', markeredgewidth=1.5,
                            zorder=21)
    rubber_band, = ax.plot([], [], ':', color='yellow', linewidth=1.5,
                           zorder=19)

    # Status bar
    status_text = ax.text(
        0.5, 0.01, "", transform=ax.transAxes, fontsize=9,
        color="white", fontweight="bold", ha="center", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.85),
        zorder=30)

    # Info text
    info_text = ax.text(
        0.02, 0.97, "", transform=ax.transAxes, fontsize=9,
        color="cyan", fontweight="bold", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.85),
        zorder=30)

    # Zone rendering artists
    zone_patches = []
    zone_edge_lines = []
    zone_labels = []
    zone_vert_markers = []

    # --- Helper functions ---

    def get_active_data():
        active_idx = [i for i in range(len(objects)) if active[i]]
        if not active_idx:
            return np.zeros((0, 2)), np.zeros((0, 2)), [], active_idx
        base = coords_2d[active_idx] + individual_offsets[active_idx]
        transformed = apply_offset(base, state["tx"], state["ty"], state["rot"])
        pixels = world_to_pixel(transformed, resolution, origin, img_h)
        c = [colors[i] for i in active_idx]
        return transformed, pixels, c, active_idx

    def redraw_objects():
        world, pixels, c, act_idx = get_active_data()
        if len(pixels) > 0:
            scatter.set_offsets(pixels)
            scatter.set_facecolors(c)
            scatter.set_sizes([120] * len(pixels))
        else:
            scatter.set_offsets(np.zeros((0, 2)))
        scatter.set_visible(state["show_objects"])

        # Update annotations
        for ann in obj_annotations:
            ann.set_visible(False)
        if state["show_objects"]:
            for j, i in enumerate(act_idx):
                if j < len(pixels):
                    obj_annotations[i].xy = (pixels[j, 0], pixels[j, 1])
                    obj_annotations[i].set_visible(True)

        # Selection ring
        sel = state["selected"]
        if sel is not None and active[sel] and state["mode"] == "align":
            sel_pos = [j for j, i in enumerate(act_idx) if i == sel]
            if sel_pos:
                sp = pixels[sel_pos[0]]
                sel_ring.set_data([sp[0]], [sp[1]])
            else:
                sel_ring.set_data([], [])
        else:
            sel_ring.set_data([], [])

    def redraw_zones():
        for p in zone_patches:
            p.remove()
        zone_patches.clear()
        for l in zone_edge_lines:
            l.remove()
        zone_edge_lines.clear()
        for t in zone_labels:
            t.remove()
        zone_labels.clear()
        for m in zone_vert_markers:
            m.remove()
        zone_vert_markers.clear()

        # Recompute hierarchy so visuals stay in sync
        compute_zone_hierarchy(zones)

        for i, zone in enumerate(zones):
            verts_w = zone["vertices_world"]
            if len(verts_w) < 3:
                continue
            verts_px = world_to_pixel(np.array(verts_w), resolution, origin, img_h)
            color = generate_zone_color(i)
            is_selected = (state["selected_zone"] == i)
            depth = zone.get("depth", 0)

            # Visual style varies by nesting depth
            if depth == 0:
                # Top-level: solid border, normal fill
                lw = 3.5 if is_selected else 2.0
                alpha_fill = 0.35 if is_selected else 0.2
                linestyle = '-'
                base_zorder = 4
            else:
                # Nested: dashed border, lighter fill, higher zorder
                lw = 3.0 if is_selected else 2.0
                alpha_fill = 0.30 if is_selected else 0.15
                linestyle = '--' if depth == 1 else ':'
                base_zorder = 4 + depth * 2  # nested zones render on top

            patch = mpatches.Polygon(verts_px, closed=True,
                                     facecolor=color, edgecolor=color,
                                     alpha=alpha_fill, linewidth=0,
                                     zorder=base_zorder)
            ax.add_patch(patch)
            zone_patches.append(patch)

            closed_px = np.vstack([verts_px, verts_px[0:1]])
            line, = ax.plot(closed_px[:, 0], closed_px[:, 1],
                            color=color, linewidth=lw,
                            linestyle=linestyle,
                            zorder=base_zorder + 1,
                            solid_capstyle='round')
            zone_edge_lines.append(line)

            if state["show_zone_labels"]:
                cpx = verts_px.mean(axis=0)
                display_name = get_zone_display_name(zone, zones)
                fontsize = max(7, 10 - depth)  # slightly smaller for nested
                txt = ax.text(cpx[0], cpx[1], display_name,
                              fontsize=fontsize, fontweight="bold",
                              color="white", ha="center", va="center",
                              bbox=dict(boxstyle="round,pad=0.3",
                                        facecolor=color, alpha=0.8),
                              zorder=15 + depth)
                zone_labels.append(txt)

            if is_selected:
                vm, = ax.plot(verts_px[:, 0], verts_px[:, 1], 's',
                              color=color, markersize=8,
                              markeredgecolor='white', markeredgewidth=2,
                              zorder=16 + depth)
                zone_vert_markers.append(vm)

    def update_draw_preview():
        verts = state["draw_verts_px"]
        if not verts:
            draw_line.set_data([], [])
            draw_markers.set_data([], [])
            return
        arr = np.array(verts)
        draw_line.set_data(arr[:, 0], arr[:, 1])
        draw_markers.set_data(arr[:, 0], arr[:, 1])

    def update_rubber_band(mx, my):
        verts = state["draw_verts_px"]
        if not verts:
            rubber_band.set_data([], [])
            return
        last = verts[-1]
        first = verts[0]
        rubber_band.set_data([last[0], mx, first[0]],
                             [last[1], my, first[1]])

    def update_status():
        mode = state["mode"]
        n_active = sum(active)
        n_deleted = len(active) - n_active
        n_zones = len(zones)
        base = f"tx={state['tx']:.2f}  ty={state['ty']:.2f}  rot={state['rot']:.1f}°"
        base += f"  |  {n_active} objects"
        if n_deleted:
            base += f" ({n_deleted} del)"
        base += f"  |  {n_zones} zones"

        if mode == "align":
            base += "  |  ALIGN MODE"
        elif mode == "add_object":
            base += "  |  ADD OBJECT (click to place)"
        elif mode == "zone_draw":
            n_v = len(state["draw_verts_px"])
            base += f"  |  DRAWING ZONE ({n_v} verts)"
        elif mode == "zone_select":
            sel = state["selected_zone"]
            name = zones[sel]["name"] if sel is not None else "?"
            base += f"  |  ZONE: {name}"
        status_text.set_text(base)

    def update_info():
        mode = state["mode"]
        if mode == "align" and state["selected"] is not None and active[state["selected"]]:
            obj = objects[state["selected"]]
            off = individual_offsets[state["selected"]]
            info = f"Selected: {obj['label']}  (id={obj['id']}, pts={obj['num_points']})"
            if np.any(off != 0):
                info += f"\n  offset: dx={off[0]:.2f} dy={off[1]:.2f}"
            info_text.set_text(info)
        elif mode == "zone_select" and state["selected_zone"] is not None:
            zone = zones[state["selected_zone"]]
            n_obj = len(zone.get("object_ids", []))
            display_name = get_zone_display_name(zone, zones)
            depth = zone.get("depth", 0)
            info = f"Zone: {display_name}  ({len(zone['vertices_world'])} verts, {n_obj} objects)"
            if depth > 0:
                info += f"\n  depth={depth}, parent={zone.get('parent', '?')}"
            info_text.set_text(info)
        else:
            info_text.set_text("")

    def refresh():
        redraw_objects()
        redraw_zones()
        update_draw_preview()
        update_status()
        update_info()
        fig.canvas.draw_idle()

    def close_polygon():
        verts_px = state["draw_verts_px"]
        verts_world = state["draw_verts_world"]
        if len(verts_px) < 3:
            print("  Need at least 3 vertices")
            return

        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        name = simpledialog.askstring("Zone Name",
                                      "Enter name for this zone:",
                                      parent=root)
        root.destroy()

        if not name:
            name = f"Zone_{len(zones) + 1}"

        zones.append({
            "name": name,
            "vertices_world": verts_world,
            "object_ids": [],
            "objects": [],
            "parent": None,
            "depth": 0,
        })
        # Recompute hierarchy after adding
        compute_zone_hierarchy(zones)
        parent = zones[-1].get("parent")
        parent_info = f" (inside {parent})" if parent else ""
        print(f"  Created zone: {name} ({len(verts_world)} vertices){parent_info}")

        state["draw_verts_px"] = []
        state["draw_verts_world"] = []
        state["mode"] = "align"
        rubber_band.set_data([], [])
        refresh()

    def find_nearest_object(click_px, click_py):
        _, pixels, _, act_idx = get_active_data()
        if len(pixels) == 0:
            return None
        dists = np.sqrt((pixels[:, 0] - click_px)**2 +
                        (pixels[:, 1] - click_py)**2)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 20:
            return act_idx[min_idx]
        return None

    def find_zone_at(px_x, px_y):
        """Find the zone at the click point, preferring the deepest (most nested) one."""
        wx, wy = pixel_to_world(px_x, px_y, resolution, origin, img_h)
        best_idx = None
        best_depth = -1
        for i in range(len(zones)):
            verts = zones[i]["vertices_world"]
            if len(verts) < 3:
                continue
            if ShapelyPolygon(verts).contains(Point(wx, wy)):
                depth = zones[i].get("depth", 0)
                if depth > best_depth:
                    best_depth = depth
                    best_idx = i
        return best_idx

    def find_nearest_vertex(px_x, px_y, zone_idx, threshold=15):
        verts_px = world_to_pixel(np.array(zones[zone_idx]["vertices_world"]),
                                  resolution, origin, img_h)
        dists = np.sqrt((verts_px[:, 0] - px_x)**2 +
                        (verts_px[:, 1] - px_y)**2)
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold:
            return int(min_idx)
        return None

    def do_save():
        offset_data = {
            "translate_x": round(state["tx"], 4),
            "translate_y": round(state["ty"], 4),
            "rotate_deg": round(state["rot"], 2),
            "scale": 1.0,
        }

        # Build updated objects with individual offsets applied
        updated_objects = []
        for i, obj in enumerate(objects):
            if not active[i]:
                continue
            obj_copy = dict(obj)
            off = individual_offsets[i]
            if np.any(off != 0):
                c = dict(obj_copy["centroid"])
                if args.world_axes == "xz":
                    c["x"] += float(off[0])
                    c["z"] += float(off[1])
                else:
                    c["x"] += float(off[0])
                    c["y"] += float(off[1])
                obj_copy["centroid"] = c
            updated_objects.append(obj_copy)

        # Recompute hierarchy and object membership
        if zones:
            compute_zone_hierarchy(zones)
        world, _, _, act_idx = get_active_data()
        active_objects = [objects[i] for i in act_idx]
        if zones:
            compute_objects_in_zones(zones, active_objects, world)

        # Determine each object's deepest zone location
        obj_location = {}
        if zones:
            for i, obj in enumerate(active_objects):
                if i >= len(world):
                    continue
                pt = Point(world[i])
                best_zone = None
                best_depth = -1
                for z in zones:
                    verts = z["vertices_world"]
                    if len(verts) < 3:
                        continue
                    if ShapelyPolygon(verts).contains(pt):
                        d = z.get("depth", 0)
                        if d > best_depth:
                            best_depth = d
                            best_zone = z
                if best_zone is not None:
                    obj_location[obj["id"]] = {
                        "zone": best_zone["name"],
                        "parent_zone": best_zone.get("parent", None),
                        "full_path": get_zone_display_name(best_zone, zones),
                    }

        # Build objects with location
        combined_objects = []
        for obj in updated_objects:
            obj_out = dict(obj)
            loc = obj_location.get(obj["id"])
            if loc:
                obj_out["location"] = loc
            else:
                obj_out["location"] = {"zone": None, "parent_zone": None, "full_path": None}
            combined_objects.append(obj_out)

        # Build zones with children
        combined_zones = []
        if zones:
            children_map = {}
            for z in zones:
                children_map.setdefault(z["name"], [])
            for z in zones:
                parent = z.get("parent")
                if parent and parent in children_map:
                    children_map[parent].append(z["name"])

            for z in zones:
                combined_zones.append({
                    "name": z["name"],
                    "parent": z.get("parent", None),
                    "depth": z.get("depth", 0),
                    "children": children_map.get(z["name"], []),
                    "vertices_world": [[round(v[0], 4), round(v[1], 4)]
                                       for v in z["vertices_world"]],
                    "objects": [{"id": o["id"], "label": o["label"]}
                                for o in combined_objects
                                if o["location"]["zone"] == z["name"]
                                or o["id"] in z.get("object_ids", [])],
                })

        # Save single combined file
        combined_data = {
            "metadata": {
                "created": datetime.now().isoformat(),
                "world_axes": args.world_axes,
                "alignment": offset_data,
                "num_objects": len(combined_objects),
                "num_zones": len(combined_zones),
            },
            "objects": combined_objects,
            "zones": combined_zones,
        }
        with open(combined_path, "w") as f:
            json.dump(combined_data, f, indent=2)

        n_kept = sum(active)
        print(f"  Saved: {combined_path} ({n_kept} objects, {len(combined_zones)} zones)")

    # --- Event handlers ---

    def on_click(event):
        if event.inaxes != ax:
            return
        # Middle or right click → start panning
        if event.button in (2, 3):
            state["panning"] = True
            state["pan_start_px"] = (event.x, event.y)
            state["pan_start_xlim"] = ax.get_xlim()
            state["pan_start_ylim"] = ax.get_ylim()
            return
        if event.button != 1:
            return
        mode = state["mode"]

        if mode == "add_object":
            add_object_at(event.xdata, event.ydata)
            return

        if mode == "zone_draw":
            if event.dblclick:
                close_polygon()
                return
            wx, wy = pixel_to_world(event.xdata, event.ydata,
                                     resolution, origin, img_h)
            state["draw_verts_px"].append([event.xdata, event.ydata])
            state["draw_verts_world"].append([wx, wy])
            update_draw_preview()
            update_status()
            fig.canvas.draw_idle()

        elif mode == "align":
            # Try to select an object marker
            idx = find_nearest_object(event.xdata, event.ydata)
            if idx is not None:
                state["selected"] = idx
                state["dragging"] = True
                state["drag_start_px"] = (event.xdata, event.ydata)
                state["drag_start_offset"] = individual_offsets[idx].copy()
                refresh()
            else:
                # Try to select a zone
                zi = find_zone_at(event.xdata, event.ydata)
                if zi is not None:
                    state["mode"] = "zone_select"
                    state["selected_zone"] = zi
                    state["selected"] = None
                    refresh()

        elif mode == "zone_select":
            sel = state["selected_zone"]
            if sel is not None:
                vi = find_nearest_vertex(event.xdata, event.ydata, sel)
                if vi is not None:
                    state["dragging_vertex"] = (sel, vi)
                    state["drag_start_px"] = (event.xdata, event.ydata)
                    return

            # Try different zone or object
            zi = find_zone_at(event.xdata, event.ydata)
            if zi is not None and zi != state["selected_zone"]:
                state["selected_zone"] = zi
                refresh()
            else:
                # Try object
                idx = find_nearest_object(event.xdata, event.ydata)
                if idx is not None:
                    state["mode"] = "align"
                    state["selected_zone"] = None
                    state["selected"] = idx
                    state["dragging"] = True
                    state["drag_start_px"] = (event.xdata, event.ydata)
                    state["drag_start_offset"] = individual_offsets[idx].copy()
                    refresh()
                elif zi is None:
                    state["selected_zone"] = None
                    state["mode"] = "align"
                    refresh()

    def on_release(event):
        if state["panning"]:
            state["panning"] = False
            state["pan_start_px"] = None
            return
        state["dragging"] = False
        state["drag_start_px"] = None
        if state["dragging_vertex"] is not None:
            state["dragging_vertex"] = None
            refresh()

    def on_motion(event):
        # Panning — use screen pixels so it works even when cursor leaves axes
        if state["panning"] and state["pan_start_px"] is not None:
            dx_screen = event.x - state["pan_start_px"][0]
            dy_screen = event.y - state["pan_start_px"][1]
            # Convert screen pixels to data coordinates
            xlim = state["pan_start_xlim"]
            ylim = state["pan_start_ylim"]
            fig_w, fig_h = fig.get_size_inches() * fig.dpi
            bbox = ax.get_position()
            ax_w = bbox.width * fig_w
            ax_h = bbox.height * fig_h
            dx_data = -dx_screen * (xlim[1] - xlim[0]) / ax_w
            dy_data = dy_screen * (ylim[1] - ylim[0]) / ax_h
            ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
            ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
            fig.canvas.draw_idle()
            return

        if event.inaxes != ax:
            return

        # Rubber band in zone draw mode
        if state["mode"] == "zone_draw" and state["draw_verts_px"]:
            update_rubber_band(event.xdata, event.ydata)
            fig.canvas.draw_idle()

        # Object dragging
        if state["dragging"] and state["mode"] == "align":
            sel = state["selected"]
            if sel is None or not active[sel]:
                return
            dx_px = event.xdata - state["drag_start_px"][0]
            dy_px = event.ydata - state["drag_start_px"][1]
            dx_world = dx_px * resolution
            dy_world = -dy_px * resolution
            theta = np.radians(-state["rot"])
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dx_unrot = dx_world * cos_t - dy_world * sin_t
            dy_unrot = dx_world * sin_t + dy_world * cos_t
            individual_offsets[sel] = state["drag_start_offset"] + np.array([dx_unrot, dy_unrot])
            redraw_objects()
            update_status()
            update_info()
            fig.canvas.draw_idle()

        # Zone vertex dragging
        if state["dragging_vertex"] is not None:
            zi, vi = state["dragging_vertex"]
            wx, wy = pixel_to_world(event.xdata, event.ydata,
                                     resolution, origin, img_h)
            zones[zi]["vertices_world"][vi] = [wx, wy]
            redraw_zones()
            update_status()
            fig.canvas.draw_idle()

    def on_key(event):
        if event.key is None:
            return

        mode = state["mode"]
        step = 0.1
        rot_step = 5.0
        if "shift" in (event.key or ""):
            step = 0.02
            rot_step = 1.0
        if "ctrl" in (event.key or ""):
            step = 0.5
            rot_step = 15.0
        key = event.key.split("+")[-1]

        # --- Global keys ---
        if key == "n":
            state["mode"] = "zone_draw"
            state["draw_verts_px"] = []
            state["draw_verts_world"] = []
            state["selected"] = None
            state["selected_zone"] = None
            rubber_band.set_data([], [])
            print("  Zone draw mode: click to add vertices, Enter to close")
            refresh()
            return

        if key == "s":
            do_save()
            return

        if key in ("q",):
            do_save()
            print(f"\n  tx={state['tx']:.2f}  ty={state['ty']:.2f}  rot={state['rot']:.1f}°")
            print(f"  {sum(active)}/{len(active)} objects, {len(zones)} zones")
            plt.close(fig)
            return

        if key == "t":
            state["show_objects"] = not state["show_objects"]
            refresh()
            return

        if key == "l":
            state["show_zone_labels"] = not state["show_zone_labels"]
            refresh()
            return

        # --- Add object keys ---
        if mode == "add_object":
            if key == "escape":
                state["mode"] = "align"
                print("  Add object cancelled")
                refresh()
            return

        # --- Zone draw keys ---
        if mode == "zone_draw":
            if key in ("enter", "return"):
                close_polygon()
            elif key == "backspace":
                if state["draw_verts_px"]:
                    state["draw_verts_px"].pop()
                    state["draw_verts_world"].pop()
                    update_draw_preview()
                    rubber_band.set_data([], [])
                    update_status()
                    fig.canvas.draw_idle()
            elif key == "escape":
                state["mode"] = "align"
                state["draw_verts_px"] = []
                state["draw_verts_world"] = []
                rubber_band.set_data([], [])
                print("  Drawing cancelled")
                refresh()
            return

        # --- Zone select keys ---
        if mode == "zone_select":
            if key in ("d", "delete"):
                sel = state["selected_zone"]
                if sel is not None:
                    removed = zones.pop(sel)
                    state["zone_undo_stack"].append((sel, removed))
                    state["selected_zone"] = None
                    state["mode"] = "align"
                    print(f"  Deleted zone: {removed['name']} (Z to undo)")
                    refresh()
            elif key == "z":
                if state["zone_undo_stack"]:
                    idx, zone = state["zone_undo_stack"].pop()
                    zones.insert(min(idx, len(zones)), zone)
                    state["selected_zone"] = min(idx, len(zones) - 1)
                    print(f"  Restored zone: {zone['name']}")
                    refresh()
            elif key == "escape":
                state["selected_zone"] = None
                state["mode"] = "align"
                refresh()
            return

        # --- Align mode keys ---
        if mode == "align":
            if key == "right":
                state["tx"] += step
            elif key == "left":
                state["tx"] -= step
            elif key == "up":
                state["ty"] += step
            elif key == "down":
                state["ty"] -= step
            elif key == "r":
                state["rot"] += rot_step
            elif key == "e":
                state["rot"] -= rot_step
            elif key in ("d", "delete"):
                sel = state["selected"]
                if sel is not None and active[sel]:
                    active[sel] = False
                    state["obj_undo_stack"].append(sel)
                    print(f"  Deleted: {objects[sel]['label']} (id={objects[sel]['id']})")
                    state["selected"] = None
                else:
                    return
            elif key == "u":
                if state["obj_undo_stack"]:
                    idx = state["obj_undo_stack"].pop()
                    active[idx] = True
                    state["selected"] = idx
                    print(f"  Restored: {objects[idx]['label']} (id={objects[idx]['id']})")
                else:
                    return
            elif key == "a":
                state["mode"] = "add_object"
                state["selected"] = None
                print("  ADD OBJECT mode: click on the map to place a new object")
                refresh()
                return
            elif key == "z":
                # Also allow Z for zone undo in align mode
                if state["zone_undo_stack"]:
                    idx, zone = state["zone_undo_stack"].pop()
                    zones.insert(min(idx, len(zones)), zone)
                    print(f"  Restored zone: {zone['name']}")
            elif key == "escape":
                if state["selected"] is not None:
                    state["selected"] = None
                else:
                    print("  Press Q to save+quit")
            else:
                return
            refresh()

    def on_scroll(event):
        if event.inaxes != ax:
            return
        # Zoom in/out centered on cursor
        zoom_factor = 0.8 if event.button == 'up' else 1.25
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Get cursor position in data coords
        cx, cy = event.xdata, event.ydata
        # Compute new limits centered on cursor
        new_width = (xlim[1] - xlim[0]) * zoom_factor
        new_height = (ylim[1] - ylim[0]) * zoom_factor
        # Fraction of cursor position within current view
        rx = (cx - xlim[0]) / (xlim[1] - xlim[0])
        ry = (cy - ylim[0]) / (ylim[1] - ylim[0])
        ax.set_xlim(cx - rx * new_width, cx + (1 - rx) * new_width)
        ax.set_ylim(cy - ry * new_height, cy + (1 - ry) * new_height)
        fig.canvas.draw_idle()

    def add_object_at(px_x, px_y):
        """Add a new object marker at the given pixel position."""
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)
        label = simpledialog.askstring("Add Object",
                                       "Enter label for this object:",
                                       parent=root)
        root.destroy()
        if not label:
            print("  Add object cancelled")
            state["mode"] = "align"
            refresh()
            return

        # Convert pixel to world coords, then undo the global offset to get
        # the raw centroid that will be stored in the scene graph.
        wx, wy = pixel_to_world(px_x, px_y, resolution, origin, img_h)
        # Undo global translation
        ux = wx - state["tx"]
        uy = wy - state["ty"]
        # Undo global rotation
        theta = np.radians(-state["rot"])
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        raw_x = ux * cos_t - uy * sin_t
        raw_y = ux * sin_t + uy * cos_t

        # Build a new object entry
        max_id = max((o["id"] for o in objects), default=0) + 1
        if args.world_axes == "xz":
            centroid = {"x": float(raw_x), "y": 0.0, "z": float(raw_y)}
        else:
            centroid = {"x": float(raw_x), "y": float(raw_y), "z": 0.0}

        new_obj = {
            "id": max_id,
            "label": label,
            "centroid": centroid,
            "confidence": 1.0,
            "num_points": 100,
            "manually_added": True,
        }

        # Append to all tracking arrays
        objects.append(new_obj)
        active.append(True)
        new_coord = np.array([[raw_x, raw_y]])
        nonlocal coords_2d, individual_offsets
        coords_2d = np.vstack([coords_2d, new_coord]) if len(coords_2d) > 0 else new_coord
        individual_offsets = np.vstack([individual_offsets, [[0.0, 0.0]]]) if len(individual_offsets) > 0 else np.array([[0.0, 0.0]])
        colors.append(generate_color(label))

        # Add annotation for new object
        ann = ax.annotate(label, (0, 0),
                          fontsize=7, color="yellow", fontweight="bold",
                          xytext=(5, 5), textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.2",
                                    facecolor="black", alpha=0.75),
                          zorder=11, visible=False)
        obj_annotations.append(ann)

        print(f"  Added object: {label} (id={max_id})")
        state["mode"] = "align"
        state["selected"] = len(objects) - 1
        refresh()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    refresh()

    print("\n" + "=" * 60)
    print("Scene Graph Editor")
    print("=" * 60)
    print(f"  {len(objects)} objects, {len(zones)} zones")
    print(f"  tx={state['tx']:.2f}  ty={state['ty']:.2f}  rot={state['rot']:.1f}°")
    print()
    print("ALIGN mode (default):")
    print("  Arrows       Move all (Shift=fine, Ctrl=coarse)")
    print("  R / E        Rotate CW / CCW")
    print("  Click        Select marker → Drag to move")
    print("  D            Delete selected marker")
    print("  U            Undo object delete")
    print("  A            Add new object (click to place)")
    print()
    print("ZONE mode (press N):")
    print("  N            Start drawing new zone")
    print("  Click        Add vertex")
    print("  Enter/Dbl    Close polygon")
    print("  Backspace    Undo last vertex")
    print("  Click zone   Select → drag vertices / D to delete")
    print("  Z            Undo zone delete")
    print()
    print("General:  Scroll=zoom  Middle/Right-drag=pan  S=save  Q=save+quit  T=toggle objects  L=toggle labels")
    print()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
