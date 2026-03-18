#!/usr/bin/env python3
"""
Simple matplotlib visualizer for the dummy scene graph.
Shows zones, objects, and robot pose — no RViz or ROS2 needed.

Usage:
    python3 visualize_scene.py              # show dummy data
    python3 visualize_scene.py --save       # save to scene_graph.png
"""

import argparse
import math

import matplotlib
matplotlib.use("Agg")  # headless-safe, always saves to file
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import numpy as np


# ---------------------------------------------------------------------------
# Same dummy data as dummy_scene_graph_publisher.py
# ---------------------------------------------------------------------------

ZONES = [
    {
        "name": "kitchen",
        "display_name": "Kitchen",
        "vertices_world": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "color": "#4CAF50",
    },
    {
        "name": "living_room",
        "display_name": "Living Room",
        "vertices_world": [[4, 0], [10, 0], [10, 5], [4, 5]],
        "color": "#2196F3",
    },
    {
        "name": "hallway",
        "display_name": "Hallway",
        "vertices_world": [[0, 3], [4, 3], [4, 5], [0, 5]],
        "color": "#FF9800",
    },
]

OBJECTS = [
    {"label": "fridge", "x": 1.0, "y": 1.0, "icon": "🧊"},
    {"label": "microwave", "x": 3.0, "y": 1.5, "icon": "📦"},
    {"label": "chair", "x": 5.5, "y": 2.0, "icon": "🪑"},
    {"label": "table", "x": 6.5, "y": 2.5, "icon": "🔲"},
    {"label": "couch", "x": 7.0, "y": 4.0, "icon": "🛋"},
    {"label": "lamp", "x": 8.5, "y": 1.0, "icon": "💡"},
]

ROBOT_X = 2.0
ROBOT_Y = 1.5
ROBOT_YAW_DEG = 0.0


def draw_scene(save_path="scene_graph.png"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # Draw zones
    for zone in ZONES:
        verts = zone["vertices_world"]
        polygon = plt.Polygon(
            verts, closed=True,
            facecolor=zone["color"], alpha=0.15,
            edgecolor=zone["color"], linewidth=2,
        )
        ax.add_patch(polygon)

        # Zone label at centroid
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        ax.text(cx, cy, zone["display_name"],
                ha="center", va="center",
                fontsize=14, fontweight="bold",
                color=zone["color"], alpha=0.6)

    # Draw objects
    for obj in OBJECTS:
        ax.plot(obj["x"], obj["y"], "o", markersize=12,
                color="#E91E63", markeredgecolor="white", markeredgewidth=1.5,
                zorder=5)
        ax.annotate(
            obj["label"],
            (obj["x"], obj["y"]),
            textcoords="offset points",
            xytext=(0, 14),
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#E91E63", alpha=0.85),
        )

    # Draw robot
    ax.plot(ROBOT_X, ROBOT_Y, "s", markersize=16,
            color="#9C27B0", markeredgecolor="white", markeredgewidth=2,
            zorder=10)
    # Heading arrow
    yaw_rad = math.radians(ROBOT_YAW_DEG)
    arrow_len = 0.6
    dx = arrow_len * math.cos(yaw_rad)
    dy = arrow_len * math.sin(yaw_rad)
    ax.annotate("",
                xy=(ROBOT_X + dx, ROBOT_Y + dy),
                xytext=(ROBOT_X, ROBOT_Y),
                arrowprops=dict(arrowstyle="->", color="#9C27B0", lw=2.5),
                zorder=10)
    ax.annotate(
        "ROBOT",
        (ROBOT_X, ROBOT_Y),
        textcoords="offset points",
        xytext=(0, -18),
        ha="center", va="top",
        fontsize=9, fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#9C27B0",
                  edgecolor="white", alpha=0.9),
    )

    # Styling
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 6)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlabel("X (metres)", fontsize=11)
    ax.set_ylabel("Y (metres)", fontsize=11)
    ax.set_title("Scene Graph — Dummy Test Environment", fontsize=14, fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E91E63",
               markersize=10, label="Object"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#9C27B0",
               markersize=10, label="Robot"),
        patches.Patch(facecolor="#4CAF50", alpha=0.3, label="Kitchen"),
        patches.Patch(facecolor="#2196F3", alpha=0.3, label="Living Room"),
        patches.Patch(facecolor="#FF9800", alpha=0.3, label="Hallway"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize dummy scene graph")
    parser.add_argument("--save", default="scene_graph.png",
                        help="Output image path (default: scene_graph.png)")
    args = parser.parse_args()
    draw_scene(args.save)


if __name__ == "__main__":
    main()
