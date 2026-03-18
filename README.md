# Guide-LLM

LLM-driven navigation agent for vision-impaired users using semantic scene graph + Claude tool_use + Nav2.

> Based on [Guide-LLM](https://arxiv.org/abs/2410.20666) by Sangmim Song et al.

## Architecture

```
              /transcribed_utterances_guide_llm (String)
                          │
 (external voice I/O)     │
                          ▼
User (voice/terminal) ──▶ scene_graph_agent.py ──▶ Claude API (tool_use)
                                  │  ▲                      │
                                  │  │                      ▼
                           ┌──────┴──┴──────┐        14 tool calls
                           │ scene_graph    │       (navigate, query,
                           │ _publisher.py  │        list, spatial,
                           │                │        describe, orient,
                           │                │        cancel, status)
                           │ /objects       │              │
                           │ /zones         │              ▼
                           └────────────────┘        tool results
                                  │                  back to Claude
                           ┌──────┴─────┐                 │
                           │ LiDAR SLAM │                 ▼
                           │ TF: map →  │         natural language
                           │ base_link  │           response
                           └────────────┘                 │
                                  │                       ▼
                           ┌──────┴─────┐      /speech_feedback (String)
                           │   Nav2     │◀── /goal_pose (PoseStamped)
                           │ (planner)  │──▶ /navigate_to_pose status
                           └────────────┘◀── /cmd_vel (Twist, zero-vel stop)
```

## Components

| File | Purpose |
|------|---------|
| `scene_graph_agent.py` | Claude tool_use agent — 14 tools, TF pose lookup, Nav2 goals + status, voice + terminal I/O |
| `scene_graph_publisher.py` | Publishes objects + zones from pre-built scene graph as RViz markers + JSON topics |

## Claude Tools (14)

### Localization

| Tool | Input | Output | Description |
|------|-------|--------|-------------|
| `get_robot_pose` | — | x, y, yaw_deg, current_zone | TF lookup (`map` → `base_link`). Returns position, orientation, and which zone the robot is in. |
| `describe_surroundings` | radius (default: 3.0m) | current_zone, nearby objects with distance + relative direction | Finds all objects within radius. Each object includes distance in meters and relative direction from robot's facing (e.g. "to your left", "ahead", "behind to your right"). Uses 8-direction compass relative to robot yaw. |

### Spatial Queries

| Tool | Input | Output | Description |
|------|-------|--------|-------------|
| `find_nearest` | object_type | label, distance_m, direction, zone | Substring-matches all objects (e.g. "chair" matches "chair_cluster_0004"), returns the closest one with distance, relative direction, and containing zone. |
| `distance_to` | destination | destination, destination_type, distance_m, direction | Straight-line distance and relative direction to a zone (centroid) or object. No navigation — just spatial info. |
| `describe_route` | destination | destination, total_distance_m, zones_along_path, objects_along_path | Previews what zones are crossed and what objects lie within a 2m corridor along the straight-line path to a destination. |
| `orient_me` | target | target, direction_from_current, turn_degrees, message | Rotates the robot in place to face a target zone or object. Publishes a PoseStamped at current position with computed yaw via Nav2. |

### Navigation

| Tool | Input | Output | Description |
|------|-------|--------|-------------|
| `navigate_to_zone` | zone_name | success, goal position | Computes zone centroid from polygon vertices, publishes PoseStamped to `/goal_pose`. |
| `navigate_to_object` | object_label | success, goal position | Looks up object map position, publishes PoseStamped to `/goal_pose`. |
| `cancel_navigation` | — | success | Cancels active navigation and journey. Publishes zero-velocity Twist to `/cmd_vel` for immediate stop, then calls Nav2's cancel service. |
| `navigate_waypoint_journey` | destination (string or array) | waypoints, stops, zone_path | Plans and executes a multi-zone waypoint journey. Accepts a single destination or a list for multi-stop trips (e.g. `["kitchen", "elevator", "lobby"]`). Auto-advances through waypoints, pauses 2s at intermediate stops. |
| `get_journey_status` | — | status, progress, stops, narration | Returns current journey progress including waypoint index, stop info (e.g. "stop 2 of 3: kitchen"), and narration text. |
| `get_navigation_status` | — | status, destination, message | Checks Nav2 action status. Returns: `navigating`, `arrived`, `failed`, `canceled`, or `idle`. |

### Scene Graph Query

| Tool | Input | Output | Description |
|------|-------|--------|-------------|
| `list_zones` | — | zone names | Returns only zone name strings. Lean — no polygon data. |
| `list_objects` | — | object labels | Returns only unique object label strings. Lean — no position data. |
| `query_zone` | zone_name | polygon, objects in zone | Round-trip query to scene_graph_publisher. Returns zone boundary and contained objects. |
| `query_object` | object_label | position, confidence | Round-trip query to scene_graph_publisher. Returns object details. |

### Tool Call Flow

```
"where are the chairs?"
  → describe_surroundings(radius=5.0)
  → returns nearby objects with relative directions
  → Claude: "There's a chair cluster 1.5 meters to your left,
             and another 2.8 meters behind you to the right."

"take me to the kitchen"
  → list_zones()                                # find exact name
  → navigate_waypoint_journey("kitchen_area")   # multi-zone waypoint journey
  → Claude: "Route planned through the hallway — 5 waypoints. Let's go!"

"take me to the kitchen, then the elevator"
  → navigate_waypoint_journey(["kitchen_area", "elevator_area"])
  → Claude: "Multi-stop route planned — kitchen then elevator, 8 waypoints."

"are we there yet?"
  → get_journey_status()
  → Claude: "We're 60% there — entering the kitchen now (stop 1 of 2)."

"stop"
  → cancel_navigation()              # publishes zero-velocity Twist + Nav2 cancel
  → Claude: "Done, I've stopped the robot."

"where's the nearest chair?"
  → find_nearest("chair")
  → returns closest matching object with distance + direction
  → Claude: "The nearest chair is chair_cluster_0004, about 1.5 meters
             to your left in the HRI Lab."

"how far is the kitchen?"
  → distance_to("kitchen_area")
  → Claude: "The kitchen is about 8.2 meters ahead to your right."

"what's on the way to the kitchen?"
  → describe_route("kitchen_area")
  → returns zones crossed + objects along path
  → Claude: "On the way to the kitchen you'll pass through the hallway.
             There's a door 3 meters along the path and a table near
             the halfway point."

"face towards the kitchen"
  → orient_me("kitchen_area")
  → robot rotates in place
  → Claude: "Turning 45 degrees to the right to face the kitchen."
```

## Quick Start

All commands assume you're in the `guide-llm/` directory.

### 1. Map server

```bash
ros2 run nav2_map_server map_server --ros-args \
    -p yaml_filename:=$(pwd)/data/maps/level_12_v5.yaml
ros2 lifecycle set /map_server configure
ros2 lifecycle set /map_server activate
```

### 2. Scene graph publisher

```bash
python3 scene_graph_publisher.py \
    --scene-graph data/scene_graph/scene_graph.json \
    --load-offset data/scene_graph/map_offset.json \
    --load-zones data/scene_graph/location_zones.json \
    --frame-id map --world-axes xz
```

### 3. Agent

```bash
export ANTHROPIC_API_KEY=sk-...
python3 scene_graph_agent.py --model claude-sonnet-4-6
```

### 4. RViz2 (optional)

```bash
rviz2  # Add: /map, /scene_graph/markers, /scene_graph/zone_markers, /goal_pose
```

### 5. Fake robot pose for testing (no SLAM)

```bash
ros2 run tf2_ros static_transform_publisher 1.0 2.0 0 0 0 0 map base_link
```

## CLI Options

### scene_graph_agent.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `claude-sonnet-4-6` | Claude model ID |
| `--api-key` | `$ANTHROPIC_API_KEY` | Anthropic API key |
| `--max-tokens` | `1024` | Max tokens per response |
| `--frame-id` | `map` | TF frame for goals |
| `--base-frame` | `base_link` | Robot base TF frame |

### scene_graph_publisher.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--scene-graph` | (required) | Path to scene_graph.json |
| `--load-offset` | — | map_offset.json for alignment |
| `--load-zones` | — | location_zones.json for zone definitions |
| `--frame-id` | `map` | TF frame |
| `--world-axes` | `xy` | 3D→2D projection (`xz` for Y-up, `xy` for Z-up) |
| `--align-from-pose` | — | Live alignment via RViz2 "2D Pose Estimate" |

## ROS2 Topics

| Topic | Type | Dir | Purpose |
|-------|------|-----|---------|
| `/scene_graph/objects` | String | Sub | Object data (latched, received once) |
| `/scene_graph/zones` | String | Sub | Zone data (latched, received once) |
| `/scene_graph/query` | String | Pub | Query object by label |
| `/scene_graph/query_result` | String | Sub | Object query response |
| `/scene_graph/zone_query` | String | Pub | Query zone by name |
| `/scene_graph/zone_result` | String | Sub | Zone query response |
| `/goal_pose` | PoseStamped | Pub | Nav2 navigation goal |
| `/cmd_vel` | Twist | Pub | Zero-velocity stop on cancel |
| `/navigate_to_pose/_action/status` | GoalStatusArray | Sub | Nav2 navigation feedback |
| `/navigate_to_pose/_action/cancel_goal` | CancelGoal | Srv | Cancel active navigation |
| `/transcribed_utterances_guide_llm` | String | Sub | Voice input (subscribe to external STT node) |
| `/speech_feedback` | String | Pub | Agent response text (consumed by external TTS node) |
| TF: `map` → `base_link` | — | Lookup | Robot pose from SLAM |

## Example Session

```
You: hi, where am I?
  [get_robot_pose] {"x": 1.23, "y": -0.45, "yaw_deg": 90.0, "current_zone": "hri_lab"}
[Assistant] You're in the HRI Lab, facing north.

You: what's around me?
  [describe_surroundings] {"current_zone": "hri_lab", "nearby_objects": [
    {"label": "chair_cluster_0004", "distance": 1.5, "direction": "to your left"},
    {"label": "chair_cluster_0014", "distance": 2.8, "direction": "behind to your right"}
  ], "count": 2}
[Assistant] You're in the HRI Lab. There's a chair cluster about 1.5 meters to
your left, and another about 2.8 meters behind you to the right.

You: take me to the kitchen
  [navigate_waypoint_journey("kitchen_area")] {"success": true, "zone_path": ["hri_lab", "hallway_0", "kitchen_area"],
    "total_waypoints": 5, "stops": ["kitchen_area"], "total_stops": 1}
[Assistant] Route planned through the hallway — 5 waypoints. Let's go!

You: are we there yet?
  [get_journey_status] {"status": "executing", "current_waypoint": 3, "total_waypoints": 5,
    "progress_percent": 60.0, "narration": "Waypoint 4 of 5: enter kitchen_area. 60.0% complete."}
[Assistant] We're about 60% there — just entering the kitchen area now.

You: stop
  [cancel_navigation] {"success": true, "message": "Navigation cancelled"}
[Assistant] Done, I've stopped the robot.

You: take me to the kitchen, then the elevator, then the lobby
  [navigate_waypoint_journey(["kitchen_area", "elevator_area", "lobby"])]
    {"success": true, "total_waypoints": 12, "stops": ["kitchen_area", "elevator_area", "lobby"],
     "total_stops": 3}
[Assistant] Multi-stop route planned — kitchen, elevator, then lobby (12 waypoints, 3 stops).

You: how are we doing?
  [get_journey_status] {"status": "executing", "current_waypoint": 4, "progress_percent": 33.3,
    "current_stop": 1, "total_stops": 3,
    "narration": "Waypoint 5 of 12: exit hallway_0. (stop 1 of 3: kitchen_area) 33.3% complete."}
[Assistant] We're heading to stop 1 of 3 (the kitchen) — about a third of the way through.

You: where's the nearest chair?
  [find_nearest] {"label": "chair_cluster_0004", "distance_m": 1.5, "direction": "to your left", "zone": "hri_lab"}
[Assistant] The nearest chair is about 1.5 meters to your left, in the HRI Lab.

You: how far is the kitchen?
  [distance_to] {"destination": "kitchen_area", "destination_type": "zone", "distance_m": 8.2, "direction": "ahead to your right"}
[Assistant] The kitchen is about 8.2 meters ahead to your right.

You: face towards the kitchen
  [orient_me] {"target": "kitchen_area", "direction_from_current": "ahead to your right", "turn_degrees": 42.3, "message": "Rotating to face kitchen_area (42.3 degrees right)"}
[Assistant] Turning about 42 degrees to the right to face the kitchen.
```

## Context Management

The agent is designed to keep Claude's context window lean:

- **System prompt**: ~15 lines — counts, tool descriptions, instructions. No object/zone data.
- **list tools**: Return only name strings, not full data.
- **query tools**: Fetch details for one specific item on demand.
- **describe_surroundings**: Returns only nearby objects within radius, not the full scene graph.
- **Subscriptions**: Objects/zones topics received once then unsubscribed.

## Project Structure

```
guide-llm/
├── scene_graph_agent.py          # LLM agent (14 Claude tools + Nav2 + voice I/O)
├── scene_graph_publisher.py      # Scene graph → ROS2 topics + RViz markers
├── data/
│   ├── maps/                     # Occupancy grid maps for Nav2
│   │   ├── level_12_v5.yaml      #   Building floor plan
│   │   ├── level_12_v5.pgm
│   │   ├── song_map_lidar_.yaml  #   LiDAR-generated map
│   │   └── song_map_lidar_.pgm
│   └── scene_graph/              # Scene graph + zone definitions
│       ├── scene_graph.json      #   Objects with 3D centroids
│       ├── location_zones.json   #   Named zone polygons
│       └── map_offset.json       #   SLAM→map alignment offset
├── setup.py
├── setup.cfg
├── package.xml
├── resource/guide_llm
└── README.md
```

## Citation

```bibtex
@article{song2024guidellm,
  author  = {Song, Sangmim and Kodagoda, Sarath and Gunatilake, Amal and Carmichael, Marc G. and Thiyagarajan, Karthick and Martin, Jodi},
  title   = {Guide-LLM: An Embodied LLM Agent and Text-Based Topological Map for Robotic Guidance of People with Visual Impairments},
  journal = {arXiv preprint arXiv:2410.20666},
  year    = {2024},
  doi     = {10.48550/arXiv.2410.20666}
}
```

## License

MIT — Sangmim Song (Sangmim.Song@student.uts.edu.au)
