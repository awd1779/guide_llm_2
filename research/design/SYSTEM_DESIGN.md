# System Design: Architecture, Perception, and Vision Integration

**Status**: Complete design, ready for Phase 1 implementation
**Last updated**: 2026-03-23

---

## Quick Navigation

This document consolidates:
- System architecture and data flow
- Perception hierarchy (when to use vision vs. scene graph)
- Vision integration design (YOLO on-demand)
- Tool definitions and their triggers

---

## 1. System Architecture Overview

### Block Diagram

```
SENSORS (Always Running)
├─ LiDAR → SLAM/Nav2 localization
├─ RGB-D Camera → On-demand perception
└─ Scene Graph (static map)

PERCEPTION LAYER (Smart Triggering)
├─ Vision Processor: YOLO on-demand
├─ Scene Graph Fusion: Merge detections
├─ Perception Hierarchy: Decide when YOLO needed
└─ Tool Dispatcher: Route to correct tool

LLM AGENT (Qwen 3.5 4B, fine-tuned)
├─ 12 tools for navigation
├─ Tool-calling format (OpenAI API)
└─ Safety-first design (always confirm before nav)

MONITORING LAYER (Background)
├─ Zone Awareness Monitor: Entry/arrival announcements
├─ Obstacle Monitor: Detect blocking obstacles
├─ Hazard Detector: Wet floors, people, stairs
└─ Alert Handler: User responses

OUTPUT
├─ Speech: Agent responses, announcements, warnings
└─ Navigation: Goals to Nav2
```

### Data Flow

```
User Input (voice/text)
    ↓
LLM Agent receives message
    ↓
Is perception needed? (Perception Hierarchy decision)
    ├─ NO (Tier 0-2) → Use scene graph only (in-memory lookup)
    └─ YES (Tier 3-4) → Trigger YOLO perception (not yet implemented)
    ↓
Tool calling via OpenAI API format
    ↓
Tool execution (navigate_to, query, describe_surroundings, etc.)
    ↓
Tool result returned to LLM
    ↓
LLM generates natural language response
    ↓
Speech output + Optional monitoring (zones, obstacles, hazards)
```

---

## 2. Perception Hierarchy: When to Use Vision vs. Scene Graph

**Principle**: Use **static scene graph by default**. Only trigger YOLO for **current environment perception** queries.

### Decision Tree (Tier-Based)

```
User Query
    ↓
┌─────────────────────────────────────────┐
│ Tier 0-2: Static Knowledge              │
│ → Scene graph ONLY (no YOLO needed)     │
├─────────────────────────────────────────┤
│ • List zones: "What rooms exist?"       │
│ • Navigate: "Take me to kitchen"        │
│ • Route query: "What's the route?"      │
│ • Robot pose: "Where am I?"             │
│ • Distance: "How far to kitchen?"       │
│ • Navigation status: "Are we there?"    │
│                                         │
│ Speed: TBD (measure on target hardware) │
│ Cost: On-device (no API calls)          │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Tier 3-4: Current Scene Perception      │
│ → YOLO perception NEEDED                │
├─────────────────────────────────────────┤
│ • Describe surroundings: "What's here?" │
│ • Find object: "Where's the chair?"     │
│ • Verify exists: "Is there a table?"    │
│ • Scene understanding: "What do you see?"│
│                                         │
│ Speed: TBD (not yet implemented)       │
│ Cost: On-device (no API calls)          │
└─────────────────────────────────────────┘
```

### Tier Details

**Tier 0: Pure Scene Graph (Static Structure)**
- Questions about zones, topology, layout
- Tools: `list_all`, `navigate_to`, `distance_to`, `describe_route`
- Example: "How do I get to the kitchen?" → Use scene graph only

**Tier 1: Scene Graph + Robot State (No Vision)**
- Current navigation status, robot pose, what's in static map
- Tools: `get_robot_pose`, `get_navigation_status`
- Example: "Are we there yet?" → Check nav status, no YOLO needed

**Tier 2: Ambiguous Queries (No Immediate Perception)**
- Queries about objects that exist in map, but user asking "tell me about X"
- Tools: `query` (static object description)
- Example: "Tell me about chairs" → Describe from scene graph

**Tier 3: Current Scene Perception (YOLO Needed)**
- What's visible right now from camera
- Tools: `describe_surroundings`, `find_nearest`
- Example: "What's in front of me?" → Run YOLO, merge with scene graph

**Tier 4: Complex Perception (YOLO + Reasoning)**
- Multi-step perception with follow-ups
- Example sequence:
  1. "Find a chair" → YOLO detects chair at 1.5m left
  2. "Is it safe?" → Check for obstacles between robot and chair
  3. "Take me there" → Navigate to chair

---

## 3. Vision Integration Design

### On-Demand YOLO Architecture

**Principle**: Don't run YOLO continuously. Only invoke it when user asks perception questions.

### Flow Diagram

```
Tool Call: describe_surroundings(radius=3.0)
    ↓
Vision Processor receives request
    ↓
RGB-D Camera frame capture
    ↓
YOLO inference on frame (latency TBD — not yet implemented)
    ↓
Post-processing: Filter by confidence >0.7
    ↓
Scene Graph Fusion:
├─ Get objects in zone from scene graph
├─ Merge with YOLO detections
└─ Resolve conflicts (YOLO takes precedence if >0.9)
    ↓
Return structured result
{
  "perception_source": "yolo",
  "objects": [
    {"label": "chair", "distance_m": 2.1, "direction": "ahead"},
    {"label": "table", "distance_m": 3.0, "direction": "left"}
  ],
  "current_zone": "hri_lab"
}
    ↓
LLM responds naturally (no raw numbers)
"There's a chair right in front of you, about 2 meters away..."
```

### Confidence Filtering

**Key Principle**: LLM never sees uncertain detections.

- YOLO confidence threshold: **>0.7**
- Detections below 0.7 filtered at perception layer
- Result sent to LLM: Only high-confidence detections
- Effect: Model learns "if in perception result, it's real"
- No confidence scores passed to LLM training

### Scene Graph Fusion Algorithm

1. **Get robot's current zone** from localization
2. **Load scene graph objects in zone**
3. **Run YOLO on current frame**
4. **For each YOLO detection** (confidence > 0.7):
   - Check if object already in scene graph for this zone
   - If YOLO confidence > 0.9: Use YOLO (prefer real perception)
   - If YOLO confidence 0.7-0.9: Trust scene graph if exists, else use YOLO
   - Update distance based on YOLO depth estimation
5. **Merge results**: Scene graph objects + new YOLO detections
6. **Return to LLM** with `perception_source` signal

---

## 4. Available Tools (12 Total)

### Navigation Tools
- **navigate_to**(destination): Navigate to zone or object
- **get_navigation_status**(): Check if arrived, still moving, failed
- **cancel_navigation**(): Cancel current navigation goal
- **replan_route**(destination, avoid_obstacles): Replan to new destination or re-route around obstacles

### Perception Tools
- **describe_surroundings**(radius): Describe nearby objects (uses YOLO if Tier 3-4)
- **find_nearest**(object_type): Find closest object matching keyword (uses YOLO)

### Scene Graph Tools
- **list_all**(): List all zones and objects
- **query**(name): Get details about zone or object
- **distance_to**(destination): Get distance without navigating
- **describe_route**(destination): Preview path to destination

### Robot State Tools
- **get_robot_pose**(): Current position and zone
- **orient_me**(target): Rotate to face target

---

## 5. Key Design Decisions

### 1. Scene Graph as Primary
- **Why**: Fast (on-device), always available, reliable
- **How**: Start with static map, enrich with real detections only when needed
- **Result**: <100ms for structural queries, ~300ms for perception queries

### 2. On-Demand YOLO (Not Continuous)
- **Why**: Saves compute, battery, thermal load
- **How**: Only run YOLO when Perception Hierarchy says "Tier 3-4"
- **Result**: Model can run on Jetson Orin continuously without overheating

### 3. Confidence Filtering at Perception Layer
- **Why**: Keep LLM training simple; model never learns uncertainty
- **How**: Only pass high-confidence (>0.7) detections to LLM
- **Result**: Model learns "if in result, it's reliable"

### 4. Tool Calling with Fine-Tuned Small LLM
- **Why**: Edge-deployable, domain-specific accuracy, low latency
- **How**: Fine-tune Qwen 3.5 4B on navigation scenarios with tool examples
- **Result**: Better performance on specific tasks than generic large models

### 5. Structured Scene Graph Format
- **Why**: Interpretable, accessible, no complex visual understanding needed
- **How**: Text-based zones + objects + topology
- **Result**: Works with screen readers, easy to debug, VI-friendly

---

## 6. Deployment Architecture

### Hardware
- **Robot**: Go2 with Jetson Orin (target deployment platform — not yet benchmarked)
- **Sensors**: LiDAR + RGB-D camera + IMU
- **ROS2**: Nav2 stack for navigation

### Software Stack
```
├─ LLM Inference: vLLM or similar (4B model, VRAM TBD on Jetson)
├─ YOLO: YOLOv8 Nano (not yet implemented)
├─ Navigation: Nav2 + SLAM
├─ Scene Graph: In-memory graph (JSON)
└─ ROS2 Nodes:
   ├─ scene_graph_agent_local.py (LLM + tool dispatcher)
   ├─ vision_processor.py (YOLO on-demand)
   ├─ zone_awareness_monitor.py (announcements)
   ├─ obstacle_monitor.py (blocking detection)
   └─ hazard_detector.py (specific hazards)
```

### Latency Budget (Projected — Must Be Measured on Target Hardware)
- User input → LLM: TBD
- LLM decision + tool call: TBD
- Tool execution:
  - Scene graph query: Expected fast (in-memory JSON lookup)
  - YOLO perception: TBD (not yet implemented)
  - Nav2 planning: TBD
- Total: TBD — benchmark on Jetson Orin before publishing any numbers

---

## 7. What's Next

**Phase 1 (Weeks 1-2)**: Implement vision processor + scene graph fusion
**Phase 2 (Weeks 2-3)**: Generate dataset + fine-tune model
**Phase 3 (Weeks 3-4)**: Implement monitoring (zones, obstacles, hazards)
**Phase 4 (Weeks 4-8)**: Integration, testing, deployment

See `IMPLEMENTATION.md` for detailed roadmap.

---

**References**: References stored in `/home/ubuntu/guide-llm_2/references/references.bib`
