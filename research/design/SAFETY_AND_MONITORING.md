# Safety and Monitoring: Zones, Obstacles, and Hazards

**Status**: Design complete, ready for Phase 3 implementation
**Last updated**: 2026-03-23

---

## Overview

Three independent monitors run in background during robot operation:

1. **Zone Awareness Monitor** - Tell user when entering/leaving zones
2. **Obstacle Monitor** - Detect and alert about blocking obstacles mid-navigation
3. **Hazard Detector** - Identify specific safety hazards (wet floors, people, stairs)

---

## 1. Zone Awareness Monitor (Non-Intrusive Announcements)

### Design Principle

Only announce **zone entry** and **arrival** to avoid overwhelming user. Queue if agent is speaking.

### When to Announce

```
Robot crosses zone boundary
    ↓
Is agent currently speaking?
├─ YES → Queue announcement until agent finishes
└─ NO → Announce immediately
    ↓
Wait 5 seconds before next announcement (throttle)
```

### Announcements

**On Zone Entry** (crossing boundary):
```
"You're entering the kitchen"
"Entering hallway 1"
```

**On Arrival** (reached navigation goal):
```
"You've arrived at the kitchen"
"We're here at hri_lab"
```

**NOT announced**:
- Passing through intermediate zones (only start/end)
- Zone names repeatedly
- Technical details

### Implementation

```python
class ZoneAwarenessMonitor:
    def __init__(self):
        self.current_zone = None
        self.last_announcement_time = 0
        self.min_interval = 5  # seconds
        self.agent_speaking = False
        self.pending_announcements = []

    def on_robot_moved(self, x, y, zone_map):
        """Called when robot localization updates."""
        new_zone = self._get_zone_at_position(x, y, zone_map)

        if new_zone != self.current_zone:
            # Zone changed
            self._queue_announcement(f"You're in the {new_zone}")
            self.current_zone = new_zone

    def on_navigation_goal_reached(self, destination):
        """Called when Nav2 reports goal reached."""
        self._queue_announcement(f"We've arrived at the {destination}")

    def _queue_announcement(self, text):
        """Queue announcement to be played when agent finishes speaking."""
        current_time = time.time()
        if current_time - self.last_announcement_time > self.min_interval:
            if self.agent_speaking:
                self.pending_announcements.append(text)
            else:
                self._speak(text)
                self.last_announcement_time = current_time

    def on_agent_speaks(self):
        self.agent_speaking = True

    def on_agent_finishes(self):
        """Called when agent response finishes."""
        self.agent_speaking = False
        # Play any pending announcements
        for announcement in self.pending_announcements:
            self._speak(announcement)
        self.pending_announcements = []

    def _speak(self, text):
        """Play text via speaker."""
        # Integration with speech output system
        publish_to_speech_output(text)
```

### ROS2 Node

```python
# perception/zone_awareness_monitor.py

class ZoneMonitor(Node):
    def __init__(self):
        super().__init__("zone_awareness_monitor")
        self.monitor = ZoneAwarenessMonitor()

        # Subscribe to localization
        self.create_subscription(
            PoseStamped, "/tf_pose", self.on_pose_update, 10
        )

        # Subscribe to navigation goal status
        self.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status",
            self.on_goal_status, 10
        )

        # Subscribe to agent speaking state
        self.create_subscription(
            Bool, "/guide_llm/agent_speaking",
            self.on_agent_speaking, 10
        )
```

---

## 2. Obstacle Monitor (Dynamic Replanning)

### When Triggered

```
During active navigation (nav goal sent)
    ↓
Every 100ms, check LiDAR for obstacles
    ↓
Obstacle blocks planned path?
├─ NO → Continue normally
└─ YES → Alert user, offer choices
    ↓
User choice: Go around / Wait / Cancel
    ├─ Go around → Replan via alternative zones
    ├─ Wait → Pause navigation until obstacle clears
    └─ Cancel → Stop navigation
```

### Detection Logic

```python
def check_obstacle_blocking_path(lidar_scan, planned_path, robot_pose):
    """
    Check if obstacle blocks planned navigation path.

    Args:
        lidar_scan: Current 360° LiDAR scan
        planned_path: Nav2 planned path (list of waypoints)
        robot_pose: Current robot position

    Returns:
        (is_blocked, obstacle_distance, obstacle_angle)
    """
    # Check first 2 meters of planned path
    lookahead_distance = 2.0

    # Get waypoints ahead on path
    waypoints_ahead = [w for w in planned_path
                      if distance(w, robot_pose) < lookahead_distance]

    # Check LiDAR ranges toward those waypoints
    for waypoint in waypoints_ahead:
        angle_to_waypoint = atan2(waypoint.y - robot.y, waypoint.x - robot.x)
        # Check ±30° cone around direction
        ranges = lidar_scan.ranges[angle_idx-30:angle_idx+30]

        min_range = min(ranges)
        if min_range < 0.5:  # Object within 50cm
            return True, min_range, angle_to_waypoint

    return False, None, None
```

### User Interaction

**Agent detects obstacle**:
```
Agent (LLM-generated): "Someone is in the way. Should I go around, wait, or stop?"
```

**User responds**:
```
User: "Go around"
→ Agent calls replan_route(avoid_obstacles=True)
→ Nav2 finds alternative path
→ Resume navigation

User: "Wait"
→ Agent pauses navigation (cancel goal, keep destination)
→ Periodically checks if obstacle cleared
→ When cleared, resume

User: "Stop"
→ Agent calls cancel_navigation()
→ Returns control to user
```

### Implementation

```python
class ObstacleMonitor:
    def __init__(self, nav_client, llm_agent):
        self.nav_client = nav_client
        self.llm_agent = llm_agent
        self.current_goal = None
        self.is_navigating = False

    def on_navigation_start(self, goal):
        """Called when navigation goal is set."""
        self.current_goal = goal
        self.is_navigating = True

    def on_lidar_scan(self, scan):
        """Called every 100ms with new LiDAR scan."""
        if not self.is_navigating:
            return

        blocked, distance, angle = self.check_obstacle_blocking_path(
            scan, self.planned_path, self.robot_pose
        )

        if blocked:
            self._on_obstacle_detected(distance, angle)

    def _on_obstacle_detected(self, distance, angle):
        """Generate LLM alert about obstacle."""
        # Ask LLM to generate contextual warning
        context = {
            "obstacle_distance": distance,
            "obstacle_direction": angle,
            "current_goal": self.current_goal,
            "current_zone": self.current_zone
        }

        # LLM generates warning
        warning = self.llm_agent.generate_obstacle_warning(context)
        # "Someone is in the way. Want me to go around or wait?"

        self.pending_user_response = self.wait_for_user_choice()

        if self.pending_user_response == "go_around":
            self.nav_client.replan(destination=self.current_goal,
                                   avoid_obstacles=True)
        elif self.pending_user_response == "wait":
            self._start_obstacle_wait_loop()
```

### Zone-Based Replanning

When user says "go around", use zone connectivity graph to find alternatives:

```python
def replan_avoiding_obstacles(current_zone, goal_zone, zone_graph):
    """
    Find alternative path using zone connectivity.

    Example:
    ┌─────────────┐         ┌─────────────┐
    │  hri_lab    │─────────│   kitchen   │
    └─────────────┘         └─────────────┘
                                   │
                            ┌──────┴──────┐
                            │  hallway_1  │
                            └─────────────┘

    If direct path blocked, try:
    hri_lab → hallway_1 → kitchen (alternative)
    """
    # Build graph excluding blocked zone
    alt_graph = zone_graph.copy()

    # Use A* or Dijkstra to find alternative
    path = dijkstra(alt_graph, current_zone, goal_zone)

    if path:
        # Replan through alternative zones
        waypoints = [zone_centroid(z) for z in path]
        send_goal_to_nav2(waypoints)
    else:
        # No alternative path exists
        agent_says("No alternative route available")
```

---

## 3. Hazard Detector (Safety-Critical Alerts)

### Hazards to Detect

```python
HAZARDS = {
    "wet_floor_sign": {
        "object_label": "wet_floor_sign",
        "priority": "high",
        "warning": "The floor might be slippery. Please be careful."
    },
    "person": {
        "object_label": "person",
        "priority": "high",
        "warning": "There's someone nearby."
    },
    "stairs": {
        "object_label": "stairs",
        "priority": "critical",
        "warning": "Be careful, there are stairs ahead."
    },
    "low_doorway": {
        "object_label": "low_doorway",
        "priority": "medium",
        "warning": "The doorway ahead is low. Please duck."
    },
    "obstacle": {
        "object_label": "obstacle",
        "priority": "high",
        "warning": "There's an obstacle in the path."
    }
}
```

### Timing

```
During navigation:
├─ Every 1 second: Run YOLO for hazard detection
├─ For each detected hazard:
│  ├─ Check distance to robot
│  ├─ Check if already alerted (avoid repeat)
│  └─ If new hazard: Generate LLM warning
└─ User responds: Continue / Wait / Go around / Cancel
```

### LLM-Generated Warnings (Contextual)

Instead of generic warnings, use LLM to generate context-aware messages:

```python
def generate_hazard_warning(hazard_type, distance, location, context):
    """
    Use LLM to generate natural, contextual warning.

    Examples:

    Hazard: wet_floor_sign, distance: 1.5m, location: kitchen
    LLM: "There's a wet floor sign in the kitchen just ahead.
          The floor might be slippery, so be extra careful."

    Hazard: person, distance: 2m, location: hallway
    LLM: "I see someone in the hallway ahead.
          They're about 2 meters away. Should I wait?"

    Hazard: stairs, distance: 1m, location: entrance
    LLM: "Watch out! There are stairs just ahead.
          Be very careful where you step."
    """
    prompt = f"""
    Generate a natural, spoken warning for a VI user about a {hazard_type}.
    Context:
    - Distance: {distance}m
    - Location: {location}
    - Current activity: {context['current_activity']}

    Make it natural, brief (1-2 sentences), and actionable.
    """

    warning = llm.generate(prompt)
    speak(warning)

    # Wait for user response
    response = wait_for_user_input(timeout=5)
    return response  # "continue", "wait", "cancel"
```

### Implementation

```python
class HazardDetector:
    def __init__(self, yolo_processor, llm_agent):
        self.yolo = yolo_processor
        self.llm = llm_agent
        self.alerted_hazards = {}  # Track what we've warned about
        self.hazard_timer = PeriodicTimer(1.0)  # Check every 1 sec

    def on_navigation_update(self, robot_pose, current_zone):
        """Check for hazards during navigation."""
        if not self.hazard_timer.should_run():
            return

        # Run YOLO to detect hazards
        detections = self.yolo.detect_hazards(confidence_threshold=0.7)

        for detection in detections:
            hazard_type = detection['label']
            distance = detection['distance_m']

            # Have we already warned about this?
            hazard_id = f"{hazard_type}_{int(distance)}"
            if hazard_id in self.alerted_hazards:
                continue  # Already warned

            # Generate warning
            warning = self.llm.generate_hazard_warning(
                hazard_type, distance, current_zone
            )
            self.speak(warning)

            # Get user response
            response = self.wait_for_response(timeout=5)

            if response == "continue":
                self.alerted_hazards[hazard_id] = True
            elif response == "wait":
                self.on_wait_response(hazard_type)
            elif response == "cancel":
                self.on_cancel_response()
```

---

## 4. System Integration

### Event Flow

```
Robot Operating
    ├─ Zone Monitor: Every 10ms
    │  └─ Check if crossed zone boundary
    │
    ├─ Obstacle Monitor: Every 100ms (if navigating)
    │  └─ Check LiDAR for path blockage
    │
    └─ Hazard Detector: Every 1000ms (if navigating)
       └─ Run YOLO for specific hazards

User Input
    ├─ If navigating: Check monitors
    ├─ Obstacle present? → Ask about it
    ├─ Hazard detected? → Warn about it
    └─ Otherwise: Process normal user request
```

### Priority System

```
If multiple events simultaneous:

1. CRITICAL: Obstacle blocking path
   → Stop navigation, ask user immediately

2. HIGH: Hazard detected (stairs, person)
   → Warn user, offer to continue/wait/stop

3. MEDIUM: Zone entry/arrival
   → Announce when agent finishes speaking

4. LOW: Navigation progress
   → Update only on request
```

---

## 5. Configuration

```python
# monitoring_config.py

MONITORING_CONFIG = {
    "zone_awareness": {
        "enabled": True,
        "announce_entry": True,
        "announce_arrival": True,
        "min_announcement_interval": 5.0,  # seconds
        "queue_if_agent_speaking": True,
    },

    "obstacle_detection": {
        "enabled": True,
        "check_interval": 0.1,  # 100ms
        "lookahead_distance": 2.0,  # meters
        "min_obstacle_distance": 0.5,  # meters (alert threshold)
        "enable_replanning": True,
    },

    "hazard_detection": {
        "enabled": True,
        "check_interval": 1.0,  # 1 second
        "yolo_confidence_threshold": 0.7,
        "hazard_types": ["wet_floor_sign", "person", "stairs", ...],
        "use_llm_warnings": True,
    }
}
```

---

## 6. Testing Scenarios

- [ ] Zone entry announcement (queue if speaking)
- [ ] Obstacle blocks path (offer go around / wait / cancel)
- [ ] Hazard detection during navigation (wet floor warning)
- [ ] Multiple hazards (prioritize by criticality)
- [ ] User timeout waiting for response (default action?)
- [ ] Zone boundary near obstacle (both triggered together?)

---

## 7. Future Enhancements

- [ ] Learning user preferences ("Always go around" vs. "Always wait")
- [ ] Proactive hazard prediction (knowing what's usually in each zone)
- [ ] User profile for warning frequency
- [ ] Haptic feedback for obstacles (if robot supports it)
- [ ] Integration with Go2 built-in bump sensors

---

**Next Step**: See `IMPLEMENTATION.md` for Phase 3 timeline.

**References**: References stored in `/home/ubuntu/guide-llm_2/references/references.bib`
