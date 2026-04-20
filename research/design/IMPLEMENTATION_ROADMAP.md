# Guide-LLM v2: Complete Implementation Roadmap

**Goal**: Build a production-ready LLM-driven navigation system for blind/low-vision users on Go2 robot with vision integration, hazard detection, and dynamic replanning.

**Timeline**: 6-8 weeks (4 phases)

---

## System Overview (How It All Fits)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Guide-LLM v2 System                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SENSORS (Always Running)                                                   │
│  ├─ LiDAR → SLAM/Nav2 (localization, path planning)                        │
│  └─ RGB-D Camera → Available for on-demand perception                      │
│                                                                              │
│  PERCEPTION (On-Demand or Continuous During Nav)                           │
│  ├─ Vision Processor: YOLO on RGB-D (on-demand for queries)                │
│  ├─ Hazard Detector: YOLO periodically during navigation (hazard detection) │
│  └─ Scene Graph Fusion: Merge detections into static scene graph           │
│                                                                              │
│  DECISION LAYER (Core LLM Agent)                                           │
│  ├─ Scene Graph Agent: Orchestrates all decisions                          │
│  ├─ Perception Hierarchy: Decides when YOLO is needed                      │
│  ├─ Tool Calling: 10 tools for navigation, perception, querying            │
│  └─ LLM (Qwen 3.5 4B): Makes decisions, generates responses                │
│                                                                              │
│  CONTEXT AWARENESS (Always Monitoring)                                      │
│  ├─ Zone Awareness Monitor: Announce zone entries/arrivals                 │
│  ├─ Obstacle Monitor: Detect blocking obstacles, replan                    │
│  └─ Hazard Monitor: Detect specific hazards, warn user                     │
│                                                                              │
│  OUTPUT                                                                      │
│  ├─ Speech Feedback (TTS): Agent responses, announcements, warnings        │
│  └─ Navigation Commands: Goals to Nav2                                      │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Dependencies

```
PHASE 1: Foundation (Weeks 1-2)
├─ Core systems: Vision processor, scene graph fusion, perception hierarchy
├─ Setup: Perception detection, tool integration
└─ Result: System can query and see things via YOLO

    ↓ (blocks Phase 2)

PHASE 2: Training & Model (Weeks 2-3)
├─ Depends on: Phase 1 (know what YOLO outputs)
├─ Tasks: Generate realistic dataset, fine-tune model
└─ Result: New Qwen model trained on YOLO confidence distributions

    ↓ (blocks Phase 3)

PHASE 3: Awareness & Safety (Weeks 3-4)
├─ Depends on: Phase 1 (perception), Phase 2 (new model)
├─ Tasks: Zone awareness, obstacle detection, hazard detection
└─ Result: System tells user about zones, obstacles, hazards

    ↓ (no block)

PHASE 4: Integration & Testing (Weeks 4-8)
├─ Parallel work: Deploy, test, refine
├─ Tasks: End-to-end testing, user testing, optimization
└─ Result: Production-ready system
```

---

## Detailed Phase Breakdown

# PHASE 1: PERCEPTION FOUNDATION (Weeks 1-2)

## Goal
Build the **vision integration layer** so the system can detect objects in real-time and answer perception-based questions.

## Components to Build

### 1. Vision Processor (`perception/vision_processor.py`)
**Purpose**: Subscribe to RGB-D, run YOLO on demand

```python
# New file: perception/vision_processor.py
- Load YOLOv8n model
- Subscribe to /camera/rgb/image_raw + /camera/depth/image_raw
- Keep latest frames in buffer
- Service: /vision/get_detections
  - Input: none (use latest frame)
  - Output: [{"label": "chair", "confidence": 0.92, "depth_m": 2.1}, ...]
- Latency target: <300ms per call
```

**Deliverable**: Node that can be called on-demand to get object detections

### 2. Scene Graph Fusion (`perception/scene_graph_fusion.py`)
**Purpose**: Merge YOLO detections with static scene graph

```python
# New file: perception/scene_graph_fusion.py
- Function: fuse_vision_into_scene_graph()
  - Input: static scene graph + YOLO detections + robot pose
  - Output: enriched scene graph with real detections merged in
- Handle:
  - Matching detections to existing objects
  - Adding new detected objects temporarily
  - Coordinate transforms (camera frame → world frame)
  - Confidence scores attached to each object
```

**Deliverable**: Utility that enriches scene graph with real perception

### 3. Perception Hierarchy (`perception/perception_hierarchy.py`)
**Purpose**: Decide WHEN YOLO is needed based on user query

```python
# New file: perception/perception_hierarchy.py
- Function: needs_real_perception(user_input: str) -> bool
- Implement tiers:
  - Tier 0: NO (static questions) - "What zones exist?"
  - Tier 1: NO (robot state) - "Where am I?"
  - Tier 2: MAYBE (query) - "Is there a chair?" → "Is there a chair NOW?"
  - Tier 3: YES (current scene) - "What's in front of me?"
  - Tier 4: YES (object search) - "Find a chair"
- Simple keyword matching + heuristics
```

**Deliverable**: Decision logic for when to trigger YOLO

### 4. Modify Scene Graph Agent (`scene_graph_agent_local.py`)
**Purpose**: Integrate vision processor + perception hierarchy

```python
# Modify: scene_graph_agent_local.py
- Add perception trigger:
  if needs_real_perception(user_input):
    perception_data = call_vision_processor()
    enriched_sg = fuse_vision(perception_data)
  else:
    enriched_sg = static_scene_graph

- Modify tools to accept enriched_sg:
  describe_surroundings(enriched_sg)
  find_nearest(enriched_sg)
  query(enriched_sg)

- Track active perception (for training data generation)
```

**Deliverable**: Agent that triggers YOLO for appropriate queries

## Testing (Phase 1)

```
✓ Test vision_processor:
  - Start node, call /vision/get_detections service
  - Verify YOLO runs, returns detections with confidence+depth
  - Measure latency (<300ms)

✓ Test scene_graph_fusion:
  - Fuse sample YOLO detections into scene graph
  - Verify enriched SG has both static + real objects
  - Check coordinate transforms

✓ Test perception_hierarchy:
  - Run decision logic on 20 sample queries
  - Verify correct tier classification (0-4)

✓ Test agent integration:
  - User: "What zones exist?" → NO YOLO (fast)
  - User: "What's in front?" → YES YOLO (slow, real)
  - Verify responses based on enriched SG
```

## Deliverables (Phase 1)

- [ ] `perception/vision_processor.py` — YOLO service
- [ ] `perception/scene_graph_fusion.py` — Fusion logic
- [ ] `perception/perception_hierarchy.py` — Tier decision
- [ ] Modified `scene_graph_agent_local.py` — Integration
- [ ] Test results: Vision processor latency <300ms, perception hierarchy accuracy >95%

---

# PHASE 2: TRAINING & MODEL (Weeks 2-3)

## Goal
Generate **realistic training data** that reflects actual YOLO outputs, then fine-tune a new model.

## Components to Build

### 1. Collect Real YOLO Data
**Purpose**: Understand actual YOLO confidence distributions

```python
# Script: data_collection/collect_yolo_stats.py
- Deploy perception/vision_processor.py to Go2
- Walk through environments, collect 100+ YOLO detections
- Log: object label, confidence, depth accuracy
- Analyze: distribution of confidences per object type

Output: yolo_statistics.json
{
  "chair": {"mean_conf": 0.90, "std": 0.05, "count": 45},
  "table": {"mean_conf": 0.88, "std": 0.06, "count": 38},
  "person": {"mean_conf": 0.92, "std": 0.04, "count": 12},
  ...
}
```

### 2. Modify Dataset Generation (`finetune/generate_dataset.py`)
**Purpose**: Generate training data with realistic YOLO confidence

```python
# Modify: finetune/generate_dataset.py
- Add parameter: yolo_confidence_model='realistic'
- New function: sample_yolo_confidence(object_type, model='realistic')
  - If 'perfect': return 0.95 (old synthetic approach)
  - If 'realistic': sample from distribution based on collected stats
  - Use yolo_statistics.json

- New function: generate_perception_failure_scenario()
  - Blur scenarios
  - Wrong angle scenarios
  - Occlusion scenarios
  - Low light scenarios

- Modify dataset composition:
  - 40% Tier 0-2 (scene graph only, no perception)
  - 60% Tier 3-4 (YOLO-based, with confidence scores)

- Tool result format:
  {
    "perception_source": "yolo",
    "objects": [
      {
        "label": "chair",
        "distance_m": 2.1,
        "confidence": 0.92,  # Real YOLO confidence
        "source": "yolo_detection"
      }
    ]
  }
```

### 3. Generate New Dataset
**Purpose**: Create 1500-sample dataset with realistic perception

```python
# Script: finetune/generate_full_dataset.py
python finetune/generate_full_dataset.py \
  --num_samples 1500 \
  --yolo_confidence_model realistic \
  --include_failures true \
  --output_train finetune/data/train_v2.jsonl \
  --output_val finetune/data/val_v2.jsonl

Output:
├─ finetune/data/train_v2.jsonl (1350 samples)
├─ finetune/data/val_v2.jsonl (150 samples)
└─ dataset_stats.json
   {
     "perception_distribution": {
       "scene_graph_only": 0.40,
       "yolo_based": 0.60
     },
     "avg_yolo_confidence": 0.83,
     ...
   }
```

### 4. Fine-tune New Model
**Purpose**: Train Qwen 3.5 4B on realistic perception data

```python
# Script: finetune/finetune_qwen_v2.py
python finetune/finetune_qwen_v2.py \
  --train_file finetune/data/train_v2.jsonl \
  --val_file finetune/data/val_v2.jsonl \
  --num_epochs 3 \
  --output_dir finetune/output/qwen3.5-4b-vision-aware

Hyperparameters (same as before):
- per_device_train_batch_size: 1
- gradient_accumulation_steps: 4
- learning_rate: 2e-4
- warmup_steps: 10
- max_seq_length: 2048
- bf16: true
```

### 5. Evaluate New Model
**Purpose**: Test that model handles realistic perception

```python
# Script: finetune/validate_v2.py
python finetune/validate_v2.py \
  --model_path finetune/output/qwen3.5-4b-vision-aware/... \
  --test_suite test_suites/vision_aware_tests.json

Tests:
- Tool selection accuracy (same 48-case suite)
- Confidence awareness:
  - Low confidence detections: Does model use hedging language?
  - Perception failures: Does model ask for help?
- Multi-turn degradation
- Quantization robustness (fp16 → q4_k_m)

Expected results:
- Tool accuracy: >96% (same as before)
- Confidence awareness: >80% (NEW - hedging on <0.75 confidence)
- Failure recovery: 100% (NEW - handles perception failures)
```

## Testing (Phase 2)

```
✓ YOLO statistics collected:
  - Verify distribution represents real YOLO performance
  - Check all object types covered

✓ Dataset generated:
  - Verify 1500 samples created
  - Check: 40% scene graph only, 60% with YOLO data
  - Inspect samples: Do they look realistic?

✓ Model trained:
  - Verify training converges (loss decreases)
  - Verify eval loss is reasonable
  - Training time: ~2-3 hours on A100/V100

✓ Model evaluated:
  - Tool accuracy >96%
  - Confidence awareness metrics baseline
  - Quantization works (model.gguf created)
```

## Deliverables (Phase 2)

- [ ] `data_collection/collect_yolo_stats.py` — Data collection script
- [ ] `yolo_statistics.json` — Real YOLO confidence distributions
- [ ] Modified `finetune/generate_dataset.py` — Realistic data generation
- [ ] `finetune/data/train_v2.jsonl`, `val_v2.jsonl` — New dataset
- [ ] `finetune/output/qwen3.5-4b-vision-aware/...` — Trained model
- [ ] Validation report: Tool accuracy >96%, confidence awareness >80%

---

# PHASE 3: AWARENESS & SAFETY (Weeks 3-4)

## Goal
Add **context awareness systems** that monitor navigation and alert user to zones, obstacles, hazards.

## Components to Build

### 1. Zone Awareness Monitor (`perception/zone_awareness_monitor.py`)
**Purpose**: Announce zone entries and arrivals

```python
# New file: perception/zone_awareness_monitor.py
- Continuously monitor robot pose (TF lookup every 200ms)
- Detect zone transitions (point-in-polygon test)
- Announce zone entry: "You're in the kitchen."
- Announce navigation arrival: "You've arrived."
- Queue announcements if agent is speaking
- Configuration:
  - announce_zone_entry: True
  - announce_arrival: True
  - min_interval_seconds: 5.0
  - queue_if_agent_speaking: True

Output: Publishes to /zone_awareness/announcement
```

### 2. Obstacle Monitor (`perception/obstacle_monitor.py`)
**Purpose**: Detect generic blocking obstacles and replan

```python
# New file: perception/obstacle_monitor.py
- Monitor LiDAR during navigation
- Detect obstacles not in scene graph (within 1.5m ahead)
- Alert user: "Something is blocking the way."
- Get user response: go around / wait / cancel
- Implement replanning (zone-based alternative paths)
- Configuration:
  - blocking_distance_threshold: 1.5
  - check_interval: 0.2
  - min_alert_interval: 3.0

Output: Publishes to /obstacle_alert/detected
```

### 3. Hazard Detector (`perception/hazard_detector.py`)
**Purpose**: Detect specific hazards (wet floor, people, etc.)

```python
# New file: perception/hazard_detector.py
- Periodically run YOLO during navigation (every 1 second)
- Check detections against HAZARD_OBJECTS config
- Alert on hazards: wet_floor_sign, person, low_doorway, steps, etc.
- Use LLM to generate contextual warning
- Get user response: continue / wait / go around
- Configuration:
  - confidence_threshold: 0.70
  - distance_threshold_m: 3.0
  - min_alert_interval_s: 5.0

Output: Publishes to /hazard_alert/detected
```

### 4. Hazard Configuration (`perception/hazard_config.py`)
**Purpose**: Define what hazards to detect and how to warn

```python
# New file: perception/hazard_config.py
HAZARD_OBJECTS = {
    'wet_floor_sign': {
        'type': 'hazard',
        'severity': 'medium',
        'hazard_type': 'slippery',
    },
    'person': {
        'type': 'hazard',
        'severity': 'high',
        'hazard_type': 'collision',
    },
    'steps': {
        'type': 'warning',
        'severity': 'high',
        'hazard_type': 'trip_fall',
    },
    # ... more as config
}

HAZARD_DETECTION_THRESHOLDS = {
    'confidence_min': 0.7,
    'distance_max': 3.0,
    'min_alert_interval': 5.0,
}
```

### 5. Modify Scene Graph Agent (`scene_graph_agent_local.py`)
**Purpose**: Handle all monitor alerts and responses

```python
# Modify: scene_graph_agent_local.py
- Subscribe to:
  - /zone_awareness/announcement
  - /obstacle_alert/detected
  - /hazard_alert/detected

- Implement handlers:
  - _on_zone_announcement(): Pass to TTS
  - _on_obstacle_alert(): Generate response, ask user
  - _on_hazard_alert(): Use LLM to warn, ask user

- Implement response handlers:
  - _handle_obstacle_response(): Replan / wait / cancel
  - _handle_hazard_response(): Continue / wait / replan

- State tracking:
  - is_agent_speaking: Used to queue announcements
  - active_hazard: Track current hazard
  - navigation_target: For arrival detection
```

### 6. Modify Tools (`scene_graph_agent_local.py`)
**Purpose**: Add new tool for hazard recovery

```python
# Potentially add new tool (optional):
- replan_around_hazard(hazard_type: str)
  - Input: type of hazard to avoid
  - Output: new navigation goal via alternative route

(Or: reuse navigate_to with alternative zone)
```

## Testing (Phase 3)

```
✓ Zone awareness:
  - Walk through zones, listen for announcements
  - Verify: Zone entry "You're in [zone]" sounds natural
  - Verify: Arrival "You've arrived" works
  - Test queuing: Agent speaking + zone entry → queued

✓ Obstacle detection:
  - Place obstacle in planned path during navigation
  - Verify: Alert triggered "Something is blocking the way"
  - User response: "Go around" → Replan works
  - User response: "Wait" → Pauses, resumes when obstacle moves

✓ Hazard detection:
  - Place wet floor sign in path
  - Verify: Detected (confidence >70%, distance <3m)
  - Verify: LLM-generated warning makes sense
  - User response: "Go around" → Reroutes
  - User response: "Continue" → Keeps going (user aware)
```

## Deliverables (Phase 3)

- [ ] `perception/zone_awareness_monitor.py` — Zone announcements
- [ ] `perception/obstacle_monitor.py` — Generic obstacle detection
- [ ] `perception/hazard_detector.py` — Hazard-specific detection
- [ ] `perception/hazard_config.py` — Hazard definitions
- [ ] Modified `scene_graph_agent_local.py` — Alert handling + responses
- [ ] Test results: All monitors working, user responses working

---

# PHASE 4: INTEGRATION & TESTING (Weeks 4-8)

## Goal
Deploy to Go2, test with real navigation, optimize for production.

## Components to Integrate

### 1. ROS2 Launch File (`launch/guide_llm_complete.py`)
**Purpose**: Start all nodes together

```python
# New file: launch/guide_llm_complete.py
def generate_launch_description():
    return LaunchDescription([
        # Existing (robot, SLAM, Nav2, voice)
        Node(package='go2_driver', ..., name='go2_driver'),
        Node(package='cartographer_ros', ..., name='cartographer'),
        Node(package='nav2_bringup', ..., name='nav2'),
        Node(package='voice_interface', ..., name='voice_interface'),

        # Scene graph agent (core)
        Node(
            package='guide_llm',
            executable='scene_graph_agent_local.py',
            name='scene_graph_agent',
            parameters=[...]
        ),

        # Perception (Phase 1)
        Node(
            package='guide_llm',
            executable='vision_processor.py',
            name='vision_processor',
        ),

        # Awareness & Safety (Phase 3)
        Node(
            package='guide_llm',
            executable='zone_awareness_monitor.py',
            name='zone_awareness_monitor',
            parameters=[
                {'announce_zone_entry': True},
                {'announce_arrival': True},
            ]
        ),

        Node(
            package='guide_llm',
            executable='obstacle_monitor.py',
            name='obstacle_monitor',
            parameters=[
                {'blocking_distance_threshold': 1.5},
                {'min_alert_interval': 3.0},
            ]
        ),

        Node(
            package='guide_llm',
            executable='hazard_detector.py',
            name='hazard_detector',
            parameters=[
                {'confidence_threshold': 0.70},
                {'distance_threshold_m': 3.0},
            ]
        ),
    ])
```

### 2. Jetson Deployment
**Purpose**: Prepare model for edge device

```bash
# Step 1: Quantize model from Phase 2
python finetune/quantize.py \
  --input finetune/output/qwen3.5-4b-vision-aware/merged_16bit \
  --output finetune/output/qwen3.5-4b-vision-aware/gguf \
  --method q4_k_m

# Step 2: Create Ollama Modelfile
cat > finetune/output/guide-llm-4b.Modelfile << EOF
FROM finetune/output/qwen3.5-4b-vision-aware/gguf/model.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 2048
PARAMETER stop "<|im_end|>"
EOF

# Step 3: Import to Ollama on Jetson
ollama create guide-llm-4b -f guide-llm-4b.Modelfile

# Step 4: Update scene_graph_agent to use new model
# In scene_graph_agent_local.py:
base_url = "http://localhost:11434/v1"  # Ollama
model = "guide-llm-4b"  # New quantized model
```

### 3. Performance Benchmarking
**Purpose**: Measure latency and accuracy on Jetson

```python
# Script: benchmarks/measure_latency.py
Tests:
- Vision processor latency (YOLO):
  - Cold start (model load): <5s
  - Warm run (inference): <300ms
  - Depth extraction: <50ms
  - Total: <400ms per query

- LLM latency (on Jetson):
  - Tier 0-2 query (scene graph only): <100ms
  - Tier 3-4 query (with YOLO): <1000ms total
    - YOLO: 400ms
    - Fusion: 50ms
    - LLM inference: 500ms
  - Multi-turn (no perception): <200ms per turn

- Memory usage:
  - LiDAR + Nav2: ~3 GB
  - YOLO + perception: ~2 GB
  - LLM (q4_k_m): ~5 GB
  - Total: ~10 GB / 16 GB available

- Zone awareness:
  - Zone transition detection: <50ms
  - Announcement latency: <500ms

- Hazard detection:
  - Per-scan YOLO latency: <400ms
  - Hazard alert publish: <50ms
  - User response parsing: <100ms
```

### 4. User Testing Plan
**Purpose**: Test with actual blind/low-vision users

```
Test Objectives:
1. Navigation accuracy
   - Can user reach destinations safely?
   - Does replanning work when blocked?

2. Hazard awareness
   - Does wet floor sign warning help?
   - Is warning timing appropriate?

3. Zone announcements
   - Helpful or distracting?
   - Volume/timing appropriate?

4. Response times
   - Are responses under 2 seconds?
   - Does YOLO delay feel acceptable?

5. Usability
   - Natural language feel?
   - Can user give varied commands?
   - Recovery from errors?

Test protocol:
- N=3-5 VI participants
- Task: Navigate from point A → B in known environment
- Measure: Success rate, time, user feedback
- Iterate: Adjust thresholds, prompts, announcements
```

### 5. Optimization & Refinement
**Purpose**: Performance tuning based on real usage

```
If latency too high:
- Reduce YOLO inference frequency during nav
  (Check hazards every 2s instead of 1s)
- Use YOLOv8n-fast variant
- Enable Jetson optimization (CUDA optimization)

If accuracy too low:
- Re-fine-tune model with more diverse data
- Adjust YOLO confidence thresholds
- Add more conversation types

If announcements overwhelming:
- Reduce announcement frequency
- Add "quiet mode" option
- Only announce hazards, skip zones

If replanning fails:
- Improve zone connectivity mapping
- Add more granular waypoints (not just zones)
- Better fallback strategies
```

## Testing (Phase 4)

```
✓ Integration testing:
  - All nodes start without errors
  - ROS topics connected properly
  - No VRAM overrun

✓ End-to-end navigation:
  - User: "Take me to kitchen" → Full flow works
  - YOLO triggered for perception queries
  - Zone announcements at right times
  - Hazard detection + warning works
  - Replanning on obstacles works

✓ Performance benchmarks:
  - Latency measurements on Jetson
  - VRAM usage stable
  - No thermal throttling

✓ User feedback:
  - Task completion rate >80%
  - Natural language feel confirmed
  - Hazard warnings helpful
  - Announcements not intrusive
```

## Deliverables (Phase 4)

- [ ] `launch/guide_llm_complete.py` — Complete launch file
- [ ] Quantized model deployed to Jetson
- [ ] Benchmarking results (latency, VRAM, accuracy)
- [ ] User testing protocol + results
- [ ] Production tuning completed
- [ ] Documentation for deployment and operation

---

## File Structure (Final)

```
guide-llm_2/
├── README.md
│
├── scene_graph_agent_local.py          ← MODIFIED (Phase 1, 3)
│
├── perception/                         ← NEW FOLDER
│   ├── __init__.py
│   ├── vision_processor.py             ← Phase 1
│   ├── scene_graph_fusion.py           ← Phase 1
│   ├── perception_hierarchy.py         ← Phase 1
│   ├── zone_awareness_monitor.py       ← Phase 3
│   ├── obstacle_monitor.py             ← Phase 3
│   ├── hazard_detector.py              ← Phase 3
│   └── hazard_config.py                ← Phase 3
│
├── launch/                             ← NEW FOLDER
│   └── guide_llm_complete.py           ← Phase 4
│
├── finetune/
│   ├── generate_dataset.py             ← MODIFIED (Phase 2)
│   ├── finetune_qwen_v2.py             ← Phase 2
│   ├── validate_v2.py                  ← Phase 2
│   ├── quantize.py                     ← Phase 4
│   ├── data/
│   │   ├── train_v2.jsonl              ← Phase 2
│   │   └── val_v2.jsonl                ← Phase 2
│   └── output/
│       └── qwen3.5-4b-vision-aware/... ← Phase 2, 4
│
├── data_collection/
│   └── collect_yolo_stats.py           ← Phase 2
│
├── benchmarks/
│   └── measure_latency.py              ← Phase 4
│
├── brain_storm/
│   ├── SYSTEM_ARCHITECTURE.md
│   ├── TRAINING_STRATEGY.md
│   ├── VISION_INTEGRATION_DESIGN.md
│   ├── PERCEPTION_HIERARCHY.md
│   ├── ZONE_AWARENESS_MINIMAL.md
│   ├── DYNAMIC_OBSTACLE_REPLANNING.md
│   ├── HAZARD_DETECTION_ALERTS.md
│   └── IMPLEMENTATION_ROADMAP.md        ← THIS FILE
│
├── data/
│   ├── scene_graph/
│   │   └── scene_with_zones.json
│   └── maps/
│       └── level12v6.yaml + .pgm
│
└── references/
    └── references.bib
```

---

## Timeline Summary

| Phase | Duration | Focus | Key Deliverable |
|-------|----------|-------|-----------------|
| **1** | Weeks 1-2 | Perception foundation | Vision processor + perception hierarchy |
| **2** | Weeks 2-3 | Training | Realistic dataset + fine-tuned model |
| **3** | Weeks 3-4 | Safety awareness | Zone/obstacle/hazard monitors |
| **4** | Weeks 4-8 | Deployment + testing | Production system on Jetson + user testing |

**Total**: 6-8 weeks to production

---

## Key Assumptions

1. **Hardware**: Go2 with Jetson Orin (16 GB), RGB-D camera, LiDAR
2. **Software**: ROS2, Nav2, Cartographer, Ollama
3. **Data**: Access to real YOLO performance metrics + user testing
4. **Team**: 1-2 engineers (can parallelize Phase 3)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| YOLO too slow on Jetson | Use YOLOv8n (nano), not full. Test early. |
| Memory overrun (LLM + YOLO) | Use q4_k_m quantization. Monitor VRAM during nav. |
| False positive hazard alerts | Tune confidence threshold (0.7 minimum). Test on variants. |
| Replanning fails (no alternative) | Implement fallback: return to start or hold position. |
| User confused by announcements | Minimize (only zone + arrival). Add "quiet mode" option. |
| Training data mismatch | Collect real YOLO stats first (Phase 2). |

---

## Success Metrics

✓ **Phase 1**: Vision processor works, <300ms latency, detections accurate
✓ **Phase 2**: New model trained, tool accuracy >96%, confidence awareness >80%
✓ **Phase 3**: All monitors working, alerts felt natural, responses appropriate
✓ **Phase 4**: Full system on Jetson, user testing >80% success rate, <2s latency per query

