# System Design & Implementation

Complete architecture, design, and implementation roadmap for Guide-LLM v2.

---

## Documents

### [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) ⭐ START HERE

Complete system architecture covering:
- Block diagram and data flow
- Perception Hierarchy (Tier 0-4 decision tree)
- Vision integration (on-demand YOLO)
- All 12 tools and their triggers
- Hardware/software stack
- Latency budget

**Read this**: To understand how the system works (30 min)

---

### [DATASET_AND_TRAINING.md](DATASET_AND_TRAINING.md)

Journey-based dataset generation strategy:
- Why journey-based scenarios (multi-stop navigation)
- Journey definition and templates
- Conversation generation patterns
- Dataset composition (1500 samples)
- Training principles (no confidence scores)
- Implementation checklist

**Read this**: To understand training data strategy or generate dataset (Phase 2)

---

### [SAFETY_AND_MONITORING.md](SAFETY_AND_MONITORING.md)

Safety and monitoring systems:
- Zone Awareness Monitor (non-intrusive announcements)
- Obstacle Monitor (dynamic obstacle detection & replanning)
- Hazard Detector (safety-critical hazards)
- LLM-generated contextual warnings
- ROS2 node examples

**Read this**: To implement Phase 3 monitoring systems

---

### [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md)

Complete 6-8 week implementation plan:
- 4 phases with dependencies
- Week-by-week breakdown
- Files to create/modify
- Testing strategy
- Risk mitigation

**Read this**: Before starting Phase 1 implementation

---

## How to Use

**For understanding the system** (engineering):
1. Read SYSTEM_DESIGN.md (architecture)
2. Read DATASET_AND_TRAINING.md (training strategy)
3. Skim SAFETY_AND_MONITORING.md (monitoring)

**For implementation** (engineering):
1. Read IMPLEMENTATION_ROADMAP.md (6-8 week plan)
2. Follow Phase 1 → Phase 2 → Phase 3 → Phase 4
3. Reference specific docs for each phase

**For research/publication** (researchers):
1. See parent directory: RESEARCH_AND_PUBLICATION.md
2. These design docs support research validation

---

**Version**: 2026-03-23
**Status**: Design complete, ready for Phase 1 implementation
