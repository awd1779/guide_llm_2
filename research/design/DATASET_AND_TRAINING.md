# Dataset Generation and Training Strategy

**Status**: Design complete, ready for implementation
**Target**: 1500 conversation samples for fine-tuning
**Last updated**: 2026-03-23

---

## Quick Summary

We generate training data using **journey-based scenarios**:
- Multi-stop navigation (2-4 stops per journey)
- Mid-journey user interruptions ("What's around us?", "Are we there?")
- Grounded in actual scene_graph.json from your environment
- Total: ~800-1000 conversations across variations and randomizations
- Split: 600 Tier 0-2 (scene graph only) + 900 Tier 3-4 (YOLO-based)

---

## 1. Journey-Based Scenarios (Core Approach)

### Why Journeys, Not Single Actions?

**Old approach** (problematic):
- Single conversation: "User: Take me to kitchen" → "Agent: navigate_to(kitchen)" → Done
- Problem: Doesn't reflect real blind user behavior
- Problem: Model doesn't learn multi-turn context

**New approach** (journey-based):
- Multi-leg navigation: hri_lab → kitchen → hallway_1 → back to hri_lab
- Mid-journey interruptions: User asks "What's around us?" during navigation
- Real behavior: Mirrors actual navigation with status checks, perception queries
- Result: Model learns context maintenance across multiple tool calls

### Journey Definition

```python
JOURNEY = {
    "id": "journey_1",
    "name": "hri_lab → kitchen → hri_lab",
    "start_zone": "hri_lab",
    "stops": [
        {
            "zone": "kitchen",
            "arrival_actions": ["query", "describe_surroundings"],
            "mid_navigation_interruptions": ["What's around us?", "Are we there?"]
        },
        {
            "zone": "hri_lab",
            "arrival_actions": ["get_robot_pose"],
            "mid_navigation_interruptions": []
        }
    ],
    "variations": 3  # Different conversation orderings
}
```

### Your Environment's Natural Journeys

Based on your scene_graph.json topology:

```
hri_lab (5 objects: 2 chairs, tv, table, home)
    ↓
hallway_0 (empty connector)
    ↓ ↘
kitchen (empty)   hallway_1 (empty)
                      ↓
                  hallway_2 (empty)
```

**Journey Templates** (~10-15 total):

1. **hri_lab ↔ kitchen** (2 stops)
   - Simple round trip: "go to kitchen and back home"

2. **hri_lab → kitchen → hallway_1** (3 stops)
   - Longer exploration

3. **Full loop**: hri_lab → kitchen → hallway_1 → hallway_2 → back
   - Explore all zones

4. **Variants with different interruptions**
   - Some journeys have mid-nav questions
   - Some just reach destination
   - Some involve canceling and replanning

---

## 2. Conversation Generation Pattern

### Example: Single Journey Conversation

```python
# Journey: hri_lab → kitchen → hri_lab

Turn 1: USER → AGENT
User: "Take me to the kitchen"

Turn 2: AGENT uses tool
Agent: calls navigate_to("kitchen")
Tool result: {"status": "goal_sent", "destination": "kitchen"}
Agent: "Heading to the kitchen now."

Turn 3: MID-NAVIGATION INTERRUPTION (60% chance)
User: "What's around us?"
Agent: calls describe_surroundings(radius=3.0)
Tool result: {"objects": [...], "current_zone": "hallway_0"}
Agent: "We're in the hallway, moving toward the kitchen."

Turn 4: AGENT checks arrival
Agent: calls get_navigation_status()
Tool result: {"status": "succeeded", "destination": "kitchen"}
Agent: "We've arrived at the kitchen!"

Turn 5: STOP ACTIONS (at destination)
User: "What's in the kitchen?"
Agent: calls query("kitchen")
Tool result: {"objects": []}  # Kitchen is empty
Agent: "This is the kitchen. It's empty right now."

Turn 6: NEXT LEG
User: "Take me back to hri_lab"
Agent: calls navigate_to("hri_lab")
Tool result: {"status": "goal_sent", "destination": "hri_lab"}
Agent: "On our way back home."

... (similar pattern continues)

Turn N: END
User: "Thanks!"
Agent: "You're welcome!"
```

### Conversation Generation Code Template

```python
def generate_journey_conversation(journey, scene_graph, variation_num):
    """
    Generate conversation for a multi-stop journey.

    Args:
        journey: Journey definition with stops and actions
        scene_graph: Scene graph JSON
        variation_num: Which variation (different random choices)

    Returns:
        {
            "journey_id": journey["id"],
            "messages": [system_prompt, ...conversation...],
            "tools": TOOLS
        }
    """
    msgs = []
    current_zone = journey["start_zone"]

    for stop_idx, stop in enumerate(journey["stops"]):
        next_zone = stop["zone"]

        # ===== NAVIGATION LEG =====
        if stop_idx == 0:
            msgs.append({
                "role": "user",
                "content": f"Take me to the {next_zone}"
            })
        else:
            msgs.append({
                "role": "user",
                "content": random.choice([
                    f"Take me to the {next_zone}",
                    f"Let's go to {next_zone}",
                    f"Go to {next_zone} next"
                ])
            })

        # Agent navigates
        msgs.extend(_tool_turn("navigate_to",
            {"destination": next_zone},
            {"status": "goal_sent", "destination": next_zone}
        ))
        msgs.append({
            "role": "assistant",
            "content": f"Heading to the {next_zone} now."
        })

        # ===== MID-NAVIGATION INTERRUPTION (optional) =====
        if random.random() < 0.6:  # 60% chance
            interruption = random.choice([
                "What's around us?",
                "Are we there yet?",
                "Where are we now?",
                "What's ahead?"
            ])
            msgs.append({"role": "user", "content": interruption})

            if "around" in interruption.lower():
                msgs.extend(_tool_turn(
                    "describe_surroundings",
                    {"radius": 3.0},
                    {"objects": [...], "current_zone": current_zone}
                ))
            elif "where" in interruption.lower():
                msgs.extend(_tool_turn(
                    "get_navigation_status",
                    {},
                    {"status": "executing", "destination": next_zone}
                ))

        # ===== ARRIVAL =====
        msgs.extend(_tool_turn(
            "get_navigation_status",
            {},
            {"status": "succeeded", "destination": next_zone}
        ))
        msgs.append({
            "role": "assistant",
            "content": f"We've arrived at the {next_zone}!"
        })

        # ===== STOP ACTIONS =====
        for action in stop["arrival_actions"]:
            if action == "query":
                msgs.append({
                    "role": "user",
                    "content": f"What's in the {next_zone}?"
                })
                msgs.extend(_tool_turn(
                    "query",
                    {"name": next_zone},
                    {"zone": next_zone, "objects": [...]}
                ))

            elif action == "describe_surroundings":
                msgs.append({
                    "role": "user",
                    "content": "Describe what's here"
                })
                msgs.extend(_tool_turn(
                    "describe_surroundings",
                    {"radius": 5.0},
                    {"objects": [...]}
                ))

            elif action == "get_robot_pose":
                msgs.append({
                    "role": "user",
                    "content": "Where are we?"
                })
                msgs.extend(_tool_turn(
                    "get_robot_pose",
                    {},
                    {"current_zone": next_zone}
                ))

        current_zone = next_zone

    # ===== END =====
    msgs.append({"role": "user", "content": "Thanks!"})
    msgs.append({
        "role": "assistant",
        "content": "You're welcome! Let me know if you need anything else."
    })

    return {
        "journey_id": journey["id"],
        "messages": [system_prompt] + msgs,
        "tools": TOOLS
    }
```

---

## 3. Dataset Composition

### Total: 1500 Samples

```
TIER 0-2 (Scene Graph Only): 600 samples (40%)
├─ Keep best existing examples
├─ Simple navigation, queries, status checks
├─ No YOLO perception needed
└─ Fast execution (<100ms)

TIER 3-4 (YOLO-Based): 900 samples (60%)  [NEW]
├─ Describe surroundings with YOLO
├─ Find nearest object
├─ Complex perception queries
└─ Slower but more accurate (~300ms)
```

### Generation Strategy

**Week 1: Define Journey Templates**
- ~10-15 journey templates
- Based on actual scene_graph.json topology
- Each template lists possible stops and actions

**Week 2: Generate Conversations**
- For each journey × variation (3-5 variations per journey):
  - Generate base navigation leg
  - Randomly add 0-2 mid-nav interruptions
  - Add stop actions
  - Randomize user language (paraphrasing)
- Output: ~800-1000 base conversations

**Week 2-3: Expand & Randomize**
- Random variations in:
  - Tool parameter choices
  - Conversation phrasing
  - Interruption timing
  - Stop action ordering
- Final dataset: 1500 samples

---

## 4. Mid-Navigation Interruptions

### Common User Questions During Navigation

```python
MID_NAV_INTERRUPTIONS = {
    "perception": [
        "What's around us?",
        "What do you see?",
        "What's ahead?",
        "Describe what's around me",
    ],
    "status": [
        "Are we there yet?",
        "How much further?",
        "Where are we now?",
        "What's our current location?",
    ],
    "safety": [
        "Is it safe to go forward?",
        "Any obstacles?",
        "Is the path clear?",
    ],
    "cancel": [
        "Stop, I changed my mind",
        "Let's go somewhere else instead",
        "Actually, take me to hallway_1 instead",
    ]
}
```

**Each journey**: 0-2 random interruptions (60% chance of at least one)

---

## 5. Training Data Format

### JSONL Format (OpenAI Chat API)

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a navigation assistant for blind/low-vision users..."
    },
    {
      "role": "user",
      "content": "Take me to the kitchen"
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_xyz",
          "type": "function",
          "function": {
            "name": "navigate_to",
            "arguments": "{\"destination\": \"kitchen\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_xyz",
      "content": "{\"status\": \"goal_sent\", \"destination\": \"kitchen\"}"
    },
    {
      "role": "assistant",
      "content": "Heading to the kitchen now."
    }
  ],
  "tools": [...]
}
```

---

## 6. Key Training Principles

### No Confidence Scores in Data
- Model never sees uncertainty
- YOLO confidence filtering happens at perception layer
- Result: Model learns "if in result, it's reliable"

### Only High-Confidence Detections
- YOLO threshold: >0.7
- Tool result signal: `"perception_source": "yolo"`
- Model learns: "yolo source = real perception, scene_graph = static"

### Realistic Tool Patterns
- Not all conversations have all tools
- Tool selection depends on query type
- Model learns: "this query → this tool, not that tool"

### Natural Language Variations
- Same journey, different phrasings
- Paraphrased user requests
- Diverse agent responses
- Result: Model generalizes beyond exact templates

---

## 7. Expected Model Performance

### Tool Selection Accuracy
- After fine-tuning: >95%
- Baseline (no fine-tuning): ~60-70%

### Parameter Correctness
- Correctly formatted function calls: >90%
- No hallucinated parameters: >98%

### Response Quality
- No raw coordinates: >95%
- No tool name leakage: >99%
- Natural spoken language: >90%

---

## 8. Implementation Checklist

- [ ] Define 10-15 journey templates
- [ ] Implement `generate_journey_conversation()` function
- [ ] Test generation on sample journeys
- [ ] Generate 1500 samples
- [ ] Split into train/val (90/10)
- [ ] Save as JSONL format
- [ ] Validate JSON syntax
- [ ] Spot-check 20-30 conversations
- [ ] Ready for fine-tuning

---

**Next Step**: See `IMPLEMENTATION.md` for detailed timeline and Phase 2 details.

**References**: References stored in `/home/ubuntu/guide-llm_2/references/references.bib`
