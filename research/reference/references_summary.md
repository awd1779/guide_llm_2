# Literature Review: Natural Language Interfaces for Vision-Impaired Navigation

A summary of key references informing the fine-tuning dataset design for Guide-LLM v2.

---

## 1. Standards & Design Guidelines

### Wayfindr Open Standard (ITU-T F.921) [wayfindr2018]

The Wayfindr Open Standard is an ITU-approved international standard for audio-based indoor navigation for persons with vision impairment. Key design principles relevant to our fine-tuning:

- **"Less is more"**: PVI do not like long or detailed instructions. Trials showed that shorter, concise instructions are preferred because PVI need cognitive bandwidth for their remaining senses.
- **Active verbs required**: Instructions must include action words. "Walk forward to the door" is effective; "The door is in front of you" does not imply action and some users will not move.
- **Piloting/chaining**: PVI commonly divide routes into memorizable segments based on landmarks. The system should present information in segments, not as a complete route dump.
- **Minimal distraction**: PVI pay considerable attention to remaining senses. Technology should assist without overwhelming.
- **Direction communication**: Three methods exist — clock-face (popular with older O&M-trained PVI, but younger PVI find it confusing), cardinal coordinates (difficult because PVI must translate to egocentric first), and body-centered/egocentric (most universally understood). There is no single preferred method; preferences depend on O&M training, travel experience, and familiarity with metaphors.

**Relevance to fine-tuning**: Train model to default to egocentric body-centered directions, use active verbs, keep responses short, and describe routes as landmark-anchored segments.

---

## 2. Spatial Cognition & Representation

### Fortin et al. (2008) — Wayfinding in the Blind [fortin2008]

Study of 38 participants (early blind, late blind, sighted controls) found that blind individuals possess superior navigational skills on route learning tasks and show increased hippocampal volume. Blind people build effective spatial representations through non-visual modalities.

**Relevance**: PVI are capable navigators — the system should provide useful spatial information, not over-simplify. They can build mental maps from verbal descriptions.

### Hersh & Ramirez (2022) — Route Descriptions and Spatial Knowledge of BPS People [hersh2022]

Analysis of blind and partially sighted people's route descriptions across two routes with different characteristics. Key findings:
- BPS people prefer **egocentric and route-based representations** over allocentric/survey ones.
- Environmental information is acquired **serially** — cognitive load should be reduced by focusing on essential information.
- Recommendations for electronic travel aids: provide information incrementally, use landmarks, support route segmentation.

**Relevance to fine-tuning**: Train model for progressive disclosure (essential info first, details on request) and landmark-anchored descriptions.

### Jeamwatthanachai et al. (2019) — Indoor Navigation by Blind People [jeamwatthanachai2019]

Interviews with 30 visually impaired people and 15 experts about navigation behaviors in unfamiliar indoor spaces (universities, hospitals, malls, museums). Key findings:
- Indoor navigation is the second most important feature after outdoor navigation.
- Key needs: obstacle detection, destination finding (rooms, staircases, elevators).
- PVI develop strategies based on tactile cues, sound reflections, and verbal descriptions from others.
- Unfamiliar spaces are significantly more challenging — prior verbal description of the space helps.

**Relevance to fine-tuning**: The model should proactively describe the environment (zone layout, nearby objects) when users enter unfamiliar areas, and use sensory-relevant descriptions.

---

## 3. LLM-Based Assistive Navigation Systems

### Song et al. (2024) — Guide-LLM [song2024guidellm]

(Our prior work.) Embodied LLM agent using a text-based topological map for robotic guidance of PVI. Uses GPT-4 with commonsense reasoning for path planning. The topological map simplifies the environment into straight paths and right-angle turns.

**Limitations addressed in v2**: Cloud-dependent (requires internet), simulation-only evaluation (iGibson), text-only LLM interaction (no structured tool calling), no fine-tuning for PVI-specific language.

### Kaniwa et al. (2024) — ChitChatGuide [kaniwa2024]

LLM-based conversational system for assisting PVI in shopping mall exploration. User study with 11 PVI participants. Key findings:
- PVI valued the LLM's ability to handle **vague and context-based questions** — they don't always know exactly what to ask.
- **Personalized, in-situ descriptions** were both useful and enjoyable.
- Conversational interaction enabled users to explore unfamiliar environments more independently.
- The system **increased enjoyment** of the exploration experience.
- For tour planning, natural contextual conversations helped users without specific destinations.

**Relevance to fine-tuning**: Train model to handle implicit/vague queries ("I'm tired", "what's interesting here?"), provide personalized responses based on current context, and maintain conversational flow.

### Tokmurziyev et al. (2025) — LLM-Glasses [tokmurziyev2025]

Wearable navigation system combining GPT-4o reasoning, YOLO-World object detection, and haptic feedback via glasses. Three user studies: haptic pattern recognition, VICON-based navigation, and LLM-guided video evaluation.

**Key difference from our work**: Uses cloud GPT-4o (latency/privacy concerns), glasses form factor (not robot), haptic output (not verbal). Our system uses local fine-tuned LLM on a mobile robot with verbal output.

---

## 4. Spatial Language & Route Descriptions

### Nicholson & Kulyukin (2010) — Verbal Route Directions for Blind Navigation [nicholson2010]

Analysis of indoor and outdoor route description corpora collected from VI travelers via online surveys. Key findings:
- VI travelers prefer **landmark-based** route descriptions over metric distances.
- Good landmarks for VI: auditory (sounds), tactile (floor texture changes), olfactory (smells), and structural (walls, doorways, corners).
- Route descriptions follow a **segment-landmark-action** pattern: "Walk until you reach [landmark], then [action]."
- Manual rule development on small corpora due to limited annotated data available.

**Relevance to fine-tuning**: Train model to use segment-landmark-action pattern. Describe objects as landmarks with sensory properties, not coordinates.

### Kattenbeck et al. (2019) — Modelling Verbal Indoor Route Descriptions for VI Travellers [kattenbeck2019]

Formal representation of human spatial knowledge in verbal indoor route descriptions. Identifies characteristics and strategies for knowledge extraction from route description corpora. Proposes a graph representation linking landmarks, paths, and actions.

**Relevance to fine-tuning**: The model should structure its descriptions similarly to how humans naturally give route directions to VI people.

---

## 5. Guide Robot Design

### ACM THRI (2025) — Guide Robot Characteristics for BLV Users [guideRobot2025]

Multi-method investigation of guide robot behaviors and features for blind and low-vision users. Key findings:
- Participants had **differing preferences** on the degree of path information — some could infer turns from proprioceptive cues, others wanted explicit verbal or haptic feedback.
- Robot should provide **configurable verbosity** — some users want minimal information, others want detailed descriptions.
- Trust calibration is important — users need to understand what the robot can and cannot do.

**Relevance to fine-tuning**: Consider training the model to adapt verbosity based on user feedback. Include examples where the model asks "Would you like me to describe what's around you?" rather than always dumping information.

---

## Summary: Key Principles for Fine-Tuning Dataset

Based on the literature, the fine-tuning dataset should enforce these principles:

| # | Principle | Source |
|---|---|---|
| 1 | Egocentric body-centered directions by default | Wayfindr, Hersh2022 |
| 2 | Short, concise responses (1-2 sentences) | Wayfindr "less is more" |
| 3 | Active verbs ("walk forward", "head to") | Wayfindr |
| 4 | Landmark-anchored descriptions, not metric | Nicholson2010, Kulyukin |
| 5 | Route segmentation (piloting/chaining) | Wayfindr, Hersh2022 |
| 6 | Progressive disclosure (essentials first) | Hersh2022, cognitive load |
| 7 | Handle vague/implicit queries | ChitChatGuide |
| 8 | Personalized, context-aware responses | ChitChatGuide |
| 9 | Always confirm before robot movement | Guide-LLM v1 |
| 10 | No visual assumptions ("you can see") | Accessibility best practice |
| 11 | Natural distance language (not metres) | Wayfindr, Nicholson2010 |
| 12 | Adaptable verbosity | THRI Guide Robot study |

---

## 6. Tool Calling Evaluation Benchmarks

### BFCL — Berkeley Function Calling Leaderboard [patil2025bfcl]

The gold standard for LLM tool calling evaluation, published at ICML 2025.
Uses **Abstract Syntax Tree (AST) matching** to compare predicted function
calls against ground truth — checks function name, parameter names, types,
and values. Evaluates 2000 question-answer pairs across serial, parallel,
and multi-turn function calls. Introduces **Relevance Detection**: given an
irrelevant query, does the model correctly NOT call any tool?

**Metrics adopted**: AST Accuracy (our Tool Selection Accuracy + Argument
Correctness), Relevance Detection (our Rejection Accuracy), Overall Accuracy.

### ToolTalk [farn2024tooltalk]

Microsoft's multi-turn conversational tool benchmark, NeurIPS 2024. Uses 28
tools grouped into 7 plugins with fully simulated tool execution.

**Key metrics**:
- **Tool Call Precision**: of the tools called, how many were correct?
- **Tool Call Recall**: of the tools that should have been called, how many
  were found?
- **Incorrect Tool Call Rate**: how often was a wrong tool called?

Evaluates at the **action level** (each tool call scored independently).

**Metrics adopted**: Precision, Recall, F1 (harmonic mean).

### ToolSandbox [lu2025toolsandbox]

Apple's stateful, conversational benchmark, NAACL 2025. Includes implicit
state dependencies between tools, trajectory matching via AST, and
**milestone scoring** for intermediate checkpoints.

**Key innovation**: Verifies actual system state after execution, not just
the sequence of calls. Uses geometric mean across sub-metrics.

**Relevance to our work**: Our memory tools (remember/recall_location,
where_was_i) create stateful dependencies similar to ToolSandbox's design.

### CONFETTI [alkhouli2025confetti]

Amazon's conversational function-calling benchmark. 109 conversations,
313 user turns, 86 APIs. Performs **turn-level evaluation** targeting:
follow-ups, goal correction/switching, ambiguous goals, implicit goals.

**Relevance to our work**: PVI users often have ambiguous/implicit goals
("I'm tired", "somewhere warm") — exactly the conversational complexities
CONFETTI targets.

### Tool Hallucination Metrics [zhao2024toolhallucination]

Categorizes tool hallucinations into two types:
- **Tool Selection Hallucination (TSH)**: model picks a tool that shouldn't
  be used (e.g., navigates without confirmation)
- **Tool Usage Hallucination / Parameter Hallucination Rate (PHR)**: model
  fabricates parameter values not present in the query or system prompt
  (e.g., invents a zone name instead of calling list_zones first)

**Metrics adopted**: PHR for measuring argument fabrication.

---

## Summary: Evaluation Metrics (Grounded in Benchmarks)

| Our Metric | Definition | Source Benchmark |
|---|---|---|
| Tool Selection Accuracy (TSA) | Correct tool / total cases | BFCL AST Accuracy |
| Tool Precision | Correct calls / all calls made | ToolTalk |
| Tool Recall | Correct calls / expected calls | ToolTalk |
| Tool F1 | Harmonic mean of precision and recall | ToolTalk |
| Argument Correctness (AC) | Correct args / cases with expected args | BFCL AST parameter matching |
| Parameter Hallucination Rate (PHR) | Fabricated values / total values | Zhao2024 |
| Rejection Accuracy (RA) | Correctly avoided rejected tools | BFCL Relevance Detection |
| Confusion Rate (CR) | Wrong tool in confusing pair | Novel (domain-specific) |
| Turn-Level Consistency | Accuracy at turn N | CONFETTI turn-level eval |
| Latency | Seconds to final response | BFCL |
