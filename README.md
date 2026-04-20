# Guide-LLM v2 — Go2 Robot Deployment

Fine-tuned Qwen 3.5 2B running locally on the Go2 robot via Ollama for vision-impaired navigation assistance.

> Based on [Guide-LLM v1](https://arxiv.org/abs/2410.20666) by Sangmim Song et al.

## Quick Start

```bash
# 1. Clone this branch
git clone -b go2-minimal https://github.com/awd1779/guide_llm_2.git
cd guide_llm_2

# 2. Setup (installs Ollama, downloads model from HuggingFace, imports)
bash setup_ollama.sh

# 3. Run
ollama serve &
python3 scene_graph_agent_local.py --model guide-llm-2b-q8
```

## What's in this branch

```
guide_llm_2/
├── setup_ollama.sh                # One-command setup (Ollama + model download)
├── models/
│   └── Modelfile-2b-q8            # Ollama model config (Qwen chat template)
├── scene_graph_agent_local.py     # Navigation agent (ROS2 + Ollama)
├── dummy_scene_graph_publisher.py # Mock scene graph for testing without robot
├── data/
│   ├── maps/level12v6.*           # Occupancy grid map for Nav2
│   └── scene_graph/               # Scene graph + zone definitions
├── package.xml                    # ROS2 package manifest
├── setup.py / setup.cfg           # ROS2 package config
└── README.md
```

## Model

| Property | Value |
|----------|-------|
| Base model | Qwen 3.5 2B |
| Fine-tuning | LoRA on 1,563 tool-calling examples (8 scenario types) |
| Quantization | Q8_0 GGUF (1.9 GB) |
| HuggingFace | [SMSong/guide-llm-2b-q8](https://huggingface.co/SMSong/guide-llm-2b-q8) |
| Inference | Ollama (~3s per response on A10G) |

### Evaluation Results (2B)

| Metric | Score |
|--------|-------|
| Tool Selection Accuracy | 89.4% |
| Sequence Accuracy | 79.8% |
| Argument Accuracy | 100% |
| End-to-End Completion | 86.7% |

## Running on the Go2 Robot

### Prerequisites

- Go2 Docker container running (with ROS2 Humble, Nav2)
- Jetson Orin with NVIDIA GPU access
- Network connection for initial model download (~1.9 GB)

### Inside the Docker container

```bash
# Terminal 1: Start robot systems
go2run

# Terminal 2: Start scene graph publisher
go2_scenegraph_pub

# Terminal 3: Start Ollama + agent
ollama serve &
python3 scene_graph_agent_local.py --model guide-llm-2b-q8
```

### Testing without robot

```bash
# Terminal 1: Mock scene graph
python3 dummy_scene_graph_publisher.py

# Terminal 2: Agent
python3 scene_graph_agent_local.py --model guide-llm-2b-q8
```

## How It Works

```
User speech → STT → agent → Ollama (Qwen 3.5 2B) → tool calls → ROS2 → Nav2 → robot moves
                                                   ↓
                                            tool responses
                                                   ↓
                                          natural language → TTS → spoken to user
```

The agent bypasses Ollama's native tool calling (broken for Qwen 3.5) by:
1. Injecting tool definitions into the system prompt (Qwen chat template format)
2. Parsing XML tool calls (`<function=name><parameter=key>value</parameter>`) from raw text
3. Feeding tool responses as `<tool_response>` messages

### Available Tools (11)

| Tool | Purpose |
|------|---------|
| `get_robot_pose` | Current position and zone |
| `describe_surroundings` | Nearby objects with distance and direction |
| `list_all` | All zones and objects |
| `find_nearest` | Closest instance of an object type |
| `distance_to` | Distance and direction to a destination |
| `navigate_to` | Start navigation to a destination |
| `orient_me` | Rotate to face a target |
| `describe_route` | Preview route to destination |
| `get_navigation_status` | Check navigation progress |
| `cancel_navigation` | Stop navigation |
| `replan_route` | Recalculate route |

### Tool Call Flow

```
User: "Take me to the kitchen"

Agent → list_all()
Tool  → {"zones": [{"name": "hri_lab"}, {"name": "kitchen"}, ...]}

Agent → distance_to("kitchen")
Tool  → {"destination": "kitchen", "distance_m": 8.45, "direction": "ahead"}

Agent → "The kitchen is a bit further away, about 8 metres ahead. Shall I take you there?"
User  → "Yes"

Agent → navigate_to("kitchen")
Tool  → {"status": "goal_sent", "destination": "kitchen"}

Agent → "Heading to the kitchen now, follow me!"
```

## ROS2 Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/scene_graph/objects` | String | Object data from scene graph |
| `/scene_graph/zones` | String | Zone data from scene graph |
| `/goal_pose` | PoseStamped | Navigation goal for Nav2 |
| `/navigate_to_pose/_action/status` | GoalStatusArray | Nav2 status feedback |
| `/speech_feedback` | String | Agent responses (for TTS) |
| `/transcribed_utterances_guide_llm` | String | Voice input (from STT) |
| TF: `map` → `base_link` | — | Robot pose from SLAM |

## CLI Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `guide-llm-2b-q8` | Ollama model name |
| `--base-url` | `http://localhost:11434/v1` | Ollama API endpoint |
| `--frame-id` | `map` | TF frame for navigation goals |
| `--base-frame` | `base_link` | Robot base TF frame |

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
