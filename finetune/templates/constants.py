"""Constants for template-based dataset generation."""

# =============================================================================
# CANONICAL SYSTEM PROMPT - FIXED, NO VARIATION
# =============================================================================

CANONICAL_SYSTEM_PROMPT = """You are Guide-LLM, a navigation assistant for vision-impaired users. You help users move around indoor spaces using spoken, natural English.

TOOL ORDERING RULES — follow these exactly, every time:
- Known location (zone/object): list_all → distance_to → [ask confirmation] → navigate_to
- Find nearest instance: find_nearest → distance_to → [ask confirmation] → navigate_to
- Describe surroundings: get_robot_pose → describe_surroundings
- Where am I / reorientation: get_robot_pose → orient_me

SPEECH RULES:
- 1-2 sentences max. Spoken English only.
- Distance: <1m "right here", 1-3m "nearby", 3-6m "a short walk away", >6m "a bit further away"
- Never say zone codes, coordinates, or bearings. Use relative directions: ahead, left, right, behind."""

# =============================================================================
# ENVIRONMENT - ZONES AND OBJECTS
# =============================================================================

ZONES = ["hri_lab", "kitchen", "hallway_0", "hallway_1", "hallway_2"]

OBJECTS = ["chair", "tv", "table", "home", "microwave", "sink", "door"]

# Object → zone mapping (which zone objects are found in)
OBJECT_ZONES = {
    "chair": "hri_lab",
    "tv": "hri_lab",
    "table": "hri_lab",
    "home": "hri_lab",  # home base
    "microwave": "kitchen",
    "sink": "kitchen",
    "door": "hallway_0",
}

# Distance mappings for distance_to responses (meters, deterministic)
DISTANCE_MATRIX = {
    ("hri_lab", "kitchen"): 8.45,
    ("hri_lab", "hallway_0"): 2.1,
    ("hri_lab", "hallway_1"): 5.3,
    ("hri_lab", "hallway_2"): 12.8,
    ("kitchen", "hri_lab"): 8.45,
    ("kitchen", "hallway_0"): 10.2,
    ("kitchen", "hallway_1"): 5.9,
    ("kitchen", "hallway_2"): 14.1,
    ("hallway_0", "hri_lab"): 2.1,
    ("hallway_0", "kitchen"): 10.2,
    ("hallway_0", "hallway_1"): 8.5,
    ("hallway_0", "hallway_2"): 16.3,
    ("hallway_1", "hri_lab"): 5.3,
    ("hallway_1", "kitchen"): 5.9,
    ("hallway_1", "hallway_0"): 8.5,
    ("hallway_1", "hallway_2"): 8.0,
    ("hallway_2", "hri_lab"): 12.8,
    ("hallway_2", "kitchen"): 14.1,
    ("hallway_2", "hallway_0"): 16.3,
    ("hallway_2", "hallway_1"): 8.0,
}

# Directions for each origin-destination pair
DIRECTION_MATRIX = {
    ("hri_lab", "kitchen"): "ahead",
    ("hri_lab", "hallway_0"): "ahead-right",
    ("hri_lab", "hallway_1"): "right",
    ("hri_lab", "hallway_2"): "behind-right",
    ("kitchen", "hri_lab"): "behind",
    ("kitchen", "hallway_0"): "left",
    ("kitchen", "hallway_1"): "behind-left",
    ("kitchen", "hallway_2"): "behind",
    ("hallway_0", "hri_lab"): "behind-left",
    ("hallway_0", "kitchen"): "right",
    ("hallway_0", "hallway_1"): "ahead",
    ("hallway_0", "hallway_2"): "ahead",
    ("hallway_1", "hri_lab"): "left",
    ("hallway_1", "kitchen"): "ahead-right",
    ("hallway_1", "hallway_0"): "behind",
    ("hallway_1", "hallway_2"): "ahead",
    ("hallway_2", "hri_lab"): "behind-left",
    ("hallway_2", "kitchen"): "behind",
    ("hallway_2", "hallway_0"): "behind",
    ("hallway_2", "hallway_1"): "behind",
}

# Map distance (meters) to vocabulary
def distance_to_vocab(distance_m):
    """Convert distance in meters to natural language vocabulary."""
    if distance_m < 1.0:
        return "right here"
    elif distance_m < 3.0:
        return "nearby"
    elif distance_m < 6.0:
        return "a short walk away"
    else:
        return "a bit further away"

# =============================================================================
# TOOLS DEFINITION (for Qwen template)
# =============================================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_robot_pose",
            "description": "Get the robot's current position (x, y, yaw) and zone",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_surroundings",
            "description": "Describe objects and areas around the robot within a radius",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius_m": {
                        "type": "number",
                        "description": "Search radius in meters (default 3.0)",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_all",
            "description": "List all available zones and objects in the environment",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query",
            "description": "Query detailed information about a specific zone or object",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Zone or object name to query",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearest",
            "description": "Find the nearest instance of an object type",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_type": {
                        "type": "string",
                        "description": "Type of object to find (e.g., 'chair', 'door')",
                    }
                },
                "required": ["object_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "distance_to",
            "description": "Get distance and direction to a destination (zone or object)",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Name of destination zone or object",
                    }
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_route",
            "description": "Get description of the route to a destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Name of destination zone or object",
                    }
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "orient_me",
            "description": "Rotate robot to face a target location",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Zone or object name to face towards",
                    }
                },
                "required": ["target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_to",
            "description": "Start navigation to a destination (zone or object)",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "type": "string",
                        "description": "Name of destination zone or object",
                    }
                },
                "required": ["destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_navigation_status",
            "description": "Check the status of current navigation",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_navigation",
            "description": "Cancel the current navigation",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replan_route",
            "description": "Replan the route to the current destination",
            "parameters": {
                "type": "object",
                "properties": {
                    "avoid_obstacles": {
                        "type": "boolean",
                        "description": "Whether to recalculate to avoid obstacles",
                    }
                },
                "required": [],
            },
        },
    },
]
