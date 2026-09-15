"""Shared small multimodal input encoding for training and OpenVINO inference."""

import re

import numpy as np
from PIL import Image

from bimind.instruction import parse_instruction

LABELS = (
    "open_drawer",
    "hold_drawer",
    "pick_object",
    "place_object",
    "release_drawer",
    "finish",
    "stop",
)
VOCABULARY = tuple(
    "open the drawer and place object on table retrieve utensil from take put it".split()
)
INSTRUCTIONS = (
    "Open the drawer and place the object on the table.",
    "Retrieve the utensil from the drawer.",
    "Take the object from the drawer and put it on the table.",
)
IMAGE_SIZE = 96
ROBOT_FEATURES = (
    "left_holds_handle",
    "right_holds_object",
    "left_opening_commanded",
    "right_opening_commanded",
    "drawer_monitor_active",
    "placement_confirmed",
)


def encode_inputs(rgb, instruction, robot_state):
    """Encode only RGB, command tokens, and six existing robot feedback flags.

    Args:
        rgb: HWC uint8 RGB camera image.
        instruction: Supported natural-language command.
        robot_state: Observation containing arms and holding fields.

    Returns:
        Named float32 batch-one tensors matching the exported model.

    Raises:
        ValueError: An image, instruction, or feedback flag is invalid.
        KeyError: Required robot feedback is absent.
    """
    parse_instruction(instruction)
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) < 1
    ):
        raise ValueError("RGB must be a nonempty HWC uint8 image")
    resized = Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    image = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    tokens = re.findall(r"[a-z]+", instruction.lower())
    text = np.array([tokens.count(word) for word in VOCABULARY], dtype=np.float32)
    text /= max(float(text.sum()), 1.0)
    left, right = (robot_state["arms"][arm] for arm in ("left", "right"))
    for arm in (left, right):
        if not isinstance(arm["holding_objects"], list):
            raise ValueError("holding_objects must be a list")
    flags = [
        "drawer_handle" in left["holding_objects"],
        "drawer_object" in right["holding_objects"],
        left["gripper"]["opening_commanded"],
        right["gripper"]["opening_commanded"],
        robot_state["holding"]["drawer_monitor_active"],
        robot_state["holding"]["placement_confirmed"],
    ]
    if any(not isinstance(flag, bool) for flag in flags):
        raise ValueError("Robot feedback flags must be booleans")
    return {
        "rgb": np.ascontiguousarray(image[None]),
        "instruction": text[None],
        "robot_state": np.array([flags], dtype=np.float32),
    }
