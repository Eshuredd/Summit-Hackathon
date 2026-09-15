"""Build the minimal planner boundary from RGB and robot feedback."""

from copy import deepcopy

from bimind.instruction import parse_instruction
from bimind.perception import estimate_drawer_state, render_camera


def build_observation(instruction, perception, robot_state, placement_confirmed=False):
    """Allowlist inference features; placement is a successful skill acknowledgement.

    Args:
        instruction: Original natural-language command.
        perception: Image-only estimator output.
        robot_state: Existing Skills feedback; scene truth is excluded.
        placement_confirmed: Whether this run acknowledged a successful place skill.

    Returns:
        Minimal planner observation without object poses or drawer truth.
    """
    collisions = robot_state.get("arm_arm_collisions", {})
    failure = robot_state.get("failure")
    if collisions.get("active") or collisions.get("total_count", 0):
        failure = "arm-arm collision reported"
    return {
        "instruction": parse_instruction(instruction),
        "drawer": {"visual_state": perception["state"], "confidence": perception["confidence"]},
        "arms": {
            arm: {
                key: deepcopy(values[key])
                for key in ("joint_positions", "gripper", "holding_objects")
                if key in values
            }
            for arm, values in robot_state["arms"].items()
        },
        "holding": {
            "drawer_monitor_active": robot_state["drawer"]["hold_monitor_active"],
            "placement_confirmed": placement_confirmed,
        },
        "failure": failure,
    }


def observe(skills, instruction, placement_confirmed=False):
    """Render the fixed RGB camera and build a fresh inference observation.

    Args:
        skills: Existing Skills session.
        instruction: Original natural-language command.
        placement_confirmed: Successful placement acknowledgement for this run.

    Returns:
        RGB frame and the corresponding minimal planner observation.
    """
    controller = skills._controller
    rgb = render_camera(controller.model, controller.data)
    perception = estimate_drawer_state(rgb)
    return rgb, build_observation(
        instruction, perception, skills.get_scene_state(), placement_confirmed
    )
