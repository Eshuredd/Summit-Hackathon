"""Instruction and inference-boundary safety contracts."""

import pytest

from bimind.instruction import parse_instruction
from bimind.observation import build_observation
from bimind.planner import next_action


@pytest.mark.parametrize(
    "command",
    [
        "Open the drawer and place the object on the table.",
        "Retrieve the utensil from the drawer.",
        "Take the object from the drawer and put it on the table.",
    ],
)
def test_supported(command):
    """All requested variants normalize identically."""
    assert parse_instruction(command) == {"task": "drawer_to_table"}


@pytest.mark.parametrize(
    "command",
    [
        "close the drawer",
        "",
        "retrieve the knife",
        "Retrieve the utensil from the drawer and throw it.",
        None,
    ],
)
def test_unsupported(command):
    """Reject unsupported commands in full rather than matching substrings."""
    with pytest.raises(ValueError):
        parse_instruction(command)


def observation(visual="closed", confidence=0.9):
    """Build a robot snapshot containing deliberately conflicting truth."""
    return build_observation(
        "Retrieve the utensil from the drawer.",
        {"state": visual, "confidence": confidence},
        {
            "drawer": {
                "is_open": True,
                "is_closed": False,
                "position": 0.06,
                "hold_monitor_active": False,
            },
            "arms": {
                arm: {"holding_objects": [], "joint_positions": [0] * 6, "gripper": {"position": 0}}
                for arm in ("left", "right")
            },
            "objects": {"secret": "truth"},
            "failure": None,
        },
    )


def test_boundary_and_conflicting_truth():
    """No scene truth is forwarded and visual closed wins over privileged open."""
    state = observation()
    assert set(state) == {"instruction", "drawer", "arms", "holding", "failure"}
    assert set(state["drawer"]) == {"visual_state", "confidence"}
    state["drawer"].update(is_open=True, is_closed=False)
    assert next_action(state)["skill"] == "open_drawer"


@pytest.mark.parametrize(
    "visual,confidence",
    [
        ("partial", 1.0),
        ("unknown", 1.0),
        ("closed", 0.64),
        ("open", float("nan")),
        ("open", -1),
        ("open", True),
    ],
)
def test_uncertainty_stops_even_after_placement(visual, confidence):
    """Uncertain RGB never permits release or finish either."""
    state = observation(visual, confidence)
    state["holding"]["placement_confirmed"] = True
    assert next_action(state)["skill"] == "STOP"


def test_configurable_threshold():
    """Apply inclusive confidence gating with no privileged fallback."""
    state = observation(confidence=0.65)
    assert next_action(state)["skill"] == "open_drawer"
    assert next_action(state, 0.66)["skill"] == "STOP"


@pytest.mark.parametrize("split", [True, False])
def test_occluded_housing_requires_multiple_visible_fragments(split):
    """A split housing can be combined; one insufficient fragment stays unknown."""
    import numpy as np

    from bimind.perception import _select_housing

    mask = np.zeros((480, 640), dtype=bool)
    mask[180:231, 321:383] = True
    if split:
        mask[192:241, 264:285] = True
    result = _select_housing(mask, (274, 220, 432, 306, 7048))
    if split:
        assert result[:4] == (264, 180, 382, 240)
    else:
        assert result is None
