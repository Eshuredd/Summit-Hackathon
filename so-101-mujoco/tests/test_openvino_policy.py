"""Multimodal encoding, OpenVINO inference, and fail-closed dispatch contracts."""

from copy import deepcopy

import numpy as np
import pytest

from bimind.openvino_policy import DEFAULT_MODEL, OpenVINOPolicy
from bimind.policy_features import INSTRUCTIONS, LABELS, VOCABULARY, encode_inputs
from planner_drawer_task import select_action


@pytest.fixture
def state():
    """Provide the six robot flags and a confident camera observation."""
    return {
        "instruction": {"task": "drawer_to_table"},
        "drawer": {"visual_state": "closed", "confidence": 0.9},
        "arms": {
            arm: {"holding_objects": [], "gripper": {"opening_commanded": False}}
            for arm in ("left", "right")
        },
        "holding": {"drawer_monitor_active": False, "placement_confirmed": False},
        "failure": None,
    }


def test_feature_boundary(state):
    """Privileged scene values cannot affect model tensors."""
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    expected = encode_inputs(rgb, INSTRUCTIONS[0], state)
    state.update(objects={"position": [999, 0, 0]}, execution_stage="finish")
    state["drawer"].update(is_open=True, is_closed=False, position=0.06)
    actual = encode_inputs(rgb, INSTRUCTIONS[0], state)
    assert set(actual) == {"rgb", "instruction", "robot_state"}
    assert actual["rgb"].shape == (1, 3, 96, 96)
    assert actual["instruction"].shape == (1, len(VOCABULARY))
    assert actual["robot_state"].shape == (1, 6)
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])
        assert actual[key].dtype == np.float32


def test_raw_instruction_is_encoded(state):
    """Equivalent supported commands keep distinct token representations."""
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    encoded = [encode_inputs(rgb, command, state)["instruction"] for command in INSTRUCTIONS]
    assert not np.array_equal(encoded[0], encoded[1])
    assert not np.array_equal(encoded[1], encoded[2])


@pytest.mark.parametrize("rgb", [None, np.zeros((4, 4)), np.zeros((4, 4, 3))])
def test_invalid_rgb_rejected(state, rgb):
    """Malformed camera inputs cannot silently become valid features."""
    with pytest.raises(ValueError):
        encode_inputs(rgb, INSTRUCTIONS[0], state)


class FakePolicy:
    """Record classifier calls and return a controlled prediction."""

    def __init__(self, skill="place_object", confidence=0.99):
        """Set the controlled model output."""
        self.prediction = {"skill": skill, "confidence": confidence}
        self.calls = 0

    def predict_next_skill(self, rgb, instruction, robot_state):
        """Record an invocation without using the symbolic planner."""
        self.calls += 1
        return self.prediction


def test_learned_action_is_not_replaced_by_symbolic(state, monkeypatch):
    """The model selects the skill; symbolic planning is never called in this mode."""

    def forbidden(*args):
        """Fail if the learned path silently delegates to the teacher."""
        raise AssertionError("symbolic planner was called")

    monkeypatch.setattr("planner_drawer_task.planner.next_action", forbidden)
    policy = FakePolicy()
    action = select_action(None, INSTRUCTIONS[0], state, policy=policy)
    assert action["skill"] == "place_object"
    assert action["args"] == ["right", "table_target"]
    assert policy.calls == 1


@pytest.mark.parametrize(
    "label,confidence",
    [
        ("place_object", 0.64),
        ("place_object", float("nan")),
        ("place_object", True),
        ("invented_skill", 1.0),
        ("stop", 0.99),
    ],
)
def test_bad_predictions_stop(state, label, confidence):
    """Low confidence, invalid outputs, and stop labels never dispatch a skill."""
    action = select_action(None, INSTRUCTIONS[0], state, policy=FakePolicy(label, confidence))
    assert action["skill"] == "STOP" and action["args"] == []


@pytest.mark.parametrize("visual,confidence", [("unknown", 1), ("partial", 1), ("open", 0.5)])
def test_camera_gate_precedes_model(state, visual, confidence):
    """Uncertain drawer perception prevents any model-selected movement."""
    state["drawer"] = {"visual_state": visual, "confidence": confidence}
    policy = FakePolicy()
    assert select_action(None, INSTRUCTIONS[0], state, policy=policy)["skill"] == "STOP"
    assert policy.calls == 0


def test_failure_precedes_model(state):
    """Controller failure never permits another model invocation."""
    state["failure"] = "lost contact"
    policy = FakePolicy()
    assert select_action(None, INSTRUCTIONS[0], state, policy=policy)["skill"] == "STOP"
    assert policy.calls == 0


def test_runtime_low_confidence_and_invalid_input(state):
    """The public prediction API itself enforces confidence and input validation."""
    policy = OpenVINOPolicy.__new__(OpenVINOPolicy)
    policy.confidence_threshold = 0.65
    policy.infer_logits = lambda inputs: np.zeros(len(LABELS), dtype=np.float32)
    rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    assert policy.predict_next_skill(rgb, INSTRUCTIONS[0], state)["skill"] == "stop"
    assert policy.predict_next_skill(rgb, "Close the drawer.", state)["skill"] == "stop"
    bad = deepcopy(state)
    bad["holding"]["placement_confirmed"] = "yes"
    assert policy.predict_next_skill(rgb, INSTRUCTIONS[0], bad)["skill"] == "stop"


def test_exported_openvino_model_matches_collected_normal_steps():
    """Load the actual exported artifact and predict each original baseline step."""
    import json
    from pathlib import Path

    from PIL import Image

    if not DEFAULT_MODEL.exists():
        pytest.skip("Run train_policy.py to create the deployment artifact")
    policy = OpenVINOPolicy()
    root = Path(__file__).resolve().parents[1] / "policy_dataset"
    rows = [json.loads(line) for line in (root / "samples.jsonl").read_text().splitlines()]
    seen = set()
    for row in rows:
        label = row["teacher_next_skill"]
        if label in seen:
            continue
        seen.add(label)
        rgb = np.asarray(Image.open(root / row["rgb"]).convert("RGB"))
        prediction = policy.predict_next_skill(rgb, row["instruction"], row["observation"])
        assert prediction["skill"] == label
    assert seen == set(LABELS)
