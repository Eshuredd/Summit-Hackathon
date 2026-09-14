"""Skill contracts, failure containment, and read-only observation regressions."""

import json

import numpy as np
import pytest

from bimind.skills import Skills


def test_scene_state_is_measured_and_read_only():
    """Observation must not step physics or infer ownership from closed commands."""
    skills = Skills()
    c = skills._controller
    qpos, qvel, ctrl, now = c.data.qpos.copy(), c.data.qvel.copy(), c.data.ctrl.copy(), c.data.time
    state = skills.get_scene_state()
    json.dumps(state, allow_nan=False)
    assert state["drawer"]["position"] == 0
    assert state["drawer"]["is_closed"] and not state["drawer"]["is_open"]
    assert state["objects"]["drawer_object"]["position"] == [0.22, -0.11, 0.031]
    assert state["targets"]["table_target"]["position"] == [0.2, 0.08, 0.015]
    assert not state["arms"]["left"]["holding"] and not state["arms"]["right"]["holding"]
    assert not state["arm_arm_collisions"]["active"]
    np.testing.assert_array_equal(c.data.qpos, qpos)
    np.testing.assert_array_equal(c.data.qvel, qvel)
    np.testing.assert_array_equal(c.data.ctrl, ctrl)
    assert c.data.time == now


@pytest.mark.parametrize(
    "method,args",
    [
        ("open_drawer", ("right",)),
        ("pick_object", ("right", "missing")),
        ("place_object", ("right", "missing")),
        ("lift_object", ("right", "drawer_object")),
        ("release_drawer", ("left",)),
        ("move_home", ("third",)),
        ("handoff_object", ("left", "right", "drawer_object")),
        ("finish", ()),
    ],
)
def test_invalid_requests_return_failure_without_motion(method, args):
    """Unsupported roles, names, and ordering must not invoke a trajectory."""
    skills = Skills()
    assert skills.reset()["success"]
    c = skills._controller
    before = c.data.qpos.copy(), c.data.ctrl.copy(), c.data.time
    result = getattr(skills, method)(*args)
    assert not result["success"] and result["skill"] == method
    assert result["reason"] and result["phase"] == "RESET"
    assert result["state"]["failure"]["skill"] == method
    json.dumps(result, allow_nan=False)
    np.testing.assert_array_equal(c.data.qpos, before[0])
    np.testing.assert_array_equal(c.data.ctrl, before[1])
    assert c.data.time == before[2]


def test_motion_failure_latches_and_preserves_phase(monkeypatch):
    """A controller failure must stop subsequent calls without further simulation."""
    skills = Skills()
    assert skills.reset()["success"]
    c = skills._controller

    def fail():
        """Exercise the existing real missing-contact guard in a specific phase."""
        c.phase = "LEFT_GRASP_HANDLE"
        c.tick(("left",))

    monkeypatch.setattr(c, "open_drawer_sequence", fail)
    failure = skills.open_drawer("left")
    assert not failure["success"]
    assert failure["phase"] == "LEFT_GRASP_HANDLE"
    assert "Lost left" in failure["reason"]
    now = c.data.time
    assert not skills.move_home("right")["success"]
    assert c.data.time == now
    assert skills.get_run_report()["failure"] == failure["reason"]


def test_sessions_are_independent_and_reset_is_not_teleport():
    """Separate sessions must have separate data, and reset cannot replay mid-task."""
    first, second = Skills(), Skills()
    assert first.reset()["success"]
    assert second.get_scene_state()["time"] == 0
    before = first._controller.data.time
    assert not first.reset()["success"]
    assert first._controller.data.time == before
    assert not second.get_scene_state()["failure"]
