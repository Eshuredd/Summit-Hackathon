"""Synthetic one-step rules and observe/plan/act runner contracts."""

from copy import deepcopy

import pytest

from bimind.planner import next_action, object_at_target
from planner_drawer_task import run_task


@pytest.fixture
def state():
    """Return a closed-drawer observation independent of workflow history."""
    return {
        "failure": None,
        "instruction": {"task": "drawer_to_table"},
        "holding": {"drawer_monitor_active": False, "placement_confirmed": False},
        "drawer": {"visual_state": "closed", "confidence": 0.9},
        "arms": {"left": {"holding_objects": []}, "right": {"holding_objects": []}},
        "objects": {"drawer_object": {"position": [0.22, -0.11, 0.031]}},
        "targets": {"table_target": {"position": [0.2, 0.08, 0.015]}},
    }


def open_drawer(state, held=False):
    """Prepare an open drawer with optional measured left support."""
    state["drawer"]["visual_state"] = "open"
    state["holding"]["drawer_monitor_active"] = held
    state["arms"]["left"]["holding_objects"] = ["drawer_handle"] if held else []


def at_target(state):
    """Place the synthetic cube at the actual target coordinates."""
    state["holding"]["placement_confirmed"] = True
    state["objects"]["drawer_object"]["position"] = list(
        state["targets"]["table_target"]["position"]
    )


def test_closed_drawer(state):
    """A closed drawer requests exactly one opening action."""
    assert next_action(state) == {
        "skill": "open_drawer",
        "args": ["left"],
        "reason": "drawer closed",
    }


@pytest.mark.parametrize("contact,monitor", [(False, False), (True, False), (False, True)])
def test_open_drawer_without_maintained_hold(state, contact, monitor):
    """Both actual handle support and the active monitor are required before picking."""
    open_drawer(state)
    state["holding"]["drawer_monitor_active"] = monitor
    state["arms"]["left"]["holding_objects"] = ["drawer_handle"] if contact else []
    assert next_action(state)["skill"] == "hold_drawer"


def test_held_drawer(state):
    """Measured left support permits the right pick."""
    open_drawer(state, held=True)
    action = next_action(state)
    assert action["skill"] == "pick_object" and action["args"] == ["right", "drawer_object"]


@pytest.mark.parametrize("already_at_target", [False, True])
def test_object_held_by_right(state, already_at_target):
    """A held object must be placed/released even when its coordinates match the target."""
    open_drawer(state, held=True)
    state["arms"]["right"]["holding_objects"] = ["drawer_object"]
    if already_at_target:
        at_target(state)
    assert next_action(state)["skill"] == "place_object"


def test_placed_object_releases_drawer_before_any_repick(state):
    """Completion predicates take precedence over the right hand being empty."""
    open_drawer(state, held=True)
    at_target(state)
    assert next_action(state)["skill"] == "release_drawer"


@pytest.mark.parametrize("drawer_stays_open", [True, False])
def test_placed_and_released_finishes(state, drawer_stays_open):
    """A completed placement must never trigger another opening or holding cycle."""
    if drawer_stays_open:
        open_drawer(state)
    at_target(state)
    assert next_action(state)["skill"] == "finish"


def test_failure_has_highest_priority():
    """A failure-only observation needs no other fields to stop motion planning."""
    result = next_action({"failure": {"phase": "RIGHT_GRASP_OBJECT", "reason": "lost contact"}})
    assert result["skill"] == "STOP" and result["args"] == []
    assert "lost contact" in result["reason"]


@pytest.mark.parametrize(
    "xy,z,expected",
    [
        (0.029, 0.004, True),
        (0.031, 0.0, False),
        (0.0, 0.006, False),
        (0.0, 0.08, False),
    ],
)
def test_target_uses_measured_xyz(state, xy, z, expected):
    """The position check must reject XY misses and a cube hovering over the marker."""
    state["targets"]["table_target"]["position"] = [0.0, 0.0, 0.015]
    state["objects"]["drawer_object"]["position"] = [xy, 0.0, 0.015 + z]
    assert object_at_target(state) == expected


def test_exact_xy_boundary_is_not_placed(state):
    """Thirty millimetres itself is outside the existing strict placement tolerance."""
    state["targets"]["table_target"]["position"] = [0, 0, 0]
    state["objects"]["drawer_object"]["position"] = [0.03, 0, 0]
    assert not object_at_target(state)


def test_history_cannot_override_observation(state):
    """Stale SUCCESS or workflow stages must not substitute for measured placement."""
    open_drawer(state, held=True)
    expected = next_action(state)
    for stage in ("new", "picked", "placed", "released", "complete"):
        state.update(execution_stage=stage, phase="SUCCESS")
        assert next_action(state) == expected


def test_planner_is_stateless_and_does_not_mutate(state):
    """Repeated planning is pure and does not advance an internal task index."""
    before = deepcopy(state)
    assert next_action(state) == next_action(state)
    assert state == before
    open_drawer(state, held=True)
    assert next_action(state)["skill"] == "pick_object"
    assert next_action(before)["skill"] == "open_drawer"


@pytest.mark.parametrize("position", [[float("nan"), 0, 0], [0, float("inf"), 0], [1, 2]])
def test_privileged_coordinates_are_ignored(state, position):
    """Privileged coordinates are no longer planner features."""
    state["objects"]["drawer_object"]["position"] = position
    assert next_action(state)["skill"] == "open_drawer"


def test_partial_drawer_and_collision_stop(state):
    """Do not invent unvalidated recovery paths or move after a reported collision."""
    state["drawer"]["visual_state"] = "partial"
    assert next_action(state)["skill"] == "STOP"
    state["drawer"]["visual_state"] = "open"
    state["failure"] = "arm-arm collision reported"
    assert next_action(state)["skill"] == "STOP"


class FakeSkills:
    """Expose observable state changes for runner tests without any simulated motion."""

    def __init__(self, state, stalled=False, fail=False):
        """Keep a private synthetic observation and record each API call."""
        self.state = deepcopy(state)
        self.calls = []
        self.observations = 0
        self.stalled = stalled
        self.fail = fail
        self.done = False

    def reset(self):
        """Represent successful session initialization."""
        return {"success": True}

    def get_scene_state(self):
        """Return a new snapshot, not the prior skill result's state."""
        self.observations += 1
        return deepcopy(self.state)

    def open_drawer(self, arm):
        """Allow tests to inject a changed goal state, a stall, or an explicit failure."""
        self.calls.append(("open_drawer", arm))
        if self.fail:
            self.state["failure"] = {"reason": "contact failed"}
            return {"success": False, "reason": "contact failed"}
        if not self.stalled:
            # The next observation already satisfies the goal. A hardcoded runner
            # would incorrectly continue with hold/pick/place.
            open_drawer(self.state)
            at_target(self.state)
        return {"success": True, "state": {"deliberately": "not a usable observation"}}

    def finish(self):
        """Record goal validation without requiring additional manipulation calls."""
        self.calls.append(("finish",))
        self.done = True
        return {"success": True}

    def get_run_report(self):
        """Return enough of the real reporting contract to inspect runner behavior."""
        return {"success": self.done, "failure": None if self.done else "incomplete"}


def test_runner_reobserves_and_replans(state):
    """Unexpected fresh observations must change the next selected skill."""
    skills = FakeSkills(state)
    report = run_task(skills)
    assert report["success"]
    assert skills.calls == [("open_drawer", "left"), ("finish",)]
    assert skills.observations == 2
    assert [row["action"]["skill"] for row in report["planner_trace"]] == ["open_drawer", "finish"]


def test_runner_bounds_nonprogressing_loop(state):
    """No state progress must produce a failed report after the bounded decision count."""
    skills = FakeSkills(state, stalled=True)
    result = run_task(skills, max_steps=3)
    assert not result["success"] and len(skills.calls) == 3
    assert "exceeded 3" in result["planner_stop_reason"]


def test_runner_stops_after_skill_failure(state):
    """A failed skill is observed and planned as STOP without another actuator call."""
    skills = FakeSkills(state, fail=True)
    result = run_task(skills)
    assert not result["success"] and len(skills.calls) == 1
    assert result["planner_trace"][-1]["action"]["skill"] == "STOP"


@pytest.fixture(autouse=True)
def fake_camera(monkeypatch):
    """Supply explicit synthetic camera observations for runner unit tests."""
    monkeypatch.setattr(
        "planner_drawer_task.observe",
        lambda skills, instruction, placed: (None, skills.get_scene_state()),
    )
