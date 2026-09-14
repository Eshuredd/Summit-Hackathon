"""Regression checks separating window close from manipulation failure."""

import importlib
import json
import sys
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from bimanual_handoff import Handoff
from bimind.controllers import DrawerController
from bimind.skills import Skills
from viewer_lifecycle import ViewerClosed


class FakeViewer:
    """A passive-viewer test double requiring no desktop window."""

    def __init__(self, running=False, close_on_sync=False):
        """Model a closed viewer or one that closes during its next sync."""
        self.running = running
        self.close_on_sync = close_on_sync
        self.closes = 0
        self.opt = SimpleNamespace(geomgroup=[True] * 6)
        self.cam = SimpleNamespace(lookat=np.zeros(3))

    def is_running(self):
        """Return whether the user has left this viewer open."""
        return self.running

    def sync(self):
        """Optionally emulate clicking X during synchronization."""
        if self.close_on_sync:
            self.running = False

    def close(self):
        """Record explicit resource cleanup."""
        self.closes += 1
        self.running = False


@pytest.mark.parametrize("controller_type", [DrawerController, Handoff])
def test_closed_viewer_does_not_step_or_change_controls(controller_type):
    """Termination must be detected before physics or the next actuator update."""
    controller = controller_type()
    controller.viewer = FakeViewer()
    before = controller.data.qpos.copy(), controller.data.ctrl.copy(), controller.data.time
    with pytest.raises(ViewerClosed):
        controller.tick()
    with pytest.raises(ViewerClosed):
        controller.move("left", np.ones(6) * 0.1)
    np.testing.assert_array_equal(controller.data.qpos, before[0])
    np.testing.assert_array_equal(controller.data.ctrl, before[1])
    assert controller.data.time == before[2]


def test_skills_user_close_is_not_failure():
    """A mid-skill close terminates once, without a failure latch or any later motion."""
    skills = Skills()
    viewer = FakeViewer(running=True, close_on_sync=True)
    skills._controller.viewer = viewer
    result = skills.reset()
    assert result["terminated"] and not result["success"] and result["reason"] is None
    assert result["state"]["failure"] is None
    now = skills._controller.data.time
    assert skills.open_drawer("left")["terminated"]
    assert skills._controller.data.time == now
    report = skills.get_run_report()
    assert report["terminated"] and report["failure"] is None and not report["success"]
    skills.close()
    assert viewer.closes == 1


@pytest.mark.parametrize(
    "failure",
    [
        "Lost left opposing support",
        "Arm-arm collision",
        "Object dropped more than 10 mm",
        "Planner path intersects",
    ],
)
def test_physical_failure_stays_failure_even_when_viewer_closes(monkeypatch, failure):
    """A simultaneous window close cannot erase an already detected physical failure."""
    skills = Skills()
    viewer = FakeViewer(running=True)
    skills._controller.viewer = viewer

    def fail():
        """Raise the controller failure while emulating a concurrent close."""
        viewer.running = False
        raise RuntimeError(failure)

    monkeypatch.setattr(skills._controller, "reset_sequence", fail)
    result = skills.reset()
    assert not result["success"] and not result.get("terminated")
    assert result["reason"] == failure
    report = skills.get_run_report()
    assert report["failure"] == failure and not report.get("terminated")
    assert not skills.move_home("left").get("terminated")
    skills.close()


@pytest.mark.parametrize(
    "module_name,extra",
    [
        ("bimanual_drawer_task", []),
        ("planner_drawer_task", []),
        ("bimanual_handoff", []),
        ("dual_pick_place", ["--arm", "left"]),
    ],
)
def test_interactive_main_never_restarts_on_close(monkeypatch, module_name, extra):
    """Even --runs 10 must launch one viewer, clean up, and return normally on X."""
    module = importlib.import_module(module_name)
    viewers, reports = [], []

    def launch(*args, **kwargs):
        """Return an already-closed window and count attempted launches."""
        viewer = FakeViewer()
        viewers.append(viewer)
        return viewer

    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(
        module.Path, "write_text", lambda path, text: reports.append(json.loads(text))
    )
    monkeypatch.setattr(sys, "argv", [module_name, "--viewer", "--runs", "10", *extra])
    module.main()
    assert len(viewers) == 1 and viewers[0].closes == 1
    rows = reports[0] if isinstance(reports[0], list) else reports[0]["results"]
    assert len(rows) == 1 and rows[0]["terminated"]
    assert rows[0]["failure"] is None and not rows[0]["success"]


@pytest.mark.parametrize(
    "module_name",
    ["bimanual_drawer_task", "planner_drawer_task", "bimanual_handoff", "dual_pick_place"],
)
def test_interactive_main_preserves_failure_exit(monkeypatch, module_name):
    """Physical errors must still exit nonzero and release viewer resources."""
    module = importlib.import_module(module_name)
    viewers = []

    def launch(*args, **kwargs):
        """Return an open viewer for the injected manipulation failure."""
        viewer = FakeViewer(running=True)
        viewers.append(viewer)
        return viewer

    def fail(*args):
        """Represent a genuine trajectory failure without special cancellation semantics."""
        raise RuntimeError("Lost grasp")

    if module_name == "bimanual_handoff":
        monkeypatch.setattr(Handoff, "run", fail)
    elif module_name == "dual_pick_place":

        def sequence(*args):
            """Raise from the same generator boundary as the physical controller."""
            yield from ()
            fail()

        monkeypatch.setattr(module.PickPlace, "sequence", sequence)
    else:
        monkeypatch.setattr(DrawerController, "reset_sequence", fail)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(module.Path, "write_text", lambda *args: None)
    extra = ["--arm", "left"] if module_name == "dual_pick_place" else []
    monkeypatch.setattr(sys, "argv", [module_name, "--viewer", "--runs", "10", *extra])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    assert len(viewers) == 1 and viewers[0].closes == 1
