"""Negative checks for drawer safety and unchanged dual-arm collision geometry."""

import xml.etree.ElementTree as ET

import numpy as np

from bimind.controllers import DrawerController as DrawerTask
from dual_pick_place import ASSETS


def test_scene_integrity():
    """Verify robot geometry is copied intact and the drawer is entirely passive."""
    original = ET.parse(ASSETS / "dual_scene.xml").getroot()
    drawer = ET.parse(ASSETS / "drawer_scene.xml").getroot()
    for arm in ("left", "right"):
        source = original.find(f"worldbody/body[@name='{arm}_base']")
        target = drawer.find(f"worldbody/body[@name='{arm}_base']")
        assert [(e.tag, e.attrib) for e in source.iter()] == [
            (e.tag, e.attrib) for e in target.iter()
        ]
    task = DrawerTask()
    assert task.model.neq == 0
    assert task.model.nu == 12
    assert task.opening() == 0
    assert np.allclose(task.model.joint("drawer_slide").range, [0, 0.06])
    assert not task.supports("left") and not task.supports("right")


def test_missing_support():
    """An untouched handle must never count as a grasp."""
    task = DrawerTask()
    try:
        task.tick(("left",))
    except RuntimeError as error:
        assert "Lost left" in str(error)
    else:
        raise AssertionError("Missing handle support was accepted")


def test_closed_drawer_guard():
    """Retrieval must reject a closed drawer even without a requested support check."""
    task = DrawerTask()
    task.maintain = True
    task.min_held_open = 0.06
    try:
        task.tick()
    except RuntimeError as error:
        assert "maintain drawer" in str(error)
    else:
        raise AssertionError("Closed drawer was accepted during retrieval")


def test_drop_guard():
    """A 20 mm unintended descent must fail the transport guard."""
    task = DrawerTask()
    task.transport = True
    task.transport_peak = task.position()[2] + 0.02
    try:
        task.tick()
    except RuntimeError as error:
        assert "dropped more than" in str(error)
    else:
        raise AssertionError("Transport drop was accepted")


if __name__ == "__main__":
    test_scene_integrity()
    test_missing_support()
    test_closed_drawer_guard()
    test_drop_guard()
    print("Drawer scene integrity and negative guard checks passed.")
