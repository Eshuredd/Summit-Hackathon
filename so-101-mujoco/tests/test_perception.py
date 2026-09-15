"""Tests for deterministic RGB drawer perception."""

from __future__ import annotations

import inspect
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bimind.perception import estimate_drawer_state, render_camera

SCENE = Path(__file__).parents[1] / "assets" / "drawer_scene.xml"


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    """Load the drawer scene once for perception tests."""
    return mujoco.MjModel.from_xml_path(str(SCENE))


def render_at(model: mujoco.MjModel, position: float) -> np.ndarray:
    """Render a test image at a privileged ground-truth drawer position."""
    data = mujoco.MjData(model)
    address = model.jnt_qposadr[model.joint("drawer_slide").id]
    data.qpos[address] = position
    mujoco.mj_forward(model, data)
    return render_camera(model, data)


def test_render_shape_dtype_and_no_state_mutation(model: mujoco.MjModel) -> None:
    """Rendering returns RGB uint8 and leaves all dynamic state untouched."""
    data = mujoco.MjData(model)
    data.qpos[:] = np.linspace(0.001, 0.02, model.nq)
    data.qvel[:] = np.linspace(-0.01, 0.01, model.nv)
    data.ctrl[:] = np.linspace(-0.1, 0.1, model.nu)
    mujoco.mj_forward(model, data)
    before = data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), data.time
    rgb = render_camera(model, data, width=320, height=240)
    assert rgb.shape == (240, 320, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    np.testing.assert_array_equal(data.ctrl, before[2])
    assert data.time == before[3]


@pytest.mark.parametrize(
    ("position", "expected"),
    [(0.0, "closed"), (0.03, "partial"), (0.06, "open")],
)
def test_three_way_drawer_classification(
    model: mujoco.MjModel, position: float, expected: str
) -> None:
    """Canonical closed, partial, and open images receive distinct labels."""
    assert estimate_drawer_state(render_at(model, position))["state"] == expected


@pytest.mark.parametrize(
    "malformed",
    [None, np.zeros((20, 20, 3), dtype=np.uint8), np.zeros((64, 64), dtype=np.uint8)],
)
def test_malformed_image_is_unknown(malformed: object) -> None:
    """Malformed observations fail closed with an unknown state."""
    result = estimate_drawer_state(malformed)  # type: ignore[arg-type]
    assert result["state"] == "unknown"
    assert result["confidence"] == 0.0


def test_estimator_has_no_simulator_state_input(model: mujoco.MjModel) -> None:
    """Inference is stable when simulator state changes but its RGB input does not."""
    rgb = render_at(model, 0.03)
    data = mujoco.MjData(model)
    address = model.jnt_qposadr[model.joint("drawer_slide").id]
    before = estimate_drawer_state(rgb)
    data.qpos[address] = 0.06
    mujoco.mj_forward(model, data)
    after = estimate_drawer_state(rgb)
    assert before == after
    assert tuple(inspect.signature(estimate_drawer_state).parameters) == ("rgb",)


@pytest.mark.parametrize("gain", [0.8, 1.2])
def test_moderate_brightness_change(model: mujoco.MjModel, gain: float) -> None:
    """Color-ratio geometry survives moderate uniform brightness variation."""
    rgb = render_at(model, 0.03)
    changed = np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    assert estimate_drawer_state(changed)["state"] == "partial"


def test_small_image_translation(model: mujoco.MjModel) -> None:
    """Relative geometry survives a small translation in the image plane."""
    rgb = render_at(model, 0.06)
    shifted = np.zeros_like(rgb)
    shifted[5:, 7:] = rgb[:-5, :-7]
    assert estimate_drawer_state(shifted)["state"] == "open"
