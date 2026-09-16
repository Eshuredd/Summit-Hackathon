"""Deterministic RGB perception helpers for the bimanual drawer scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy import ndimage

DEFAULT_CAMERA = "drawer_overview"
CLOSED_MAX_DISPLACEMENT = 0.727
OPEN_MIN_DISPLACEMENT = 0.892
TRAVEL_X_WEIGHT = 0.90
ZERO_HORIZONTAL_OFFSET = 0.126


def render_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str = DEFAULT_CAMERA,
    width: int = 640,
    height: int = 480,
    debug_path: str | Path | None = None,
) -> np.ndarray:
    """Render an RGB observation without advancing or modifying simulation state.

    Args:
        model: MuJoCo model containing the named fixed camera.
        data: Current MuJoCo simulation data.
        camera_name: Name of the RGB camera to render.
        width: Output width in pixels.
        height: Output height in pixels.
        debug_path: Optional path at which to save the rendered frame.

    Returns:
        An RGB ``uint8`` array with shape ``(height, width, 3)``.

    Raises:
        ValueError: If either output dimension is not positive.
    """
    if width <= 0 or height <= 0:
        raise ValueError("camera width and height must be positive")

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera_name)
        rgb = np.asarray(renderer.render(), dtype=np.uint8).copy()
    finally:
        renderer.close()

    if debug_path is not None:
        _save_debug_frame(rgb, Path(debug_path))
    return rgb


def estimate_drawer_state(rgb: np.ndarray) -> dict[str, Any]:
    """Estimate closed, partially-open, or open state from one RGB image only.

    The measurement projects the image displacement from the detected stationary
    housing origin to the detected moving drawer silhouette onto the visible drawer
    travel direction, then normalizes by detected housing width. No simulator object,
    coordinates, or joint state are accepted.

    Args:
        rgb: RGB image with shape ``(height, width, 3)``.

    Returns:
        A dictionary containing ``state``, ``confidence``, and diagnostic
        ``measurement`` values. Invalid or unrecognized images return ``unknown``.
    """
    problem = _validate_image(rgb)
    if problem is not None:
        return _unknown(problem)

    image = np.asarray(rgb, dtype=np.float32) + 1.0
    red, green, blue = np.moveaxis(image, -1, 0)

    # Chromatic ratios are substantially less sensitive to uniform illumination than
    # raw RGB thresholds. The first mask finds the wood drawer; the second finds the
    # blue-gray fixed housing rather than relying on a fixed image region.
    drawer_mask = (red / green > 1.25) & (green / blue > 1.08) & (red > 24.0)
    housing_mask = (
        (green / red > 1.17)
        & (blue / green > 1.12)
        & (blue / green < 1.22)
        & (red > 20.0)
    )

    drawer = _largest_component(drawer_mask)
    housing = _select_housing(housing_mask, drawer)
    if drawer is None:
        return _unknown("drawer_not_detected")
    if housing is None:
        return _unknown("housing_not_detected")

    drawer_x0, drawer_y0, drawer_x1, drawer_y1, drawer_area = drawer
    housing_x0, housing_y0, housing_x1, housing_y1, housing_area = housing
    housing_width = housing_x1 - housing_x0 + 1
    if housing_width < max(12, round(rgb.shape[1] * 0.08)):
        return _unknown("housing_too_small")

    vertical_displacement = (drawer_y1 - housing_y0) / housing_width
    horizontal_offset = (drawer_x0 - housing_x0) / housing_width
    displacement = vertical_displacement + TRAVEL_X_WEIGHT * (
        horizontal_offset - ZERO_HORIZONTAL_OFFSET
    )
    if displacement <= CLOSED_MAX_DISPLACEMENT:
        state = "closed"
        margin = CLOSED_MAX_DISPLACEMENT - displacement
    elif displacement >= OPEN_MIN_DISPLACEMENT:
        state = "open"
        margin = displacement - OPEN_MIN_DISPLACEMENT
    else:
        state = "partial"
        margin = min(
            displacement - CLOSED_MAX_DISPLACEMENT,
            OPEN_MIN_DISPLACEMENT - displacement,
        )

    image_area = rgb.shape[0] * rgb.shape[1]
    detection_quality = min(1.0, drawer_area / (image_area * 0.012))
    margin_quality = min(1.0, margin / 0.05)
    confidence = float(np.clip(0.50 + 0.25 * detection_quality + 0.25 * margin_quality, 0, 1))
    return {
        "state": state,
        "confidence": confidence,
        "measurement": {
            "normalized_displacement": float(displacement),
            "normalized_vertical_displacement": float(vertical_displacement),
            "normalized_horizontal_offset": float(horizontal_offset),
            "drawer_bbox": [drawer_x0, drawer_y0, drawer_x1, drawer_y1],
            "housing_bbox": [housing_x0, housing_y0, housing_x1, housing_y1],
            "drawer_area_px": drawer_area,
            "housing_area_px": housing_area,
            "closed_max": CLOSED_MAX_DISPLACEMENT,
            "open_min": OPEN_MIN_DISPLACEMENT,
        },
    }


def _validate_image(rgb: object) -> str | None:
    """Return an error code for malformed image input, otherwise ``None``."""
    if not isinstance(rgb, np.ndarray):
        return "image_must_be_numpy_array"
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return "image_must_have_shape_h_w_3"
    if rgb.shape[0] < 32 or rgb.shape[1] < 32:
        return "image_too_small"
    if not np.issubdtype(rgb.dtype, np.number):
        return "image_must_be_numeric"
    if not np.all(np.isfinite(rgb)):
        return "image_contains_non_finite_values"
    return None


def _largest_component(mask: np.ndarray) -> tuple[int, int, int, int, int] | None:
    """Return the bounding box and area of the largest connected mask component."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    label = int(np.argmax(areas))
    area = int(areas[label])
    if area < mask.size * 0.001:
        return None
    rows, columns = np.nonzero(labels == label)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
        area,
    )


def _select_housing(
    mask: np.ndarray,
    drawer: tuple[int, int, int, int, int] | None,
) -> tuple[int, int, int, int, int] | None:
    """Select the fixed housing component by size and overlap with the drawer."""
    if drawer is None:
        return None
    labels, count = ndimage.label(mask)
    if count == 0:
        return None
    drawer_x0, drawer_y0, drawer_x1, drawer_y1, _ = drawer
    candidates: list[tuple[float, tuple[int, int, int, int, int]]] = []
    fragments = []
    for label in range(1, count + 1):
        rows, columns = np.nonzero(labels == label)
        area = len(columns)
        if area < mask.size * 0.001:
            continue
        x0, x1 = int(columns.min()), int(columns.max())
        y0, y1 = int(rows.min()), int(rows.max())
        horizontal_overlap = max(0, min(x1, drawer_x1) - max(x0, drawer_x0) + 1)
        drawer_width = drawer_x1 - drawer_x0 + 1
        overlap_fraction = horizontal_overlap / drawer_width
        vertically_relevant = y0 < drawer_y0 and y1 >= drawer_y0
        if vertically_relevant and horizontal_overlap > 0:
            fragments.append((x0, y0, x1, y1, area))
        if overlap_fraction < 0.45 or not vertically_relevant:
            continue
        score = area * (1.0 + overlap_fraction)
        candidates.append((score, (x0, y0, x1, y1, area)))
    if candidates:
        return max(candidates, key=lambda item: item[0])[1]
    # A gripper can split the housing into separate visible components. Combine
    # only substantial housing-colored fragments bordering the drawer in this RGB.
    if len(fragments) >= 2:
        x0 = min(part[0] for part in fragments)
        y0 = min(part[1] for part in fragments)
        x1 = max(part[2] for part in fragments)
        y1 = max(part[3] for part in fragments)
        overlap = max(0, min(x1, drawer_x1) - max(x0, drawer_x0) + 1)
        if overlap / (drawer_x1 - drawer_x0 + 1) >= 0.45:
            return x0, y0, x1, y1, sum(part[4] for part in fragments)
    return None


def _unknown(reason: str) -> dict[str, Any]:
    """Create a stable unknown-state response."""
    return {"state": "unknown", "confidence": 0.0, "measurement": {"reason": reason}}


def _save_debug_frame(rgb: np.ndarray, path: Path) -> None:
    """Save a debug RGB frame using the project's Matplotlib dependency."""
    from matplotlib import image as mpl_image

    path.parent.mkdir(parents=True, exist_ok=True)
    mpl_image.imsave(path, rgb)
