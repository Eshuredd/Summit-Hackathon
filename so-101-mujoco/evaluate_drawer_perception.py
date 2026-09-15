"""Evaluate image-only drawer perception against privileged MuJoCo truth."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from bimind.perception import estimate_drawer_state, render_camera

SCENE = Path(__file__).parent / "assets" / "drawer_scene.xml"
POSITIONS = tuple(np.linspace(0.0, 0.06, 7))


def truth_state(position: float) -> str:
    """Convert privileged drawer position to the current task semantics.

    Args:
        position: Drawer slide displacement in metres.

    Returns:
        The ground-truth state label used only for evaluation.
    """
    if abs(position) <= 0.001:
        return "closed"
    if position >= 0.048:
        return "open"
    return "partial"


def run_evaluation(
    debug_dir: Path | None = None,
    show: bool = False,
    frame_delay: float = 0.35,
) -> list[dict[str, Any]]:
    """Run the physical-position sweep under controlled visual variations.

    Args:
        debug_dir: Optional directory in which every RGB sample is saved.
        show: Whether to open an interactive preview of the camera observations.
        frame_delay: Seconds to display each sample before advancing.

    Returns:
        One result dictionary per rendered sample.
    """
    scenarios = (
        ("baseline", 1.0, (0.0, 0.0, 0.0), (0.0, 0.0), "original"),
        ("lighting_low", 0.70, (0.0, 0.0, 0.0), (0.0, 0.0), "original"),
        ("lighting_high", 1.30, (0.0, 0.0, 0.0), (0.0, 0.0), "original"),
        ("camera_left", 1.0, (-0.01, 0.005, 0.0), (0.0, 0.0), "original"),
        ("camera_right", 1.0, (0.01, -0.005, 0.005), (0.0, 0.0), "original"),
        ("world_x", 1.0, (0.0, 0.0, 0.0), (0.01, 0.0), "original"),
        ("world_y", 1.0, (0.0, 0.0, 0.0), (0.0, -0.01), "original"),
        ("background_gray", 1.0, (0.0, 0.0, 0.0), (0.0, 0.0), "gray"),
    )
    preview = _CameraPreview(frame_delay) if show else None
    results: list[dict[str, Any]] = []
    for seed, (name, light, camera_delta, world_delta, background) in enumerate(scenarios):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        data = mujoco.MjData(model)
        _configure_visual_variation(model, light, camera_delta, world_delta, background)
        drawer_joint = model.joint("drawer_slide")
        drawer_address = model.jnt_qposadr[drawer_joint.id]
        for position in POSITIONS:
            data.qpos[drawer_address] = position
            mujoco.mj_forward(model, data)
            debug_path = None
            if debug_dir is not None:
                debug_path = debug_dir / f"{name}_{round(position * 1000):02d}mm.png"
            rgb = render_camera(model, data, debug_path=debug_path)
            estimate = estimate_drawer_state(rgb)

            # Privileged state is deliberately read only after image-only inference.
            truth_position = float(data.qpos[drawer_address])
            expected = truth_state(truth_position)
            passed = estimate["state"] == expected
            result = {
                "seed": seed,
                "scenario": name,
                "drawer_truth": truth_position,
                "truth_state": expected,
                "visual_state": estimate["state"],
                "confidence": estimate["confidence"],
                "measurement": estimate["measurement"],
                "pass": passed,
            }
            results.append(result)
            if preview is not None:
                preview.update(rgb, result)
            print(
                f"seed={seed} scenario={name} drawer_truth={truth_position:.3f} "
                f"visual_state={estimate['state']} confidence={estimate['confidence']:.2f} "
                f"{'PASS' if passed else 'FAIL'}"
            )
    if preview is not None:
        preview.hold()
    return results


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate and per-position/per-scenario accuracy statistics.

    Args:
        results: Sample records returned by :func:`run_evaluation`.

    Returns:
        JSON-serializable accuracy summary.
    """
    groups: dict[str, dict[str, list[bool]]] = {
        "by_position_mm": defaultdict(list),
        "by_scenario": defaultdict(list),
    }
    for result in results:
        groups["by_position_mm"][f"{result['drawer_truth'] * 1000:.0f}"].append(result["pass"])
        groups["by_scenario"][result["scenario"]].append(result["pass"])

    def accuracies(group: dict[str, list[bool]]) -> dict[str, float]:
        """Calculate accuracy for each named result group."""
        return {name: sum(values) / len(values) for name, values in group.items()}

    return {
        "samples": len(results),
        "correct": sum(result["pass"] for result in results),
        "accuracy": sum(result["pass"] for result in results) / len(results),
        "by_position_mm": accuracies(groups["by_position_mm"]),
        "by_scenario": accuracies(groups["by_scenario"]),
    }


def _configure_visual_variation(
    model: mujoco.MjModel,
    light_scale: float,
    camera_delta: tuple[float, float, float],
    world_delta: tuple[float, float],
    background: str,
) -> None:
    """Apply controlled appearance changes without changing drawer mechanics."""
    model.vis.headlight.diffuse[:] *= light_scale
    model.vis.headlight.ambient[:] *= light_scale
    model.light_diffuse[:] *= light_scale
    model.camera("drawer_overview").pos[:] += camera_delta
    for body_name in ("drawer_housing", "drawer", "target"):
        model.body(body_name).pos[:2] += world_delta
    if background == "gray":
        texture = model.texture("groundplane")
        start = int(model.tex_adr[texture.id])
        length = int(model.tex_width[texture.id] * model.tex_height[texture.id] * 3)
        pixels = model.tex_data[start : start + length].reshape(-1, 3)
        luminance = np.mean(pixels, axis=1, keepdims=True)
        pixels[:] = np.clip(luminance * np.array([0.9, 0.95, 1.0]), 0, 255).astype(np.uint8)


class _CameraPreview:
    """Interactive display for the exact RGB frame passed to perception."""

    def __init__(self, frame_delay: float) -> None:
        """Create a Matplotlib window without changing simulation state.

        Args:
            frame_delay: Seconds to show each evaluation sample.
        """
        if frame_delay <= 0:
            raise ValueError("frame delay must be positive")
        from matplotlib import pyplot

        self._pyplot = pyplot
        self._frame_delay = frame_delay
        self._patches: list[Any] = []
        pyplot.ion()
        self._figure, self._axes = pyplot.subplots(num="Drawer perception camera")
        self._image = self._axes.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self._axes.set_axis_off()
        self._figure.tight_layout()

    def update(self, rgb: np.ndarray, result: dict[str, Any]) -> None:
        """Show one observation with detected image geometry overlaid.

        Args:
            rgb: Exact RGB observation supplied to the estimator.
            result: Evaluation result and visual measurements for the frame.
        """
        if not self._pyplot.fignum_exists(self._figure.number):
            return
        from matplotlib.patches import Rectangle

        self._image.set_data(rgb)
        for patch in self._patches:
            patch.remove()
        self._patches.clear()
        measurement = result["measurement"]
        for key, color in (("drawer_bbox", "lime"), ("housing_bbox", "cyan")):
            if key not in measurement:
                continue
            x0, y0, x1, y1 = measurement[key]
            patch = Rectangle(
                (x0, y0), x1 - x0 + 1, y1 - y0 + 1, fill=False, edgecolor=color, linewidth=2
            )
            self._axes.add_patch(patch)
            self._patches.append(patch)
        self._axes.set_title(
            f"{result['scenario']} | {result['drawer_truth'] * 1000:.0f} mm | "
            f"visual={result['visual_state']} ({result['confidence']:.2f}) | "
            f"truth={result['truth_state']}"
        )
        self._figure.canvas.draw_idle()
        self._pyplot.pause(self._frame_delay)

    def hold(self) -> None:
        """Keep the final camera observation visible until the window is closed."""
        if self._pyplot.fignum_exists(self._figure.number):
            self._pyplot.ioff()
            self._pyplot.show()


def main() -> None:
    """Run evaluation and optionally write detailed JSON and debug frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="optional detailed JSON output path")
    parser.add_argument("--debug-dir", type=Path, help="optional rendered-frame directory")
    parser.add_argument(
        "--show",
        action="store_true",
        help="show the exact camera frames and detected geometry during evaluation",
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.35,
        help="seconds per preview frame (default: 0.35)",
    )
    args = parser.parse_args()
    results = run_evaluation(args.debug_dir, show=args.show, frame_delay=args.frame_delay)
    summary = summarize(results)
    print(json.dumps(summary, indent=2))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "results": results}, indent=2))


if __name__ == "__main__":
    main()
