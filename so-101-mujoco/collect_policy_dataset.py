"""Collect RGB and allowlisted features with the current symbolic planner as teacher."""

import argparse
import copy
import json
from pathlib import Path

import mujoco
from matplotlib import image

from bimind.instruction import DEFAULT_INSTRUCTION, parse_instruction
from bimind.observation import build_observation
from bimind.perception import estimate_drawer_state, render_camera
from bimind.planner import next_action
from bimind.skills import Skills
from evaluate_drawer_perception import _configure_visual_variation
from planner_drawer_task import run_task

VARIATIONS = (
    ("baseline", 1.0, (0, 0, 0), (0, 0), "original"),
    ("lighting_low", 0.7, (0, 0, 0), (0, 0), "original"),
    ("lighting_high", 1.3, (0, 0, 0), (0, 0), "original"),
    ("camera_left", 1.0, (-0.01, 0.005, 0), (0, 0), "original"),
    ("camera_right", 1.0, (0.01, -0.005, 0.005), (0, 0), "original"),
    ("world_x", 1.0, (0, 0, 0), (0.01, 0), "original"),
    ("world_y", 1.0, (0, 0, 0), (0, -0.01), "original"),
    ("background_gray", 1.0, (0, 0, 0), (0, 0), "gray"),
)


def collect(output, runs=1, instruction=DEFAULT_INSTRUCTION, confidence_threshold=0.65):
    """Save teacher decisions under existing perturbations in isolated render copies.

    Args:
        output: New dataset directory, to avoid overwriting prior samples.
        runs: Number of complete physical teacher episodes.
        instruction: Natural-language command stored with every sample.
        confidence_threshold: Shared planner confidence gate.

    Returns:
        Number of saved training samples.
    """
    parse_instruction(instruction)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for episode in range(runs):
        skills = Skills(scene="drawer")

        def save(rgb, observation, action):
            """Write only image, inference features, and teacher output."""
            number = len(records)
            filename = f"frame_{number:06d}.png"
            image.imsave(output / filename, rgb)
            records.append(
                {
                    "rgb": filename,
                    "instruction": instruction,
                    "observation": observation,
                    "teacher_next_skill": action["skill"].lower(),
                    "teacher_args": action["args"],
                    "episode": episode,
                }
            )

        def record_step(rgb, observation, action):
            """Re-render the current step without modifying the live controller."""
            for _, light, camera, world, background in VARIATIONS:
                model = copy.copy(skills._controller.model)
                data = copy.copy(skills._controller.data)
                _configure_visual_variation(model, light, camera, world, background)
                mujoco.mj_forward(model, data)
                frame = render_camera(model, data)
                varied = copy.deepcopy(observation)
                estimate = estimate_drawer_state(frame)
                varied["drawer"] = {
                    "visual_state": estimate["state"],
                    "confidence": estimate["confidence"],
                }
                save(frame, varied, next_action(varied, confidence_threshold))
            if action["skill"] == "open_drawer":
                # Synthetic safety examples use cloned truth only to generate frames.
                for position in (0.01, 0.03, 0.04):
                    model = copy.copy(skills._controller.model)
                    data = copy.copy(skills._controller.data)
                    data.qpos[model.joint("drawer_slide").qposadr[0]] = position
                    mujoco.mj_forward(model, data)
                    frame = render_camera(model, data)
                    varied = build_observation(
                        instruction, estimate_drawer_state(frame), skills.get_scene_state()
                    )
                    save(frame, varied, next_action(varied, confidence_threshold))

        try:
            report = run_task(
                skills,
                instruction=instruction,
                confidence_threshold=confidence_threshold,
                on_step=record_step,
            )
            if not report["success"]:
                raise RuntimeError(report["failure"])
        finally:
            skills.close()
            (output / "samples.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
    return len(records)


def main():
    """Collect complete teacher episodes without training a policy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("policy_dataset"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    args = parser.parse_args()
    if args.runs < 1 or not 0 <= args.confidence_threshold <= 1:
        parser.error("runs must be positive and confidence threshold must be in [0, 1]")
    try:
        parse_instruction(args.instruction)
    except ValueError as error:
        parser.error(str(error))
    count = collect(args.output, args.runs, args.instruction, args.confidence_threshold)
    print(f"Policy training samples collected: {count}")


if __name__ == "__main__":
    main()
