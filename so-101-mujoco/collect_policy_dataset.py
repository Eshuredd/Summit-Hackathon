"""Collect live planner decisions for the baseline and randomized drawer episodes."""

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


def collect(
    output,
    runs=1,
    seeds=tuple(range(10)),
    instruction=DEFAULT_INSTRUCTION,
    confidence_threshold=0.65,
):
    """Save one live sample per planner decision for the baseline and randomized scenes.

    Args:
        output: New dataset directory, to avoid overwriting prior samples.
        runs: Number of baseline drawer episodes to record.
        seeds: Randomized drawer seeds to evaluate and record.
        instruction: Natural-language command stored with every sample.
        confidence_threshold: Shared planner confidence gate.

    Returns:
        Number of saved training samples.
    """
    parse_instruction(instruction)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []

    def save(rgb, observation, action, episode, scenario=None):
        """Write a single live RGB frame and the teacher decision for that step."""
        number = len(records)
        filename = f"frame_{number:06d}.png"
        image.imsave(output / filename, rgb)

        row = {
            "rgb": filename,
            "instruction": instruction,
            "observation": observation,
            "teacher_next_skill": action["skill"].lower(),
            "teacher_args": action["args"],
            "episode": episode,
        }

        if scenario is not None:
            row["scenario"] = scenario

        records.append(row)

    def record_baseline_step(rgb, observation, action, skills, episode_name):
        for scenario, light, camera, world, background in VARIATIONS:
            model = copy.copy(skills._controller.model)
            data = copy.copy(skills._controller.data)

            _configure_visual_variation(
                model,
                light,
                camera,
                world,
                background,
            )

            mujoco.mj_forward(model, data)
            frame = render_camera(model, data)

            varied = copy.deepcopy(observation)
            estimate = estimate_drawer_state(frame)

            varied["drawer"] = {
                "visual_state": estimate["state"],
                "confidence": estimate["confidence"],
            }

            save(
                frame,
                varied,
                next_action(varied, confidence_threshold),
                episode_name,
                scenario=scenario,
            )

        if action["skill"] == "open_drawer":
            for position in (0.01, 0.03, 0.04):
                model = copy.copy(skills._controller.model)
                data = copy.copy(skills._controller.data)

                data.qpos[
                    model.joint("drawer_slide").qposadr[0]
                ] = position

                mujoco.mj_forward(model, data)
                frame = render_camera(model, data)

                varied = build_observation(
                    instruction,
                    estimate_drawer_state(frame),
                    skills.get_scene_state(),
                )

                save(
                    frame,
                    varied,
                    next_action(varied, confidence_threshold),
                    episode_name,
                )

    def record_randomized_step(rgb, observation, action, episode_name):
        save(
            rgb,
            observation,
            action,
            episode_name,
        )

    for episode in range(runs):
        skills = Skills(scene="drawer")
        try:
            report = run_task(
                skills,
                instruction=instruction,
                confidence_threshold=confidence_threshold,
                on_step=lambda rgb, observation, action, skills=skills, episode_name=episode: (
                    record_baseline_step(
                        rgb,
                        observation,
                        action,
                        skills,
                        episode_name,
                    )
                ),
            )
            if not report["success"]:
                print(f"Baseline episode {episode} failed: {report.get('failure')}", flush=True)
        finally:
            skills.close()
            (output / "samples.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )

    for seed in seeds:
        skills = Skills(scene="drawer", seed=seed)
        try:
            episode_name = f"randomized_{seed}"
            report = run_task(
                skills,
                instruction=instruction,
                confidence_threshold=confidence_threshold,
                on_step=lambda rgb, observation, action, episode_name=episode_name: (
                    record_randomized_step(
                        rgb,
                        observation,
                        action,
                        episode_name,
                    )
                ),
            )
            if not report["success"]:
                print(f"Randomized episode {episode_name} failed: {report.get('failure')}", flush=True)
        finally:
            skills.close()
            (output / "samples.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )

    (output / "samples.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    return len(records)


def main():
    """Collect complete baseline episodes and the randomized teacher samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("policy_dataset"))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    args = parser.parse_args()
    if args.runs < 1 or not 0 <= args.confidence_threshold <= 1:
        parser.error("runs must be positive and confidence threshold must be in [0, 1]")
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    try:
        parse_instruction(args.instruction)
    except ValueError as error:
        parser.error(str(error))
    count = collect(
        args.output,
        args.runs,
        tuple(args.seeds),
        args.instruction,
        args.confidence_threshold,
    )
    print(f"Policy training samples collected: {count}")


if __name__ == "__main__":
    main()
