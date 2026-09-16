"""Collect live planner decisions for the baseline and randomized drawer episodes."""

import argparse
import json
from pathlib import Path

from matplotlib import image

from bimind.instruction import DEFAULT_INSTRUCTION, parse_instruction
from bimind.skills import Skills
from planner_drawer_task import run_task


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

    def save(rgb, observation, action, episode):
        """Write a single live RGB frame and the teacher decision for that step."""
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

    def record_step(rgb, observation, action, episode_name):
        """Store exactly one live sample per planner decision."""
        save(rgb, observation, action, episode_name)

    for episode in range(runs):
        skills = Skills(scene="drawer")
        try:
            report = run_task(
                skills,
                instruction=instruction,
                confidence_threshold=confidence_threshold,
                on_step=lambda rgb, observation, action, episode_name=episode: record_step(
                    rgb, observation, action, episode_name
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
                on_step=lambda rgb, observation, action, episode_name=episode_name: record_step(
                    rgb, observation, action, episode_name
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
