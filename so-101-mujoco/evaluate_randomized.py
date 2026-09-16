"""Evaluate the existing drawer task on reproducible fresh randomized scenes."""

import argparse
import json
from pathlib import Path

from bimind.instruction import DEFAULT_INSTRUCTION
from bimind.skills import Skills
from planner_drawer_task import run_task


def evaluate(seeds, policy_name, output):
    """Run every seed, preserving failures and closing each scene.

    Args:
        seeds: Nonnegative integer seeds.
        policy_name: Symbolic or OpenVINO policy selector.
        output: JSON report path.

    Returns:
        Aggregate evaluation report using existing task diagnostics.
    """
    policy = None
    if policy_name == "openvino":
        from bimind.openvino_policy import OpenVINOPolicy

        policy = OpenVINOPolicy()
    results = []
    for seed in seeds:
        print(f"\n[SEED {seed}]", flush=True)
        skills = None
        metadata = None
        try:
            skills = Skills(scene="drawer", seed=seed)
            metadata = skills.randomization
            for key, value in metadata.items():
                print(f"{key}={value}", flush=True)
            report = run_task(skills, instruction=DEFAULT_INSTRUCTION, policy=policy)
        except Exception as error:
            report = skills.get_run_report() if skills is not None else {}
            report.update(success=False, failure=f"{type(error).__name__}: {error}")
        finally:
            if skills is not None:
                skills.close()
        report.update(
            seed=seed,
            randomization=metadata,
            placement_error=report.get("object_final_xy_error"),
            collisions=report.get("arm_arm_collisions"),
        )
        report.setdefault("planner_trace", [])
        results.append(report)
        print(
            f"result={'SUCCESS' if report['success'] else 'FAILURE'}\n"
            f"placement_error={report['placement_error']}\ncollisions={report['collisions']}\n"
            f"failure={report['failure']}",
            flush=True,
        )
    successes = sum(r["success"] for r in results)
    summary = dict(
        policy=policy_name,
        instruction=DEFAULT_INSTRUCTION,
        runs=len(results),
        successes=successes,
        success_rate=successes / len(results),
        results=results,
    )
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nRandomized evaluation:\nSUCCESS = {successes} / {len(results)}", flush=True)
    return summary


def main():
    """Parse the minimal evaluation command and fail after all unsuccessful trials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("symbolic", "openvino"), default="openvino")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be nonnegative")
    filename = (
        "randomized_results.json"
        if args.policy == "openvino"
        else "randomized_symbolic_results.json"
    )
    report = evaluate(args.seeds, args.policy, args.output or Path(__file__).with_name(filename))
    raise SystemExit(0 if report["successes"] == report["runs"] else 1)


if __name__ == "__main__":
    main()
