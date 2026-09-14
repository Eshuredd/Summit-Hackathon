"""Headless execution regressions for the high-level skill API."""

import argparse
import contextlib
import io
import json
from pathlib import Path

from bimanual_handoff import Handoff
from bimind.skills import Skills


def execute(scene, calls):
    """Run skill calls in a fresh session and return their measured report.

    Args:
        scene: Scene supported by Skills.
        calls: Sequence of method names and positional arguments.

    Returns:
        Final skill execution report.
    """
    skills = Skills(scene)
    try:
        for name, args in calls:
            if not getattr(skills, name)(*args)["success"]:
                break
        return skills.get_run_report()
    finally:
        skills.close()


def main():
    """Compare API handoff against the original and exercise other supported paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--drawer-report", type=Path)
    args = parser.parse_args()
    if args.drawer_report:
        baseline = json.loads(
            (Path(__file__).parent / "tests/fixtures/drawer_baseline.json").read_text()
        )
        rows = json.loads(args.drawer_report.read_text())
        for row in rows:
            changed = [
                k
                for k, value in baseline.items()
                if k not in ("run", "runtime") and row[k] != value
            ]
            if changed:
                raise AssertionError(f"Drawer run {row['run']} changed: {changed}")
        if len(rows) != args.runs:
            raise AssertionError(f"Expected {args.runs} drawer runs, found {len(rows)}")
        print(f"All {len(rows)} drawer runs exactly match pre-refactor diagnostics.")
        return
    if args.runs < 1:
        parser.error("--runs must be positive")
    with contextlib.redirect_stdout(io.StringIO()):
        original = Handoff()
        original.run()
    baseline = dict(
        diagnostics=original.history,
        final_position=original.position().tolist(),
        max_cube_drop=original.max_drop,
        arm_arm_collisions=original.collisions,
    )
    results = []
    calls = [
        ("reset", ()),
        ("pick_object", ("left", "object")),
        ("lift_object", ("left", "object")),
        ("handoff_object", ("left", "right", "object")),
        ("place_object", ("right", "table_target")),
        ("finish", ()),
    ]
    for run in range(args.runs):
        with contextlib.redirect_stdout(io.StringIO()):
            result = execute("handoff", calls)
        matches = all(result[k] == v for k, v in baseline.items())
        results.append(dict(case="handoff", run=run + 1, baseline_matches=matches, **result))
        print(
            f"Handoff API {run + 1}/{args.runs}: success={result['success']}, "
            f"exact baseline={matches}",
            flush=True,
        )
    for arm in ("left", "right"):
        calls = [
            ("reset", ()),
            ("move_home", (arm,)),
            ("pick_object", (arm, "object")),
            ("place_object", (arm, "table_target")),
            ("finish", ()),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            result = execute("handoff", calls)
        results.append(dict(case=f"{arm}_pick_place", **result))
        print(
            f"{arm} independent API pick/place: {result['success']} {result['failure']}", flush=True
        )
    calls = [
        ("reset", ()),
        ("open_drawer", ("left",)),
        ("hold_drawer", ("left",)),
        ("pick_object", ("right", "drawer_object")),
        ("place_object", ("right", "table_target")),
        ("release_drawer", ("left",)),
        ("finish", ()),
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        result = execute("drawer", calls)
    baseline = json.loads(
        (Path(__file__).parent / "tests/fixtures/drawer_baseline.json").read_text()
    )
    matches = all(result[k] == v for k, v in baseline.items() if k not in ("run", "runtime"))
    results.append(dict(case="drawer_implicit_lift", baseline_matches=matches, **result))
    print(
        f"Drawer implicit lift: success={result['success']}, exact baseline={matches}", flush=True
    )
    (Path(__file__).parent / "skill_regression_results.json").write_text(
        json.dumps(results, indent=2)
    )
    if not all(r["success"] and r.get("baseline_matches", True) for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
