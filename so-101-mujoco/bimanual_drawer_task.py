"""Run the deterministic drawer task entirely through the high-level skill API."""

import argparse
import json
from pathlib import Path

from bimind.skills import Skills


def run_task(skills: Skills) -> dict:
    """Execute the proven sequence, stopping immediately on any failed skill.

    Args:
        skills: A newly constructed drawer-scene skill session.

    Returns:
        The complete trial report, including structured skill outcomes.
    """
    calls = (
        (skills.reset, ()),
        (skills.open_drawer, ("left",)),
        (skills.hold_drawer, ("left",)),
        (skills.pick_object, ("right", "drawer_object")),
        (skills.lift_object, ("right", "drawer_object")),
        (skills.place_object, ("right", "table_target")),
        (skills.release_drawer, ("left",)),
        (skills.finish, ()),
    )
    for skill, args in calls:
        result = skill(*args)
        if not result["success"]:
            print(f"[DRAWER] FAILURE in {result['phase']}: {result['reason']}", flush=True)
            break
    return skills.get_run_report()


def main():
    """Run deterministic headless trials and retain the existing report filenames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    results = []
    for run in range(args.runs):
        skills = Skills(scene="drawer", viewer=args.viewer)
        try:
            results.append(dict(run=run + 1, **run_task(skills)))
        finally:
            skills.close()
        print(json.dumps(results[-1]), flush=True)
    path = Path(__file__).parent / (
        "drawer_viewer_results.json" if args.viewer else "drawer_results.json"
    )
    path.write_text(json.dumps(results, indent=2))
    successes = sum(r["success"] for r in results)
    print(f"Drawer success: {successes}/{args.runs}", flush=True)
    if successes != args.runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
