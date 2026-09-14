"""Observe, plan one action, execute one skill, and observe the drawer again."""

import argparse
import json
from pathlib import Path
from typing import Any

from bimind import planner
from bimind.skills import Skills

# This dispatch whitelist is an API boundary, not an ordered task sequence.
SKILLS = frozenset(
    {"open_drawer", "hold_drawer", "pick_object", "place_object", "release_drawer", "finish"}
)


def describe(state: dict[str, Any]) -> str:
    """Summarize one fresh observation for the console log."""
    drawer = state["drawer"]
    status = "closed" if drawer["is_closed"] else "open" if drawer["is_open"] else "partially open"
    left = state["arms"]["left"]["holding_objects"]
    right = state["arms"]["right"]["holding_objects"]
    return (
        f"drawer is {status}; left holding={left}, monitor={drawer['hold_monitor_active']}; "
        f"right holding={right}; object_at_target={planner.object_at_target(state)}"
    )


def run_task(skills: Skills, max_steps: int = 16) -> dict[str, Any]:
    """Dispatch one fresh planner decision per iteration and stop on any failure.

    Args:
        skills: New drawer-scene session; initialization performs its existing reset.
        max_steps: Upper bound on decisions to prevent a nonprogressing loop.

    Returns:
        The unchanged physical report plus planner observations, decisions, and any stop reason.
    """
    trace = []
    stop_reason = None
    reset = skills.reset()
    if not reset["success"]:
        stop_reason = reset["reason"]
    else:
        for step in range(max_steps):
            state = skills.get_scene_state()
            print(f"[OBSERVE] {describe(state)}", flush=True)
            action = planner.next_action(state)
            name, args = action["skill"], action["args"]
            invocation = f"{name}({', '.join(args)})"
            print(f"[PLAN] {action['reason']} -> {invocation}", flush=True)
            entry = dict(step=step + 1, observation=state, action=action)
            trace.append(entry)
            if name == "STOP":
                stop_reason = action["reason"]
                break
            if name not in SKILLS:
                stop_reason = f"Planner returned unsupported skill: {name}"
                break
            print(f"[ACT] {invocation}", flush=True)
            result = getattr(skills, name)(*args)
            entry["result"] = result
            if not result["success"]:
                # Observe once more and record the planner's terminal failure decision.
                state = skills.get_scene_state()
                terminal = planner.next_action(state)
                print(f"[OBSERVE] skill failed: {result['reason']}", flush=True)
                print(f"[PLAN] {terminal['reason']} -> {terminal['skill']}", flush=True)
                trace.append(dict(step=step + 2, observation=state, action=terminal))
                stop_reason = result["reason"]
                break
            if name == "finish":
                print(f"[OBSERVE] {describe(skills.get_scene_state())}; task complete", flush=True)
                break
        else:
            stop_reason = f"Planner exceeded {max_steps} decisions without completion"
    report = skills.get_run_report()
    if stop_reason is not None:
        report.update(success=False, failure=stop_reason)
    return dict(report, planner_trace=trace, planner_stop_reason=stop_reason)


def main():
    """Run deterministic headless planner trials and save an independent report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args()
    if args.runs < 1 or args.max_steps < 1:
        parser.error("--runs and --max-steps must be positive")
    results = []
    for run in range(args.runs):
        skills = Skills(scene="drawer")
        try:
            report = dict(run=run + 1, **run_task(skills, args.max_steps))
            results.append(report)
        finally:
            skills.close()
        print(
            f"Planner trial {run + 1}/{args.runs}: success={report['success']}, "
            f"failure={report['failure']}",
            flush=True,
        )
    path = Path(__file__).parent / "planner_drawer_results.json"
    path.write_text(json.dumps(results, indent=2))
    successes = sum(r["success"] for r in results)
    print(f"Planner drawer success: {successes}/{args.runs}", flush=True)
    if successes != args.runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
