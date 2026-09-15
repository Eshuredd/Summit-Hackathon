"""Observe, plan one action, execute one skill, and observe the drawer again."""

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Any

from bimind import planner
from bimind.instruction import DEFAULT_INSTRUCTION, parse_instruction
from bimind.observation import observe
from bimind.skills import Skills

# This dispatch whitelist is an API boundary, not an ordered task sequence.
SKILLS = frozenset(
    {"open_drawer", "hold_drawer", "pick_object", "place_object", "release_drawer", "finish"}
)


# Arguments are fixed bindings to the existing Skills API, never network outputs.
SKILL_ARGS = {
    "open_drawer": ["left"],
    "hold_drawer": ["left"],
    "pick_object": ["right", "drawer_object"],
    "place_object": ["right", "table_target"],
    "release_drawer": ["left"],
    "finish": [],
}


def select_action(rgb, instruction, state, confidence_threshold=0.65, policy=None):
    """Select a symbolic or learned action with fail-closed learned-policy gates.

    Args:
        rgb: Fresh camera image.
        instruction: Original natural-language command.
        state: Minimal camera and robot observation.
        confidence_threshold: Required perception and model confidence.
        policy: OpenVINO policy object, or None for the existing symbolic planner.

    Returns:
        Existing action schema with a fixed Skills API argument binding.
    """
    if policy is None:
        return planner.next_action(state, confidence_threshold)
    if state.get("failure") is not None:
        return {"skill": "STOP", "args": [], "reason": f"active failure: {state['failure']}"}
    drawer = state.get("drawer", {})
    confidence = drawer.get("confidence")
    if (
        drawer.get("visual_state") not in ("closed", "open")
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not isfinite(confidence)
        or not confidence_threshold <= confidence <= 1
    ):
        return {"skill": "STOP", "args": [], "reason": "unsafe or uncertain RGB drawer state"}
    prediction = policy.predict_next_skill(rgb, instruction, state)
    name, confidence = prediction.get("skill"), prediction.get("confidence")
    if (
        name not in SKILL_ARGS
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not isfinite(confidence)
        or not confidence_threshold <= confidence <= 1
    ):
        return {
            "skill": "STOP",
            "args": [],
            "reason": f"OpenVINO stop, unsupported label, or low confidence: {prediction}",
        }
    return {
        "skill": name,
        "args": list(SKILL_ARGS[name]),
        "reason": f"OpenVINO confidence={confidence:.4f}",
    }


def run_task(
    skills: Skills,
    max_steps: int = 16,
    instruction: str = DEFAULT_INSTRUCTION,
    confidence_threshold: float = 0.65,
    on_step=None,
    policy=None,
) -> dict[str, Any]:
    """Dispatch one fresh planner decision per iteration and stop on any failure.

    Args:
        skills: New drawer-scene session; initialization performs its existing reset.
        max_steps: Upper bound on decisions to prevent a nonprogressing loop.
        instruction: Natural-language command.
        confidence_threshold: Minimum accepted visual confidence.
        on_step: Optional dataset callback receiving RGB, observation, and action.
        policy: OpenVINO policy instance; None keeps symbolic planning.

    Returns:
        The unchanged physical report plus planner observations, decisions, and any stop reason.
    """
    parse_instruction(instruction)
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence threshold must be between zero and one")
    print(f"[COMMAND] {instruction}", flush=True)
    placement_confirmed = False
    trace = []
    stop_reason = None
    reset = skills.reset()
    if reset.get("terminated"):
        print("[VIEWER] Closed by user; exiting without another trial.", flush=True)
        return dict(skills.get_run_report(), planner_trace=trace, planner_stop_reason=None)
    if not reset["success"]:
        stop_reason = reset["reason"]
    else:
        for step in range(max_steps):
            rgb, state = observe(skills, instruction, placement_confirmed)
            print(
                f"[PERCEPTION] drawer={state['drawer']['visual_state']} "
                f"confidence={state['drawer']['confidence']:.2f}",
                flush=True,
            )
            action = select_action(rgb, instruction, state, confidence_threshold, policy)
            if on_step is not None:
                on_step(rgb, state, action)
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
            print(f"[ACT] {'success' if result['success'] else 'failure'}", flush=True)
            if name == "place_object" and result["success"]:
                placement_confirmed = True
            if result.get("terminated"):
                print("[VIEWER] Closed by user; exiting without another trial.", flush=True)
                break
            if not result["success"]:
                # Observe once more and record the planner's terminal failure decision.
                rgb, state = observe(skills, instruction, placement_confirmed)
                state["failure"] = result["reason"]
                terminal = select_action(rgb, instruction, state, confidence_threshold, policy)
                if on_step is not None:
                    on_step(rgb, state, terminal)
                print(f"[OBSERVE] skill failed: {result['reason']}", flush=True)
                print(f"[PLAN] {terminal['reason']} -> {terminal['skill']}", flush=True)
                trace.append(dict(step=step + 2, observation=state, action=terminal))
                stop_reason = result["reason"]
                break
            if name == "finish":
                print("[OBSERVE] task complete", flush=True)
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
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--policy", choices=("symbolic", "openvino"), default="symbolic")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--device", default="CPU")
    args = parser.parse_args()
    try:
        parse_instruction(args.instruction)
    except ValueError as error:
        parser.error(str(error))
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be between zero and one")
    if args.runs < 1 or args.max_steps < 1:
        parser.error("--runs and --max-steps must be positive")
    policy = None
    if args.policy == "openvino":
        from bimind.openvino_policy import DEFAULT_MODEL, OpenVINOPolicy

        policy = OpenVINOPolicy(args.model or DEFAULT_MODEL, args.device, args.confidence_threshold)
        print(f"[POLICY] OpenVINO device={policy.device_name} precision={policy.precision}")
    results = []
    trial_count = 1 if args.viewer else args.runs
    for run in range(trial_count):
        skills = Skills(scene="drawer", viewer=args.viewer)
        try:
            report = dict(
                run=run + 1,
                **run_task(
                    skills,
                    args.max_steps,
                    args.instruction,
                    args.confidence_threshold,
                    policy=policy,
                ),
                policy=args.policy,
            )
            results.append(report)
        finally:
            skills.close()
        print(
            f"Planner trial {run + 1}/{trial_count}: success={report['success']}, "
            f"failure={report['failure']}",
            flush=True,
        )
    filename = (
        "planner_drawer_results.json"
        if args.policy == "symbolic"
        else "planner_openvino_results.json"
    )
    path = Path(__file__).parent / filename
    path.write_text(json.dumps(results, indent=2))
    successes = sum(r["success"] for r in results)
    if any(r.get("terminated") for r in results):
        print("Planner run terminated by user; no manipulation failure.", flush=True)
    else:
        print(f"Planner drawer success: {successes}/{len(results)}", flush=True)
    if any(not r["success"] and not r.get("terminated") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
