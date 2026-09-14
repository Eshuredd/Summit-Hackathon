# One-step symbolic drawer planner

Run headlessly from the project directory on the validated MuJoCo 3.10.0 runtime:

```powershell
.\.venv\Scripts\python.exe planner_drawer_task.py --runs 10
```

The runner initializes a new `Skills(scene="drawer")` session and performs its
existing reset. Each subsequent iteration fetches `get_scene_state()`, calls
`bimind.planner.next_action(state)`, logs the reason, and executes just that skill.
It observes again before making another decision. There is no ordered action
list, task index, motion execution, or model inference in the planner.

## Action schema

```json
{"skill": "open_drawer", "args": ["left"], "reason": "drawer closed"}
```

`STOP` with empty `args` is a terminal failure decision. Every decision has a
human-readable reason. `finish` is an actual skill call; only its successful
validation completes a run. The runner dispatches through an unordered whitelist
of supported skill names, stops on skill failure, and limits execution to 16
decisions by default (`--max-steps`).

## Rules, in priority order

1. Stop on active failure, reported arm-arm collision, or invalid observation.
2. If the object is at the target and the right arm has released it, release the
   drawer if the left arm still holds its handle; otherwise finish when the hold
   monitor is also inactive.
3. For an unfinished placement, open a closed drawer.
4. Hold an open drawer if either measured left support or hold monitoring is absent.
5. If the drawer is maintained and the right arm holds the cube, place it.
6. Otherwise, with the drawer maintained, pick the cube with the right arm.

The planner rejects partial-opening recovery and unsupported ownership rather
than inventing new trajectories. The existing Skills preconditions remain in
force: requesting `hold_drawer` without handle support can return a skill failure.
This is deterministic planning for the validated session, not arbitrary-state
recovery. No workflow stages or phase labels choose the next manipulation.

`object_at_target` uses actual object and target XYZ: Euclidean XY error must be
strictly below 0.03 m, and absolute Z error below 0.005 m. The Z check prevents a
cube hovering above the marker from counting as placed. If the right arm still
holds a cube at those coordinates, it must still execute placement/release.
Completion rules precede opening/holding rules because the drawer remains open
after the successful task. The final physical checks remain in `Skills.finish`.

`place_object` already includes the validated lift when needed; the planner
therefore does not need to select a separate lift action.

## Example decisions

```text
[PLAN] drawer closed -> open_drawer(left)
[ACT] open_drawer(left)
[OBSERVE] drawer is open; left holding=['drawer_handle'], monitor=False; ...
[PLAN] drawer open but left hold is not maintained -> hold_drawer(left)
[PLAN] drawer held; right is not holding object -> pick_object(right, drawer_object)
[PLAN] right holds object away from table target -> place_object(right, table_target)
[PLAN] object at table target; left still holds drawer -> release_drawer(left)
[PLAN] object at table target and drawer released -> finish()
```

An OBSERVE and ACT log accompanies each manipulation decision in the full trace.
`planner_drawer_results.json` contains the existing physical diagnostics plus
`planner_trace` (fresh observations, one-action decisions, and skill results) and
`planner_stop_reason`. Existing task result files are not overwritten.

## Tests and baseline comparison

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe validate_skills.py --drawer-report planner_drawer_results.json --runs 10
```

Synthetic tests cover closed/open/held drawers, right-hand ownership, target
placement, failure priority, XY/Z boundaries, stale workflow history, stateless
planning, malformed positions, and collisions. Runner tests inject changed
observations to prove replanning, plus failure and nonprogress cases.

The planner, runner, and tests are additions. The existing Skills API,
controllers, scene, grasp implementation, and deterministic scripted runner
remain unchanged.


## Optional interactive run

`planner_drawer_task.py --viewer` uses one interactive session. Closing its window
terminates execution normally; it does not produce a planner FAILURE or start a
new trial. The runner closes its Skills resources in `finally`. Without
`--viewer`, execution remains headless and uses the requested trial count.
