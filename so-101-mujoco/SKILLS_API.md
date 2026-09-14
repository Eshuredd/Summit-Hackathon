# High-level manipulation skills

`bimind.skills.Skills` owns one independent MuJoCo session. Calls are synchronous:
a skill executes its validated trajectory and returns measured state. Import it
from this repository's project directory, using the project virtual environment.

```python
from bimind.skills import Skills

skills = Skills(scene="drawer")
try:
    calls = [
        (skills.reset, ()),
        (skills.open_drawer, ("left",)),
        (skills.hold_drawer, ("left",)),
        (skills.pick_object, ("right", "drawer_object")),
        (skills.place_object, ("right", "table_target")),
        (skills.release_drawer, ("left",)),
        (skills.finish, ()),
    ]
    for skill, args in calls:
        result = skill(*args)
        if not result["success"]:
            print(result)
            break
    print(skills.get_run_report())
finally:
    skills.close()
```

`pick_object` establishes opposing-pad support. `lift_object` is separately
available. `place_object` and `handoff_object` run that same calibrated lift first
if it has not already happened. Both drawer call styles produce the same motion.

## Modules

- `bimind/skills.py`: public API, preconditions, structured results, live state,
  failure containment, and trial reports.
- `bimind/controllers.py`: the extracted drawer controller and adapters for the
  existing free-cube controller. Trajectory values, durations, physics stepping,
  contact checks, and collision preflight are preserved.
- `bimanual_drawer_task.py`: CLI arguments, high-level skill ordering, and JSON
  reporting. It contains no MuJoCo controls, IK, or grasp implementation.
- `test_skills.py`: observation, result, ordering, and failure-isolation tests.
- `validate_skills.py`: physical execution comparisons against the original
  handoff controller and the saved drawer baseline.
- `tests/fixtures/drawer_baseline.json`: the successful pre-refactor drawer trial,
  including all original phase snapshots.

## API and supported paths

| Call | Behavior |
| --- | --- |
| `reset()` | Run initial settling once in a new session. |
| `open_drawer(arm)` | Left approaches, seats its grip, and pulls 60 mm. |
| `hold_drawer(arm)` | Left holds and enables monitoring on every subsequent physics step. |
| `pick_object(arm, object_name)` | Establish a real grasp, without lifting. |
| `lift_object(arm, object_name)` | Execute the cached calibrated lift. |
| `place_object(arm, target_name)` | Lift if needed, transfer, lower, release, and retreat to HOME. |
| `handoff_object(from_arm, to_arm, object_name)` | Transfer left to right, verify receiver support, and retreat the donor. |
| `release_drawer(arm)` | Release the left handle after placement, retreat, and return HOME. |
| `move_home(arm)` | Move an empty arm HOME, preserving the other arm's support checks. |
| `get_scene_state()` | Read observations without stepping simulation. |
| `finish()` | Validate completion and emit SUCCESS. |
| `get_run_report()` | Return original diagnostics plus skill results and final state. |
| `close()` | Close an optional viewer without changing physics. |

For `scene="drawer"`, the validated roles are left drawer operation and right
retrieval of `drawer_object`. For `scene="handoff"`, either arm can pick/place
`object`, and handoff is left-to-right. Both scenes expose `table_target`. The
existing XML names `target` and `placement_target` are accepted aliases.
Unsupported roles, names, or call order return failures before motion.

For a handoff session, call `reset`, `pick_object("left", "object")`, optionally
`lift_object`, `handoff_object("left", "right", "object")`,
`place_object("right", "table_target")`, and `finish`.

These are validated deterministic paths, not a general planner for arbitrary
objects, targets, directions, or scenes. Each session performs one sequence;
create a new instance for the next reset. No natural-language or model layer is
included. MuJoCo 3.10.0 and the existing lockfile remain enforced.

## Result and observation schemas

Every manipulation skill returns a JSON-compatible dictionary:

```text
{
  success: bool,
  skill: str,
  arm: "left" | "right" | null,
  reason: str | null,
  phase: str,
  state: SceneState
}
```

Handoff results additionally include `from_arm` and `to_arm`. A failure retains
the precise controller phase and live object/robot state. The first physical or planning failure
stops the session: further skill calls return failures without motion. There is
no rollback, automatic regrasp, or reset of live object positions. Create a new
session to recover. Scene construction/runtime configuration errors raise before
a usable session exists.

`SceneState` is:

```text
{
  scene, time, phase, execution_stage,
  drawer: null | {
    position, limits, open_fraction, is_open, is_closed,
    hold_monitor_active, handle_position
  },
  objects: { object_name: { position: [x,y,z], quaternion: [w,x,y,z] } },
  targets: { table_target: { position: [x,y,z] } },
  arms: {
    left|right: {
      joint_positions, controls, end_effector_position,
      gripper: { position, command, velocity, opening_commanded },
      holding: bool, holding_objects: [names]
    }
  },
  arm_arm_collisions: { active, active_count, total_count, pairs },
  failure: null | { skill, phase, reason }
}
```

Positions are metres, joint angles are radians, and time is simulation seconds.
`holding_objects` requires opposing inner-pad forces, rather than a closed-gripper
command or remembered ownership. `execution_stage` is workflow history; holding
is measured independently. Drawer opening is not clamped, so compliant limit
excursions remain visible.

## Headless validation

```powershell
.\.venv\Scripts\python.exe -m pytest -q test_skills.py test_drawer_guards.py test_handoff_guards.py test_dual_scene.py
.\.venv\Scripts\python.exe bimanual_drawer_task.py --runs 10
.\.venv\Scripts\python.exe validate_skills.py --drawer-report drawer_results.json --runs 10
.\.venv\Scripts\python.exe validate_skills.py --runs 10
```

The physical API validation runs the original handoff once, compares ten API
handoffs against it, tests each arm's independent pick/place, and checks the
implicit-lift drawer sequence against its baseline. It writes
`skill_regression_results.json` without overwriting `handoff_results.json`.


## Viewer termination

Viewer close is separate from task failure. A skill interrupted by the window X
returns `success: false`, `reason: null`, `terminated: true`, and
`termination_reason: "viewer_closed"`. The session report also has
`failure: null`; the physical failure latch is untouched. Further calls perform
no motion. Runners exit normally and close resources without another trial.
Genuine grasp, collision, drop, and planning failures retain their failure reason
and nonzero runner exit status, even if the window also closes. Interactive
runners execute one trial; headless `--runs` behavior is unchanged.
