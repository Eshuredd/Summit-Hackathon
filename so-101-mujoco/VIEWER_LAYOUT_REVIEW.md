# Dual-arm layout and viewer lifecycle review

## Orientation audit

The layout in `build_dual_scene.py`, `assets/dual_scene.xml`, and
`assets/drawer_scene.xml` intentionally uses opposite yaw rotations. No scene or
base transform was changed during this review.

| Robot | World base XYZ (m) | Yaw about world Z | Nominal local +X forward in world |
| --- | --- | --- | --- |
| Left | `(0, -0.04, 0)` | `+90 degrees` | `+Y` |
| Right | `(0.36, 0.08, 0)` | `-90 degrees` | `-Y` |

The model's nominal forward direction is local +X, verified by zero-pan forward
kinematics: its HOME gripper is approximately `(0.39136, -0.00001, 0.22647)` in
the single-arm frame. The mounting transforms send local -Y inward (+X for left,
-X for right). The robots therefore turn their shoulders into the shared central
workspace; their zero-pan forward axes do not point straight at each other.
Rotating them merely to align their appearance would invalidate the working paths.

The robots have identical kinematic ranges, and their useful reachable regions
overlap around the drawer and table. Position-only IK from both bases reaches the
closed/open handle, exposed cube, table target, and a shared point above the
workspace with less than 0.02 mm residual at each sampled point. These are
kinematic point-reach checks, not proof that every grasp orientation/path is
collision-free. The successful physical drawer task and shared-object handoff
provide the separate contact/collision evidence.

The layout is functional but the current trajectories have tight joint margins.
An every-physics-step audit of the successful drawer sequence measured:

| Arm / joint | Minimum remaining margin | Phase |
| --- | --- | --- |
| Left shoulder pan | 4.03 degrees | LEFT_APPROACH_HANDLE |
| Left shoulder lift | 11.45 degrees | LEFT_RETREAT |
| Left wrist flex | at upper stop | LEFT_RETREAT |
| Right shoulder pan | at upper stop | RIGHT_RETREAT |
| Right wrist flex | 1.00 degree | RIGHT_RETREAT |

Small compliant excursions at the stops were approximately 0.004 degree at the
left wrist and 0.007 degree at the right shoulder. Neither robot has a materially
smaller intrinsic reach envelope, but the role-specific configurations are not
uniformly comfortable: the right shoulder retreat and left wrist retreat are
limit-constrained. The current task still completes without arm-arm collisions;
there is no demonstrated unreachable task point requiring a base rotation.
Physics, limits, and trajectories remain unchanged. Detailed measurements are in
`orientation_audit.json`.

## Root cause and fix

The drawer controller formerly converted `viewer.is_running() == False` into a
`RuntimeError`. Skills caught it as a physical failure, and the drawer runner's
outer `--runs` loop constructed a new scene/viewer. The original handoff runner
also iterated multiple interactive trials, and did not check for early window
closure during its motion loops. The dual pick/place loop did stop on close, but
incorrectly classified it as manipulation failure.

`viewer_lifecycle.ViewerClosed` now represents user cancellation independently of
`RuntimeError`. Controllers check the viewer before physics/command updates and
around viewer sync. Skills expose explicit termination without setting their
physical failure latch. Interactive drawer, planner, and handoff runners execute
one trial and always close the viewer in `finally`. Dual pick/place recognizes
closure as cancellation and does not run placement-failure validation on an
interrupted cycle. Genuine failures retain nonzero exit status.

`simulate.py` already had a `while viewer.is_running()` loop. It now checks again
after sync before reset detection, preventing a closed window from triggering
that branch. Explicit resets while a live viewer remains open are preserved.
`pick_place.py` already has a context-managed single viewer and checks liveness
before reset logic on the next iteration; it needs no change. `validate_skills.py`
uses only headless sessions and already closes them in `finally`.

Tests use passive-viewer doubles (no desktop window) to exercise close before
motion, close during sync, no automatic relaunch with `--runs 10`, cleanup, and
physical failures concurrent with close. Headless regression compares physical
diagnostics against the saved baseline.
