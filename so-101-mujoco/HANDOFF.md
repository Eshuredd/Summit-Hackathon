# Deterministic SO-101 handoff

Run from this project directory with the validated project environment:

```powershell
.\.venv\Scripts\python.exe bimanual_handoff.py --runs 10
```

The script uses MuJoCo 3.10.0 and the existing `assets/dual_scene.xml` unchanged.
It does not modify `dual_pick_place.py`, the single-arm controller, the cube,
actuator forces, collision masks, or jaw meshes. The other environment at
`bindGIT/.venv` is not the validated runtime.

The left arm uses the existing successful pickup at `(0.18, 0, 0.015)` m and
presents the cube near `(0.18, 0.04, 0.14)` m. The actual grasp offset means the
achieved presentation is a few millimeters from that nominal target. The receiver
is planned against the live achieved cube position, not just the nominal target.

The right gripper approaches approximately 45 degrees downward from its side of
the workspace, closing across the two exposed side faces. A folded staging pose
avoids sweeping its fingers through the donor housing. Bounded pose IK sets the
receiving orientation; it does not move or attach the cube. Only servo controls
move the live robot. Separate scratch data is used for kinematic path checks.

After 0.25 seconds of uninterrupted valid opposing-pad contact on both arms, the
left jaw opens partially to 0.34 rad. This releases its pinch without swinging its
finger into the receiver. It withdraws 65 mm along its own tool axis while retaining
orientation, then right-only support is checked for one second. The left fully
opens only after clearing the receiver, then returns HOME.

The right transfers the cube to the existing target, lowers slowly, and releases
only near the table with low cube velocity. It retreats upward before returning
HOME. Final placement requires XY error below 30 mm, center Z within 5 mm of
15 mm, table contact, and low cube velocity.

Safety checks run at each 1 ms physics step. Any arm-arm contact or loss of required
opposing-pad support fails the trial. Motion paths are also sampled before moving;
closing targets cannot be checked as free-space joint paths because the cube stops
the jaw before the servo target. Closing still uses full live collision checks.
No collision exclusions, welds, fake attachments, or live cube teleports are used.

The maximum transfer drop is measured from the cube height immediately before left
release through right-only verification and left's return HOME. More than 10 mm
fails the trial. Planned lowering at the destination is excluded from that metric.

`handoff_results.json` contains each trial's runtime, verification flags, phase
diagnostics, cube/end-effector positions, contact counts, supporting arm(s), maximum
drop, arm-arm collisions, final placement error, and any failure reason. The script
exits nonzero if any trial fails. Runs are identical deterministic resets, not
randomized robustness tests.
