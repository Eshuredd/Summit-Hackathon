# Deterministic bimanual drawer task

Run from `so-101-mujoco` using the validated project environment:

```powershell
.\.venv\Scripts\python.exe bimanual_drawer_task.py --runs 10
```

The script checks the existing lockfile versions, including MuJoCo 3.10.0.
Headless execution is the default. `--viewer` remains optional and closes when
its trial finishes; closing it early records a failure. The requested final
validation uses headless execution only.

## Scene layout

All coordinates are world XYZ in metres. The existing dual SO-101 robot bodies,
actuators, and collision geometry are copied unchanged from `dual_scene.xml`.

| Element | Initial pose / setting |
| --- | --- |
| Drawer tray origin | `(0.19, -0.11, 0.008)` |
| Drawer slide axis | positive world Y |
| Slide limits | `[0, 0.060]`, initially `0` |
| Handle centre | `(0.12, -0.035, 0.050)` |
| Cube centre | `(0.22, -0.11, 0.031)` |
| Table target, cube centre | `(0.20, 0.08, 0.015)` |

The drawer is a passive, lightweight tray with a floor, four walls, a visible
handle, and a surrounding housing. The slide constraint represents its guides;
there is no drawer actuator, return motor, attachment, or equality constraint.
The left robot pulls the handle through physical contacts and holds its servo
command while the right robot retrieves and places the cube.

## Control and validation

The controller reuses the handoff's arm-arm path preflight, actual collision
checks, calibrated grasp planner, and opposing inner-pad force checks. The
left handle approach includes a 5 mm seating correction because a constrained
handle cannot move sideways into the fixed pad like a free cube. The pull uses
pose IK to preserve the grasp orientation. The right arm uses the original
calibrated lift, followed by short placement-transfer waypoints. Only robot
actuator controls change during execution. Kinematic calculations write to
private scratch data; live object and slide positions are never assigned.

Each physics step checks arm-arm contacts and required grasp support. During
retrieval the drawer must remain at least 48 mm open, with opposing left-pad
support. Before lifting, the cube must be grasped inside the tray. Its bottom
must clear the 44 mm drawer rim by at least 10 mm before moving toward the table;
this route travels away from the housing roof. Release requires the cube to be
near the table and within the target tolerance. Final success requires table
contact, less than 30 mm XY error, and settled object velocity.

`drawer_results.json` records every trial and its phase snapshots, including:

- Initial, maximum, final, and minimum retrieval drawer displacement.
- Maximum handle and object contact counts and opposing support at phase boundaries.
- Object lift height, final position, XY error, and unintended transport descent.
- Inside-drawer grasp and rim-clearance evidence, collision count, and failure phase.

The drop metric is peak-to-current descent during lateral transport; commanded
lowering onto the table is excluded. MuJoCo's compliant joint limit can permit
small excursions beyond 60 mm, which are reported without clamping.

The ten trials are identical deterministic resets, not randomized robustness
tests. The right retreat can nudge the resting cube; the final success check runs
after both arms retreat, so it includes that displacement.

Run the structural and negative guard checks with:

```powershell
.\.venv\Scripts\python.exe test_drawer_guards.py
```

These verify unchanged robot geometry, a passive drawer with no welds, and
rejection of missing handle support, a closed drawer during retrieval, and an
excessive transport drop. Existing handoff files and results are not rewritten
by the drawer task.
