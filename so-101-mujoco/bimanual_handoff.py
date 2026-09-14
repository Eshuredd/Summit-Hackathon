"""Deterministic physical handoff with strict contact and collision validation."""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import simulate as grasp
from dual_pick_place import ASSETS, ArmBackend, check_runtime, verify_layout
from viewer_lifecycle import ViewerClosed, check_viewer


class Handoff:
    """Execute slow arm motions and fail closed on collision or lost support."""

    def __init__(self, viewer=False):
        """Load the validated dual scene without changing physical properties."""
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS / "dual_scene.xml"))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        self.backends = {a: ArmBackend(self.model, self.data, a) for a in ("left", "right")}
        self.cube = grasp.get_cube_geom_id(self.model)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if viewer else None
        if self.viewer:
            self.viewer.opt.geomgroup[3] = False
            self.viewer.cam.lookat[:] = [0.18, 0.04, 0.1]
            self.viewer.cam.distance = 0.8
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -55
        self.phase = "INIT"
        self.max_drop = 0.0
        self.transfer_z = None
        self.collisions = 0
        self.history = []
        self.left_initial_hold = False
        self.dual_verified = False
        self.right_only_verified = False
        self.require_left_clear = False

    def position(self):
        """Return the live cube center."""
        return self.data.body("target").xpos.copy()

    def supports(self, arm):
        """Require opposing inner-pad forces from the selected arm."""
        return bool(
            self.backends[arm].cube_contacts_fixed_and_moving_jaw(self.model, self.data, self.cube)
        )

    def arm_contacts(self, data=None):
        """Return all actual cross-arm contacts, leaving collision detection enabled."""
        data = self.data if data is None else data
        left, right = self.backends["left"].allowed, self.backends["right"].allowed
        return [
            c
            for c in data.contact
            if c.dist <= 0
            and ((c.geom1 in left and c.geom2 in right) or (c.geom2 in left and c.geom1 in right))
        ]

    def diagnostics(self):
        """Print the cube, both end effectors, counts, and current support state."""
        counts = {
            a: len(
                grasp.get_finger_cube_contacts(self.model, self.data, self.cube, list(b.allowed))
            )
            for a, b in self.backends.items()
        }
        supports = [a for a in self.backends if self.supports(a)]
        row = dict(
            phase=self.phase,
            cube=self.position().tolist(),
            left_ee=self.data.site("left_gripperframe").xpos.tolist(),
            right_ee=self.data.site("right_gripperframe").xpos.tolist(),
            contacts=counts,
            arm_arm_contacts=len(self.arm_contacts()),
            support=supports,
            cube_z=float(self.position()[2]),
        )
        self.history.append(row)
        print(json.dumps(row))

    def sync_viewer(self):
        """Refresh the MuJoCo viewer at the physics timestep rate when enabled."""
        if self.viewer is None:
            return
        check_viewer(self.viewer)
        start = time.perf_counter()
        self.viewer.sync()
        check_viewer(self.viewer)
        time.sleep(max(0, self.model.opt.timestep - (time.perf_counter() - start)))

    def tick(self, required=()):
        """Advance physics and enforce real contact safety at every timestep."""
        check_viewer(self.viewer)
        mujoco.mj_step(self.model, self.data)
        contacts = self.arm_contacts()
        if contacts:
            self.collisions += len(contacts)
            pairs = [
                (self.model.geom(c.geom1).name, self.model.geom(c.geom2).name, c.dist)
                for c in contacts
            ]
            raise RuntimeError(f"Arm-arm collision: {pairs}")
        for arm in required:
            if not self.supports(arm):
                raise RuntimeError(f"Lost {arm} opposing support")
        if self.require_left_clear and grasp.get_finger_cube_contacts(
            self.model, self.data, self.cube, list(self.backends["left"].allowed)
        ):
            raise RuntimeError("Left contacted cube during the right-only support check")
        if self.transfer_z is not None:
            self.max_drop = max(self.max_drop, self.transfer_z - self.position()[2])
            if self.max_drop > 0.01:
                raise RuntimeError("Cube dropped more than 10 mm during transfer")
        self.sync_viewer()

    def move(self, arm, target, duration=3, required=()):
        """Move only the selected six controls while the other arm holds its command."""
        check_viewer(self.viewer)
        ids = self.backends[arm].indices
        start = self.data.ctrl[ids].copy()
        target = np.asarray(target)[ids] if len(target) == 12 else np.asarray(target)
        # A closing servo target lies inside the cube, so its physical stopping angle
        # comes from contact dynamics; enforce collision safety on every real step.
        if self.phase != "RIGHT_GRASP":
            self.preflight(arm, target)
        steps = int(duration / self.model.opt.timestep)
        for i in range(steps):
            check_viewer(self.viewer)
            alpha = 0.5 * (1 - np.cos(np.pi * (i + 1) / steps))
            self.data.ctrl[ids] = start + alpha * (target - start)
            self.tick(required)

    def preflight(self, arm, target, start=None):
        """Reject a sampled joint path with any arm-arm overlap before execution."""
        ids = self.backends[arm].indices
        addresses = self.model.jnt_qposadr[self.model.actuator_trnid[ids, 0]]
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        start = self.data.qpos[addresses].copy() if start is None else start
        for alpha in np.linspace(0, 1, 101):
            check_viewer(self.viewer)
            scratch.qpos[addresses] = start + alpha * (target - start)
            mujoco.mj_forward(self.model, scratch)
            if self.arm_contacts(scratch):
                raise RuntimeError(f"Planned {arm} path intersects other arm at {alpha:.2f}")

    def hold(self, seconds, required=()):
        """Allow the physical contacts to settle under unchanged servo controls."""
        for _ in range(int(seconds / self.model.opt.timestep)):
            self.tick(required)

    def mark(self, phase):
        """Start a named phase and record its incoming physical state."""
        self.phase = phase
        print(f"[HANDOFF] {phase}")
        self.diagnostics()

    def cube_target(self, arm, center):
        """Move a held cube using the achieved end-effector offset."""
        backend = self.backends[arm]
        site = backend.get_end_effector_site_id(self.model)
        offset = self.data.site_xpos[site] - self.position()
        return backend.solve_site_position_ik(
            self.model, site, center + offset, self.data.ctrl, grasp.GRIPPER_CLOSED_VALUE
        )

    def receive_pose(self, center, offset, opening=0.5):
        """Fit a side-on receiver to the cube using bounded five-joint pose IK."""
        backend = self.backends["right"]
        ids = backend.indices
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        joints = self.model.actuator_trnid[ids, 0]
        addresses = self.model.jnt_qposadr[joints]
        z_axis = backend.base - center
        z_axis[2] = 0
        z_axis /= np.linalg.norm(z_axis)
        z_axis = (z_axis + np.array([0, 0, 1])) / np.sqrt(2)
        x_axis = np.cross([0, 0, 1], z_axis)
        x_axis /= np.linalg.norm(x_axis)
        desired = np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))
        body = self.model.body("right_gripper").id

        def residual(q):
            """Match the cube proxy and receiver orientation without a global branch jump."""
            scratch.qpos[addresses[:5]] = q
            scratch.qpos[addresses[5]] = opening
            mujoco.mj_forward(self.model, scratch)
            rotation = scratch.xmat[body].reshape(3, 3)
            point = scratch.xpos[body] + rotation @ offset
            return np.r_[
                20 * (point - center), Rotation.from_matrix(desired.T @ rotation).as_rotvec()
            ]

        best = None
        for seed in (
            self.data.qpos[addresses[:5]],
            [1.1, -0.4, 0.5, 0.2, 1.5],
            [1.1, 0.4, -0.5, -0.2, -1.5],
        ):
            bounds = self.model.actuator_ctrlrange[ids[:5]].T
            solved = least_squares(residual, np.clip(seed, *bounds), bounds=bounds, max_nfev=250)
            if best is None or np.linalg.norm(solved.fun) < np.linalg.norm(best.fun):
                best = solved
        residual(best.x)
        result = np.r_[best.x, opening]
        print(
            "Receiver IK residual",
            best.fun,
            "q",
            result,
            "planned cross contacts",
            len(self.arm_contacts(scratch)),
        )
        if np.linalg.norm(best.fun[:3]) > 0.08:
            raise RuntimeError("Receiver IK position error exceeds 4 mm")
        return result

    def withdraw_pose(self, arm, distance):
        """Retract along the gripper axis while preserving its present orientation."""
        ids = self.backends[arm].indices
        addresses = self.model.jnt_qposadr[self.model.actuator_trnid[ids, 0]]
        body = self.model.body(f"{arm}_gripper").id
        desired = self.data.xmat[body].reshape(3, 3).copy()
        position = self.data.xpos[body] + desired[:, 2] * distance
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos

        def residual(q):
            """Match palm translation and preserve orientation around the other fingers."""
            scratch.qpos[addresses[:5]] = q
            mujoco.mj_forward(self.model, scratch)
            rotation = scratch.xmat[body].reshape(3, 3)
            return np.r_[
                20 * (scratch.xpos[body] - position),
                Rotation.from_matrix(desired.T @ rotation).as_rotvec(),
            ]

        solved = least_squares(
            residual, self.data.qpos[addresses[:5]], bounds=self.model.actuator_ctrlrange[ids[:5]].T
        )
        return np.r_[solved.x, self.data.ctrl[ids[5]]]

    def run(self):
        """Pick left, present, establish right support, release left, and place right."""
        self.mark("LEFT_PICK")
        left = self.backends["left"]
        with contextlib.redirect_stdout(io.StringIO()):
            targets = left.get_validated_auto_targets(
                self.model, self.data, left.get_end_effector_site_id(self.model)
            )
        for name in grasp.AUTO_SEQUENCE:
            self.move(
                "left",
                targets[name],
                3 if name in ("CLOSE_GRIPPER", "GRASP_DEPTH") else 1.5,
                required=("left",) if name == "LIFT" else (),
            )
            if name == "CLOSE_GRIPPER":
                self.hold(0.5, ("left",))
                self.left_initial_hold = True
        self.mark("LEFT_TO_CENTER")
        center = np.array([0.18, 0.04, 0.14])
        self.move("left", self.cube_target("left", center), 4, ("left",))
        self.hold(0.5, ("left",))
        center = self.position()
        donor = self.data.body("left_gripper")
        offset = donor.xmat.reshape(3, 3).T @ (center - donor.xpos)
        self.mark("RIGHT_APPROACH")
        direction = self.backends["right"].base - center
        direction[2] = 0
        direction /= np.linalg.norm(direction)
        approach = self.receive_pose(center + (direction + [0, 0, 1]) * 0.035, offset)
        receive = self.receive_pose(center, offset)
        candidates = [
            np.array([0, s, e, w, 0, 0.5])
            for s in (-0.8, -1.3, 0.5)
            for e in (1.4, -0.8)
            for w in (1.2, 0)
        ]
        for staging in candidates:
            try:
                self.preflight("right", staging)
                self.preflight("right", approach, start=staging)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError("No collision-free receiving staging path")
        print("Receiver staging", staging)
        self.move("right", staging, 4, ("left",))
        self.move("right", approach, 4, ("left",))
        self.move("right", receive, 4, ("left",))
        self.mark("RIGHT_GRASP")
        closed = receive.copy()
        closed[5] = grasp.GRIPPER_CLOSED_VALUE
        self.move("right", closed, 4, ("left",))
        self.hold(0.25, ("left", "right"))
        if self.position()[2] < 0.05:
            raise RuntimeError("Cube is not safely above the table for transfer")
        self.dual_verified = True
        self.mark("DUAL_CONTACT_VERIFIED")
        self.transfer_z = float(self.position()[2])
        self.mark("LEFT_RELEASE")
        opened = self.data.ctrl[left.indices].copy()
        opened[5] = 0.34
        self.move("left", opened, 4, ("right",))
        retreat = self.withdraw_pose("left", 0.065)
        self.move("left", retreat, 4, ("right",))
        if grasp.get_finger_cube_contacts(self.model, self.data, self.cube, list(left.allowed)):
            raise RuntimeError("Left still contacts cube after withdrawal")
        self.require_left_clear = True
        self.hold(1, ("right",))
        self.right_only_verified = True
        self.mark("RIGHT_SUPPORT_VERIFIED")
        opened = self.data.ctrl[left.indices].copy()
        opened[5] = grasp.GRIPPER_OPEN_VALUE
        self.move("left", opened, 1.5, ("right",))
        self.move("left", targets["HOME"], 4, ("right",))
        self.transfer_z = None
        self.mark("RIGHT_TO_TARGET")
        destination = self.model.site("placement_target").pos.copy()
        self.move("right", self.cube_target("right", destination + [0, 0, 0.06]), 4, ("right",))
        self.mark("PLACE")
        self.move("right", self.cube_target("right", destination + [0, 0, 0.002]), 4, ("right",))
        self.hold(0.5)
        bottom = (
            self.position()[2]
            - np.abs(self.data.geom_xmat[self.cube].reshape(3, 3)[2])
            @ self.model.geom_size[self.cube]
        )
        dof = self.model.jnt_dofadr[self.model.joint("target_joint").id]
        if (
            not -0.001 <= bottom <= 0.006
            or np.linalg.norm(self.data.qvel[dof : dof + 3]) > 0.02
            or np.linalg.norm(self.position()[:2] - destination[:2]) >= 0.03
        ):
            raise RuntimeError("Release blocked: cube is not settled near target surface")
        opened = self.data.ctrl[self.backends["right"].indices].copy()
        opened[5] = grasp.GRIPPER_OPEN_VALUE
        self.move("right", opened, 4)
        self.hold(1)
        right = self.backends["right"]
        site = right.get_end_effector_site_id(self.model)
        retreat = right.solve_site_position_ik(
            self.model,
            site,
            self.data.site_xpos[site] + [0, 0, 0.08],
            self.data.ctrl,
            grasp.GRIPPER_OPEN_VALUE,
        )
        self.move("right", retreat, 4)
        self.move("right", grasp.PRESET_POSES["HOME"], 4)
        self.hold(1)
        if np.linalg.norm(self.position()[:2] - destination[:2]) >= 0.03:
            raise RuntimeError("Final XY placement error")
        if abs(self.position()[2] - 0.015) > 0.005:
            raise RuntimeError("Cube is not resting on table")
        floor = self.model.geom("floor").id
        if not any(
            {int(c.geom1), int(c.geom2)} == {floor, self.cube} and c.dist <= 0
            for c in self.data.contact
        ):
            raise RuntimeError("Cube has no table contact after placement")
        if np.linalg.norm(self.data.qvel[dof : dof + 6]) > 0.02:
            raise RuntimeError("Cube has not settled after placement")
        if not (self.left_initial_hold and self.dual_verified and self.right_only_verified):
            raise RuntimeError("Missing transfer verification")
        self.mark("SUCCESS")
        if self.viewer:
            try:
                while self.viewer.is_running():
                    self.hold(0.1)
            except ViewerClosed:
                pass  # The task already succeeded; close ends only the display loop.


def main():
    """Run headless handoff trials and write diagnostics for every outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    try:
        runtime = check_runtime()
    except ValueError as error:
        parser.error(
            str(error).replace(
                str(ASSETS.parent / "dual_pick_place.py"), str(Path(__file__).resolve())
            )
        )
    results = []
    trial_count = 1 if args.viewer else args.runs
    for run in range(trial_count):
        experiment = Handoff(viewer=args.viewer)
        terminated = False
        failure = None
        try:
            verify_layout(experiment.model)
            experiment.run()
            failure = None
        except ViewerClosed:
            terminated = True
            print("[VIEWER] Closed by user; exiting without another trial.")
        except (RuntimeError, ValueError) as error:
            failure = str(error)
            print(f"[HANDOFF] FAILURE in {experiment.phase}: {failure}")
            experiment.diagnostics()
        finally:
            if experiment.viewer is not None:
                experiment.viewer.close()
        results.append(
            dict(
                run=run + 1,
                success=failure is None and not terminated,
                failure=failure,
                phase=experiment.phase,
                max_cube_drop=experiment.max_drop,
                arm_arm_collisions=experiment.collisions,
                left_initial_hold=experiment.left_initial_hold,
                dual_contact_before_release=experiment.dual_verified,
                right_only_support=experiment.right_only_verified,
                runtime=runtime,
                final_position=experiment.position().tolist(),
                target_position=experiment.model.site("placement_target").pos.tolist(),
                final_xy_error=float(
                    np.linalg.norm(
                        experiment.position()[:2]
                        - experiment.model.site("placement_target").pos[:2]
                    )
                ),
                diagnostics=experiment.history,
            )
        )
        if terminated:
            results[-1].update(terminated=True, termination_reason="viewer_closed")
            break
    (ASSETS.parent / "handoff_results.json").write_text(json.dumps(results, indent=2))
    successes = sum(r["success"] for r in results)
    if any(r.get("terminated") for r in results):
        print("Handoff run terminated by user; no manipulation failure.")
    else:
        print(f"Handoff success: {successes}/{len(results)}")
    print(f"Maximum transfer drop: {max(r['max_cube_drop'] for r in results):.6f} m")
    print(f"Arm-arm collisions: {sum(r['arm_arm_collisions'] for r in results)}")
    if any(r["failure"] is not None for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
