"""Deterministic contact-only pick and place, shared by headless and viewer runs."""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import simulate as grasp


class PickPlace:
    """Execute one physical pick/place cycle without modifying grasp geometry."""

    def __init__(self, model, data, destination, backend=grasp):
        """Initialize a cycle from the reset scene and desired cube-center destination."""
        self.grasp = backend
        self.model, self.data = model, data
        self.destination = np.asarray(destination, dtype=float)
        self.cube = self.grasp.get_cube_geom_id(model)
        self.site = self.grasp.get_end_effector_site_id(model)
        self.cube_dof = int(model.jnt_dofadr[model.joint("target_joint").id])
        self.table_z = float(model.geom("floor").pos[2])
        self.initial = data.body("target").xpos.copy()
        self.failure = None
        self.max_z = self.initial[2]
        self.min_transfer_bottom = np.inf
        self.phase = "HOME"
        self.result = None

    def bottom(self):
        """Return the cube's lowest world-space height, including its rotation."""
        rotation = self.data.geom_xmat[self.cube].reshape(3, 3)
        return float(
            self.data.geom_xpos[self.cube, 2]
            - np.abs(rotation[2]) @ self.model.geom_size[self.cube]
        )

    def move(self, name, target, duration, held=False):
        """Yield one control update per simulation step along a smooth joint path."""
        self.phase = name
        print(f"[AUTO] {name}")
        start = self.data.ctrl.copy()
        count = int(np.ceil(duration / self.model.opt.timestep))
        for i in range(count):
            if held and not self.grasp.cube_contacts_fixed_and_moving_jaw(
                self.model, self.data, self.cube
            ):
                raise RuntimeError(f"Opposing pad contact lost during {name}")
            if name == "ABOVE_TARGET":
                self.min_transfer_bottom = min(self.min_transfer_bottom, self.bottom())
                if self.bottom() < 0.025:
                    raise RuntimeError("Insufficient cube clearance during lateral transfer")
            alpha = 0.5 * (1 - np.cos(np.pi * (i + 1) / count))
            self.data.ctrl[:] = (1 - alpha) * start + alpha * target
            yield

    def cartesian_target(self, position, seed, opening):
        """Solve a site target using the existing grasp IK implementation."""
        return self.grasp.solve_site_position_ik(self.model, self.site, position, seed, opening)

    def sequence(self):
        """Run the successful grasp, then transfer, lower, release, and retreat."""
        with contextlib.redirect_stdout(io.StringIO()):
            targets = self.grasp.get_validated_auto_targets(self.model, self.data, self.site)
        for name in self.grasp.AUTO_SEQUENCE:
            duration = self.grasp.AUTO_VERIFY_HOLD_SECONDS if name == "GRASP_DEPTH" else 1.5
            if name == "CLOSE_GRIPPER":
                duration = self.grasp.GRIPPER_CLOSE_SECONDS
            yield from self.move(name, targets[name], duration, held=name == "LIFT")
            if name == "CLOSE_GRIPPER":
                tracker = self.grasp.GraspContactTracker()
                confirmed = False
                for _ in range(int(self.grasp.GRASP_SETTLE_SECONDS / self.model.opt.timestep)):
                    confirmed = tracker.update(
                        self.data.time,
                        self.grasp.cube_contacts_fixed_and_moving_jaw(
                            self.model, self.data, self.cube
                        ),
                    )
                    if confirmed:
                        break
                    yield
                if not confirmed:
                    raise RuntimeError("Opposing pad contact did not settle before lift")

        # Use the achieved cube/site offset; the cube is never attached or teleported.
        offset = self.data.site_xpos[self.site] - self.data.body("target").xpos
        above_cube = self.destination.copy()
        above_cube[2] = max(self.data.body("target").xpos[2], self.table_z + 0.065)
        above = self.cartesian_target(
            above_cube + offset, self.data.ctrl, self.grasp.GRIPPER_CLOSED_VALUE
        )
        yield from self.move("ABOVE_TARGET", above, 3.0, held=True)

        offset = self.data.site_xpos[self.site] - self.data.body("target").xpos
        resting_cube = self.destination.copy()
        resting_cube[2] += 0.002
        lower = self.cartesian_target(
            resting_cube + offset, self.data.ctrl, self.grasp.GRIPPER_CLOSED_VALUE
        )
        yield from self.move("LOWER_TO_TARGET", lower, 4.0, held=True)
        yield from self.move("SETTLE_AT_TARGET", lower, 0.5)
        position = self.data.body("target").xpos.copy()
        if (
            np.linalg.norm(position[:2] - self.destination[:2]) >= 0.025
            or not -0.001 <= self.bottom() - self.table_z <= 0.006
            or np.linalg.norm(self.data.qvel[self.cube_dof : self.cube_dof + 3]) > 0.02
        ):
            raise RuntimeError("Release blocked: cube is not safely near the target surface")
        release = self.grasp.get_gripper_target(self.model, lower, self.grasp.GRIPPER_OPEN_VALUE)
        yield from self.move("OPEN_GRIPPER_AT_TARGET", release, 3.0)
        yield from self.move("RELEASE_SETTLE", release, 1.0)
        retreat_pos = self.data.site_xpos[self.site].copy() + [0, 0, 0.08]
        retreat = self.cartesian_target(retreat_pos, release, self.grasp.GRIPPER_OPEN_VALUE)
        yield from self.move("RETREAT", retreat, 3.0)
        home = self.grasp.get_gripper_target(
            self.model, targets["HOME"], self.grasp.GRIPPER_OPEN_VALUE
        )
        yield from self.move("HOME", home, 3.0)
        yield from self.move("FINAL_SETTLE", home, 1.0)

    def validate(self):
        """Print placement outcome after the cube has had time to settle."""
        final = self.data.body("target").xpos.copy()
        distance = float(np.linalg.norm(final[:2] - self.destination[:2]))
        floor = self.model.geom("floor").id
        floor_contact = any(
            {int(c.geom1), int(c.geom2)} == {floor, self.cube} and c.dist <= 0
            for c in self.data.contact
        )
        resting = (
            abs(final[2] - self.destination[2]) < 0.005
            and np.linalg.norm(self.data.qvel[self.cube_dof : self.cube_dof + 6]) < 0.02
            and floor_contact
        )
        success = self.failure is None and distance < 0.03 and resting
        self.result = dict(
            success=bool(success),
            initial=self.initial.tolist(),
            target=self.destination.tolist(),
            final=final.tolist(),
            xy_distance=distance,
            final_z=float(final[2]),
            failure=self.failure,
        )
        print(f"Cube initial position: {self.initial}")
        print(f"Target position (cube center): {self.destination}")
        print(f"Cube final position: {final}")
        print(f"XY distance: {distance:.6f} m; final cube Z: {final[2]:.6f} m")
        print(
            f"Maximum cube Z: {self.max_z:.6f} m; "
            f"transfer clearance: {self.min_transfer_bottom:.6f} m"
        )
        print("SUCCESS" if success else f"FAILURE: {self.failure or 'placement/rest check'}")
        return self.result


def main():
    """Run repeatable headless resets or an interactive passive viewer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--target", type=float, nargs=3, help="Desired final cube-center XYZ")
    parser.add_argument("--report", type=Path, default=Path("pick_place_results.json"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "assets/scene.xml"))
    destination = (
        model.site("placement_target").pos.copy()
        if args.target is None
        else np.asarray(args.target)
    )
    if not np.all(np.isfinite(destination)):
        parser.error("Target coordinates must be finite")
    model.geom("placement_marker").pos[:2] = destination[:2]
    data = mujoco.MjData(model)

    def reset():
        """Reset physics and the controller together."""
        mujoco.mj_resetData(model, data)
        grasp.SO101Robot(model=model, data=data).reset("HOME")
        controller = PickPlace(model, data, destination)
        return controller, controller.sequence()

    def tick(controller, steps):
        """Advance exactly one physical step, reporting completion or failure."""
        try:
            next(steps)
        except StopIteration:
            controller.validate()
            return False
        except RuntimeError as error:
            controller.failure = str(error)
            controller.validate()
            return False
        mujoco.mj_step(model, data)
        controller.max_z = max(controller.max_z, float(data.body("target").xpos[2]))
        return True

    if args.headless:
        successes = 0
        results = []
        for run in range(args.runs):
            print(f"\nReset {run + 1}/{args.runs} (deterministic, no randomization)")
            controller, steps = reset()
            while tick(controller, steps):
                pass
            successes += controller.result["success"]
            results.append(controller.result)
        print(f"Success rate: {successes}/{args.runs}")
        args.report.write_text(
            json.dumps(
                dict(successes=successes, runs=args.runs, randomized=False, results=results),
                indent=2,
            )
        )
        if successes != args.runs:
            raise SystemExit(1)
    else:
        controller, steps = reset()
        active = True
        restart_requested = False

        def key_callback(key):
            """Keep manual pose/gripper controls; R requests a complete reset."""
            nonlocal active, restart_requested
            if key in (82, 114):
                restart_requested = True
            elif key in (49, 50, 51, 52, 32):
                active = False
                if key == 32:
                    index = model.actuator("gripper").id
                    data.ctrl[index] = (
                        grasp.GRIPPER_OPEN_VALUE
                        if data.ctrl[index] < 0.25
                        else grasp.GRIPPER_CLOSED_VALUE
                    )
                else:
                    pose = ("HOME", "REACH", "PICK", "STOW")[key - 49]
                    data.ctrl[:] = grasp.get_named_pose_target(model, pose)
                print("[MANUAL] Automatic cycle stopped; press R to reset and restart.")

        print("Viewer: R resets/restarts; 1–4 select poses; Space toggles the jaw.")
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.opt.geomgroup[3] = False
            previous_time = data.time
            while viewer.is_running():
                start = time.perf_counter()
                if restart_requested or data.time < previous_time:
                    controller, steps = reset()
                    active = True
                    restart_requested = False
                if active:
                    active = tick(controller, steps)
                else:
                    mujoco.mj_step(model, data)
                previous_time = data.time
                viewer.sync()
                time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))


if __name__ == "__main__":
    main()
