"""Deterministic passive-drawer manipulation using only robot contact forces."""

import argparse
import contextlib
import io
import json

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

import simulate as grasp
from bimanual_handoff import Handoff
from dual_pick_place import ASSETS, ArmBackend, check_runtime, verify_layout


class DrawerTask(Handoff):
    """Reuse collision preflight and opposing-pad checks for a passive drawer."""

    def __init__(self, viewer=False):
        """Reset a separate scene before starting contact-only physics."""
        self.model = mujoco.MjModel.from_xml_path(str(ASSETS / "drawer_scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.data.ctrl[:] = np.tile(grasp.PRESET_POSES["HOME"], 2)
        for i in range(self.model.nu):
            self.data.qpos[self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]] = (
                self.data.ctrl[i]
            )
        mujoco.mj_forward(self.model, self.data)
        self.backends = {a: ArmBackend(self.model, self.data, a) for a in ("left", "right")}
        self.cube = grasp.get_cube_geom_id(self.model)
        self.handle = self.model.geom("drawer_handle").id
        self.slide = self.model.jnt_qposadr[self.model.joint("drawer_slide").id]
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if viewer else None
        if self.viewer:
            self.viewer.opt.geomgroup[3] = False
            self.viewer.cam.lookat[:] = [0.18, 0, 0.07]
            self.viewer.cam.distance = 0.75
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -45
        self.phase = "RESET"
        self.history = []
        self.collisions = 0
        self.max_drop = 0.0
        self.max_open = self.opening()
        self.initial_open = self.opening()
        self.initial_object = self.position()
        self.max_z = self.initial_object[2]
        self.maintain = False
        self.min_held_open = None
        self.grasp_inside = False
        self.clear = False
        self.transport = False
        self.transport_peak = None
        self.handle_contacts = 0
        self.object_contacts = 0

    def opening(self):
        """Return the live passive slide displacement in metres."""
        return float(self.data.qpos[self.slide])

    def supports(self, arm):
        """Require opposing inner pad forces on the handle or cube."""
        geom = self.handle if arm == "left" else self.cube
        return bool(
            self.backends[arm].cube_contacts_fixed_and_moving_jaw(self.model, self.data, geom)
        )

    def counts(self, arm, geom):
        """Count actual contact points with the selected manipulated geometry."""
        return len(
            grasp.get_finger_cube_contacts(
                self.model, self.data, geom, list(self.backends[arm].allowed)
            )
        )

    def tick(self, required=()):
        """Step physics, checking collisions, grasp support, and drawer clearance."""
        mujoco.mj_step(self.model, self.data)
        self.max_open = max(self.max_open, self.opening())
        self.max_z = max(self.max_z, self.position()[2])
        self.handle_contacts = max(self.handle_contacts, self.counts("left", self.handle))
        self.object_contacts = max(self.object_contacts, self.counts("right", self.cube))
        contacts = self.arm_contacts()
        self.collisions += len(contacts)
        if contacts:
            raise RuntimeError("Arm-arm collision")
        for arm in required:
            if not self.supports(arm):
                raise RuntimeError(f"Lost {arm} opposing pad support")
        if self.maintain:
            self.min_held_open = min(self.min_held_open, self.opening())
            if self.opening() < 0.048 or not self.supports("left"):
                raise RuntimeError("Left failed to maintain drawer open with handle support")
        if self.transport:
            self.transport_peak = max(self.transport_peak, self.position()[2])
            self.max_drop = max(self.max_drop, self.transport_peak - self.position()[2])
            if self.max_drop > 0.01:
                raise RuntimeError("Object dropped more than 10 mm during transport")
        if self.viewer and not self.viewer.is_running():
            raise RuntimeError("Viewer closed before task completion")
        self.sync_viewer()

    def mark(self, phase):
        """Emit the required phase marker and a measured state snapshot."""
        self.phase = phase
        print(f"[DRAWER] {phase}", flush=True)
        row = dict(
            phase=phase,
            drawer=self.opening(),
            object=self.position().tolist(),
            handle_contacts=self.counts("left", self.handle),
            object_contacts=self.counts("right", self.cube),
            left_support=self.supports("left"),
            right_support=self.supports("right"),
            arm_arm_contacts=len(self.arm_contacts()),
        )
        self.history.append(row)
        print(json.dumps(row), flush=True)

    def plan_grasp(self, arm, center):
        """Use the existing calibrated jaw alignment in a private planning model."""
        backend = self.backends[arm]
        scratch = mujoco.MjData(backend.local)
        adr = backend.local.jnt_qposadr[backend.local.joint("target_joint").id]
        scratch.qpos[adr : adr + 3] = backend.rotation.T @ (center - backend.base)
        mujoco.mj_forward(backend.local, scratch)
        with contextlib.redirect_stdout(io.StringIO()):
            targets = grasp.get_validated_auto_targets(backend.local, scratch, backend.local_site)
        return {name: backend.expand(q) for name, q in targets.items()}

    def translate(self, arm, delta, duration=4, required=()):
        """Follow short Cartesian segments using only actuator position commands."""
        backend = self.backends[arm]
        site = backend.get_end_effector_site_id(self.model)
        start = self.data.site_xpos[site].copy()
        opening = self.data.ctrl[backend.indices[5]]
        desired = self.data.body(f"{arm}_gripper").xmat.reshape(3, 3).copy()
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        addresses = self.model.jnt_qposadr[self.model.actuator_trnid[backend.indices, 0]]
        for fraction in np.linspace(0, 1, 31)[1:]:
            position = start + np.asarray(delta) * fraction
            if arm == "left" and self.phase == "LEFT_PULL_OPEN":

                def residual(q):
                    """Preserve the handle grasp orientation along the constrained slide."""
                    scratch.qpos[addresses[:5]] = q
                    mujoco.mj_forward(self.model, scratch)
                    rotation = scratch.body(f"{arm}_gripper").xmat.reshape(3, 3)
                    return np.r_[
                        30 * (scratch.site_xpos[site] - position),
                        Rotation.from_matrix(desired.T @ rotation).as_rotvec(),
                    ]

                bounds = self.model.actuator_ctrlrange[backend.indices[:5]].T
                solution = least_squares(
                    residual, self.data.ctrl[backend.indices[:5]], bounds=bounds
                )
                target = np.r_[solution.x, opening]

            else:
                target = backend.solve_site_position_ik(
                    self.model, site, position, self.data.ctrl, opening
                )
            self.move(arm, target, duration / 30, required)

    def bottom(self):
        """Return the cube bottom accounting for its current orientation."""
        return float(
            self.position()[2]
            - np.abs(self.data.geom_xmat[self.cube].reshape(3, 3)[2])
            @ self.model.geom_size[self.cube]
        )

    def run(self):
        """Open, maintain, retrieve, place, and release with phase-specific guards."""
        self.mark("RESET")
        self.hold(0.5)
        self.mark("LEFT_APPROACH_HANDLE")
        targets = self.plan_grasp("left", self.data.geom_xpos[self.handle].copy())
        for name in ("OPEN_GRIPPER_FULLY", "ABOVE_OBJECT", "DESCEND_WITH_JAWS_OPEN", "GRASP_DEPTH"):
            self.move("left", targets[name], 3)
        self.translate("left", [0.005, 0, 0], 2)
        self.mark("LEFT_GRASP_HANDLE")
        closed = self.data.ctrl[self.backends["left"].indices].copy()
        closed[5] = grasp.GRIPPER_CLOSED_VALUE
        self.move("left", closed, 4)
        self.hold(0.3, ("left",))
        self.mark("LEFT_PULL_OPEN")
        self.translate("left", [0, 0.06, 0], 6, ("left",))
        self.hold(0.5, ("left",))
        if self.opening() < 0.048:
            raise RuntimeError("Drawer did not reach 80% travel")
        self.mark("LEFT_HOLD_OPEN")
        self.maintain = True
        self.min_held_open = self.opening()
        self.hold(0.5, ("left",))
        self.mark("RIGHT_APPROACH_OBJECT")
        targets = self.plan_grasp("right", self.position())
        for name in ("OPEN_GRIPPER_FULLY", "ABOVE_OBJECT", "DESCEND_WITH_JAWS_OPEN", "GRASP_DEPTH"):
            self.move("right", targets[name], 4, ("left",))
        self.mark("RIGHT_GRASP_OBJECT")
        self.move("right", targets["CLOSE_GRIPPER"], 4, ("left",))
        self.hold(0.3, ("left", "right"))
        relative = self.position() - self.data.body("drawer").xpos
        self.grasp_inside = bool(
            abs(relative[0]) < 0.073 and abs(relative[1]) < 0.042 and self.bottom() < 0.037
        )
        if not self.grasp_inside:
            raise RuntimeError("Object was not grasped inside drawer")
        self.mark("RIGHT_LIFT_OBJECT")
        self.move("right", targets["LIFT"], 6, ("left", "right"))
        self.clear = self.bottom() > 0.054
        if not self.clear:
            raise RuntimeError("Object did not clear drawer rim by 10 mm")
        self.mark("RIGHT_MOVE_TO_TABLE")
        self.transport = True
        self.transport_peak = self.position()[2]
        destination = self.model.site("placement_target").pos.copy()
        above = destination.copy()
        above[2] = self.position()[2]
        start = self.position()
        for fraction in np.linspace(0, 1, 31)[1:]:
            waypoint = start + fraction * (above - start)
            self.move("right", self.cube_target("right", waypoint), 0.25, ("left", "right"))
        self.transport = False
        self.mark("RIGHT_PLACE_OBJECT")
        self.move(
            "right", self.cube_target("right", destination + [0, 0, 0.002]), 6, ("left", "right")
        )
        self.hold(0.5, ("left",))
        if (
            not -0.001 <= self.bottom() <= 0.006
            or np.linalg.norm(self.position()[:2] - destination[:2]) >= 0.03
        ):
            raise RuntimeError("Release blocked: object not at table target")
        opened = self.data.ctrl[self.backends["right"].indices].copy()
        opened[5] = grasp.GRIPPER_OPEN_VALUE
        self.move("right", opened, 4, ("left",))
        self.hold(0.5, ("left",))
        self.mark("RIGHT_RETREAT")
        self.translate("right", [0.06, 0, 0.03], 4, ("left",))
        self.move("right", grasp.PRESET_POSES["HOME"], 4, ("left",))
        self.mark("LEFT_RELEASE_HANDLE")
        self.maintain = False
        opened = self.data.ctrl[self.backends["left"].indices].copy()
        opened[5] = grasp.GRIPPER_OPEN_VALUE
        self.move("left", opened, 4)
        self.mark("LEFT_RETREAT")
        self.translate("left", [0, 0, 0.08], 4)
        self.move("left", grasp.PRESET_POSES["HOME"], 4)
        self.hold(1)
        floor = self.model.geom("floor").id
        on_table = any(
            {int(c.geom1), int(c.geom2)} == {floor, self.cube} and c.dist <= 0
            for c in self.data.contact
        )
        dof = self.model.jnt_dofadr[self.model.joint("target_joint").id]
        if (
            not on_table
            or np.linalg.norm(self.position()[:2] - destination[:2]) >= 0.03
            or np.linalg.norm(self.data.qvel[dof : dof + 6]) > 0.02
        ):
            raise RuntimeError("Final placement/rest check failed")
        self.mark("SUCCESS")

    def result(self, failure):
        """Return measured outcomes, including incomplete phase evidence."""
        return dict(
            success=failure is None,
            failure=failure,
            phase=self.phase,
            drawer_initial_position=self.initial_open,
            drawer_max_opening=self.max_open,
            drawer_final_position=self.opening(),
            handle_contact_count=self.handle_contacts,
            maintains_open=self.min_held_open is not None and self.min_held_open >= 0.048,
            minimum_retrieval_opening=self.min_held_open,
            object_contact_count=self.object_contacts,
            object_lift_height=float(self.max_z - self.initial_object[2]),
            object_final_position=self.position().tolist(),
            object_final_xy_error=float(
                np.linalg.norm(self.position()[:2] - self.model.site("placement_target").pos[:2])
            ),
            arm_arm_collisions=self.collisions,
            object_drop_distance=self.max_drop,
            grasped_inside_drawer=self.grasp_inside,
            lifted_clear=bool(self.clear),
            diagnostics=self.history,
        )


def main():
    """Run repeatable resets and save independent viewer and batch reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    runtime = check_runtime()
    results = []
    for run in range(args.runs):
        task = DrawerTask(args.viewer)
        verify_layout(task.model)
        failure = None
        try:
            task.run()
        except (RuntimeError, ValueError) as error:
            failure = str(error)
            print(f"[DRAWER] FAILURE in {task.phase}: {failure}", flush=True)
        finally:
            results.append(dict(run=run + 1, runtime=runtime, **task.result(failure)))
            if task.viewer:
                task.viewer.close()
        print(json.dumps(results[-1]), flush=True)
    path = ASSETS.parent / ("drawer_viewer_results.json" if args.viewer else "drawer_results.json")
    path.write_text(json.dumps(results, indent=2))
    successes = sum(r["success"] for r in results)
    print(f"Drawer success: {successes}/{args.runs}", flush=True)
    if successes != args.runs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
