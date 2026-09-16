"""Deterministic passive-drawer manipulation using only robot contact forces."""

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
from dual_pick_place import ASSETS, ArmBackend
from viewer_lifecycle import check_viewer


class DrawerController(Handoff):
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
        check_viewer(self.viewer)
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
        """Plan drawer grasps with a small robust depth bias for the object.

        Args:
            arm: ``left`` for the handle grasp or ``right`` for the object grasp.
            center: World-frame grasp center of the handle or object.

        Returns:
            Named full-model actuator targets for the automatic grasp sequence.
        """
        backend = self.backends[arm]
        scratch = mujoco.MjData(backend.local)

        adr = backend.local.jnt_qposadr[backend.local.joint("target_joint").id]

        scratch.qpos[adr : adr + 3] = backend.rotation.T @ (center - backend.base)

        mujoco.mj_forward(backend.local, scratch)

        # The drawer object needs 1.5 mm more grasp depth.
        # Keep the left drawer-handle grasp unchanged.
        grasp_depth_bias = 0.0015 if arm == "right" else 0.0

        with contextlib.redirect_stdout(io.StringIO()):
            targets = grasp.get_validated_auto_targets(
                backend.local,
                scratch,
                backend.local_site,
                grasp_depth_bias=grasp_depth_bias,
            )

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

    def reset_sequence(self):
        """Execute the unchanged reset sequence trajectory segment."""
        self.mark("RESET")
        self.hold(0.5)

    def open_drawer_sequence(self):
        """Execute the unchanged open drawer sequence trajectory segment."""
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

    def hold_drawer_sequence(self):
        """Execute the unchanged hold drawer sequence trajectory segment."""
        self.mark("LEFT_HOLD_OPEN")
        self.maintain = True
        self.min_held_open = self.opening()
        self.hold(0.5, ("left",))

    def pick_sequence(self):
        """Execute the unchanged pick sequence trajectory segment."""
        self.mark("RIGHT_APPROACH_OBJECT")
        targets = self.plan_grasp("right", self.position())
        for name in ("OPEN_GRIPPER_FULLY", "ABOVE_OBJECT", "DESCEND_WITH_JAWS_OPEN", "GRASP_DEPTH"):
            self.move("right", targets[name], 4, ("left",))
        self.pick_targets = targets
        self.mark("RIGHT_GRASP_OBJECT")
        self.move("right", targets["CLOSE_GRIPPER"], 4, ("left",))
        self.hold(0.3, ("left", "right"))
        relative = self.position() - self.data.body("drawer").xpos
        self.grasp_inside = bool(
            abs(relative[0]) < 0.073 and abs(relative[1]) < 0.042 and self.bottom() < 0.037
        )
        if not self.grasp_inside:
            raise RuntimeError("Object was not grasped inside drawer")

    def lift_sequence(self):
        """Execute the unchanged lift sequence trajectory segment."""
        self.mark("RIGHT_LIFT_OBJECT")
        self.move("right", self.pick_targets["LIFT"], 6, ("left", "right"))
        self.clear = self.bottom() > 0.054
        if not self.clear:
            raise RuntimeError("Object did not clear drawer rim by 10 mm")

    def place_sequence(self):
        """Execute the unchanged place sequence trajectory segment."""
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

        # Lift clear of the placed object and left arm before the normal retreat.
        self.translate("right", [0, 0, 0.015], 2, ("left",))

        # Keep the existing retreat after gaining clearance.
        self.translate("right", [0.06, 0, 0.03], 4, ("left",))

        self.move("right", grasp.PRESET_POSES["HOME"], 4, ("left",))

    def release_drawer_sequence(self):
        """Execute the unchanged release drawer sequence trajectory segment."""
        self.mark("LEFT_RELEASE_HANDLE")
        self.maintain = False
        opened = self.data.ctrl[self.backends["left"].indices].copy()
        opened[5] = grasp.GRIPPER_OPEN_VALUE
        self.move("left", opened, 4)
        self.mark("LEFT_RETREAT")
        self.translate("left", [0, 0, 0.08], 4)
        self.move("left", grasp.PRESET_POSES["HOME"], 4)
        self.hold(1)

    def finish_sequence(self):
        """Execute the unchanged finish sequence trajectory segment."""
        destination = self.model.site("placement_target").pos.copy()
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


class ObjectController(Handoff):
    """Adapt the proven free-cube pick, transfer, and place paths into segments."""

    def pick_sequence(self, arm):
        """Execute the original grasp sequence up to established pad support.

        Args:
            arm: Selected robot namespace.
        """
        self.mark(f"{arm.upper()}_PICK")
        backend = self.backends[arm]
        with contextlib.redirect_stdout(io.StringIO()):
            targets = backend.get_validated_auto_targets(
                self.model, self.data, backend.get_end_effector_site_id(self.model)
            )
        self.pick_targets = targets
        for name in grasp.AUTO_SEQUENCE:
            if name == "LIFT":
                break
            self.move(arm, targets[name], 3 if name in ("CLOSE_GRIPPER", "GRASP_DEPTH") else 1.5)
            if name == "CLOSE_GRIPPER":
                self.hold(0.5, (arm,))
                if arm == "left":
                    self.left_initial_hold = True

    def lift_sequence(self, arm):
        """Run the cached calibrated lift without changing the physical grasp.

        Args:
            arm: Arm that performed the immediately preceding pick.
        """
        self.move(arm, self.pick_targets["LIFT"], 1.5, (arm,))

    def handoff_sequence(self):
        """Execute the original left-to-right handoff and donor retreat."""
        left = self.backends["left"]
        targets = self.pick_targets
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

    def place_sequence(self, arm):
        """Execute the original table placement and retreat.

        Args:
            arm: Arm currently supporting the cube.
        """
        self.mark(f"{arm.upper()}_TO_TARGET")
        destination = self.model.site("placement_target").pos.copy()
        self.move(arm, self.cube_target(arm, destination + [0, 0, 0.06]), 4, (arm,))
        self.mark("PLACE")
        self.move(arm, self.cube_target(arm, destination + [0, 0, 0.002]), 4, (arm,))
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
        opened = self.data.ctrl[self.backends[arm].indices].copy()
        opened[5] = grasp.GRIPPER_OPEN_VALUE
        self.move(arm, opened, 4)
        self.hold(1)
        right = self.backends[arm]
        site = right.get_end_effector_site_id(self.model)
        retreat = right.solve_site_position_ik(
            self.model,
            site,
            self.data.site_xpos[site] + [0, 0, 0.08],
            self.data.ctrl,
            grasp.GRIPPER_OPEN_VALUE,
        )
        self.move(arm, retreat, 4)
        self.move(arm, grasp.PRESET_POSES["HOME"], 4)
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
