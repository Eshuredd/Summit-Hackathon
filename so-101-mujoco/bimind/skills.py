"""Instance-scoped, synchronous manipulation skills with measured scene state."""

from collections.abc import Callable
from typing import Any

import numpy as np

import simulate as grasp
from dual_pick_place import check_runtime, verify_layout

from .controllers import DrawerController, ObjectController


class Skills:
    """Own a scene and expose only validated contact-based manipulation paths.

    Args:
        scene: ``drawer`` for retrieval or ``handoff`` for free-cube manipulation.
        viewer: Whether to display the same physical execution in a viewer.
    """

    def __init__(self, scene: str = "drawer", viewer: bool = False):
        """Load the selected scene and enforce the validated runtime before motion."""
        if scene not in ("drawer", "handoff"):
            raise ValueError(f"Unknown scene: {scene}")
        self.runtime = check_runtime()
        self.scene = scene
        self._controller = (
            DrawerController(viewer) if scene == "drawer" else ObjectController(viewer)
        )
        verify_layout(self._controller.model)
        self._stage = "new"
        self._owner = None
        self._failure = None
        self.results = []
        self.object_name = "drawer_object" if scene == "drawer" else "object"

    def get_scene_state(self) -> dict[str, Any]:
        """Read live state without stepping physics or changing any controls.

        Returns:
            JSON-compatible drawer, object, target, robot, and collision measurements.
            Holding names require the existing opposing inner-pad force check.
        """
        c = self._controller
        model, data = c.model, c.data
        objects = {
            self.object_name: {
                "position": c.position().tolist(),
                "quaternion": data.body("target").xquat.tolist(),
            }
        }
        drawer = None
        if self.scene == "drawer":
            drawer = dict(
                position=c.opening(),
                limits=[0.0, 0.06],
                open_fraction=c.opening() / 0.06,
                is_open=c.opening() >= 0.048,
                is_closed=abs(c.opening()) <= 0.001,
                hold_monitor_active=c.maintain,
                handle_position=data.geom_xpos[c.handle].tolist(),
            )
        arms = {}
        for arm, backend in c.backends.items():
            joints = model.actuator_trnid[backend.indices, 0]
            held = []
            if backend.cube_contacts_fixed_and_moving_jaw(model, data, c.cube):
                held.append(self.object_name)
            if self.scene == "drawer" and backend.cube_contacts_fixed_and_moving_jaw(
                model, data, c.handle
            ):
                held.append("drawer_handle")
            gripper = joints[5]
            position = float(data.qpos[model.jnt_qposadr[gripper]])
            command = float(data.ctrl[backend.indices[5]])
            arms[arm] = dict(
                joint_positions=data.qpos[model.jnt_qposadr[joints]].tolist(),
                controls=data.ctrl[backend.indices].tolist(),
                end_effector_position=data.site(f"{arm}_gripperframe").xpos.tolist(),
                gripper=dict(
                    position=position,
                    command=command,
                    velocity=float(data.qvel[model.jnt_dofadr[gripper]]),
                    opening_commanded=command >= grasp.GRIPPER_OPEN_VALUE - 0.01,
                ),
                holding=bool(held),
                holding_objects=held,
            )
        contacts = c.arm_contacts()
        return dict(
            scene=self.scene,
            time=float(data.time),
            phase=c.phase,
            execution_stage=self._stage,
            drawer=drawer,
            objects=objects,
            targets={"table_target": {"position": model.site("placement_target").pos.tolist()}},
            arms=arms,
            arm_arm_collisions=dict(
                active=bool(contacts),
                active_count=len(contacts),
                total_count=c.collisions,
                pairs=[
                    [model.geom(int(x.geom1)).name, model.geom(int(x.geom2)).name] for x in contacts
                ],
            ),
            failure=self._failure,
        )

    def _execute(self, skill: str, arm: str | None, action: Callable[[], None]) -> dict[str, Any]:
        """Run a skill and latch failures so later calls cannot continue unsafe motion."""
        reason = None
        try:
            if self._failure is not None:
                raise RuntimeError("Execution stopped after failure; create a new Skills session")
            if arm is not None and arm not in ("left", "right"):
                raise ValueError(f"Unknown arm: {arm}")
            action()
        except (RuntimeError, ValueError, AssertionError) as error:
            reason = str(error)
            if self._failure is None:
                self._failure = dict(skill=skill, phase=self._controller.phase, reason=reason)
        result = dict(
            success=reason is None,
            skill=skill,
            arm=arm,
            reason=reason,
            phase=self._controller.phase,
            state=self.get_scene_state(),
        )
        self.results.append(result)
        return result

    def _require(self, condition: bool, reason: str):
        """Reject an unsupported request or invalid precondition before actuator motion."""
        if not condition:
            raise ValueError(reason)

    def _object(self, name: str):
        """Resolve the public object name or the existing XML body name."""
        self._require(name in (self.object_name, "target"), f"Unknown object: {name}")

    def _holding(self, arm: str, name: str) -> bool:
        """Check physical support rather than treating execution history as attachment."""
        return name in self.get_scene_state()["arms"][arm]["holding_objects"]

    def reset(self) -> dict[str, Any]:
        """Settle a newly constructed scene once; never teleport an active scene.

        Returns:
            A structured skill result. Create a new session for another trial.
        """

        def action():
            """Run only the original initial settling steps."""
            self._require(self._stage == "new", "Reset requires a new Skills session")
            if self.scene == "drawer":
                self._controller.reset_sequence()
            self._stage = "ready"

        return self._execute("reset", None, action)

    def open_drawer(self, arm: str) -> dict[str, Any]:
        """Approach, grasp, and pull the passive drawer along its proven path.

        Args:
            arm: ``left``; reverse-role drawer paths are not validated.

        Returns:
            Structured result with the measured opening and robot state.
        """

        def action():
            """Enforce the supported role before opening."""
            self._require(
                self.scene == "drawer" and arm == "left",
                "Drawer opening is validated for left in drawer scene only",
            )
            self._require(self._stage == "ready", "open_drawer requires reset")
            self._controller.open_drawer_sequence()
            self._stage = "open"

        return self._execute("open_drawer", arm, action)

    def hold_drawer(self, arm: str) -> dict[str, Any]:
        """Enable per-step opening and handle-support monitoring through retrieval.

        Args:
            arm: ``left``.

        Returns:
            Structured result after the original hold settling interval.
        """

        def action():
            """Begin persistent monitoring under the current servo commands."""
            self._require(
                self.scene == "drawer" and arm == "left",
                "Drawer hold is validated for left in drawer scene only",
            )
            self._require(
                self._stage == "open" and self._holding(arm, "drawer_handle"),
                "hold_drawer requires a grasped open drawer",
            )
            self._controller.hold_drawer_sequence()
            self._stage = "held"

        return self._execute("hold_drawer", arm, action)

    def pick_object(self, arm: str, object_name: str) -> dict[str, Any]:
        """Approach and establish opposing-pad support without yet lifting.

        Args:
            arm: Right for drawer retrieval; either arm for the free cube.
            object_name: ``drawer_object`` or ``object`` according to the scene.

        Returns:
            Structured result after physical grasp confirmation.
        """

        def action():
            """Use the scene's original calibrated grasp implementation."""
            self._object(object_name)
            if self.scene == "drawer":
                self._require(
                    arm == "right" and self._stage == "held",
                    "Drawer pick requires right arm and held drawer",
                )
                self._controller.pick_sequence()
            else:
                self._require(
                    self._stage == "ready", "Object pick requires reset and an unowned cube"
                )
                self._controller.pick_sequence(arm)
            self._owner = arm
            self._stage = "picked"

        return self._execute("pick_object", arm, action)

    def _lift(self, arm: str):
        """Execute one calibrated lift; callers validate ownership and object identity."""
        self._require(
            self._stage == "picked" and self._owner == arm,
            "Lift requires a preceding pick by this arm",
        )
        self._require(self._holding(arm, self.object_name), "Object has no opposing-pad support")
        if self.scene == "drawer":
            self._controller.lift_sequence()
        else:
            self._controller.lift_sequence(arm)
        self._stage = "lifted"

    def lift_object(self, arm: str, object_name: str) -> dict[str, Any]:
        """Lift using the exact cached trajectory from the preceding grasp.

        Args:
            arm: Arm that picked the object.
            object_name: Public scene object name or ``target``.

        Returns:
            Structured result including the measured lifted object position.
        """

        def action():
            """Resolve the object before executing its calibrated lift."""
            self._object(object_name)
            self._lift(arm)

        return self._execute("lift_object", arm, action)

    def place_object(self, arm: str, target_name: str) -> dict[str, Any]:
        """Lift if needed, transfer, place, and perform the proven retreat to HOME.

        Args:
            arm: Arm physically holding the object.
            target_name: ``table_target`` or the XML alias ``placement_target``.

        Returns:
            Structured result after release, retreat, and physical placement validation.
        """

        def action():
            """Preserve both explicit-lift and pick-then-place call sequences."""
            self._require(
                target_name in ("table_target", "placement_target"),
                f"Unknown target: {target_name}",
            )
            self._require(
                self._owner == arm and self._stage in ("picked", "lifted", "transferred"),
                "place_object requires an object held by this arm",
            )
            self._require(
                self._holding(arm, self.object_name), "Object has no opposing-pad support"
            )
            if self._stage == "picked":
                self._lift(arm)
            if self.scene == "drawer":
                self._controller.place_sequence()
                self._validate_placement()
            else:
                self._controller.place_sequence(arm)
            self._owner = None
            self._stage = "placed"

        return self._execute("place_object", arm, action)

    def handoff_object(self, from_arm: str, to_arm: str, object_name: str) -> dict[str, Any]:
        """Transfer a held free cube using the original left-to-right handoff.

        Args:
            from_arm: ``left``.
            to_arm: ``right``.
            object_name: ``object`` or ``target`` in the handoff scene.

        Returns:
            Structured result with donor and receiver state and support evidence.
        """

        def action():
            """Require the validated direction and a physically supported donor."""
            self._object(object_name)
            self._require(
                self.scene == "handoff" and from_arm == "left" and to_arm == "right",
                "Only left-to-right handoff in the handoff scene is validated",
            )
            self._require(
                self._owner == from_arm and self._stage in ("picked", "lifted"),
                "Handoff requires a preceding donor pick",
            )
            self._require(
                self._holding(from_arm, self.object_name), "Donor has no opposing-pad support"
            )
            if self._stage == "picked":
                self._lift(from_arm)
            self._controller.handoff_sequence()
            self._owner = to_arm
            self._stage = "transferred"

        result = self._execute("handoff_object", from_arm, action)
        result.update(from_arm=from_arm, to_arm=to_arm)
        return result

    def release_drawer(self, arm: str) -> dict[str, Any]:
        """Release the handle and execute the original left retreat and HOME motion.

        Args:
            arm: ``left`` after successful placement.

        Returns:
            Structured result after the original final settling interval.
        """

        def action():
            """Release only after retrieval has completed safely."""
            self._require(
                self.scene == "drawer" and arm == "left" and self._stage == "placed",
                "release_drawer requires left arm after placement",
            )
            self._controller.release_drawer_sequence()
            self._stage = "released"

        return self._execute("release_drawer", arm, action)

    def move_home(self, arm: str) -> dict[str, Any]:
        """Return an empty arm to HOME using the existing collision-checked motion.

        Args:
            arm: Empty left or right arm. Held objects must be placed or handed off first.

        Returns:
            Structured result after the four-second HOME trajectory.
        """

        def action():
            """Avoid opening a loaded gripper or disabling another arm's hold guard."""
            self._require(self._stage != "new", "move_home requires reset")
            self._require(
                not self.get_scene_state()["arms"][arm]["holding"],
                "move_home cannot move an arm holding an object or handle",
            )
            required = tuple(
                a for a in ("left", "right") if a != arm and self._controller.supports(a)
            )
            self._controller.move(arm, grasp.PRESET_POSES["HOME"], 4, required)

        return self._execute("move_home", arm, action)

    def _validate_placement(self):
        """Read the same table contact, settled velocity, and XY checks without stepping."""
        c = self._controller
        floor = c.model.geom("floor").id
        on_table = any(
            {int(x.geom1), int(x.geom2)} == {floor, c.cube} and x.dist <= 0 for x in c.data.contact
        )
        dof = c.model.jnt_dofadr[c.model.joint("target_joint").id]
        self._require(
            on_table
            and np.linalg.norm(c.position()[:2] - c.model.site("placement_target").pos[:2]) < 0.03
            and np.linalg.norm(c.data.qvel[dof : dof + 6]) <= 0.02,
            "Final placement/rest check failed",
        )

    def finish(self) -> dict[str, Any]:
        """Verify the completed sequence and emit its original success phase.

        Returns:
            Structured result; completion cannot be claimed before placement and release.
        """

        def action():
            """Run the unchanged final task checks without adding physics steps."""
            expected = "released" if self.scene == "drawer" else "placed"
            self._require(self._stage == expected, "Task sequence has not completed")
            if self.scene == "drawer":
                self._controller.finish_sequence()
            else:
                self._validate_placement()
                self._controller.mark("SUCCESS")
            self._stage = "complete"

        return self._execute("finish", None, action)

    def get_run_report(self) -> dict[str, Any]:
        """Return the original drawer diagnostics plus structured skill outcomes.

        Returns:
            JSON-compatible trial report; incomplete execution is never successful.
        """
        failure = self._failure["reason"] if self._failure else None
        if self._stage != "complete" and failure is None:
            failure = "Task sequence has not completed"
        c = self._controller
        report = (
            c.result(failure)
            if self.scene == "drawer"
            else dict(
                success=failure is None,
                failure=failure,
                phase=c.phase,
                arm_arm_collisions=c.collisions,
                max_cube_drop=c.max_drop,
                final_position=c.position().tolist(),
                final_xy_error=float(
                    np.linalg.norm(c.position()[:2] - c.model.site("placement_target").pos[:2])
                ),
                left_initial_hold=c.left_initial_hold,
                dual_contact_before_release=c.dual_verified,
                right_only_support=c.right_only_verified,
                diagnostics=c.history,
            )
        )
        return dict(report, runtime=self.runtime, skills=self.results, state=self.get_scene_state())

    def close(self):
        """Close the optional viewer without stepping or changing the physics state."""
        if self._controller.viewer:
            self._controller.viewer.close()
