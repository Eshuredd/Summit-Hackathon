#!/usr/bin/env python3
"""Interactive 3D Simulation Viewer for SO-101 Robot Arm in MuJoCo."""

import os
import shutil
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.optimize import least_squares

from so101.robot import ACTUATOR_NAMES, PRESET_POSES, SO101Robot

AUTO_SEQUENCE = (
    "HOME",
    "OPEN_GRIPPER_FULLY",
    "ABOVE_OBJECT",
    "DESCEND_WITH_JAWS_OPEN",
    "GRASP_DEPTH",
    "CLOSE_GRIPPER",
    "LIFT",
)
AUTO_MOVE_SECONDS = 1.5
AUTO_HOLD_SECONDS = 0.6
AUTO_VERIFY_HOLD_SECONDS = 3.0
GRIPPER_OPEN_VALUE = 0.5
GRIPPER_CLOSED_VALUE = -0.14
ABOVE_OBJECT_Z_OFFSET = 0.08
DESCEND_PREGRASP_Z_OFFSET = 0.035
LIFT_Z_OFFSET = 0.05
GRASP_SITE_OFFSET_ABOVE_JAW_MIDPOINT = 0.006
GRIPPING_SURFACE_Z_BAND = 0.005
JAW_ALIGNMENT_TOLERANCE = 0.005
GRASP_CONTACT_HOLD_SECONDS = 0.2
GRIPPER_CLOSE_SECONDS = 3.0
GRASP_SETTLE_SECONDS = 2.0


class GraspContactTracker:
    """Track uninterrupted opposing pad contact in simulation time."""

    def __init__(self) -> None:
        """Start with no confirmed contact."""
        self.since = None

    def update(self, now: float, contact: bool) -> bool:
        """Return true only after the required continuous contact interval.

        Args:
            now: Current simulation time in seconds.
            contact: Whether both opposing inner pads carry contact force.
        """
        if not contact:
            self.since = None
        elif self.since is None or now < self.since:
            self.since = now
        return self.since is not None and now - self.since >= GRASP_CONTACT_HOLD_SECONDS


def ensure_mjpython() -> None:
    """Relaunch process under mjpython on macOS if needed for launch_passive viewer support."""
    # Check MJPYTHON_LAUNCHED to prevent recursive restart loops,
    # as embedded mjpython may report standard CPython binary names.
    if (
        sys.platform == "darwin"
        and "mjpython" not in Path(sys.executable).name
        and os.environ.get("MJPYTHON_LAUNCHED") != "1"
    ):
        import sysconfig

        venv_mjpython = Path(sys.executable).parent / "mjpython"
        mjpython_path = str(venv_mjpython) if venv_mjpython.exists() else shutil.which("mjpython")
        if mjpython_path and os.path.exists(mjpython_path):
            # When using uv or standalone CPython distributions on macOS,
            # mjpython needs DYLD_LIBRARY_PATH to locate libpython3.14.dylib.
            libdirs = [
                str(Path(sys.base_prefix) / "lib"),
                str(sysconfig.get_config_var("LIBDIR")),
            ]
            curr_dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
            valid_dirs = [d for d in libdirs if d and d != "None"]
            if curr_dyld:
                valid_dirs.append(curr_dyld)
            os.environ["DYLD_LIBRARY_PATH"] = ":".join(valid_dirs)
            os.environ["MJPYTHON_LAUNCHED"] = "1"

            print(
                f"[macOS detected] Relaunching automatically under `{Path(mjpython_path).name}` "
                "for interactive keyboard controls..."
            )
            os.execv(mjpython_path, [mjpython_path, str(Path(__file__).resolve())] + sys.argv[1:])


def print_actuator_summary(model: mujoco.MjModel) -> None:
    """Print actuator order, corresponding joints, and control ranges."""
    print("\nActuator order in model.nu / data.ctrl:")
    for act_id in range(model.nu):
        actuator_name = model.actuator(act_id).name
        joint_id = int(model.actuator_trnid[act_id][0])
        joint_name = model.joint(joint_id).name
        ctrl_min, ctrl_max = model.actuator_ctrlrange[act_id]
        print(
            f"  ctrl[{act_id}] {actuator_name} -> joint {joint_name} "
            f"ctrlrange=[{ctrl_min:.5f}, {ctrl_max:.5f}]"
        )


def get_named_pose_target(model: mujoco.MjModel, pose_name: str) -> np.ndarray:
    """Return a preset pose target in model actuator order, using actuator names."""
    target_by_actuator = dict(zip(ACTUATOR_NAMES, PRESET_POSES[pose_name], strict=True))
    return np.array([target_by_actuator[model.actuator(i).name] for i in range(model.nu)])


def get_end_effector_site_id(model: mujoco.MjModel) -> int:
    """Return the preferred end-effector site ID."""
    try:
        return model.site("gripperframe").id
    except KeyError:
        return model.site("end_effector").id


def set_controlled_joint_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target: np.ndarray,
) -> None:
    """Set robot joint qpos and ctrl values in actuator order."""
    for act_id in range(model.nu):
        actuator_name = model.actuator(act_id).name
        joint_id = model.joint(actuator_name).id
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = target[act_id]
        data.ctrl[act_id] = target[act_id]


def get_site_position_for_target(
    model: mujoco.MjModel,
    site_id: int,
    target: np.ndarray,
) -> np.ndarray:
    """Return end-effector site position for a target without mutating live data."""
    scratch = mujoco.MjData(model)
    set_controlled_joint_qpos(model, scratch, target)
    mujoco.mj_forward(model, scratch)
    return scratch.site_xpos[site_id].copy()


def get_fingertip_z_for_target(
    model: mujoco.MjModel,
    finger_geom_ids: list[int],
    target: np.ndarray,
) -> list[float]:
    """Return finger collision geom Z values for a target pose."""
    scratch = mujoco.MjData(model)
    set_controlled_joint_qpos(model, scratch, target)
    mujoco.mj_forward(model, scratch)
    return [float(scratch.geom_xpos[geom_id][2]) for geom_id in finger_geom_ids]


def get_body_geom_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    """Return geom IDs attached to a body."""
    body_id = model.body(body_name).id
    return [geom_id for geom_id in range(model.ngeom) if model.geom_bodyid[geom_id] == body_id]


def get_cube_geom_id(model: mujoco.MjModel) -> int:
    """Return the scene cube geom ID."""
    cube_geoms = get_body_geom_ids(model, "target")
    if len(cube_geoms) != 1:
        raise ValueError(f"Expected exactly one cube geom, found {len(cube_geoms)}")
    return cube_geoms[0]


def get_finger_collision_geom_ids(model: mujoco.MjModel) -> list[int]:
    """Return active gripper collision geom IDs."""
    finger_body_names = {"gripper", "moving_jaw_so101_v1"}
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if model.body(model.geom_bodyid[geom_id]).name in finger_body_names
        and model.geom_contype[geom_id] != 0
        and model.geom_conaffinity[geom_id] != 0
    ]


def get_fixed_gripping_geom_id(model: mujoco.MjModel) -> int:
    """Return the fixed-side gripping collision geom ID."""
    for geom_id in get_finger_collision_geom_ids(model):
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = model.mesh(mesh_id).name if mesh_id >= 0 else ""
        if mesh_name == "fixed_pad_5":
            return geom_id
    raise ValueError("Could not find fixed gripping surface geom")


def get_moving_gripping_geom_id(model: mujoco.MjModel) -> int:
    """Return the moving-jaw gripping collision geom ID."""
    for geom_id in get_finger_collision_geom_ids(model):
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = model.mesh(mesh_id).name if mesh_id >= 0 else ""
        if mesh_name == "moving_pad_0":
            return geom_id
    raise ValueError("Could not find moving jaw geom")


def format_geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    """Return a readable geom name, falling back to ID and body when unnamed."""
    name = model.geom(geom_id).name
    body_name = model.body(model.geom_bodyid[geom_id]).name
    return name or f"geom[{geom_id}]/{body_name}"


def describe_gripper_geom(model: mujoco.MjModel, geom_id: int) -> str:
    """Return the physical role of a gripper collision geom."""
    mesh_id = int(model.geom_dataid[geom_id])
    mesh_name = model.mesh(mesh_id).name if mesh_id >= 0 else ""
    if mesh_name == "sts3215_03a_v1":
        return "servo housing / palm-side gripper body"
    if mesh_name.startswith("fixed_pad_"):
        return "fixed gripper-side structure"
    if mesh_name.startswith("moving_pad_"):
        return "moving jaw"
    return "other"


def get_finger_cube_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    finger_geom_ids: list[int],
) -> list[tuple[str, str, float]]:
    """Return active contacts between the cube and gripper collision geoms."""
    finger_geom_set = set(finger_geom_ids)
    contacts = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        is_finger_cube = (
            geom1 == cube_geom_id
            and geom2 in finger_geom_set
            or geom2 == cube_geom_id
            and geom1 in finger_geom_set
        )
        if is_finger_cube:
            contacts.append(
                (
                    format_geom_name(model, geom1),
                    format_geom_name(model, geom2),
                    float(contact.dist),
                )
            )
    return contacts


def cube_contacts_fixed_and_moving_jaw(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    allowed_geom_ids: set[int] | None = None,
) -> bool:
    """Return whether the cube touches both fixed side and moving jaw geoms."""
    contacted_roles = set()
    # Local +X points into the gap from the fixed jaw; -X from the moving jaw.
    normals = {}
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if cube_geom_id not in {geom1, geom2}:
            continue
        if contact.dist > 0 or contact.efc_address < 0:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, contact_id, force)
        if force[0] <= 1e-4:
            continue
        other_geom = geom2 if geom1 == cube_geom_id else geom1
        if allowed_geom_ids is not None and other_geom not in allowed_geom_ids:
            continue
        role = describe_gripper_geom(model, other_geom)
        if role not in {"fixed gripper-side structure", "moving jaw"}:
            continue
        inward = data.xmat[model.geom_bodyid[other_geom]].reshape(3, 3)[:, 0].copy()
        if role == "moving jaw":
            inward *= -1
        normal = contact.frame[:3].copy()
        if geom1 == cube_geom_id:
            normal *= -1
        if np.dot(normal, inward) < 0.8 or contact.dist < -0.001:
            continue
        contacted_roles.add(role)
        normals[role] = normal
    return (
        "fixed gripper-side structure" in contacted_roles
        and "moving jaw" in contacted_roles
        and np.dot(normals["fixed gripper-side structure"], normals["moving jaw"]) < -0.8
    )


def get_mesh_vertices_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    """Return mesh vertices transformed to world coordinates for a mesh geom."""
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        raise ValueError(f"{format_geom_name(model, geom_id)} is not a mesh geom")
    vert_addr = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    mesh_vertices = np.asarray(model.mesh_vert[vert_addr : vert_addr + vert_count])
    geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)
    return data.geom_xpos[geom_id] + mesh_vertices @ geom_xmat.T


def get_gripping_surface_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
    closing_axis: np.ndarray,
    cube_top_z: float,
    use_max_projection: bool,
) -> np.ndarray:
    """Estimate a gripping surface point near cube height from mesh vertices."""
    vertices = get_mesh_vertices_world(model, data, geom_id)
    z_mask = np.abs(vertices[:, 2] - cube_top_z) <= GRIPPING_SURFACE_Z_BAND
    candidate_vertices = vertices[z_mask] if np.any(z_mask) else vertices
    projections = candidate_vertices @ closing_axis
    extreme_projection = projections.max() if use_max_projection else projections.min()
    surface_vertices = candidate_vertices[np.abs(projections - extreme_projection) <= 0.002]
    return surface_vertices.mean(axis=0)


def get_jaw_alignment(
    model: mujoco.MjModel,
    target: np.ndarray,
    cube_pos: np.ndarray,
    cube_reference_z: float,
) -> dict[str, np.ndarray | float]:
    """Return jaw surface positions, midpoint, closing axis, and cube alignment error."""
    scratch = mujoco.MjData(model)
    set_controlled_joint_qpos(model, scratch, target)
    mujoco.mj_forward(model, scratch)

    fixed_geom_id = get_fixed_gripping_geom_id(model)
    moving_geom_id = get_moving_gripping_geom_id(model)
    fixed_center = scratch.geom_xpos[fixed_geom_id].copy()
    moving_center = scratch.geom_xpos[moving_geom_id].copy()
    closing_axis = moving_center - fixed_center
    closing_axis[2] = 0.0
    axis_norm = np.linalg.norm(closing_axis)
    if axis_norm < 1e-9:
        raise ValueError("Could not derive gripper closing axis")
    closing_axis /= axis_norm

    fixed_surface = get_gripping_surface_position(
        model,
        scratch,
        fixed_geom_id,
        closing_axis,
        cube_reference_z,
        use_max_projection=True,
    )
    moving_surface = get_gripping_surface_position(
        model,
        scratch,
        moving_geom_id,
        closing_axis,
        cube_reference_z,
        use_max_projection=False,
    )
    midpoint = 0.5 * (fixed_surface + moving_surface)
    error = cube_pos - midpoint
    return {
        "fixed_surface": fixed_surface,
        "moving_surface": moving_surface,
        "midpoint": midpoint,
        "cube_pos": cube_pos,
        "error": error,
        "closing_axis": closing_axis,
        "axis_error": float(error @ closing_axis),
    }


def get_jaw_surface_gap(alignment: dict[str, np.ndarray | float]) -> float:
    """Return open free gap between opposing gripping surface points."""
    fixed_surface = np.asarray(alignment["fixed_surface"])
    moving_surface = np.asarray(alignment["moving_surface"])
    closing_axis = np.asarray(alignment["closing_axis"])
    return abs(float((moving_surface - fixed_surface) @ closing_axis))


def get_min_gripper_table_clearance(
    model: mujoco.MjModel,
    target: np.ndarray,
    gripper_geom_ids: list[int],
) -> float:
    """Return the lowest gripper mesh vertex height above the table plane."""
    scratch = mujoco.MjData(model)
    set_controlled_joint_qpos(model, scratch, target)
    mujoco.mj_forward(model, scratch)
    min_z = np.inf
    for geom_id in gripper_geom_ids:
        vertices = get_mesh_vertices_world(model, scratch, geom_id)
        min_z = min(min_z, float(vertices[:, 2].min()))
    return min_z


def get_contacts_for_target(
    model: mujoco.MjModel,
    target: np.ndarray,
    cube_geom_id: int,
    finger_geom_ids: list[int],
) -> list[tuple[str, str, float]]:
    """Return cube/gripper contacts for a target without mutating live data."""
    scratch = mujoco.MjData(model)
    set_controlled_joint_qpos(model, scratch, target)
    mujoco.mj_forward(model, scratch)
    return get_finger_cube_contacts(model, scratch, cube_geom_id, finger_geom_ids)


def print_gripper_aperture_table(
    model: mujoco.MjModel,
    grasp_target: np.ndarray,
    cube_pos: np.ndarray,
    cube_width_along_axis: float,
) -> None:
    """Print measured gripper aperture across the valid control range."""
    gripper_actuator_id = model.actuator("gripper").id
    ctrl_min, ctrl_max = model.actuator_ctrlrange[gripper_actuator_id]
    samples = np.array(
        [
            ctrl_min,
            GRIPPER_CLOSED_VALUE,
            0.0,
            0.25,
            GRIPPER_OPEN_VALUE,
            1.0,
            ctrl_max,
        ]
    )
    samples = np.unique(np.clip(samples, ctrl_min, ctrl_max))
    print("\nMeasured gripper aperture at grasp height:")
    print("| gripper ctrl | jaw surface gap |")
    print("| ------------ | --------------- |")
    for gripper_ctrl in samples:
        target = get_gripper_target(model, grasp_target, float(gripper_ctrl))
        alignment = get_jaw_alignment(model, target, cube_pos, cube_pos[2])
        gap_mm = get_jaw_surface_gap(alignment) * 1000.0
        print(f"| {gripper_ctrl:12.5f} | {gap_mm:13.2f} mm |")
    print(f"\nCube full width along closing axis: {cube_width_along_axis * 1000.0:.2f} mm")


def print_jaw_alignment(label: str, alignment: dict[str, np.ndarray | float]) -> None:
    """Print jaw/cube lateral alignment diagnostics."""
    print(f"\n[{label}] jaw alignment")
    print(f"  fixed finger position: {np.round(alignment['fixed_surface'], 4)}")
    print(f"  moving jaw position:   {np.round(alignment['moving_surface'], 4)}")
    print(f"  jaw midpoint:          {np.round(alignment['midpoint'], 4)}")
    print(f"  cube position:         {np.round(alignment['cube_pos'], 4)}")
    print(f"  midpoint error:        {np.round(alignment['error'], 4)}")
    print(f"  closing axis:          {np.round(alignment['closing_axis'], 4)}")
    print(f"  axis error:            {alignment['axis_error']:.4f} m")


def solve_site_position_ik(
    model: mujoco.MjModel,
    site_id: int,
    site_target: np.ndarray,
    seed_target: np.ndarray,
    gripper_value: float,
) -> np.ndarray:
    """Solve a small bounded numerical IK problem for the end-effector site position."""
    arm_low = model.actuator_ctrlrange[:5, 0]
    arm_high = model.actuator_ctrlrange[:5, 1]
    gripper_id = model.actuator("gripper").id
    scratch = mujoco.MjData(model)

    def site_position(arm_qpos: np.ndarray) -> np.ndarray:
        target = np.empty(model.nu)
        target[:5] = arm_qpos
        target[gripper_id] = gripper_value
        set_controlled_joint_qpos(model, scratch, target)
        mujoco.mj_forward(model, scratch)
        return scratch.site_xpos[site_id].copy()

    def residual(arm_qpos: np.ndarray) -> np.ndarray:
        position_error = site_position(arm_qpos) - site_target
        regularization = 0.02 * (arm_qpos - seed_target[:5])
        return np.concatenate([position_error * 20.0, regularization])

    seeds = [
        seed_target[:5],
        np.zeros(5),
        np.array([0.0, -0.6, 0.6, 1.2, 0.0]),
        np.array([0.2, -0.8, 0.8, 1.2, 0.0]),
        np.array([-0.2, -0.8, 0.8, 1.2, 0.0]),
    ]
    best_target = None
    best_distance = np.inf
    for seed in seeds:
        result = least_squares(
            residual,
            np.clip(seed, arm_low, arm_high),
            bounds=(arm_low, arm_high),
            max_nfev=300,
        )
        solved_arm = result.x
        distance = float(np.linalg.norm(site_position(solved_arm) - site_target))
        if distance < best_distance:
            best_distance = distance
            best_target = np.empty(model.nu)
            best_target[:5] = solved_arm
            best_target[gripper_id] = gripper_value

    if best_target is None:
        raise RuntimeError("Unable to solve end-effector position IK")
    return best_target


def get_gripper_target(
    model: mujoco.MjModel,
    base_target: np.ndarray,
    gripper_value: float,
) -> np.ndarray:
    """Return a target with only the gripper actuator changed."""
    target = base_target.copy()
    gripper_actuator_id = model.actuator("gripper").id
    target[gripper_actuator_id] = gripper_value
    return target


def get_validated_auto_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
) -> dict[str, np.ndarray]:
    """Validate automatic sequence targets against MuJoCo actuator control limits."""
    cube_geom_id = get_cube_geom_id(model)
    finger_geom_ids = get_finger_collision_geom_ids(model)
    cube_pos = data.body("target").xpos.copy()
    cube_size = model.geom_size[cube_geom_id].copy()
    grasp_site_target = cube_pos.copy()
    grasp_site_target[2] = cube_pos[2] + GRASP_SITE_OFFSET_ABOVE_JAW_MIDPOINT
    home_target = get_named_pose_target(model, "HOME")
    open_home_target = get_gripper_target(model, home_target, GRIPPER_OPEN_VALUE)
    above_object_target = solve_site_position_ik(
        model,
        site_id,
        cube_pos + np.array([0.0, 0.0, ABOVE_OBJECT_Z_OFFSET]),
        open_home_target,
        GRIPPER_OPEN_VALUE,
    )
    grasp_target = solve_site_position_ik(
        model,
        site_id,
        grasp_site_target,
        above_object_target,
        GRIPPER_OPEN_VALUE,
    )
    total_pose_correction = np.zeros(3)
    for _ in range(8):
        alignment = get_jaw_alignment(model, grasp_target, cube_pos, cube_pos[2])
        axis = np.asarray(alignment["closing_axis"])
        fixed_surface = np.asarray(alignment["fixed_surface"])
        half_width = float(np.sum(np.abs(axis) * cube_size))
        desired_fixed = cube_pos - axis * (half_width + 0.001)
        axis_error = float((desired_fixed - fixed_surface) @ axis)
        vertical_error = float(cube_pos[2] - fixed_surface[2])
        if abs(axis_error) <= 0.001 and abs(vertical_error) <= 0.001:
            break
        pose_correction = axis_error * np.asarray(alignment["closing_axis"])
        pose_correction[2] = vertical_error
        total_pose_correction += pose_correction
        grasp_site_target += pose_correction
        grasp_target = solve_site_position_ik(
            model,
            site_id,
            grasp_site_target,
            grasp_target,
            GRIPPER_OPEN_VALUE,
        )
    descend_target = solve_site_position_ik(
        model,
        site_id,
        grasp_site_target + np.array([0.0, 0.0, DESCEND_PREGRASP_Z_OFFSET]),
        above_object_target,
        GRIPPER_OPEN_VALUE,
    )
    grasp_alignment = get_jaw_alignment(model, grasp_target, cube_pos, cube_pos[2])
    cube_width_along_axis = float(
        2.0 * np.sum(np.abs(np.asarray(grasp_alignment["closing_axis"])) * cube_size)
    )
    print(f"\nApplied fixed-jaw correction: {np.round(total_pose_correction, 4)}")
    print_jaw_alignment("GRASP_DEPTH planned", grasp_alignment)
    print_gripper_aperture_table(model, grasp_target, cube_pos, cube_width_along_axis)
    open_gap = get_jaw_surface_gap(grasp_alignment)
    min_clearance = get_min_gripper_table_clearance(model, grasp_target, finger_geom_ids)
    open_contacts = get_contacts_for_target(model, grasp_target, cube_geom_id, finger_geom_ids)
    print("\nPre-close grasp checks:")
    print(f"  selected open gripper ctrl value:   {GRIPPER_OPEN_VALUE:.5f}")
    print(f"  selected closed gripper ctrl value: {GRIPPER_CLOSED_VALUE:.5f}")
    print(f"  actual open jaw surface gap:        {open_gap * 1000.0:.2f} mm")
    print(f"  fixed jaw contact-surface position: {np.round(grasp_alignment['fixed_surface'], 4)}")
    print(f"  moving jaw contact-surface position:{np.round(grasp_alignment['moving_surface'], 4)}")
    print(f"  jaw midpoint Z:                     {grasp_alignment['midpoint'][2]:.4f} m")
    print(f"  cube center Z:                      {cube_pos[2]:.4f} m")
    print(f"  vertical midpoint error:            {grasp_alignment['error'][2]:.4f} m")
    print(f"  minimum gripper/table clearance:    {min_clearance:.4f} m")
    print(f"  open-depth cube/gripper contacts:   {open_contacts}")
    if open_gap <= cube_width_along_axis:
        raise ValueError("Selected open gripper gap is not wider than the cube")
    if min_clearance <= 0.0:
        raise ValueError("Selected grasp depth would collide with the table")
    close_target = get_gripper_target(model, grasp_target, GRIPPER_CLOSED_VALUE)
    lift_target = solve_site_position_ik(
        model,
        site_id,
        grasp_site_target + np.array([0.0, 0.0, LIFT_Z_OFFSET]),
        close_target,
        GRIPPER_CLOSED_VALUE,
    )
    targets = {
        "HOME": home_target,
        "OPEN_GRIPPER_FULLY": open_home_target,
        "ABOVE_OBJECT": above_object_target,
        "DESCEND_WITH_JAWS_OPEN": descend_target,
        "GRASP_DEPTH": grasp_target,
        "CLOSE_GRIPPER": close_target,
        "LIFT": lift_target,
    }
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]
    for pose_name, target in targets.items():
        if np.any(target < ctrl_min) or np.any(target > ctrl_max):
            raise ValueError(f"Preset pose {pose_name} exceeds actuator control limits")
    return targets


def print_alignment(
    label: str,
    cube_pos: np.ndarray,
    end_effector_pos: np.ndarray,
    cube_top_z: float,
    fingertip_z_values: list[float] | None = None,
) -> None:
    """Print end-effector alignment with the cube."""
    xyz_error = end_effector_pos - cube_pos
    distance = float(np.linalg.norm(xyz_error))
    print(f"\n[{label}] alignment")
    print(f"  cube position:         {np.round(cube_pos, 4)}")
    print(f"  end-effector position: {np.round(end_effector_pos, 4)}")
    print(f"  XYZ error vector:      {np.round(xyz_error, 4)}")
    print(f"  distance to cube:      {distance:.4f} m")
    print(f"  cube top Z:            {cube_top_z:.4f} m")
    print(f"  end-effector Z:        {end_effector_pos[2]:.4f} m")
    print(f"  commanded below top:   {cube_top_z - end_effector_pos[2]:.4f} m")
    if fingertip_z_values is not None:
        print(f"  fingertip geom Z:      {np.round(fingertip_z_values, 4)}")


def print_scene_grasp_geometry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    finger_geom_ids: list[int],
) -> None:
    """Print cube and active finger collision geometry."""
    cube_pos = data.geom_xpos[cube_geom_id]
    cube_size = model.geom_size[cube_geom_id]
    print("\nCube geom:")
    print(f"  name:        {format_geom_name(model, cube_geom_id)}")
    print(f"  world pos:   {np.round(cube_pos, 4)}")
    print(f"  size:        {np.round(cube_size, 4)}")
    print(f"  top Z:       {cube_pos[2] + cube_size[2]:.4f}")
    print(f"  contype:     {int(model.geom_contype[cube_geom_id])}")
    print(f"  conaffinity: {int(model.geom_conaffinity[cube_geom_id])}")
    print("\nFinger collision geoms:")
    for geom_id in finger_geom_ids:
        print(f"  {format_geom_name(model, geom_id)}")
        print(f"    role:        {describe_gripper_geom(model, geom_id)}")
        print(f"    world pos:   {np.round(data.geom_xpos[geom_id], 4)}")
        print(f"    size:        {np.round(model.geom_size[geom_id], 4)}")
        print(f"    contype:     {int(model.geom_contype[geom_id])}")
        print(f"    conaffinity: {int(model.geom_conaffinity[geom_id])}")


def main() -> None:
    """Launch the interactive 3D simulation viewer for the SO-101 robot arm."""
    ensure_mjpython()
    scene_path = Path(__file__).parent / "assets" / "scene.xml"
    print(f"Loading SO-101 scene from {scene_path}...")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    robot = SO101Robot(model=model, data=data)

    robot.reset("HOME")
    print_actuator_summary(model)
    end_effector_site_id = get_end_effector_site_id(model)
    end_effector_site_name = model.site(end_effector_site_id).name
    print(f"\nUsing end-effector site: {end_effector_site_name}")
    cube_geom_id = get_cube_geom_id(model)
    finger_geom_ids = get_finger_collision_geom_ids(model)
    cube_top_z = data.geom_xpos[cube_geom_id][2] + model.geom_size[cube_geom_id][2]
    print_scene_grasp_geometry(model, data, cube_geom_id, finger_geom_ids)
    auto_targets = get_validated_auto_targets(model, data, end_effector_site_id)
    cube_initial_pos = data.body("target").xpos.copy()
    cube_center_z = float(cube_initial_pos[2])
    print(f"\nCube initial position: {np.round(cube_initial_pos, 4)}")

    phase_eef_positions = {
        "HOME": data.site_xpos[end_effector_site_id].copy(),
        "ABOVE_OBJECT": get_site_position_for_target(
            model,
            end_effector_site_id,
            auto_targets["ABOVE_OBJECT"],
        ),
        "DESCEND_WITH_JAWS_OPEN": get_site_position_for_target(
            model,
            end_effector_site_id,
            auto_targets["DESCEND_WITH_JAWS_OPEN"],
        ),
        "GRASP_DEPTH": get_site_position_for_target(
            model,
            end_effector_site_id,
            auto_targets["GRASP_DEPTH"],
        ),
    }
    for pose_name in phase_eef_positions:
        print_alignment(
            pose_name,
            cube_initial_pos,
            phase_eef_positions[pose_name],
            cube_top_z,
            get_fingertip_z_for_target(model, finger_geom_ids, auto_targets[pose_name]),
        )
    print_jaw_alignment(
        "GRASP_DEPTH",
        get_jaw_alignment(model, auto_targets["GRASP_DEPTH"], cube_initial_pos, cube_center_z),
    )

    print("\n" + "=" * 60)
    print("  SO-101 MuJoCo Interactive Simulation")
    print("=" * 60)
    print("  Controls:")
    print("  - Double-click objects to inspect")
    print("  - Drag with Right-click to rotate camera, Scroll to zoom")
    print("  - Drag joints/body parts with Ctrl + Right-click")
    print("  - Press [1] Key -> HOME pose  (passive mode)")
    print("  - Press [2] Key -> REACH pose (passive mode)")
    print("  - Press [3] Key -> PICK pose  (passive mode)")
    print("  - Press [4] Key -> STOW pose  (passive mode)")
    print("  - Press [Space] -> Toggle Gripper Open / Close")
    print("=" * 60 + "\n")

    pose_keys = ["HOME", "REACH", "PICK", "STOW"]
    gripper_open = True
    auto_running = True
    auto_phase_index = 0
    auto_phase_start_time = data.time
    grasp_contact_tracker = GraspContactTracker()
    auto_phase_start_ctrl = data.ctrl.copy()
    auto_phase_target_ctrl = auto_targets[AUTO_SEQUENCE[auto_phase_index]].copy()
    auto_phase_next_log_time = auto_phase_start_time
    initial_cube_z = float(cube_initial_pos[2])
    max_cube_z = initial_cube_z
    previous_sim_time = data.time

    def print_close_lift_log(label: str) -> None:
        """Print gripper/cube/contact telemetry for close and lift phases."""
        gripper_actuator_id = model.actuator("gripper").id
        gripper_joint_id = model.joint("gripper").id
        gripper_qpos_addr = model.jnt_qposadr[gripper_joint_id]
        cube_z = float(data.body("target").xpos[2])
        cube_displacement = cube_z - initial_cube_z
        contacts = get_finger_cube_contacts(model, data, cube_geom_id, finger_geom_ids)
        cube_between_surfaces = cube_contacts_fixed_and_moving_jaw(model, data, cube_geom_id)
        print(f"[{label}] gripper command: {data.ctrl[gripper_actuator_id]:.4f}")
        print(f"[{label}] gripper joint position: {data.qpos[gripper_qpos_addr]:.4f}")
        print(f"[{label}] end-effector Z: {data.site_xpos[end_effector_site_id][2]:.4f} m")
        print(f"[{label}] cube Z: {cube_z:.4f} m")
        print(f"[{label}] cube Z displacement: {cube_displacement:.4f} m")
        print(f"[{label}] cube-gripper contact count: {len(contacts)}")
        print(f"[{label}] cube-gripper contact pairs: {contacts}")
        print(f"[{label}] cube between fixed side and moving jaw: {cube_between_surfaces}")

    def print_lift_result() -> None:
        """Print final grasp/lift result."""
        lift_success = float(
            data.body("target").xpos[2]
        ) - initial_cube_z >= LIFT_Z_OFFSET - 0.005 and cube_contacts_fixed_and_moving_jaw(
            model, data, cube_geom_id
        )
        cube_final_pos = data.body("target").xpos.copy()
        grasp_error = phase_eef_positions.get("GRASP_DEPTH", np.full(3, np.nan)) - cube_initial_pos
        final_fingertip_z = [float(data.geom_xpos[geom_id][2]) for geom_id in finger_geom_ids]
        final_contacts = get_finger_cube_contacts(model, data, cube_geom_id, finger_geom_ids)
        print("\n[AUTO] Grasp/lift result")
        print(f"  cube initial position: {np.round(cube_initial_pos, 4)}")
        print(f"  ABOVE_OBJECT ee pos:   {np.round(phase_eef_positions['ABOVE_OBJECT'], 4)}")
        print(f"  GRASP_DEPTH ee pos:    {np.round(phase_eef_positions['GRASP_DEPTH'], 4)}")
        print(f"  final XYZ error:       {np.round(grasp_error, 4)}")
        print(f"  cube top Z:            {cube_top_z:.4f} m")
        print(f"  fingertip Z at stop:   {np.round(final_fingertip_z, 4)}")
        print(f"  end-effector Z at stop:{data.site_xpos[end_effector_site_id][2]:.4f} m")
        print(f"  contact occurred:      {bool(final_contacts)}")
        print(f"  contacting geoms:      {final_contacts}")
        print(f"  initial cube Z:        {initial_cube_z:.4f} m")
        print(f"  maximum cube Z:        {max_cube_z:.4f} m")
        print(f"  LIFT SUCCESS:          {lift_success}")
        print(f"  cube final position:   {np.round(cube_final_pos, 4)}")

    def begin_auto_phase(now: float) -> None:
        """Start the next automatic pose phase."""
        nonlocal auto_phase_next_log_time, auto_phase_start_time, auto_phase_start_ctrl
        nonlocal auto_phase_target_ctrl
        pose_name = AUTO_SEQUENCE[auto_phase_index]
        auto_phase_start_time = now
        auto_phase_next_log_time = now
        auto_phase_start_ctrl = data.ctrl.copy()
        auto_phase_target_ctrl = auto_targets[pose_name].copy()
        grasp_contact_tracker.since = None
        print(f"[AUTO] Phase {auto_phase_index + 1}/{len(AUTO_SEQUENCE)}: {pose_name}")
        if pose_name == "GRASP_DEPTH":
            print("[AUTO] Pausing before CLOSE_GRIPPER for visual jaw/cube alignment check.")
            print_jaw_alignment(
                "GRASP_DEPTH visual check",
                get_jaw_alignment(
                    model,
                    auto_phase_target_ctrl,
                    cube_initial_pos,
                    cube_initial_pos[2],
                ),
            )

    def restart_auto_sequence(now: float) -> None:
        """Restart the automatic pose sequence after a viewer reset."""
        nonlocal auto_phase_index, auto_running, cube_initial_pos, cube_center_z, cube_top_z
        nonlocal initial_cube_z, max_cube_z, previous_sim_time
        auto_running = True
        auto_phase_index = 0
        cube_initial_pos = data.body("target").xpos.copy()
        cube_center_z = float(cube_initial_pos[2])
        initial_cube_z = float(cube_initial_pos[2])
        max_cube_z = initial_cube_z
        cube_top_z = data.geom_xpos[cube_geom_id][2] + model.geom_size[cube_geom_id][2]
        previous_sim_time = data.time
        auto_targets.update(get_validated_auto_targets(model, data, end_effector_site_id))
        phase_eef_positions["HOME"] = data.site_xpos[end_effector_site_id].copy()
        phase_eef_positions["ABOVE_OBJECT"] = get_site_position_for_target(
            model,
            end_effector_site_id,
            auto_targets["ABOVE_OBJECT"],
        )
        phase_eef_positions["DESCEND_WITH_JAWS_OPEN"] = get_site_position_for_target(
            model,
            end_effector_site_id,
            auto_targets["DESCEND_WITH_JAWS_OPEN"],
        )
        phase_eef_positions["GRASP_DEPTH"] = get_site_position_for_target(
            model,
            end_effector_site_id,
            auto_targets["GRASP_DEPTH"],
        )
        print("[AUTO] Viewer reset detected; restarting scripted grasp/lift check.")
        print(f"Cube initial position: {np.round(cube_initial_pos, 4)}")
        for pose_name in phase_eef_positions:
            print_alignment(
                pose_name,
                cube_initial_pos,
                phase_eef_positions[pose_name],
                cube_top_z,
                get_fingertip_z_for_target(model, finger_geom_ids, auto_targets[pose_name]),
            )
        print_jaw_alignment(
            "GRASP_DEPTH",
            get_jaw_alignment(model, auto_targets["GRASP_DEPTH"], cube_initial_pos, cube_center_z),
        )
        begin_auto_phase(now)

    begin_auto_phase(auto_phase_start_time)

    def key_callback(keycode: int) -> None:
        """Handle keyboard presses within the passive viewer window.

        Args:
            keycode: The ASCII integer code of the key pressed.
        """
        nonlocal auto_running, gripper_open
        if auto_running:
            auto_running = False
            print("[AUTO] Manual key pressed; automatic sequence stopped.")

        # Key '1' -> 49, '2' -> 50, '3' -> 51, '4' -> 52
        if keycode in [49, 50, 51, 52]:
            idx = keycode - 49
            pose_name = pose_keys[idx]
            print(f"Moving robot to preset pose: {pose_name}")
            target_qpos = PRESET_POSES[pose_name].copy()
            if not gripper_open:
                target_qpos[-1] = GRIPPER_CLOSED_VALUE
            robot.set_joint_positions(target_qpos)
        elif keycode == 32:  # Spacebar
            gripper_open = not gripper_open
            state_str = "OPEN" if gripper_open else "CLOSED"
            print(f"Toggling Gripper -> {state_str}")
            cur_ctrl = robot.data.ctrl.copy()
            cur_ctrl[-1] = GRIPPER_OPEN_VALUE if gripper_open else GRIPPER_CLOSED_VALUE
            robot.set_joint_positions(cur_ctrl)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            viewer.opt.geomgroup[2] = True
            viewer.opt.geomgroup[3] = False

            while viewer.is_running():
                step_start = time.time()
                now = data.time

                if data.time < previous_sim_time:
                    restart_auto_sequence(now)

                if auto_running:
                    pose_name = AUTO_SEQUENCE[auto_phase_index]
                    if pose_name == "LIFT" and not cube_contacts_fixed_and_moving_jaw(
                        model, data, cube_geom_id
                    ):
                        auto_running = False
                        print("[AUTO] Lift stopped: cube contact was lost.")
                        print_lift_result()
                        continue
                    max_cube_z = max(max_cube_z, float(data.body("target").xpos[2]))
                    if pose_name == "GRASP_DEPTH":
                        duration = AUTO_VERIFY_HOLD_SECONDS
                    elif pose_name == "CLOSE_GRIPPER":
                        duration = GRIPPER_CLOSE_SECONDS
                    elif np.allclose(auto_phase_start_ctrl, auto_phase_target_ctrl):
                        duration = AUTO_HOLD_SECONDS
                    else:
                        duration = AUTO_MOVE_SECONDS
                    t = min((now - auto_phase_start_time) / duration, 1.0)
                    alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                    data.ctrl[:] = (1.0 - alpha) * auto_phase_start_ctrl + (
                        alpha * auto_phase_target_ctrl
                    )
                    finger_cube_contacts = get_finger_cube_contacts(
                        model,
                        data,
                        cube_geom_id,
                        finger_geom_ids,
                    )
                    if (
                        pose_name in {"DESCEND_WITH_JAWS_OPEN", "GRASP_DEPTH"}
                        and finger_cube_contacts
                        and now >= auto_phase_next_log_time
                    ):
                        print(f"[AUTO {pose_name}] open-jaw cube contacts: {finger_cube_contacts}")
                        auto_phase_next_log_time = now + 0.3
                    if pose_name in {"CLOSE_GRIPPER", "LIFT"} and now >= auto_phase_next_log_time:
                        print_close_lift_log(f"AUTO {pose_name}")
                        auto_phase_next_log_time = now + 0.3

                    if t >= 1.0:
                        if pose_name == "CLOSE_GRIPPER":
                            confirmed = grasp_contact_tracker.update(
                                now, cube_contacts_fixed_and_moving_jaw(model, data, cube_geom_id)
                            )
                            if not confirmed:
                                if now - auto_phase_start_time >= duration + GRASP_SETTLE_SECONDS:
                                    auto_running = False
                                    print(
                                        "[AUTO] Lift blocked: opposing pad contact did not settle."
                                    )
                                    print_close_lift_log("GRASP FAILED")
                                mujoco.mj_step(model, data)
                                viewer.sync()
                                previous_sim_time = data.time
                                time.sleep(max(0, model.opt.timestep - (time.time() - step_start)))
                                continue
                        data.ctrl[:] = auto_phase_target_ctrl
                        print(f"[AUTO] Reached {pose_name}")
                        if pose_name in {"CLOSE_GRIPPER", "LIFT"}:
                            print_close_lift_log(f"AUTO {pose_name} final")
                        if pose_name in phase_eef_positions:
                            phase_eef_positions[pose_name] = data.site_xpos[
                                end_effector_site_id
                            ].copy()
                            print_alignment(
                                pose_name,
                                cube_initial_pos,
                                phase_eef_positions[pose_name],
                                cube_top_z,
                                [float(data.geom_xpos[geom_id][2]) for geom_id in finger_geom_ids],
                            )
                        auto_phase_index += 1
                        if auto_phase_index < len(AUTO_SEQUENCE):
                            begin_auto_phase(now)
                        else:
                            auto_running = False
                            print("[AUTO] Sequence complete. Manual controls remain active.")
                            print_lift_result()

                mujoco.mj_step(model, data)
                viewer.sync()
                if data.time < previous_sim_time:
                    restart_auto_sequence(data.time)
                previous_sim_time = data.time

                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    except RuntimeError as err:
        if "mjpython" in str(err):
            print("Notice: macOS standard Python detected (`launch_passive` restriction).")
            print("Launching interactive native viewer window...")
            mujoco.viewer.launch(model, data)
        else:
            raise err


if __name__ == "__main__":
    main()
