#!/usr/bin/env python3
"""Preset Pose Interpolation and Kinematics Verification Script."""

from pathlib import Path

import mujoco
import numpy as np

from so101.robot import PRESET_POSES, SO101Robot


def test_preset_sequence() -> None:
    """Verify smooth trajectory interpolation across registered preset poses.

    Raises:
        AssertionError: If the maximum joint position tracking error exceeds tolerance.
    """
    scene_path = Path(__file__).resolve().parents[1] / "assets" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    robot = SO101Robot(model=model, data=data)

    print("=" * 60)
    print("  Testing SO-101 Pose Sequence Interpolation")
    print("=" * 60)

    poses_to_test = ["HOME", "REACH", "PICK", "STOW", "HOME"]

    robot.reset("HOME")

    for pose_name in poses_to_test:
        target_qpos = PRESET_POSES[pose_name]
        print(f"\n[-->] Moving smoothly to pose: {pose_name}")
        print(f"      Target Joints (rad): {np.round(target_qpos, 3)}")

        robot.move_to_pose(target_qpos, duration_sec=0.8, dt=model.opt.timestep)

        cur_qpos = robot.get_joint_positions()
        ee_pos, _ = robot.get_end_effector_pose()

        err_l2 = np.linalg.norm(cur_qpos - target_qpos)
        max_joint_err = np.max(np.abs(cur_qpos - target_qpos))
        ee_str = f"[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
        print(f"      Reached Pose EE Pos (x,y,z): {ee_str}")
        print(f"      Joint Position Error (L2 / Max): {err_l2:.4f} / {max_joint_err:.4f} rad")

        assert max_joint_err < 0.35, (
            f"Max joint error exceeded for {pose_name} ({max_joint_err:.4f} rad)"
        )

    print("\n" + "=" * 60)
    print("  All Pose Trajectory Interpolations Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_preset_sequence()
