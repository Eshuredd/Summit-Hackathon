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

from so101.robot import ACTUATOR_NAMES, PRESET_POSES, SO101Robot

AUTO_SEQUENCE = ("HOME", "REACH", "HOME")
AUTO_MOVE_SECONDS = 1.5
AUTO_HOLD_SECONDS = 0.6


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


def get_validated_auto_targets(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """Validate automatic sequence targets against MuJoCo actuator control limits."""
    targets = {}
    ctrl_min = model.actuator_ctrlrange[:, 0]
    ctrl_max = model.actuator_ctrlrange[:, 1]
    for pose_name in AUTO_SEQUENCE:
        target = get_named_pose_target(model, pose_name)
        if np.any(target < ctrl_min) or np.any(target > ctrl_max):
            raise ValueError(f"Preset pose {pose_name} exceeds actuator control limits")
        targets[pose_name] = target
    return targets


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
    auto_targets = get_validated_auto_targets(model)

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
    auto_phase_start_time = time.time()
    auto_phase_start_ctrl = data.ctrl.copy()
    auto_phase_target_ctrl = auto_targets[AUTO_SEQUENCE[auto_phase_index]].copy()
    previous_sim_time = data.time

    def begin_auto_phase(now: float) -> None:
        """Start the next automatic pose phase."""
        nonlocal auto_phase_start_time, auto_phase_start_ctrl, auto_phase_target_ctrl
        pose_name = AUTO_SEQUENCE[auto_phase_index]
        auto_phase_start_time = now
        auto_phase_start_ctrl = data.ctrl.copy()
        auto_phase_target_ctrl = auto_targets[pose_name].copy()
        print(f"[AUTO] Phase {auto_phase_index + 1}/{len(AUTO_SEQUENCE)}: {pose_name}")

    def restart_auto_sequence(now: float) -> None:
        """Restart the automatic pose sequence after a viewer reset."""
        nonlocal auto_phase_index, auto_running, previous_sim_time
        auto_running = True
        auto_phase_index = 0
        previous_sim_time = data.time
        print("[AUTO] Viewer reset detected; restarting HOME -> REACH -> HOME.")
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
                target_qpos[-1] = 0.3
            robot.set_joint_positions(target_qpos)
        elif keycode == 32:  # Spacebar
            gripper_open = not gripper_open
            state_str = "OPEN" if gripper_open else "CLOSED"
            print(f"Toggling Gripper -> {state_str}")
            cur_ctrl = robot.data.ctrl.copy()
            cur_ctrl[-1] = 0.0 if gripper_open else 0.3
            robot.set_joint_positions(cur_ctrl)

    try:
        with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

            while viewer.is_running():
                step_start = time.time()
                now = time.time()

                if data.time < previous_sim_time:
                    restart_auto_sequence(now)

                if auto_running:
                    pose_name = AUTO_SEQUENCE[auto_phase_index]
                    duration = (
                        AUTO_HOLD_SECONDS
                        if np.allclose(auto_phase_start_ctrl, auto_phase_target_ctrl)
                        else AUTO_MOVE_SECONDS
                    )
                    t = min((now - auto_phase_start_time) / duration, 1.0)
                    alpha = 0.5 * (1.0 - np.cos(np.pi * t))
                    data.ctrl[:] = (1.0 - alpha) * auto_phase_start_ctrl + (
                        alpha * auto_phase_target_ctrl
                    )

                    if t >= 1.0:
                        data.ctrl[:] = auto_phase_target_ctrl
                        print(f"[AUTO] Reached {pose_name}")
                        auto_phase_index += 1
                        if auto_phase_index < len(AUTO_SEQUENCE):
                            begin_auto_phase(now)
                        else:
                            auto_running = False
                            print("[AUTO] Sequence complete. Manual controls remain active.")

                mujoco.mj_step(model, data)
                viewer.sync()
                if data.time < previous_sim_time:
                    restart_auto_sequence(time.time())
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
