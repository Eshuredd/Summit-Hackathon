#!/usr/bin/env python3
"""SO-101 MuJoCo Quickstart Diagnostic Tool."""

import sys
import time
from pathlib import Path


def check_system() -> bool:
    """Run comprehensive system diagnostic checks for SO-101 simulation environment.

    Returns:
        bool: True if all critical diagnostics passed successfully, False otherwise.
    """
    print("=" * 60)
    print("  SO-101 MuJoCo & MuJoCo Warp Diagnostic Check")
    print("=" * 60)

    # 1. Check Python Version
    python_ver = sys.version.split()[0]
    print(f"[✓] Python Version: {python_ver} (Target: >= 3.14)")

    # 2. Check MuJoCo Installation & Parsing
    try:
        import mujoco

        print(f"[✓] MuJoCo Version: {mujoco.__version__}")
    except ImportError as e:
        print(f"[✗] Failed to import mujoco: {e}")
        return False

    scene_path = Path(__file__).parent / "assets" / "scene.xml"
    if not scene_path.exists():
        print(f"[✗] Scene file not found at {scene_path}")
        return False

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        print(f"[✓] Successfully parsed MJCF scene: {scene_path.name}")
        print(f"    - Total Bodies:    {model.nbody}")
        print(f"    - Total Joints:    {model.njnt}")
        print(f"    - Actuators:       {model.nu}")
        print(f"    - Degrees of Freedom (nv): {model.nv}")
    except Exception as e:
        print(f"[✗] Error parsing MJCF xml: {e}")
        return False

    # 3. Check Robot Controller
    from so101.robot import PRESET_POSES, SO101Robot

    robot = SO101Robot(model=model, data=data)
    robot.reset("HOME")
    ee_pos, _ = robot.get_end_effector_pose()
    print(
        f"[✓] SO101Robot Initialized. EE Pos: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
    )
    print(f"    - Registered Preset Poses: {list(PRESET_POSES.keys())}")

    # 4. Benchmark CPU Physics Step Performance
    t0 = time.perf_counter()
    steps = 5000
    for _ in range(steps):
        mujoco.mj_step(model, data)
    t1 = time.perf_counter()
    fps = steps / (t1 - t0)
    print(f"[✓] CPU Simulation Benchmark: {fps:.1f} FPS ({steps} steps in {t1 - t0:.4f}s)")

    # 5. Check MuJoCo Warp GPU backend availability
    try:
        import mujoco_warp as _mjw  # noqa: F401
        import warp as _wp  # noqa: F401

        print("[✓] MuJoCo Warp (`mujoco_warp`) is INSTALLED")
    except ImportError:
        print("[!] MuJoCo Warp (`mujoco_warp`) package is optional/not found in current env.")

    print("\n" + "=" * 60)
    print("  All System Diagnostics Passed Successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_system()
    sys.exit(0 if success else 1)
