# SO-101 MuJoCo & MuJoCo Warp Simulation Quickstart 🤖⚡

A zero-friction, modern **MuJoCo** and **MuJoCo Warp** simulation environment for the **SO-101 (SO-ARM101)** 6-DOF open-source follower/leader robot arm.

Designed in partnership with **Hugging Face LeRobot** and **The Robot Studio**, the SO-101 arm is one of the most accessible platforms for end-to-end robotics research, imitation learning, and Sim2Real policy deployment.

---

## 🌟 Features

- **Python 3.14 & `uv` First**: Modern project packaging with instant virtualenv creation and deterministic dependency resolution.
- **Accurate MJCF Model**: 6-DOF kinematics (`shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`), Feetech STS3215 servo position & torque specs, realistic contacts, and visual geoms matching the physical SO-101 build.
- **Interactive 3D Visualization**: Native GUI powered by `mujoco.viewer` with preset poses, interactive joint controls, contact forces, and automatic macOS compatibility.
- **MuJoCo Warp Acceleration**: Built-in support for `mujoco-warp` (Google DeepMind + NVIDIA Warp) to run parallel environment steps.
- **Gymnasium RL Wrapper**: `SO101ReachEnv` ready for Reinforcement Learning (RL) and Imitation Learning dataset collection compatible with Hugging Face LeRobot format.

---

## 🚀 Quickstart Guide

### Prerequisites

Ensure you have [`uv`](https://docs.astral.sh/uv/) installed (Python 3.14 will be managed automatically by `uv`).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/johannstark/so-101-mujoco.git
cd so-101-mujoco

# Synchronize dependencies with Python 3.14
uv sync
```

### 2. Run System Diagnostics

Verify your Python runtime, MuJoCo model parsing, and physics engine benchmarks:

```bash
uv run python check_env.py
```

---

## 🖥️ Interactive 3D Simulation

Launch the native interactive 3D visual simulator:

```bash
uv run python simulate.py
```

> **macOS Note**: `simulate.py` automatically detects macOS standard Python runtimes and configures `DYLD_LIBRARY_PATH` to seamlessly re-launch under `mjpython`, guaranteeing full compatibility for keyboard controls and UI callbacks.

### Interactive Controls

- **Mouse**:
  - `Right-click + Drag`: Rotate camera view
  - `Scroll`: Zoom in / out
  - `Double-click`: Select body/geom to inspect physics properties
  - `Ctrl + Right-click`: Drag robot joints/links dynamically
- **Keyboard Shortcuts** (in passive mode):
  - `1`: Move to **HOME** pose
  - `2`: Move to **REACH** pose
  - `3`: Move to **PICK** pose
  - `4`: Move to **STOW** pose
  - `Spacebar`: Toggle Gripper Open / Closed

---

## 🔗 Official Guides & Community Resources

| Resource | Link | Description |
| :--- | :--- | :--- |
| **Hugging Face LeRobot** | [huggingface/lerobot](https://github.com/huggingface/lerobot) | Main repository for LeRobot policies, datasets, and hardware tools |
| **LeRobot Documentation** | [LeRobot Docs](https://huggingface.co/docs/lerobot) | Tutorials on assembly, servo calibration, teleoperation, and dataset recording |
| **SO-ARM100 / SO-101 CAD** | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Official 3D CAD files, STL meshes, and assembly instructions |
| **MuJoCo Warp Engine** | [google-deepmind/mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | High-throughput parallel physics simulation framework |

---

## ⚡ High-Throughput Simulation with MuJoCo Warp

[MuJoCo Warp (`mujoco-warp`)](https://github.com/google-deepmind/mujoco_warp) leverages Warp kernels to simulate thousands of SO-101 robot arms in parallel.

Run the batched benchmark:

```bash
uv run python test_warp.py
```

Example usage in Python for Reinforcement Learning:

```python
from so101.warp_env import SO101WarpEnv

# Initialize parallel SO-101 environments
env = SO101WarpEnv(num_envs=128)
obs = env.reset()

# Step parallel physics environments
actions = np.random.uniform(-0.02, 0.02, size=(128, 6)).astype(np.float32)
obs, rewards, dones = env.step(actions)
```

---

## 🤖 Robot Kinematics & Joint Structure

The SO-101 model (`assets/so101.xml`) defines 6 articulated degrees of freedom matching the physical robot:

| Joint Name | Type | Range (Rad) | Description |
| :--- | :--- | :--- | :--- |
| `shoulder_pan` | Hinge (Z-axis) | $[-3.14, 3.14]$ | Base yaw rotation |
| `shoulder_lift` | Hinge (Y-axis) | $[-1.74, 1.74]$ | Upper arm pitch elevation |
| `elbow_flex` | Hinge (Y-axis) | $[-2.35, 2.35]$ | Forearm pitch flex |
| `wrist_flex` | Hinge (Y-axis) | $[-1.74, 1.74]$ | Wrist pitch angle |
| `wrist_roll` | Hinge (Z-axis) | $[-3.14, 3.14]$ | Wrist roll rotation |
| `gripper` | Hinge / Claw | $[-0.05, 0.80]$ | End-effector finger open/close |

---

## 🧪 Preset Trajectory Testing

Run programmatic pose interpolation tests:

```bash
uv run python test_poses.py
```

This script verifies smooth cosine joint trajectory generation between preset poses (`HOME`, `REACH`, `PICK`, `STOW`).

---
Made in 🇨🇴 Colombia with Love ❤️
## Deterministic single-arm pick and place

The complete cycle uses the existing contact-based grasp, transfers above the green
non-colliding marker, lowers over four seconds, releases near the table, retreats
upward, and returns HOME. The desired final cube center is `[0.18, 0.08, 0.015]`
meters. Grasp meshes and actuator settings are unchanged.

From this directory on Windows:

```powershell
.\.venv\Scripts\python.exe pick_place.py
.\.venv\Scripts\python.exe pick_place.py --headless --runs 10
```

Both modes use the same controller. Press R in the viewer to reset/restart, 1–4
for manual poses, or Space to toggle the jaw. Manual commands stop the sequence.
The original `simulate.py` grasp-only viewer remains available.

Headless mode writes `pick_place_results.json` and exits nonzero if any reset fails.
These are identical deterministic resets, not randomized robustness trials.
Success requires XY error below 3 cm, cube-center Z within 5 mm of the destination,
floor contact, and low final cube velocity. Release requires XY error below 2.5 cm,
cube-bottom height within 6 mm above the table, and linear speed below 2 cm/s.
Lateral transfer requires opposing pad contact and at least 2.5 cm of bottom
clearance. No attachments or welds are used.

## Grasp collision checks

The gripper has one fixed finger and one hinged finger. Its collision meshes are
separate convex sections clipped from the original CAD meshes, with named distal
pad sections. Only compressive contacts on the inward-facing pad surfaces count
as a grasp; palm, shell, fingertip-bottom, and one-sided contacts do not.

The automatic sequence aligns the cube beside the fixed finger, closes for three
seconds, and allows up to two additional seconds to settle. Lifting requires
0.2 seconds of uninterrupted opposing pad contact and stops if contact is lost.

Run `python check_grasp.py` to check penetration, sustained contact, lift height,
one-second retention, and rejection of one-sided contact. Add `--render` to save
`grasp_verified.png` (requires Pillow and a working OpenGL renderer).
Run `python fit_jaw_collisions.py` to regenerate the fitted collision assets from
the original STL files. The visual CAD meshes remain unchanged.
