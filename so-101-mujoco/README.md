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
