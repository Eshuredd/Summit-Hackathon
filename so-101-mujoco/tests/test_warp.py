#!/usr/bin/env python3
"""MuJoCo Warp GPU Batched Simulation Benchmark Script."""

import sys
import time

import numpy as np

from so101.warp_env import WARP_AVAILABLE, SO101WarpEnv


def main() -> None:
    """Execute the batched simulation throughput benchmark using MuJoCo Warp."""
    print("=" * 60)
    print("  MuJoCo Warp (`mujoco_warp`) Batched Simulation Benchmark")
    print("=" * 60)

    num_envs = 128
    num_steps = 200

    if not WARP_AVAILABLE:
        print(
            "[!] Notice: `mujoco_warp` package or CUDA device is not installed/available "
            "in environment."
        )
        print(
            "[!] Standard single-instance CPU physics fallback can be used via `SO101ReachEnv` "
            "or `simulate.py`."
        )
        print("[!] To install MuJoCo Warp on a GPU-enabled system: `uv add mujoco-warp warp-lang`.")
        sys.exit(0)

    print(f"Creating {num_envs} parallel SO-101 environments...")
    env = SO101WarpEnv(num_envs=num_envs)

    obs = env.reset()
    print(f"Batched initial observation shape: {obs.shape}")

    t0 = time.perf_counter()
    for _ in range(num_steps):
        # Generate random joint command actions for all environments
        actions = np.random.uniform(-0.02, 0.02, size=(num_envs, 6)).astype(np.float32)
        obs, rewards, dones = env.step(actions)

    t1 = time.perf_counter()
    total_steps = num_envs * num_steps
    sps = total_steps / (t1 - t0)

    print("\n[✓] Benchmark Completed Successfully!")
    print(f"    - Parallel Environments: {num_envs}")
    print(f"    - Physics Sub-steps:     {num_steps}")
    print(f"    - Total Simulated Steps: {total_steps:,}")
    print(f"    - Elapsed Time:          {t1 - t0:.4f} seconds")
    print(f"    - Simulation Throughput: {sps:,.0f} Steps / Second (SPS)")
    print("=" * 60)


if __name__ == "__main__":
    main()
