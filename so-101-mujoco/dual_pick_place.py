"""Run one selected SO-101 in a shared dual-arm scene; hold the other at HOME."""

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import scipy

import simulate as grasp
from pick_place import PickPlace

ASSETS = Path(__file__).parent / "assets"


def check_runtime():
    """Reject a mismatched physics/planning stack before running the locked trajectory."""
    versions = {"mujoco": mujoco.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    locked = tomllib.loads((ASSETS.parent / "uv.lock").read_text())
    expected = {p["name"]: p["version"] for p in locked["package"] if p["name"] in versions}
    print(f"Python executable: {sys.executable}")
    print("Runtime: " + ", ".join(f"{name}={version}" for name, version in versions.items()))
    mismatches = [
        f"{name}: found {version}, lock requires {expected.get(name)}"
        for name, version in versions.items()
        if expected.get(name) != version
    ]
    if mismatches:
        raise ValueError(
            "Runtime differs from the validated project uv.lock: "
            + "; ".join(mismatches)
            + "\nUse this project's interpreter, not the parent bindGIT environment:"
            + f'\n  & "{ASSETS.parent / ".venv" / "Scripts" / "python.exe"}" '
            + f'"{Path(__file__).resolve()}" '
            + " ".join(sys.argv[1:])
            + "\nOr create/sync the project environment with uv sync --frozen."
        )
    return dict(executable=sys.executable, versions=versions)


class ArmBackend:
    """Map six-axis planning into twelve controls and use live selected-arm contacts."""

    def __init__(self, model, data, arm):
        """Keep a single-arm model solely for unchanged local-frame kinematic planning."""
        self.arm = arm
        self.indices = np.array(
            [model.actuator(f"{arm}_{name}").id for name in grasp.ACTUATOR_NAMES]
        )
        self.base = data.body(f"{arm}_base").xpos.copy()
        self.rotation = data.body(f"{arm}_base").xmat.reshape(3, 3).copy()
        self.local = mujoco.MjModel.from_xml_path(str(ASSETS / "scene.xml"))
        self.local_site = grasp.get_end_effector_site_id(self.local)
        self.home = np.tile(grasp.PRESET_POSES["HOME"], 2)
        self.allowed = {
            i
            for i in range(model.ngeom)
            if model.body(model.geom_bodyid[i]).name.startswith(f"{arm}_")
        }

    def __getattr__(self, name):
        """Reuse existing constants and geometry-independent helpers."""
        return getattr(grasp, name)

    def expand(self, target):
        """Fill only the selected actuator slots; hold the other six at HOME."""
        result = self.home.copy()
        result[self.indices] = target
        return result

    def get_end_effector_site_id(self, model):
        """Resolve the selected prefixed end-effector site."""
        return model.site(f"{self.arm}_gripperframe").id

    def get_validated_auto_targets(self, model, data, site):
        """Plan the original grasp in the selected base frame without stepping physics."""
        local_data = mujoco.MjData(self.local)
        cube_joint = self.local.joint("target_joint").id
        adr = self.local.jnt_qposadr[cube_joint]
        local_data.qpos[adr : adr + 3] = self.rotation.T @ (data.body("target").xpos - self.base)
        mujoco.mj_forward(self.local, local_data)
        targets = grasp.get_validated_auto_targets(self.local, local_data, self.local_site)
        return {name: self.expand(target) for name, target in targets.items()}

    def solve_site_position_ik(self, model, site, position, seed, opening):
        """Transform world-space placement targets into the selected base frame."""
        target = grasp.solve_site_position_ik(
            self.local,
            self.local_site,
            self.rotation.T @ (position - self.base),
            seed[self.indices],
            opening,
        )
        return self.expand(target)

    def get_gripper_target(self, model, target, opening):
        """Change only the selected arm's gripper control."""
        result = target.copy()
        result[model.actuator(f"{self.arm}_gripper").id] = opening
        return result

    def cube_contacts_fixed_and_moving_jaw(self, model, data, cube):
        """Require genuine opposing pad forces from the selected robot only."""
        return grasp.cube_contacts_fixed_and_moving_jaw(model, data, cube, self.allowed)


def verify_layout(model):
    """Verify and print exact left-then-right actuator/joint ordering."""
    expected = [f"{arm}_{name}" for arm in ("left", "right") for name in grasp.ACTUATOR_NAMES]
    assert [model.actuator(i).name for i in range(model.nu)] == expected
    for i, name in enumerate(expected):
        joint = int(model.actuator_trnid[i, 0])
        assert model.joint(joint).name == name
    grasp.print_actuator_summary(model)
    assert model.neq == 0, "No attachments or welds are allowed"


def main():
    """Test deterministic resets headlessly by default, or show the same cycle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("left", "right"), required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    try:
        runtime = check_runtime()
    except ValueError as error:
        parser.error(str(error))
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "dual_scene.xml"))
    data = mujoco.MjData(model)
    verify_layout(model)
    destination = model.site("placement_target").pos.copy()
    inactive = "left" if args.arm == "right" else "right"
    idle_ids = np.array([model.actuator(f"{inactive}_{name}").id for name in grasp.ACTUATOR_NAMES])
    idle_qpos = model.jnt_qposadr[model.actuator_trnid[idle_ids, 0]]
    owner = np.array(
        [
            1
            if model.body(model.geom_bodyid[i]).name.startswith("left_")
            else 2
            if model.body(model.geom_bodyid[i]).name.startswith("right_")
            else 0
            for i in range(model.ngeom)
        ]
    )
    results = []
    for run in range(1 if args.viewer else args.runs):
        print(f"\nSelected arm: {args.arm}; reset {run + 1}; {inactive} held at HOME")
        mujoco.mj_resetData(model, data)
        data.ctrl[:] = np.tile(grasp.PRESET_POSES["HOME"], 2)
        for i in range(model.nu):
            data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]] = data.ctrl[i]
        mujoco.mj_forward(model, data)
        for _ in range(50):
            mujoco.mj_step(model, data)
        backend = ArmBackend(model, data, args.arm)
        controller = PickPlace(model, data, destination, backend=backend)
        steps = controller.sequence()
        max_idle_error = 0.0
        viewer = mujoco.viewer.launch_passive(model, data) if args.viewer else None
        if viewer:
            viewer.opt.geomgroup[3] = False
            viewer.cam.lookat[:] = [0.18, 0.04, 0.1]
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -55
        terminated = False
        try:
            while viewer is None or viewer.is_running():
                start = time.perf_counter()
                try:
                    next(steps)
                except StopIteration:
                    break
                assert np.array_equal(data.ctrl[idle_ids], grasp.PRESET_POSES["HOME"])
                mujoco.mj_step(model, data)
                controller.max_z = max(controller.max_z, float(data.body("target").xpos[2]))
                max_idle_error = max(max_idle_error, float(np.max(np.abs(data.qpos[idle_qpos]))))
                if max_idle_error > 0.02:
                    raise RuntimeError("Inactive arm moved away from HOME")
                for contact in data.contact:
                    if {owner[contact.geom1], owner[contact.geom2]} == {1, 2} and contact.dist < 0:
                        raise RuntimeError("Inter-arm collision")
                if viewer:
                    viewer.sync()
                    time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))
            else:
                terminated = True
                print("[VIEWER] Closed by user; exiting without another trial.")
        except (RuntimeError, ValueError, AssertionError) as error:
            controller.failure = str(error)
        finally:
            result = (
                dict(
                    success=False,
                    failure=None,
                    terminated=True,
                    termination_reason="viewer_closed",
                    initial=controller.initial.tolist(),
                    final=data.body("target").xpos.tolist(),
                    target=destination.tolist(),
                )
                if terminated
                else controller.validate()
            )
            result.update(arm=args.arm, max_inactive_home_error=max_idle_error, runtime=runtime)
            results.append(result)
            print(f"Inactive HOME max joint error: {max_idle_error:.6f} rad")
            if viewer:
                # Keep the completed result visible until the user closes the window.
                while controller.failure is None and not terminated and viewer.is_running():
                    start = time.perf_counter()
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    time.sleep(max(0, model.opt.timestep - (time.perf_counter() - start)))
                viewer.close()
    successes = sum(r["success"] for r in results)
    if any(r.get("terminated") for r in results):
        print("Dual pick/place terminated by user; no manipulation failure.")
    else:
        print(f"{args.arm}: {successes}/{len(results)} SUCCESS")
    report = Path(__file__).parent / f"dual_{args.arm}_results.json"
    report.write_text(
        json.dumps(
            dict(arm=args.arm, successes=successes, runs=len(results), results=results), indent=2
        )
    )
    if any(not r["success"] and not r.get("terminated") for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
