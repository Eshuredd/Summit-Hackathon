"""Run the grasp sequence headlessly and report actual contact penetration."""

import sys
from pathlib import Path

import mujoco
import numpy as np

import simulate as sim


def main(render: bool = False) -> None:
    """Measure each phase without viewer timing or rendering effects."""
    model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "assets/scene.xml"))
    data = mujoco.MjData(model)
    sim.SO101Robot(model=model, data=data).reset("HOME")
    targets = sim.get_validated_auto_targets(model, data, sim.get_end_effector_site_id(model))
    cube = sim.get_cube_geom_id(model)
    fingers = sim.get_finger_collision_geom_ids(model)
    initial_z = float(data.body("target").xpos[2])
    for phase in sim.AUTO_SEQUENCE:
        start = data.ctrl.copy()
        duration = 3.0 if phase == "GRASP_DEPTH" else 1.5
        if phase == "CLOSE_GRIPPER":
            duration = sim.GRIPPER_CLOSE_SECONDS
        total = duration + (sim.GRASP_SETTLE_SECONDS if phase == "CLOSE_GRIPPER" else 0)
        tracker = sim.GraspContactTracker()
        confirmed = False
        worst = 0.0
        for step in range(int(total / model.opt.timestep)):
            if phase == "LIFT":
                assert sim.cube_contacts_fixed_and_moving_jaw(model, data, cube), "Lost grasp"
            alpha = 0.5 * (1 - np.cos(np.pi * min((step + 1) * model.opt.timestep / duration, 1)))
            data.ctrl[:] = (1 - alpha) * start + alpha * targets[phase]
            mujoco.mj_step(model, data)
            contacts = sim.get_finger_cube_contacts(model, data, cube, fingers)
            worst = max(worst, max((-c[2] for c in contacts), default=0))
            if phase == "CLOSE_GRIPPER" and (step + 1) * model.opt.timestep >= duration:
                confirmed = tracker.update(
                    data.time, sim.cube_contacts_fixed_and_moving_jaw(model, data, cube)
                )
                if confirmed:
                    break
        both = sim.cube_contacts_fixed_and_moving_jaw(model, data, cube)
        print(
            f"{phase}: penetration={worst * 1000:.3f} mm; both jaws={both}; "
            f"cube={data.body('target').xpos}; jaw={data.qpos[5]:.3f}"
        )
        assert np.all(np.isfinite(data.qpos)), "Simulation became unstable"
        assert worst < 0.001, f"Excessive penetration during {phase}: {worst} m"
        if phase == "LIFT":
            for _ in range(int(1 / model.opt.timestep)):
                mujoco.mj_step(model, data)
                assert sim.cube_contacts_fixed_and_moving_jaw(model, data, cube), "Lost grasp"
            print(f"Cube retained after lift: {both}")
            assert both, "Lift ended without opposing pad contact"
            assert data.body("target").xpos[2] - initial_z >= 0.045, "Cube did not lift 45 mm"
        if phase == "LIFT" and render:
            renderer = mujoco.Renderer(model, height=480, width=640)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = data.body("target").xpos
            camera.distance = 0.28
            camera.azimuth = 90
            camera.elevation = -15
            options = mujoco.MjvOption()
            options.geomgroup[3] = False
            renderer.update_scene(data, camera=camera, scene_option=options)
            from PIL import Image

            Image.fromarray(renderer.render()).save(Path(__file__).parent / "grasp_verified.png")
            renderer.close()
        if phase == "CLOSE_GRIPPER" and not confirmed:
            raise AssertionError("Lift blocked: cube is not grasped by both opposing pads")
    # Neither single-sided pad contact nor shell contact may authorize lifting.
    for prefix in ("fixed_pad_", "moving_pad_"):
        ids = [i for i in fingers if model.geom(i).name.startswith(prefix)]
        types = model.geom_contype[ids].copy()
        affinities = model.geom_conaffinity[ids].copy()
        model.geom_contype[ids] = 0
        model.geom_conaffinity[ids] = 0
        mujoco.mj_forward(model, data)
        assert not sim.cube_contacts_fixed_and_moving_jaw(model, data, cube), prefix
        model.geom_contype[ids] = types
        model.geom_conaffinity[ids] = affinities
        mujoco.mj_forward(model, data)
    tracker = sim.GraspContactTracker()
    assert not tracker.update(0, True)
    assert not tracker.update(0.1, True)
    assert not tracker.update(0.15, False)
    assert not tracker.update(0.2, True)
    assert not tracker.update(0.3, True)
    assert tracker.update(0.41, True)
    assert not tracker.update(0.42, False)
    print("One-sided contact rejection and continuous-contact timer checks passed.")


if __name__ == "__main__":
    main(render="--render" in sys.argv)
