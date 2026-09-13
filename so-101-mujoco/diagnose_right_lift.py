"""Record actual right-arm lift motion and contact state without changing physics."""

import contextlib
import io
import json

import mujoco
import numpy as np

from dual_pick_place import ASSETS, ArmBackend
from pick_place import PickPlace


def main():
    """Print phase transitions, jaw motion, and pad contacts around lift."""
    model = mujoco.MjModel.from_xml_path(str(ASSETS / "dual_scene.xml"))
    data = mujoco.MjData(model)
    for _ in range(50):
        mujoco.mj_step(model, data)
    backend = ArmBackend(model, data, "right")
    controller = PickPlace(model, data, model.site("placement_target").pos, backend)
    rows = []
    joint = model.joint("right_gripper").id
    adr = model.jnt_qposadr[joint]
    dof = model.jnt_dofadr[joint]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in controller.sequence():
                mujoco.mj_step(model, data)
                if controller.phase in {"CLOSE_GRIPPER", "LIFT"}:
                    rows.append(
                        dict(
                            time=data.time,
                            phase=controller.phase,
                            command=float(data.ctrl[11]),
                            angle=float(data.qpos[adr]),
                            velocity=float(data.qvel[dof]),
                            arm=data.qpos[6:12].tolist(),
                            both=bool(
                                backend.cube_contacts_fixed_and_moving_jaw(
                                    model, data, controller.cube
                                )
                            ),
                        )
                    )
                if controller.phase == "ABOVE_TARGET":
                    break
    except RuntimeError as error:
        print(error)
        for i, contact in enumerate(data.contact):
            if controller.cube not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == controller.cube else contact.geom1
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            normal = contact.frame[:3] * (-1 if contact.geom1 == controller.cube else 1)
            axis = data.xmat[model.geom_bodyid[other]].reshape(3, 3)[:, 0]
            print(
                model.geom(other).name,
                "dist",
                contact.dist,
                "force",
                force[0],
                "normal",
                normal,
                "axis dot",
                normal @ axis,
            )
    (ASSETS.parent / "right_lift_trace.json").write_text(json.dumps(rows, indent=2))
    lift = [r for r in rows if r["phase"] == "LIFT"]
    print("Lift start:", lift[0])
    print("Lift end:", lift[-1])
    print("Max jaw speed:", max(abs(r["velocity"]) for r in lift))
    print("Jaw excursion:", np.ptp([r["angle"] for r in lift]))


if __name__ == "__main__":
    main()
