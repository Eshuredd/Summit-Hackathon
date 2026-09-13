"""Structural regressions for the generated dual scene and controller adapter."""

import mujoco
import numpy as np

import simulate as grasp
from dual_pick_place import ASSETS, ArmBackend, verify_layout


def test_dual_structure():
    """Check namespace isolation, physical jaw identity, and actuator mapping."""
    single = mujoco.MjModel.from_xml_path(str(ASSETS / "scene.xml"))
    dual = mujoco.MjModel.from_xml_path(str(ASSETS / "dual_scene.xml"))
    verify_layout(dual)
    assert dual.nq == 19 and dual.nv == 18 and dual.nu == 12
    assert dual.geom("placement_marker").contype == 0
    assert dual.geom("placement_marker").conaffinity == 0
    for arm in ("left", "right"):
        for i in range(1, single.nbody):
            name = single.body(i).name
            if name != "target":
                assert dual.body(f"{arm}_{name}").id >= 0
        for name in ("fixed_pad_4", "fixed_pad_5", "moving_pad_0", "moving_pad_1"):
            source = single.geom(name).id
            target = dual.geom(f"{arm}_{name}").id
            for field in (
                "geom_pos",
                "geom_quat",
                "geom_size",
                "geom_friction",
                "geom_solref",
                "geom_solimp",
                "geom_contype",
                "geom_conaffinity",
            ):
                np.testing.assert_allclose(
                    getattr(single, field)[source], getattr(dual, field)[target]
                )
            assert dual.mesh(int(dual.geom_dataid[target])).name == name
    data = mujoco.MjData(dual)
    mujoco.mj_forward(dual, data)
    for arm in ("left", "right"):
        backend = ArmBackend(dual, data, arm)
        command = backend.expand(np.full(6, 0.1))
        idle = np.setdiff1d(np.arange(12), backend.indices)
        np.testing.assert_array_equal(command[idle], grasp.PRESET_POSES["HOME"])
        np.testing.assert_array_equal(command[backend.indices], np.full(6, 0.1))
        assert all(
            dual.body(dual.geom_bodyid[g]).name.startswith(f"{arm}_") for g in backend.allowed
        )


if __name__ == "__main__":
    test_dual_structure()
    print("Dual scene structural checks passed.")
