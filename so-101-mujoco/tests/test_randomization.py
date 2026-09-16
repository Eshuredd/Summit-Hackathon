"""Reproducibility, physical bounds, and inference isolation for seeded scenes."""

import json

import numpy as np
import pytest

from bimind.instruction import DEFAULT_INSTRUCTION
from bimind.observation import build_observation
from bimind.policy_features import encode_inputs
from bimind.randomization import apply_randomization
from bimind.skills import Skills


def test_reproducible_fresh_scenes():
    """Identical seeds reproduce both metadata and the actual modified environment."""
    scenes = [Skills(seed=seed) for seed in (3, 3, 4)]
    try:
        a, b, c = scenes
        assert a.randomization == b.randomization != c.randomization
        json.dumps(a.randomization, allow_nan=False)
        for name in ("body_mass", "body_inertia", "geom_friction", "tex_data", "light_diffuse"):
            np.testing.assert_array_equal(
                getattr(a._controller.model, name), getattr(b._controller.model, name)
            )
        np.testing.assert_array_equal(a._controller.data.qpos, b._controller.data.qpos)
    finally:
        for scene in scenes:
            scene.close()


@pytest.mark.parametrize("seed", range(10))
def test_ranges_and_initial_clearance(seed):
    """All evaluation seeds remain physical, inside the tray, and at time zero."""
    skills = Skills(seed=seed)
    try:
        c = skills._controller
        meta = skills.randomization
        assert 0.8 <= meta["lighting_scale"] <= 1.2
        assert 0.9 <= meta["mass_scale"] <= 1.1
        assert 0.9 <= meta["friction_scale"] <= 1.1
        assert np.max(np.abs(meta["object_offset"])) <= 0.003
        assert meta["object_offset"][2] == 0
        assert meta["background_variant"] in ("original", "lighter", "darker")
        assert meta["background_rgb_delta"] in (0, 8, -8)
        assert c.data.time == 0
        assert np.all(c.model.geom_friction > 0)
        assert c.model.body("target").mass[0] > 0
        relative = c.position()[:2] - c.model.body("drawer").pos[:2]
        assert np.all(np.abs(relative) + c.model.geom_size[c.cube, :2] < [0.073, 0.042])
        assert all(
            contact.dist >= -1e-10
            for contact in c.data.contact
            if c.cube in (contact.geom1, contact.geom2)
        )
        adr = c.model.jnt_qposadr[c.model.joint("target_joint").id]
        np.testing.assert_array_equal(c.data.qpos[adr + 3 : adr + 7], [1, 0, 0, 0])
        c.data.time = 1
        with pytest.raises(ValueError, match="fresh scene"):
            apply_randomization(c.model, c.data, seed)
    finally:
        skills.close()


def test_metadata_never_becomes_policy_input():
    """Neither observation allowlisting nor feature encoding admits seed metadata."""
    skills = Skills(seed=3)
    try:
        state = skills.get_scene_state()
        perception = {"state": "closed", "confidence": 0.9}
        expected = build_observation(DEFAULT_INSTRUCTION, perception, state)
        state.update(skills.randomization, randomization=skills.randomization)
        actual = build_observation(DEFAULT_INSTRUCTION, perception, state)
        assert actual == expected
        rgb = np.zeros((96, 96, 3), dtype=np.uint8)
        before = encode_inputs(rgb, DEFAULT_INSTRUCTION, expected)
        actual.update(skills.randomization, randomization=skills.randomization)
        after = encode_inputs(rgb, DEFAULT_INSTRUCTION, actual)
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])
    finally:
        skills.close()


def test_evaluation_records_failure_and_continues(monkeypatch, tmp_path):
    """An unexpected task error closes its scene and does not skip later seeds."""
    import evaluate_randomized as evaluation

    closed = []

    class Session:
        """Provide only the scene lifecycle used by the evaluator."""

        def __init__(self, scene, seed):
            """Record the seed without creating a physics scene."""
            self.randomization = {"seed": seed}

        def get_run_report(self):
            """Return the available physical diagnostics after an exception."""
            return {"object_final_xy_error": 0.1, "arm_arm_collisions": 0}

        def close(self):
            """Record complete lifecycle cleanup."""
            closed.append(self.randomization["seed"])

    def task(skills, **kwargs):
        """Fail the first trial and complete the second."""
        if skills.randomization["seed"] == 0:
            raise RuntimeError("injected task failure")
        return dict(success=True, failure=None, **skills.get_run_report())

    monkeypatch.setattr(evaluation, "Skills", Session)
    monkeypatch.setattr(evaluation, "run_task", task)
    output = tmp_path / "results.json"
    result = evaluation.evaluate([0, 1], "symbolic", output)
    assert closed == [0, 1]
    assert result["successes"] == 1 and result["runs"] == 2
    assert result["results"][0]["failure"] == "RuntimeError: injected task failure"
    assert json.loads(output.read_text()) == result
