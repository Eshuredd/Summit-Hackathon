"""Small deterministic perturbations for fresh, validated drawer scenes."""

import mujoco
import numpy as np

LIGHT_RANGE = (0.8, 1.2)
MASS_RANGE = (0.9, 1.1)
FRICTION_RANGE = (0.9, 1.1)
XY_LIMIT = 0.003
BACKGROUND_DELTAS = (0, 8, -8)


def apply_randomization(model, data, seed):
    """Randomize a fresh baseline model once, without advancing physics.

    Args:
        model: Newly loaded drawer model; do not reuse a randomized model.
        data: Initialized scene data at time zero, before rendering or execution.
        seed: Nonnegative integer seed for the sole random generator.

    Returns:
        JSON-compatible evaluation metadata, never an inference observation.

    Raises:
        ValueError: The scene has already advanced or placement is outside the tray.
    """
    if data.time != 0:
        raise ValueError("Randomization requires a fresh scene at time zero")
    rng = np.random.default_rng(seed)
    light = float(rng.uniform(*LIGHT_RANGE))
    mass = float(rng.uniform(*MASS_RANGE))
    friction = float(rng.uniform(*FRICTION_RANGE))
    offset = np.r_[rng.uniform(-XY_LIMIT, XY_LIMIT, 2), 0.0]
    variant = int(rng.integers(len(BACKGROUND_DELTAS)))
    body = model.body("target").id
    cube = int(model.body_geomadr[body])
    adr = model.jnt_qposadr[model.joint("target_joint").id]
    position = data.qpos[adr : adr + 3] + offset
    center = model.body("drawer").pos[:2]
    # Inner wall faces are 73 mm and 42 mm from the drawer center.
    if np.any(np.abs(position[:2] - center) + model.geom_size[cube, :2] >= [0.073, 0.042]):
        raise ValueError("Randomized object lies outside drawer interior")
    model.body_mass[body] *= mass
    model.body_inertia[body] *= mass
    drawer = model.body("drawer").id
    geoms = np.flatnonzero((model.geom_bodyid == body) | (model.geom_bodyid == drawer))
    model.geom_friction[geoms] *= friction
    for field in ("ambient", "diffuse", "specular"):
        getattr(model, f"light_{field}")[:] *= light
        getattr(model.vis.headlight, field)[:] *= light
    texture = model.texture("groundplane").id
    start = int(model.tex_adr[texture])
    size = int(model.tex_width[texture] * model.tex_height[texture] * model.tex_nchannel[texture])
    pixels = model.tex_data[start : start + size]
    pixels[:] = np.clip(pixels.astype(np.int16) + BACKGROUND_DELTAS[variant], 0, 255)
    # Refresh mass-dependent constants using separate scratch data, preserving robot state.
    mujoco.mj_setConst(model, mujoco.MjData(model))
    data.qpos[adr : adr + 3] = position
    mujoco.mj_forward(model, data)
    return dict(
        seed=int(seed),
        lighting_scale=light,
        mass_scale=mass,
        friction_scale=friction,
        object_offset=offset.tolist(),
        background_variant=("original", "lighter", "darker")[variant],
        background_rgb_delta=BACKGROUND_DELTAS[variant],
    )
