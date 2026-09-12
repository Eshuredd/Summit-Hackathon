# SO-101 MuJoCo Agent Guidelines

## 1. Context Sources
- **SO-101 Arm**: 6-DOF robotic arm. Check [README.md](file:///Users/juanes/code/so-101-mujoco/README.md) for overview, [so101/robot.py](file:///Users/juanes/code/so-101-mujoco/so101/robot.py) for kinematics & preset poses (`HOME`, `REACH`, `PICK`, `STOW`), and [so101/env.py](file:///Users/juanes/code/so-101-mujoco/so101/env.py) for RL environment specs.
- **MuJoCo & Warp**: See MuJoCo documentation and [simulate.py](file:///Users/juanes/code/so-101-mujoco/simulate.py) for standard simulation loops. For GPU/CPU batched simulation, reference [so101/warp_env.py](file:///Users/juanes/code/so-101-mujoco/so101/warp_env.py), `google-deepmind/mujoco_warp`, and NVIDIA Warp docs.

## 2. Environment (`uv`)
- **Activate Virtualenv**: `source .venv/bin/activate`
- **Run Commands**: Prefer executing tests, scripts, and linters via `uv run` (e.g., `uv run ruff check .`, `uv run pytest`, `uv run python simulate.py`).
- **Sync Dependencies**: `uv sync`

## 3. Formatting & Linting (`ruff`)
- Always inspect and auto-format code using `ruff` as configured in [pyproject.toml](file:///Users/juanes/code/so-101-mujoco/pyproject.toml):
  - Check & fix lint: `uv run ruff check --fix .`
  - Format code: `uv run ruff format .`
- **Docstrings**: All modules, classes, and functions must use **Google-styled docstrings** (`Args:`, `Returns:`, `Raises:`). Max line length is **100 chars**.

## 4. Git Restrictions
> [!IMPORTANT]
> **NEVER execute `git add` or `git commit`.** The user assumes full responsibility for staging and committing changes. Read-only exploratory commands (`git status`, `git diff`, `git log`) are allowed.

## 5. External Repo Info (`gh`)
- You are authorized and encouraged to use the GitHub CLI (`gh`) to query external repositories, docs, issues, and PRs (e.g., `gh repo view google-deepmind/mujoco_warp`).
