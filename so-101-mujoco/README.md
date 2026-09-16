# SO-101 MuJoCo tiny multimodal policy

This project combines the SO-101 drawer task, a symbolic planner, and a compact
RGB + instruction + robot-state OpenVINO policy. The system keeps the original
MuJoCo scene and controller semantics while using a tiny learned classifier at the
planner boundary.

## Architecture

```text
Instruction + RGB + robot state
        ↓
Tiny multimodal OpenVINO policy
        ↓
Skills API
        ↓
Bimanual SO-101 controller
```

## Setup

From this repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-policy.txt
```

For a fresh environment with the project dependencies:

```bash
uv sync
uv run python check_env.py
```

## Symbolic and OpenVINO runs

```powershell
.\.venv\Scripts\python.exe collect_policy_dataset.py
.\.venv\Scripts\python.exe train_policy.py
.\.venv\Scripts\python.exe benchmark_policy.py
.\.venv\Scripts\python.exe planner_drawer_task.py --policy symbolic --runs 10
.\.venv\Scripts\python.exe planner_drawer_task.py --instruction "Open the drawer and place the object on the table." --policy openvino --runs 10
```

## Randomization evaluation

```powershell
.\.venv\Scripts\python.exe evaluate_randomized.py --policy symbolic --seeds 0 1 2 3 4 5 6 7 8 9
.\.venv\Scripts\python.exe evaluate_randomized.py --policy openvino --seeds 0 1 2 3 4 5 6 7 8 9
```

## Intel benchmark

```powershell
.\.venv\Scripts\python.exe benchmark_policy.py --iterations 500
```

## Final results

- 111 total samples
- 86 training
- 25 validation
- 60 randomized teacher samples
- randomized episodes 8 and 9 held out
- 6,955 parameters
- threshold 0.65
- policy artifacts under `policy_model/`

The policy remains a small, transparent classifier with a fixed allowlist and no
randomization metadata in the model inputs.

---

## Attribution and license

This project builds on the upstream SO-101 MuJoCo and MuJoCo Warp work and keeps
that upstream attribution intact. The original repository license remains in force
for the base simulation assets and upstream code paths; the policy additions in
this directory extend that work without removing the upstream copyright and
license notices.
