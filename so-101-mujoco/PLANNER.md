# RGB instruction planner

Run in the validated project environment:

```powershell
.\.venv\Scripts\python.exe planner_drawer_task.py --instruction "Open the drawer and place the object on the table." --viewer
.\.venv\Scripts\python.exe planner_drawer_task.py --runs 10
.\.venv\Scripts\python.exe collect_policy_dataset.py --output policy_dataset --runs 1
```

The deterministic parser accepts the existing drawer-to-table task only. Every
step renders `drawer_overview`, estimates drawer state from that RGB, combines it
with allowlisted robot feedback and the parsed instruction, plans one skill, and
calls the existing Skills API. `--confidence-threshold` defaults to 0.65.
Partial, unknown, malformed, and low-confidence estimates STOP before any skill,
including release or finish. The planner never reads drawer truth flags.

The observation contains `instruction`, `drawer` (visual_state and confidence),
`arms` (joints, gripper, contact-based holding names), `holding` (monitor status and
placement acknowledgement), and `failure`. No object coordinates are inference
features. Placement acknowledgement starts false on each reset and becomes true
only after the existing place skill succeeds. This is session-local completion
feedback, not object detection or arbitrary-state restart support. Existing
controller safety checks and `finish` still validate the physical outcome.

The policy collector saves PNG frames and `samples.jsonl`, with the original
natural-language instruction, minimal observation, teacher next-skill label,
arguments, and episode ID. It re-renders every step under the eight existing
visual perturbations using isolated model/data copies. Synthetic partial drawer
frames add STOP examples. Privileged coordinates only generate those frames;
they are never serialized as inference features. The output directory must be
new to prevent accidental dataset overwrite. No training is performed.

`planner_drawer_results.json` remains a diagnostics report and can contain
privileged Skills reports; it is not the policy dataset.

Validation:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe evaluate_drawer_perception.py
.\.venv\Scripts\python.exe planner_drawer_task.py --runs 10
```

## Optional learned policy

Use `--policy openvino` for the tiny multimodal classifier, or `--policy symbolic`
for the default teacher. See [POLICY.md](POLICY.md) for training, export, safety,
and benchmark details. The learned mode keeps the same Skills API dispatch.
