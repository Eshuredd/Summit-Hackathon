# Tiny multimodal OpenVINO policy

## Run

Use the project's existing validated environment. Additional dependencies are
pinned in `requirements-policy.txt`; the physics dependency lock is unchanged.

```powershell
python -m pip --python .venv/Scripts/python.exe install -r requirements-policy.txt
.\.venv\Scripts\python.exe train_policy.py
.\.venv\Scripts\python.exe benchmark_policy.py
.\.venv\Scripts\python.exe -m pytest -q tests
.\.venv\Scripts\python.exe planner_drawer_task.py --policy symbolic --runs 10
.\.venv\Scripts\python.exe planner_drawer_task.py --instruction "Open the drawer and place the object on the table." --policy openvino --runs 10
```

Add `--viewer` for a single interactive trial. OpenVINO defaults to CPU and
`policy_model/policy.xml`; `--device` and `--model` allow explicit overrides.
`--confidence-threshold` defaults to 0.65 for both camera and policy gates.

## One fixed architecture

- RGB: bilinear resize to 96x96, float32 CHW in [0,1]; three stride-two
  convolutions (8, 12, 16 channels), ReLU, 4x4 average pooling.
- Instruction: normalized counts in a fixed 14-word vocabulary, then an
  eight-unit linear layer and ReLU. The original command is encoded; all three
  supported phrasings are sampled during training.
- Robot: six flags for left handle ownership, right object ownership, each
  gripper's opening command, active drawer hold monitoring, and acknowledged
  successful placement. No coordinates, joint targets, drawer truth, or
  perception class are neural-network inputs.
- Concatenate features, 24-unit ReLU layer, seven skill logits: 6,955 parameters.

Only the existing supported instruction intent is accepted. This dataset cannot
establish general language understanding beyond those equivalent commands.

## Training and validation

The 51 Day-1 source frames are split before augmentation: 38 train and 13
validation. For each of the six normal skill classes, `camera_right` and
`background_gray` views are held out; the STOP class holds out its 40mm view.
Legacy scenario identity is recovered from the collector's documented row order.
No validation image is used in an optimizer step. This is visual holdout within
one deterministic episode, not independent robot-state or real-world validation.

Training uses seed 7, CPU Adam at 0.003, inverse-frequency class weights, and 800
full-batch steps. Training-only brightness scaling (0.8-1.2) and translations
(up to two resized pixels, edge padding) correct sparse image coverage without
changing the architecture. Equivalent supported instructions are sampled each
step. The initial unaugmented run scored 10/13; its metrics remain in
`initial_training_report.json`. Final metrics are in `training_report.json`.
The final validation score was observed after this augmentation change; it is
not an untouched external test set.

Rows and columns of the confusion matrix use this order:
`open_drawer, hold_drawer, pick_object, place_object, release_drawer, finish, stop`.

```text
2 0 0 0 0 0 0
0 2 0 0 0 0 0
0 0 2 0 0 0 0
0 0 0 2 0 0 0
0 0 0 0 2 0 0
0 0 0 0 0 2 0
0 0 0 0 0 0 1
```

## Deployment and safety

`OpenVINOPolicy.predict_next_skill(rgb, instruction, robot_state)` returns
`{"skill": ..., "confidence": ...}`; the same function is available at module
level with a cached default CPU runtime. It imports no PyTorch for inference.
Invalid inputs, nonfinite outputs, or low softmax confidence return `stop`.

In OpenVINO mode, the network chooses the next skill. The symbolic planner is
not consulted to override the prediction. The runner separately retains the
RGB partial/unknown/low-confidence STOP gate and existing failure STOP gate.
A fixed allowlist binds skill labels to existing Skills API arguments. The
network cannot supply joint targets or arbitrary method names. Controllers,
Skills semantics, physics, robot geometry, IK, grasping, and trajectories remain
unchanged. There is no symbolic fallback for a weak model prediction.

## Export and benchmark

The same trained weights are saved as a PyTorch checkpoint, checked ONNX model,
and FP32 OpenVINO IR. Conversion follows the official
[OpenVINO conversion API](https://docs.openvino.ai/2023.3/openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR.html).
Export uses the [PyTorch ONNX exporter](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html).
All 51 source-frame class predictions are checked for PyTorch/OpenVINO agreement.

`benchmark.json` reports 30 warmups and 500 timed batch-one requests, with a
single CPU inference thread and an explicit FP32 hint. Model-only timing uses
pre-encoded inputs. Full API timing includes resize, token/state encoding,
inference, softmax, and confidence checking. Neither includes compilation,
camera rendering, disk reads, or robot execution. Device identity is reported
from OpenVINO: this machine has an AMD Ryzen 9 5900HX CPU.

Symbolic and OpenVINO task reports are kept separately in
`planner_drawer_results.json` and `planner_openvino_results.json`.

## Verified results

- 105 tests passed.
- Symbolic: 10/10 successful; OpenVINO: 10/10 successful, with model-selected
  actions for all 60 decisions.
- Train: 38/38 correct; validation: 13/13 correct. The deployed API also gets
  39/39 validation frame/command combinations correct at threshold 0.65.
- ONNX: 49,529 bytes. OpenVINO IR: 45,280 bytes (XML plus BIN), FP32.
- Warm model latency: average 0.161 ms, p50 0.156 ms, p95 0.188 ms;
  sequential throughput approximately 6,212 requests/sec.
- Full prediction API: average 1.440 ms, p50 1.430 ms, p95 1.549 ms;
  approximately 695 requests/sec, excluding camera rendering and motion.

Machine-readable evidence is in `policy_model/day2_report.json`.
