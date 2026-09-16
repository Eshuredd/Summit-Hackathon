# Tiny multimodal OpenVINO policy

## Dataset and validation summary

The policy is trained on 111 total samples: 86 training and 25 validation. The
training set includes 51 baseline teacher samples plus 60 randomized teacher
samples. Validation holds out the full randomized episodes 8 and 9 while also
keeping the baseline visual holdout split used for the original deterministic
teacher data. The model has 6,955 parameters and uses a confidence threshold of
0.65.

This dataset is intentionally limited to the supported drawer instruction and the
existing planner action set. The model does not consume seed values or randomization metadata.

## Run

Use the validated project environment and the policy-specific dependencies.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-policy.txt
.\.venv\Scripts\python.exe collect_policy_dataset.py
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
  eight-unit linear layer and ReLU. Equivalent supported phrasings are sampled
  during training.
- Robot: six flags for left handle ownership, right object ownership, each
  gripper's opening command, active drawer hold monitoring, and acknowledged
  successful placement. No coordinates, joint targets, drawer truth, or
  perception class are neural-network inputs.
- Concatenate features, 24-unit ReLU layer, seven skill logits: 6,955 parameters.

Only the existing supported instruction intent is accepted.

## Training and validation

The dataset follows the mixed split requested for the project: 51 baseline
samples and 60 randomized teacher samples, with 86 training samples and 25
validation samples. Baseline samples keep the old visual holdout logic; the
randomized collection uses whole-episode validation for episodes 8 and 9. No
validation sample is used in an optimizer step.

Training uses seed 7, CPU Adam at 0.003, inverse-frequency class weights, and
800 full-batch steps. Training-only brightness scaling (0.8-1.2) and
translations (up to two resized pixels, edge padding) correct sparse image
coverage without changing the architecture. Equivalent supported instructions are
sampled each step.

Rows and columns of the confusion matrix use this order:
`open_drawer, hold_drawer, pick_object, place_object, release_drawer, finish, stop`.

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

The same trained weights are saved as a PyTorch checkpoint, a checked ONNX model,
and an FP32 OpenVINO IR. Conversion follows the official
[OpenVINO conversion API](https://docs.openvino.ai/2023.3/openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_IR.html).
Export uses the [PyTorch ONNX exporter](https://docs.pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html).

`benchmark.json` reports 30 warmups and 500 timed batch-one requests, with a
single CPU inference thread and an explicit FP32 hint. Model-only timing uses
pre-encoded inputs. Full API timing includes resize, token/state encoding,
inference, softmax, and confidence checking. Neither includes compilation,
camera rendering, disk reads, or robot execution.

## Verified results

- 111 total samples
- 86 training
- 25 validation
- 60 randomized teacher samples
- randomized episodes 8 and 9 held out
- 6,955 parameters
- threshold 0.65

Machine-readable evidence is stored in `policy_model/training_report.json` and
`policy_model/benchmark.json`.
