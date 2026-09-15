"""Train one tiny symbolic-teacher policy, evaluate, and export ONNX and OpenVINO IR."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import openvino as ov
import torch
from PIL import Image

from bimind.policy_features import INSTRUCTIONS, LABELS, ROBOT_FEATURES, VOCABULARY, encode_inputs
from bimind.policy_model import TinyPolicy


def load_dataset(directory):
    """Load source frames and split by held-out visual scenario, before text augmentation.

    Args:
        directory: Collector output containing samples.jsonl and PNG images.

    Returns:
        Records, encoded tensors, labels, and disjoint train/validation indices.
    """
    records = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    encoded, targets, train, validation = [], [], [], []
    class_counts = {}
    for index, row in enumerate(records):
        label = row["teacher_next_skill"]
        count = class_counts.get(label, 0)
        class_counts[label] = count + 1
        # Day-1 collection order has eight scenario views per normal skill and
        # three partial-drawer STOP views. New collections carry scenario names.
        scenario = row.get("scenario")
        held_out = (
            scenario in ("camera_right", "background_gray")
            if scenario is not None
            else count % 8 in (4, 7)
            if label != "stop"
            else count % 3 == 2
        )
        (validation if held_out else train).append(index)
        rgb = np.asarray(Image.open(directory / row["rgb"]).convert("RGB"))
        encoded.append(encode_inputs(rgb, row["instruction"], row["observation"]))
        targets.append(LABELS.index(label))
    tensors = {
        name: torch.from_numpy(np.concatenate([item[name] for item in encoded]))
        for name in encoded[0]
    }
    if set(np.array(targets)[train]) != set(range(len(LABELS))) or set(
        np.array(targets)[validation]
    ) != set(range(len(LABELS))):
        raise ValueError("Both splits must contain all seven skill classes")
    return records, tensors, torch.tensor(targets), train, validation


def train(dataset, output, epochs=800, seed=7):
    """Fit the fixed model on CPU and export the final epoch without validation tuning.

    Args:
        dataset: Collector dataset directory.
        output: Artifact directory.
        epochs: Fixed number of full-batch Adam steps.
        seed: Reproducible initialization and command augmentation seed.

    Returns:
        Serializable training and export report.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    records, tensors, targets, train_indices, val_indices = load_dataset(dataset)
    model = TinyPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    counts = torch.bincount(targets[train_indices], minlength=len(LABELS)).float()
    weights = counts.sum() / (len(LABELS) * counts)
    command_vectors = []
    dummy_rgb = np.zeros((96, 96, 3), dtype=np.uint8)
    for command in INSTRUCTIONS:
        command_vectors.append(
            encode_inputs(dummy_rgb, command, records[0]["observation"])["instruction"][0]
        )
    command_vectors = torch.tensor(np.array(command_vectors))
    model.train()
    for epoch in range(epochs):
        text = command_vectors[torch.randint(len(INSTRUCTIONS), (len(train_indices),))]
        images = tensors["rgb"][train_indices].clone()
        # Broaden sparse source-frame coverage without touching validation frames.
        images *= torch.empty((len(train_indices), 1, 1, 1)).uniform_(0.8, 1.2)
        images.clamp_(0, 1)
        padded = torch.nn.functional.pad(images, (2, 2, 2, 2), mode="replicate")
        offsets = torch.randint(0, 5, (len(train_indices), 2))
        images = torch.stack(
            [padded[i, :, y : y + 96, x : x + 96] for i, (y, x) in enumerate(offsets)]
        )
        logits = model(images, text, tensors["robot_state"][train_indices])
        loss = torch.nn.functional.cross_entropy(logits, targets[train_indices], weight=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"epoch={epoch + 1} loss={loss.item():.6f}", flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(**tensors)
        predictions = logits.argmax(1)
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    for index in val_indices:
        confusion[int(targets[index]), int(predictions[index])] += 1
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "policy.pt")
    example = tuple(tensors[name][:1] for name in ("rgb", "instruction", "robot_state"))
    torch.onnx.export(
        model,
        example,
        str(output / "policy.onnx"),
        input_names=["rgb", "instruction", "robot_state"],
        output_names=["logits"],
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    onnx.checker.check_model(onnx.load(output / "policy.onnx"))
    converted = ov.convert_model(str(output / "policy.onnx"))
    ov.save_model(converted, str(output / "policy.xml"), compress_to_fp16=False)
    compiled = ov.Core().compile_model(converted, "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    errors, ov_predictions = [], []
    for index in range(len(records)):
        result = compiled(
            {name: value[index : index + 1].numpy() for name, value in tensors.items()}
        )
        actual = result[compiled.output(0)][0]
        errors.append(float(np.max(np.abs(actual - logits[index].numpy()))))
        ov_predictions.append(int(actual.argmax()))
    if not np.allclose(np.array(ov_predictions), predictions.numpy()):
        raise RuntimeError("OpenVINO class predictions differ from PyTorch")
    report = {
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "train_accuracy": float(
            (predictions[train_indices] == targets[train_indices]).float().mean()
        ),
        "validation_accuracy": float(
            (predictions[val_indices] == targets[val_indices]).float().mean()
        ),
        "labels": LABELS,
        "confusion_matrix_rows_truth_columns_prediction": confusion.tolist(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "precision": "FP32",
        "onnx_export_success": True,
        "openvino_conversion_success": True,
        "openvino_max_abs_logit_error": max(errors),
        "openvino_class_agreement": 1.0,
        "epochs": epochs,
        "seed": seed,
        "vocabulary": VOCABULARY,
        "robot_features": ROBOT_FEATURES,
        "image_size": 96,
        "split": "Held-out camera_right/background_gray; legacy STOP holds out 40mm view",
        "limitation": (
            "One deterministic episode; validation measures visual perturbations, "
            "not new manipulation states"
        ),
        "training_instruction_augmentation": INSTRUCTIONS,
        "training_image_augmentation": (
            "brightness 0.8-1.2; translation +/-2 pixels with edge padding"
        ),
        "train_files": [records[i]["rgb"] for i in train_indices],
        "validation_files": [records[i]["rgb"] for i in val_indices],
        "dataset_sha256": hashlib.sha256((dataset / "samples.jsonl").read_bytes()).hexdigest(),
        "artifact_bytes": {
            name: (output / name).stat().st_size
            for name in ("policy.pt", "policy.onnx", "policy.xml", "policy.bin")
        },
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "openvino": ov.__version__,
        },
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    """Train and export the single fixed architecture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("policy_dataset"))
    parser.add_argument("--output", type=Path, default=Path("policy_model"))
    parser.add_argument("--epochs", type=int, default=800)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    train(args.dataset, args.output, args.epochs)


if __name__ == "__main__":
    main()
