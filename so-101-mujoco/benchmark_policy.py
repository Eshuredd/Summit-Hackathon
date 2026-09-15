"""Benchmark warm batch-one OpenVINO inference and preprocessing separately."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from bimind.openvino_policy import DEFAULT_MODEL, OpenVINOPolicy
from bimind.policy_features import encode_inputs


def summarize(seconds):
    """Summarize warmed synchronous measurements in milliseconds and requests/sec.

    Args:
        seconds: Individual elapsed times in seconds.

    Returns:
        Average, p50, p95, and sequential throughput.
    """
    return {
        "average_ms": float(np.mean(seconds) * 1000),
        "p50_ms": float(np.percentile(seconds, 50) * 1000),
        "p95_ms": float(np.percentile(seconds, 95) * 1000),
        "throughput_per_second": float(1 / np.mean(seconds)),
    }


def benchmark(dataset, model=DEFAULT_MODEL, device="CPU", iterations=500):
    """Measure model-only and full API timings over all dataset input frames.

    Args:
        dataset: Dataset directory, loaded before timed calls.
        model: Exported model path.
        device: OpenVINO target device.
        iterations: Number of measured requests for each timing mode.

    Returns:
        Benchmark report excluding compilation, camera rendering, and file I/O.
    """
    policy = OpenVINOPolicy(model, device)
    rows = [json.loads(line) for line in (dataset / "samples.jsonl").read_text().splitlines()]
    samples = [
        (
            np.asarray(Image.open(dataset / row["rgb"]).convert("RGB")),
            row["instruction"],
            row["observation"],
        )
        for row in rows
    ]
    encoded = [encode_inputs(*sample) for sample in samples]
    for index in range(30):
        policy.predict_next_skill(*samples[index % len(samples)])
    raw, complete = [], []
    for index in range(iterations):
        start = perf_counter()
        policy.infer_logits(encoded[index % len(encoded)])
        raw.append(perf_counter() - start)
    for index in range(iterations):
        start = perf_counter()
        policy.predict_next_skill(*samples[index % len(samples)])
        complete.append(perf_counter() - start)
    return {
        "device": device,
        "device_name": policy.device_name,
        "precision": policy.precision,
        "iterations": iterations,
        "warmup": 30,
        "batch_size": 1,
        "cpu_inference_threads": 1 if device == "CPU" else None,
        "model_only": summarize(raw),
        "full_predict_api": summarize(complete),
        "excludes": ["model load/compile", "file I/O", "camera rendering", "robot motion"],
    }


def main():
    """Run and persist a simple CPU benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("policy_dataset"))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("policy_model/benchmark.json"))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("iterations must be positive")
    report = benchmark(args.dataset, args.model, args.device, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
