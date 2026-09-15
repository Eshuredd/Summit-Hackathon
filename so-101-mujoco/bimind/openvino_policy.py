"""OpenVINO-only runtime for one tiny high-level multimodal skill classifier."""

from functools import lru_cache
from pathlib import Path

import numpy as np

from bimind.policy_features import LABELS, encode_inputs

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "policy_model" / "policy.xml"


class OpenVINOPolicy:
    """Load the exported classifier once and predict one existing skill at a time."""

    def __init__(self, model_path=DEFAULT_MODEL, device="CPU", confidence_threshold=0.65):
        """Compile FP32 OpenVINO IR or ONNX for batch-one synchronous inference.

        Args:
            model_path: Exported IR XML or ONNX file.
            device: OpenVINO device name.
            confidence_threshold: Minimum model probability allowed to authorize a skill.

        Raises:
            ValueError: Confidence threshold is outside the unit interval.
        """
        import openvino as ov

        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence threshold must be in [0, 1]")
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.core = ov.Core()
        self.device_name = self.core.get_property(device, "FULL_DEVICE_NAME")
        config = {"PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": "f32"}
        if device == "CPU":
            config["INFERENCE_NUM_THREADS"] = 1
        self.compiled = self.core.compile_model(str(model_path), device, config)
        self.request = self.compiled.create_infer_request()
        self.precision = str(self.compiled.get_property("INFERENCE_PRECISION_HINT"))

    def infer_logits(self, inputs):
        """Run already encoded tensors through the compiled OpenVINO network.

        Args:
            inputs: Dictionary of float32 batch-one model inputs.

        Returns:
            Seven class logits copied from the inference request.
        """
        self.request.infer(inputs)
        return self.request.get_output_tensor(0).data[0].copy()

    def predict_next_skill(self, rgb, instruction, robot_state):
        """Predict the next skill, failing closed on invalid inputs or low confidence.

        Args:
            rgb: Current uint8 RGB frame.
            instruction: Natural-language instruction.
            robot_state: Current arms and holding feedback; no scene truth is encoded.

        Returns:
            Skill label and softmax confidence. Low confidence returns stop.
        """
        if robot_state.get("failure") is not None:
            return {"skill": "stop", "confidence": 0.0}
        try:
            inputs = encode_inputs(rgb, instruction, robot_state)
            logits = self.infer_logits(inputs)
        except ValueError, KeyError, TypeError, RuntimeError:
            return {"skill": "stop", "confidence": 0.0}
        if logits.shape != (len(LABELS),) or not np.isfinite(logits).all():
            return {"skill": "stop", "confidence": 0.0}
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        label = LABELS[index] if confidence >= self.confidence_threshold else "stop"
        return {"skill": label, "confidence": confidence}


@lru_cache(maxsize=1)
def _default_policy():
    """Compile the default CPU policy once for the module-level convenience API."""
    return OpenVINOPolicy()


def predict_next_skill(rgb, instruction, robot_state):
    """Predict with the cached default OpenVINO policy.

    Args:
        rgb: Current uint8 RGB frame.
        instruction: Natural-language command.
        robot_state: Current arms and holding feedback.

    Returns:
        Dictionary containing skill and confidence.
    """
    return _default_policy().predict_next_skill(rgb, instruction, robot_state)
