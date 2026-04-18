# The purpose of this module is to define the visual verification interface.
# After each robot motion step, the orchestrator passes a webcam frame and a yes/no question
# to verify(). Moondream2 INT4 runs on the AMD Ryzen AI NPU to answer in under 120ms —
# fast enough to check every step without adding meaningful delay to the task.

from __future__ import annotations

from enum import Enum

import numpy as np


class Backend(str, Enum):
    NPU = "NPU"  # AMD Ryzen AI NPU via ONNX QNN — required for real-time operation (<120ms)
    GPU = "GPU"  # AMD GPU via ONNX DirectML — fallback if NPU is unavailable
    CPU = "CPU"  # ~820ms per query — too slow for closed-loop use between arm steps


VERIFICATION_QUERIES = {
    "post_pick":     "Is an object held securely in the gripper?",
    "pre_place":     "Is this shelf slot currently empty?",
    "post_place":    "Is there an item standing upright in this shelf slot?",
    "task_complete": "Is the source box empty?",
}


class VerificationEngine:
    # loads Moondream2 INT4 once on the given backend — instantiate once and reuse across steps
    def __init__(self, backend: Backend = Backend.NPU) -> None:
        raise NotImplementedError

    def query(self, frame: np.ndarray, question: str) -> tuple[bool, float]:
        # returns (yes/no answer, inference time in ms)
        raise NotImplementedError


def verify(frame: np.ndarray, query: str, backend: Backend = Backend.NPU) -> tuple[bool, float]:
    raise NotImplementedError
