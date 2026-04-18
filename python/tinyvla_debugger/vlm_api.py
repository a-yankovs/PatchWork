"""
vlm_api.py — On-device VLM verification interface stub
Owner: Sasha

Runs Moondream2 INT4 on the AMD Ryzen AI NPU via ONNX Runtime QNN
execution provider. GPU (DML) fallback available via backend parameter.

Target latency: <120ms per verify() call on NPU.
CPU baseline: ~700–900ms (too slow for closed-loop operation).

Integration note for Sasha:
  - The orchestrator calls verify() after every robot step — low latency
    is load-bearing, not a nice-to-have. Confirm sub-120ms before merging.
  - Expose the backend switch as shown below so the orchestrator can
    fall back to GPU instantly if NPU has issues during the demo.
  - latency_ms in the return tuple is what gets written to trace.jsonl
    and displayed on the dashboard — return the real number.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------

BACKEND_NPU = "npu"     # QNNExecutionProvider  (primary)
BACKEND_GPU = "gpu"     # DmlExecutionProvider   (fallback)
BACKEND_CPU = "cpu"     # CPUExecutionProvider   (last resort / tests)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify(
    frame: np.ndarray,
    query: str,
    backend: str = BACKEND_NPU,
) -> tuple[bool, float]:
    """
    Run a natural language verification query against a camera frame.

    Uses Moondream2 INT4 quantized via AMD's Olive tool, loaded on the
    AMD Ryzen AI NPU through ONNX Runtime's QNN execution provider.

    The four shelf-stocking queries the orchestrator will call:
      1. "Is an object held securely in the gripper?"  — post-pick
      2. "Is this shelf slot currently empty?"         — pre-placement check
      3. "Is there an item standing upright in this shelf slot?" — post-place
      4. "Is the source box empty?"                    — end-of-task detection

    Args:
        frame:   BGR image as numpy array (H, W, 3), uint8.
                 Captured from webcam immediately after the step completes.
        query:   Natural language yes/no question about the scene.
        backend: Execution provider — use BACKEND_NPU (default), BACKEND_GPU,
                 or BACKEND_CPU. Switch to BACKEND_GPU if NPU latency > 200ms.

    Returns:
        (result, latency_ms):
            result     — True if the VLM answers "yes", False otherwise.
            latency_ms — Wall-clock inference time in milliseconds.

    Raises:
        NotImplementedError: Until Sasha's implementation lands.

    Example:
        >>> result, ms = verify(frame, "Is an object held securely in the gripper?")
        >>> print(f"Gripper check: {result} in {ms:.1f}ms")
    """
    raise NotImplementedError(
        "vlm_api.verify not yet implemented — waiting for Sasha. "
        "Use a mock in tests: return (True, 95.0) for success scenarios."
    )
