"""
vlm_api.py — On-device VLM verification interface stub
Owner: Sasha

Runs Moondream2 INT4 on AMD ROCm GPU via ONNX Runtime ROCm execution
provider. CPU fallback available via backend parameter.

Target latency: <120ms per verify() call on ROCm GPU.
CPU baseline: ~700–900ms (too slow for closed-loop operation).

Integration note for Sasha:
  - The orchestrator calls verify() after every robot step — low latency
    is load-bearing, not a nice-to-have. Confirm sub-120ms before merging.
  - Expose the backend switch as shown below so the orchestrator can
    fall back to CPU if ROCm has issues during the demo.
  - latency_ms in the return tuple is what gets written to trace.jsonl
    and displayed on the dashboard — return the real number.
  - Install: pip install onnxruntime-rocm --break-system-packages
    (ROCm 6.x wheel; confirm ROCm version on the node with rocminfo)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Backend constants
# ---------------------------------------------------------------------------

BACKEND_ROCM = "rocm"   # ROCMExecutionProvider  (primary — ROCm GPU)
BACKEND_GPU  = BACKEND_ROCM   # alias kept for call-site compatibility
BACKEND_CPU  = "cpu"    # CPUExecutionProvider   (last resort / tests)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""vlm_api.py — On-device VLM verification via Ollama + Moondream2
Owner: Sasha

Runs Moondream2 on AMD Radeon 890M via ROCm (25/25 layers on GPU).
Custom Modelfile for yes/no verification queries.
Fully on-device, no cloud.
Latency: ~100ms per verify() call (warm).

Usage:
    from vlm_api import verify
    result, latency_ms = verify(frame, "grasp_check")

Query types:
    "grasp_check"  - is an object held in the gripper?
    "slot_empty"   - is the shelf slot empty?
    "slot_filled"  - is an item upright in the slot?
    "box_empty"    - is the source box empty?

Returns:
    (bool, float) - (result, latency_ms)
"""

import requests
import base64
import time
import cv2

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "moondream-verify"

QUERIES = {
    "grasp_check": "Is there an object being held in the gripper?",
    "slot_empty": "Is this shelf slot empty?",
    "slot_filled": "Is there an item standing upright on the shelf?",
    "box_empty": "Is the source box empty?",
}

def verify(frame, query_type):
    question = QUERIES.get(query_type)
    if not question:
        return False, 0.0

    _, buf = cv2.imencode('.jpg', frame)
    img_b64 = base64.b64encode(buf).decode()

    start = time.perf_counter()
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": question,
        "images": [img_b64],
        "stream": False
    })
    latency = (time.perf_counter() - start) * 1000
    answer = resp.json()["response"].strip().lower()
    result = answer.startswith("yes")

    return result, latency

if __name__ == "__main__":
    import numpy as np

    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    for qt in ["grasp_check", "slot_empty", "slot_filled", "box_empty"]:
        result, latency = verify(frame, qt)
        print(f"{qt:15s} | {str(result):5s} | {latency:.0f}ms")
