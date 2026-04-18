"""
vlm_api.py — On-device VLM verification via Ollama + Moondream2
Owner: Sasha

Runs Moondream2 on AMD Radeon 890M via ROCm (25/25 layers on GPU).
Custom Modelfile for yes/no verification queries.
Fully on-device, no cloud. Served via Ollama at localhost:11434.
Latency: ~100ms per verify() call (warm).

Usage:
    from vlm_api import verify, locate_object
    result, latency_ms = verify(frame, "Is an object held in the gripper?")
    hints = locate_object(frame)   # → {"visible": bool, "left": bool, "high": bool}

verify() accepts either:
  - A QUERIES dict key  ("grasp_check", "slot_empty", etc.)  — maps to the canned question
  - Any free-text string                                      — sent to Moondream2 as-is
This lets the orchestrator pass full verification_query strings from the
skill program without requiring they match a canned key.

Query type shortcuts (still supported for convenience):
    "grasp_check"       - is an object held in the gripper?
    "slot_empty"        - is this shelf slot empty?
    "slot_filled"       - is an item standing upright on the shelf?
    "box_empty"         - is the source box empty?
    "object_visible"    - is there an object visible in the pick zone?
    "box_on_shelf_area" - is there a box positioned near the shelf?
    "holding_object"    - is the robotic gripper holding something?
    "shelf_has_slot"    - is there an empty slot visible on the shelf?

Returns:
    (bool, float) - (result, latency_ms)

Install:
    ollama pull moondream   # or use custom Modelfile as moondream-verify
    Confirm ROCm offload:  ollama run moondream  (should show 25/25 GPU layers)
"""

from __future__ import annotations

import base64
import logging
import time

import cv2
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend constants (kept for call-site compatibility)
# ---------------------------------------------------------------------------

BACKEND_ROCM = "rocm"
BACKEND_GPU  = BACKEND_ROCM
BACKEND_CPU  = "cpu"

# ---------------------------------------------------------------------------
# Ollama endpoint
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "moondream-verify"

# ---------------------------------------------------------------------------
# Canned query shortcuts — verify() falls back to free-text if key not found
# ---------------------------------------------------------------------------

QUERIES: dict[str, str] = {
    # Existing verification queries
    "grasp_check":       "Is there an object being held in the gripper?",
    "slot_empty":        "Is this shelf slot empty?",
    "slot_filled":       "Is there an item standing upright on the shelf?",
    "box_empty":         "Is the source box empty?",
    # New skill routing queries (used by skill_router.py)
    "object_visible":    "Is there an object visible in the pick zone?",
    "box_on_shelf_area": "Is there a box positioned near the shelf ready to be placed on it?",
    "holding_object":    "Is the robotic gripper currently holding an object?",
    "shelf_has_slot":    "Is there an empty slot visible on the shelf?",
    # New skill verification queries
    "in_box":            "Is the object now placed inside the box?",
    "box_in_slot":       "Is a box standing upright in the shelf slot?",
}

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _ask(frame, question: str) -> tuple[bool, float]:
    """
    Encode frame as JPEG, POST to Ollama, return (bool, latency_ms).
    Raises requests.RequestException on connection failure.
    """
    _, buf = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(buf).decode()

    start = time.perf_counter()
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model":  MODEL,
            "prompt": question,
            "images": [img_b64],
            "stream": False,
        },
        timeout=5.0,
    )
    resp.raise_for_status()
    latency = (time.perf_counter() - start) * 1000

    answer = resp.json()["response"].strip().lower()
    result = answer.startswith("yes")
    logger.debug("VLM (%dms): %r → %s", latency, question[:60], result)
    return result, latency


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify(frame, query: str) -> tuple[bool, float]:
    """
    Verify a yes/no question about the current scene.

    Args:
        frame: BGR numpy array from OpenCV webcam capture.
        query: Either a QUERIES shortcut key (e.g. "grasp_check") or a
               free-text yes/no question sent directly to Moondream2.
               The orchestrator passes full verification_query strings
               from the skill program — those work here without mapping.

    Returns:
        (result: bool, latency_ms: float)
        Returns (False, 0.0) on connection error so the orchestrator can
        classify the failure and retry rather than crashing.
    """
    question = QUERIES.get(query, query)   # key → canned question, else literal
    try:
        return _ask(frame, question)
    except Exception as exc:
        logger.error("vlm_api.verify failed: %s", exc)
        return False, 0.0


def locate_object(frame) -> dict:
    """
    Run a short sequence of positional VLM queries to locate a lost object.
    Called by the orchestrator on OBJECT_NOT_FOUND before the retry.

    Returns a dict with keys:
        "visible" (bool)   — any object at all in the scene?
        "left"    (bool)   — object is left of centre
        "high"    (bool)   — object is above centre
        "latency_ms" (float) — total query latency

    The orchestrator converts these flags into z_offset_mm / approach_angle_deg
    parameter deltas before re-running the pick step.
    """
    total_latency = 0.0
    hints: dict = {}

    try:
        visible, ms = _ask(frame, "Is there any object visible in the scene?")
        total_latency += ms
        hints["visible"] = visible

        if visible:
            left, ms = _ask(frame, "Is the object located on the left side of the scene?")
            total_latency += ms
            hints["left"] = left

            high, ms = _ask(frame, "Is the object higher than the centre of the frame?")
            total_latency += ms
            hints["high"] = high
        else:
            hints["left"] = False
            hints["high"] = False

    except Exception as exc:
        logger.error("vlm_api.locate_object failed: %s", exc)
        hints.setdefault("visible", False)
        hints.setdefault("left", False)
        hints.setdefault("high", False)

    hints["latency_ms"] = total_latency
    return hints


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    logging.basicConfig(level=logging.DEBUG)
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)

    print("--- verify() shortcut keys ---")
    for qt in ["grasp_check", "slot_empty", "slot_filled", "box_empty", "object_visible"]:
        result, latency = verify(frame, qt)
        print(f"  {qt:20s} | {str(result):5s} | {latency:.0f}ms")

    print("\n--- verify() free-text query (orchestrator style) ---")
    result, latency = verify(frame, "Is an object held securely in the gripper?")
    print(f"  free-text              | {str(result):5s} | {latency:.0f}ms")

    print("\n--- locate_object() ---")
    hints = locate_object(frame)
    print(f"  visible={hints['visible']}  left={hints['left']}  high={hints['high']}  "
          f"latency={hints['latency_ms']:.0f}ms")
