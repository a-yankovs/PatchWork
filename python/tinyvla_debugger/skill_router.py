"""
skill_router.py — Webcam-to-skill classifier
Owner: Vera

Given a single webcam frame, determines which of the four canonical robot
skills is needed by running a short sequence of yes/no VLM queries.

Routing priority (evaluated top-to-bottom, first match wins):
  1. box_in_shelf       — a box is near the shelf, ready to be shelved
  2. place_in_box       — the gripper is already holding something
  3. full_pick_and_place — an object AND an empty shelf slot are both visible
  4. pick_object        — an object is visible but no shelf slot is available
  (fallback)            — defaults to full_pick_and_place when scene is ambiguous

Usage:
    from skill_router import route_skill
    from vlm_api import verify as vlm_verify

    frame  = capture_frame()          # BGR numpy array
    skill  = route_skill(frame, vlm_verify)
    # skill is one of: "pick_object", "place_in_box",
    #                  "full_pick_and_place", "box_in_shelf"
    result = await orchestrator.run_skill(skill)

The VLM callable must match the vlm_api.verify() signature:
    verify(frame, query: str) -> (bool, float)

All four output strings are keys in compiler.SKILL_REGISTRY, so
orchestrator.run_skill() will hit the fast-path registry lookup
and bypass the LLM compiler.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

# Canonical skill names — must match keys in compiler.SKILL_REGISTRY
ROUTABLE_SKILLS = [
    "pick_object",
    "place_in_box",
    "full_pick_and_place",
    "box_in_shelf",
]

_FALLBACK_SKILL = "full_pick_and_place"


class VLMCallable(Protocol):
    def __call__(self, frame: Any, query: str) -> tuple[bool, float]: ...


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def route_skill(frame: Any, vlm: VLMCallable) -> str:
    """
    Determine which skill to run based on the current scene.

    Runs at most 4 VLM queries (each ~100ms on warm Moondream2/ROCm).
    Total routing latency budget: ~400ms.

    Args:
        frame: BGR numpy array from OpenCV webcam capture.
        vlm:   Callable matching vlm_api.verify() signature.
               Accepts (frame, query_string) → (bool, float).

    Returns:
        One of: "pick_object", "place_in_box",
                "full_pick_and_place", "box_in_shelf"
    """
    skill = _route(frame, vlm)
    logger.info("SkillRouter → %s", skill)
    return skill


# ---------------------------------------------------------------------------
# Internal routing logic
# ---------------------------------------------------------------------------

def _ask(frame: Any, vlm: VLMCallable, query: str) -> bool:
    """Run one VLM query and return the boolean result (ignores latency)."""
    result, latency_ms = vlm(frame, query)
    logger.debug("  SkillRouter query (%.0fms): %r → %s", latency_ms, query[:60], result)
    return result


def _route(frame: Any, vlm: VLMCallable) -> str:
    # ------------------------------------------------------------------ #
    # Priority 1: box_in_shelf                                            #
    # A box is positioned near the shelf, waiting to be shelved.          #
    # Check this first — it's the most specific scene state.              #
    # ------------------------------------------------------------------ #
    if _ask(frame, vlm, "Is there a box positioned near the shelf ready to be placed on it?"):
        return "box_in_shelf"

    # ------------------------------------------------------------------ #
    # Priority 2: place_in_box                                            #
    # The gripper is already holding an object — skip pick, go to place.  #
    # ------------------------------------------------------------------ #
    if _ask(frame, vlm, "Is the robotic gripper currently holding an object?"):
        return "place_in_box"

    # ------------------------------------------------------------------ #
    # Priority 3 & 4: need to know what's visible                         #
    # ------------------------------------------------------------------ #
    object_visible = _ask(frame, vlm, "Is there an object visible in the pick zone?")
    shelf_has_slot = _ask(frame, vlm, "Is there an empty slot visible on the shelf?")

    if object_visible and shelf_has_slot:
        # Object to pick + destination slot → full cycle
        return "full_pick_and_place"

    if object_visible:
        # Object visible but no shelf slot (box destination assumed)
        return "pick_object"

    # ------------------------------------------------------------------ #
    # Fallback: scene ambiguous or empty — default to full cycle           #
    # ------------------------------------------------------------------ #
    logger.warning(
        "SkillRouter: scene ambiguous "
        "(object_visible=%s, shelf_has_slot=%s) — defaulting to %s",
        object_visible, shelf_has_slot, _FALLBACK_SKILL,
    )
    return _FALLBACK_SKILL


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np
    import logging

    logging.basicConfig(level=logging.DEBUG)

    # Minimal mock VLM for local testing without hardware
    _mock_answers: dict[str, bool] = {
        "box positioned near the shelf": False,
        "gripper currently holding":     False,
        "object visible in the pick zone": True,
        "empty slot visible on the shelf": True,
    }

    def mock_vlm(frame, query: str) -> tuple[bool, float]:
        for key, answer in _mock_answers.items():
            if key in query.lower():
                return answer, 42.0
        return False, 42.0

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    skill = route_skill(frame, mock_vlm)
    print(f"Routed to: {skill}")
    assert skill in ROUTABLE_SKILLS, f"Unknown skill: {skill}"
    print("OK")
