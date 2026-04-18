"""
classifier.py — Rule-based failure classifier
Owner: Vera

When a VLM verification step returns an unexpected result, the classifier
determines the root cause so the patch library can select the right fix.

Classification is purely rule-based (no ML) — it pattern-matches on which
VLM query failed and which action was running. This makes it deterministic,
testable, and fast.

Failure type → default patch mapping (from spec):
  GRASP_FAIL         → z_offset_mm: +5
  PLACEMENT_MISS     → approach_angle_deg: +10
  PLACEMENT_COLLISION→ speed_scale: 0.7   (replaces, doesn't delta)
  DROP_DURING_TRANSIT→ gripper_close_force: +0.1
  UNKNOWN_FAIL       → no default patch (surfaces to dashboard)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure type constants
# ---------------------------------------------------------------------------

GRASP_FAIL          = "GRASP_FAIL"
PLACEMENT_MISS      = "PLACEMENT_MISS"
PLACEMENT_COLLISION = "PLACEMENT_COLLISION"
DROP_DURING_TRANSIT = "DROP_DURING_TRANSIT"
OBJECT_NOT_FOUND    = "OBJECT_NOT_FOUND"   # object absent / outside reach before pick
UNKNOWN_FAIL        = "UNKNOWN_FAIL"

ALL_FAILURE_TYPES = {
    GRASP_FAIL,
    PLACEMENT_MISS,
    PLACEMENT_COLLISION,
    DROP_DURING_TRANSIT,
    OBJECT_NOT_FOUND,
    UNKNOWN_FAIL,
}

# ---------------------------------------------------------------------------
# Action groups
# ---------------------------------------------------------------------------

PICK_ACTIONS   = {"pick_from_box", "pick", "pick_object"}
PLACE_ACTIONS  = {"place_slot_1", "place_slot_2", "place_slot_3", "place", "place_in_box"}
SCAN_ACTIONS   = {"scan_shelf", "scan", "scan_scene"}
CHECK_ACTIONS  = {"check_box_empty", "check_empty"}
MOVE_ACTIONS   = {"move_box_to_shelf"}

# ---------------------------------------------------------------------------
# Query fingerprints (substrings to match, all lowercase)
# ---------------------------------------------------------------------------

_GRIPPER_PATTERNS = [
    "gripper",
    "held",
    "holding",
    "object held",
    "securely in the gripper",
    "grasped",
]

_PLACEMENT_PATTERNS = [
    "upright in shelf slot",
    "standing upright",
    "item in slot",
    "item standing",
    "placed in",
    "upright in",
]

_EMPTY_PATTERNS = [
    "slot currently empty",
    "is the slot empty",
    "shelf slot empty",
    "slot empty",
    "is slot",
]

_OBJECT_VISIBLE_PATTERNS = [
    "object visible",
    "object in the pick zone",
    "object present",
    "item visible",
    "something to pick",
    "visible in the pick zone",
    "visible in the scene",
    "an object visible",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = text.lower()
    return any(p in t for p in patterns)


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    failure_type: str
    confidence: str           # "high" | "medium" | "low"
    reasoning: str            # human-readable explanation for trace/dashboard


# ---------------------------------------------------------------------------
# FailureClassifier
# ---------------------------------------------------------------------------

class FailureClassifier:
    """
    Classifies robot execution failures from VLM query results.

    Decision tree (per spec table):
      0. Query mentions object/item visibility AND result is False on a
         scan or pick action → OBJECT_NOT_FOUND (triggers visual re-localisation)
      1. Query mentions gripper/holding → GRASP_FAIL
      2. Query mentions item upright/placement:
           first attempt  → PLACEMENT_MISS
           retry attempt  → PLACEMENT_COLLISION
      3. Query mentions slot-empty AND result is True after a place action
         (slot still empty = item dropped in transit) → DROP_DURING_TRANSIT
      4. None of the above → UNKNOWN_FAIL

    Args:
        strict: If True, raise ValueError on UNKNOWN_FAIL instead of
                returning it. Useful in tests to catch unhandled cases.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict

    def classify(
        self,
        action: str,
        verification_query: str,
        verification_result: bool,
        *,
        retry: bool = False,
        expected_result: bool = True,
    ) -> ClassificationResult:
        """
        Classify a verification failure.

        A "failure" is when verification_result != expected_result.
        Call this only when the step has failed — don't call on success.

        Args:
            action:               The robot action that was executing,
                                  e.g. "pick_from_box" or "place_slot_1".
            verification_query:   The VLM query string for this step.
            verification_result:  What the VLM actually returned.
            retry:                True if this is the second (or later) attempt
                                  at this step after a patch was already applied.
            expected_result:      What the step expected (usually True).

        Returns:
            ClassificationResult with failure_type, confidence, and reasoning.

        Raises:
            ValueError: If strict=True and failure type is UNKNOWN_FAIL.
        """
        action_lower = action.lower().strip()
        query_lower  = verification_query.lower().strip()

        result = self._classify_inner(
            action_lower, query_lower, verification_result, retry, expected_result
        )

        logger.info(
            "Classified failure: action=%r query=%r result=%s retry=%s → %s (%s)",
            action, verification_query, verification_result, retry,
            result.failure_type, result.confidence,
        )

        if self.strict and result.failure_type == UNKNOWN_FAIL:
            raise ValueError(
                f"UNKNOWN_FAIL for action={action!r} query={verification_query!r}. "
                "Add a classification rule or update the query patterns."
            )

        return result

    # ------------------------------------------------------------------
    # Internal decision tree
    # ------------------------------------------------------------------

    def _classify_inner(
        self,
        action: str,
        query: str,
        result: bool,
        retry: bool,
        expected: bool,
    ) -> ClassificationResult:

        # Rule 0 — Object not found (must come before Rule 1)
        # Triggered when: visibility/scene query returns False on a scan or pick action.
        # The object was absent before the arm even attempted to grasp.
        # → orchestrator will run visual re-localisation, then retry.
        if (
            _matches_any(query, _OBJECT_VISIBLE_PATTERNS)
            and result is False
            and action in (SCAN_ACTIONS | PICK_ACTIONS)
        ):
            return ClassificationResult(
                failure_type=OBJECT_NOT_FOUND,
                confidence="high",
                reasoning=(
                    f"VLM query '{query}' returned False during "
                    f"{'scan' if action in SCAN_ACTIONS else 'pick'} action — "
                    "object not visible in pick zone. "
                    "Orchestrator will run visual re-localisation before retry."
                ),
            )

        # Rule 1 — Grasp failure
        # Triggered when: gripper-check query returns False (object not in gripper)
        if _matches_any(query, _GRIPPER_PATTERNS) and result is False:
            return ClassificationResult(
                failure_type=GRASP_FAIL,
                confidence="high",
                reasoning=(
                    f"VLM query '{query}' returned False — "
                    "gripper did not acquire object. "
                    "Default patch: z_offset_mm +5 to lower approach point."
                ),
            )

        # Rule 2 — Placement failure (two variants)
        # Triggered when: placement-check query returns False (item not in slot)
        if _matches_any(query, _PLACEMENT_PATTERNS) and result is False:
            if retry:
                return ClassificationResult(
                    failure_type=PLACEMENT_COLLISION,
                    confidence="high",
                    reasoning=(
                        f"VLM query '{query}' returned False on retry — "
                        "item hit the slot edge (collision). "
                        "Default patch: speed_scale 0.7 to slow approach."
                    ),
                )
            return ClassificationResult(
                failure_type=PLACEMENT_MISS,
                confidence="high",
                reasoning=(
                    f"VLM query '{query}' returned False (first attempt) — "
                    "item was not placed correctly. "
                    "Default patch: approach_angle_deg +10."
                ),
            )

        # Rule 3 — Drop during transit
        # Triggered when: slot-empty check returns True AFTER a place action
        # (slot is still empty → item was dropped before reaching it)
        if (
            _matches_any(query, _EMPTY_PATTERNS)
            and result is True        # slot IS empty (unexpected after place)
            and action in PLACE_ACTIONS
        ):
            return ClassificationResult(
                failure_type=DROP_DURING_TRANSIT,
                confidence="medium",
                reasoning=(
                    f"VLM query '{query}' returned True after place action — "
                    "slot still empty, item dropped during transit. "
                    "Default patch: gripper_close_force +0.1."
                ),
            )

        # Rule 4 — Drop during transit via scan_shelf post-place
        # Edge case: scan_shelf after place shows empty slot
        if (
            _matches_any(query, _EMPTY_PATTERNS)
            and result is True
            and action in SCAN_ACTIONS
        ):
            return ClassificationResult(
                failure_type=DROP_DURING_TRANSIT,
                confidence="low",
                reasoning=(
                    "Slot empty detected during scan after expected placement. "
                    "Possible drop during transit."
                ),
            )

        # Fallback
        return ClassificationResult(
            failure_type=UNKNOWN_FAIL,
            confidence="low",
            reasoning=(
                f"No rule matched: action={action!r}, "
                f"query={query!r}, result={result}, retry={retry}. "
                "Add a rule or update query patterns."
            ),
        )

    # ------------------------------------------------------------------
    # Convenience: classify failure type string only (for patch lookup)
    # ------------------------------------------------------------------

    def get_failure_type(
        self,
        action: str,
        verification_query: str,
        verification_result: bool,
        *,
        retry: bool = False,
        expected_result: bool = True,
    ) -> str:
        """Shorthand that returns just the failure type string."""
        return self.classify(
            action,
            verification_query,
            verification_result,
            retry=retry,
            expected_result=expected_result,
        ).failure_type


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clf = FailureClassifier()

    cases = [
        ("pick_from_box",  "Is an object held securely in the gripper?", False, False),
        ("place_slot_1",   "Is there an item standing upright in shelf slot 1?", False, False),
        ("place_slot_1",   "Is there an item standing upright in shelf slot 1?", False, True),
        ("place_slot_2",   "Is this shelf slot currently empty?", True, False),
        ("scan_shelf",     "Are there empty slots visible?", True, False),
    ]

    for action, query, result, retry in cases:
        r = clf.classify(action, query, result, retry=retry)
        print(f"  {r.failure_type:25s} [{r.confidence}] — {action!r}, retry={retry}")
