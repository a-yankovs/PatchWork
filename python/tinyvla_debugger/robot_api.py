"""
robot_api.py — Robot hardware interface stub
Owner: Diya

Exposes exactly three functions that every other module codes against.
Do NOT add extra functions here — the orchestrator, tests, and mocks
all depend on this exact interface.

Integration note for Diya:
  - replay_skill must yield a StepEvent AFTER each motion step completes,
    not before. The orchestrator calls vlm_api.verify() immediately after
    each yield, so timing matters.
  - apply_patch modifies the stored trajectory parameters on disk so that
    the next replay_skill call picks them up automatically.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class StepEvent:
    """Emitted by replay_skill after each motion step completes."""
    step_id: int
    action: str
    timestamp: float
    gripper_state: str          # e.g. "open" | "closed" | "unknown"
    params_used: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Steps emitted per replay when running without real hardware
_MOCK_STEPS = ["scan_shelf", "pick_from_box", "place_slot_1", "place_slot_2", "check_box_empty"]


def record_skill(skill_name: str) -> None:
    """Blocks while operator demonstrates skill on the leader arm; saves trajectory to disk."""
    # TODO DIYA: replace with LeRobot record API
    logger.warning("robot_api.record_skill: hardware not connected — no-op for '%s'", skill_name)


def replay_skill(skill_name: str, params: dict) -> Iterator[StepEvent]:
    """Replays recorded trajectory step-by-step, yielding a StepEvent after each motion."""
    # TODO DIYA: replace with LeRobot replay API; params adjust the stored trajectory
    logger.warning("robot_api.replay_skill: hardware not connected — emitting mock events for '%s'", skill_name)
    for i, action in enumerate(_MOCK_STEPS):
        time.sleep(0.05)  # small delay so the event loop doesn't spin-lock
        yield StepEvent(step_id=i, action=action, timestamp=time.time(),
                        gripper_state="closed", params_used=dict(params))


def apply_patch(skill_name: str, patch: dict) -> None:
    """Persists parameter deltas to the stored trajectory so next replay picks them up."""
    # TODO DIYA: write patch into the on-disk trajectory metadata
    logger.warning("robot_api.apply_patch: hardware not connected — patch not persisted: %s", patch)
