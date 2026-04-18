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

from dataclasses import dataclass, field
from typing import Iterator


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

def record_skill(skill_name: str) -> None:
    """
    Record a new skill by physical demonstration.

    Blocks while the operator moves the leader arm. The full joint-position
    trajectory of the follower arm is saved to disk under `skill_name`.

    Args:
        skill_name: Snake-case identifier, e.g. "pick_from_box".

    Raises:
        NotImplementedError: Until Diya's implementation lands.
    """
    raise NotImplementedError(
        "robot_api.record_skill not yet implemented — waiting for Diya. "
        "Use a mock in tests."
    )


def replay_skill(skill_name: str, params: dict) -> Iterator[StepEvent]:
    """
    Replay a previously recorded skill trajectory.

    Applies `params` adjustments to the stored trajectory (z_offset_mm,
    speed_scale, approach_angle_deg, gripper_close_force, retry_count)
    then executes step-by-step on the follower arm, yielding a StepEvent
    after EACH step completes.

    The orchestrator calls vlm_api.verify() after each yield, so this
    generator must pause between steps until the caller resumes it.

    Args:
        skill_name: Snake-case identifier matching a recorded trajectory.
        params:     Dict with any subset of patchable parameters:
                      z_offset_mm        (float) vertical approach offset in mm
                      speed_scale        (float) replay speed multiplier, default 1.0
                      approach_angle_deg (float) wrist approach rotation in degrees
                      gripper_close_force(float) grip strength, 0.0–1.0
                      retry_count        (int)   max automatic retries per step

    Yields:
        StepEvent for each completed motion step.

    Raises:
        NotImplementedError: Until Diya's implementation lands.
    """
    raise NotImplementedError(
        "robot_api.replay_skill not yet implemented — waiting for Diya. "
        "Use a mock in tests."
    )
    # Unreachable; satisfies type checkers that expect Iterator[StepEvent]
    yield StepEvent(step_id=0, action="", timestamp=0.0, gripper_state="unknown")


def apply_patch(skill_name: str, patch: dict) -> None:
    """
    Persist parameter deltas to the stored trajectory before next replay.

    Modifies the on-disk trajectory metadata so subsequent calls to
    replay_skill automatically incorporate the patch without re-recording.

    Args:
        skill_name: Snake-case identifier matching a recorded trajectory.
        patch:      Dict of parameter deltas, e.g. {"z_offset_mm": 5}.

    Raises:
        NotImplementedError: Until Diya's implementation lands.
    """
    raise NotImplementedError(
        "robot_api.apply_patch not yet implemented — waiting for Diya. "
        "Use a mock in tests."
    )
