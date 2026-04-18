"""
robot_api.py — Robot hardware interface
Owner: Diya

# [STUB] — all three public functions raise NotImplementedError.
# Replace each body with the real LeRobot/SO-100 implementation.
# The rest of the codebase (orchestrator, tests, mocks) codes against
# this interface — do NOT change function signatures or StepEvent fields.

Integration notes for Diya:
  - replay_skill must yield a StepEvent AFTER each motion step completes,
    not before. The orchestrator calls vlm_api.verify() immediately after
    each yield, so timing matters.
  - The four skill names robot_api must handle are exactly:
      "pick_object", "place_in_box", "full_pick_and_place", "box_in_shelf"
    These match SKILL_REGISTRY keys in compiler.py.
  - apply_patch modifies stored trajectory parameters on disk so the next
    replay_skill call picks them up automatically.

To swap in the real implementation at runtime, pass the module to the orchestrator:
    import robot_api
    orc = Orchestrator(robot=robot_api)   # uses real hardware
    orc = Orchestrator(robot=MockRobotAPI())  # uses mock (testing)
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
        skill_name: One of "pick_object", "place_in_box",
                    "full_pick_and_place", "box_in_shelf".

    # [STUB] Replace body with LeRobot teleoperation recording.
    # Example: use lerobot.record() with the SO-100 arm config.
    """
    raise NotImplementedError(
        "robot_api.record_skill — [STUB] awaiting Diya's LeRobot implementation."
    )


def replay_skill(skill_name: str, params: dict) -> Iterator[StepEvent]:
    """
    Replay a previously recorded skill trajectory.

    Applies `params` adjustments to the stored trajectory, then executes
    step-by-step on the follower arm, yielding a StepEvent after EACH
    step completes. The orchestrator calls vlm_api.verify() after each
    yield, so this generator must pause between steps until resumed.

    Args:
        skill_name: One of "pick_object", "place_in_box",
                    "full_pick_and_place", "box_in_shelf".
        params:     Patchable parameters (all optional, defaults apply):
                      z_offset_mm         float  vertical approach offset in mm
                      speed_scale         float  replay speed multiplier
                      approach_angle_deg  float  wrist rotation in degrees
                      gripper_close_force float  grip strength 0.0–1.0
                      retry_count         int    max retries per step

    Yields:
        StepEvent after each completed motion step.

    # [STUB] Replace body with LeRobot trajectory replay.
    # Example: use lerobot.replay() with patched params applied to the
    # stored trajectory config before playback.
    """
    raise NotImplementedError(
        "robot_api.replay_skill — [STUB] awaiting Diya's LeRobot implementation."
    )
    yield StepEvent(step_id=0, action="", timestamp=0.0, gripper_state="unknown")  # noqa: unreachable


def apply_patch(skill_name: str, patch: dict) -> None:
    """
    Persist parameter deltas to the stored trajectory config.

    Modifies on-disk trajectory metadata so subsequent replay_skill calls
    automatically incorporate the patch without re-recording.

    Args:
        skill_name: One of the four canonical skill names.
        patch:      Parameter deltas, e.g. {"gripper_close_force": 0.8}.

    # [STUB] Replace body with a write to the LeRobot trajectory config file.
    # Example: load the YAML/JSON config for skill_name, merge patch, save.
    """
    raise NotImplementedError(
        "robot_api.apply_patch — [STUB] awaiting Diya's LeRobot implementation."
    )
