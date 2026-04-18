# The purpose of this module is to define the interface for controlling the LeRobot SO-100
# follower arm. Skills are recorded by physically demonstrating them with the leader arm,
# then replayed with optional parameter adjustments — no re-recording needed to fix a failure.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class StepEvent:
    # snapshot yielded to the orchestrator after each motion step completes
    step_id: int
    action: str
    timestamp: float
    gripper_state: float  # 0.0 = fully open, 1.0 = fully closed
    params_used: dict


def record_skill(skill_name: str) -> None:
    raise NotImplementedError


def replay_skill(skill_name: str, params: dict) -> Iterator[StepEvent]:
    # drives the follower arm through the stored trajectory, yielding one StepEvent per step
    # the orchestrator verifies each step with the VLM before allowing the next one to begin
    # supported params: z_offset_mm, speed_scale, approach_angle_deg, gripper_close_force, retry_count
    raise NotImplementedError


def apply_patch(skill_name: str, patch: dict) -> None:
    raise NotImplementedError
