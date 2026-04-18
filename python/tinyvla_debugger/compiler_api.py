# The purpose of this module is to define the natural language compiler interface.
# compile_command() takes a plain English instruction and returns a structured skill program
# that the execution engine can run step by step. Powered by Phi-3-mini on the AMD GPU.

from __future__ import annotations


# defining a structure in which compile_command() must always return
# each step includes an action name (what the robot does) and a verification query
# (the question the VLM will answer to confirm the step succeeded)
SKILL_PROGRAM_SCHEMA = {
    "skill_name": str,
    "description": str,
    "steps": [
        {
            "step_id": int,
            "action": str,             # must be in get_available_actions()
            "verification_query": str, # passed directly to vlm_api.verify() after this step runs
            "expected_result": bool,
        }
    ],
    "parameters": {
        "z_offset_mm": float,         # vertical approach offset — patched when grasp fails
        "speed_scale": float,         # replay speed multiplier — patched when collision occurs
        "approach_angle_deg": float,  # wrist rotation — patched when placement misses
        "gripper_close_force": float, # grip strength — patched when item drops in transit
    },
}


# converts a plain English command into a structured skill program using Phi-3-mini on the AMD GPU
# e.g. "put the canned goods on the middle shelf" → JSON with ordered steps and VLM queries
def compile_command(natural_language_command: str) -> dict:
    raise NotImplementedError
    # NotImplementedError is a built-in exception that indicates that the function is
    # currently a placeholder and hasn't been implemented yet.


# checks that a skill program dict has all required fields before the orchestrator tries to run it
def validate_skill_program(program: dict) -> bool:
    raise NotImplementedError


# returns the list of action names the robot arm can actually perform
def get_available_actions() -> list[str]:
    raise NotImplementedError
