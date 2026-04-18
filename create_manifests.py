"""
create_manifests.py — Bootstrap skill manifests for all four canonical skills.

Run this ONCE before using real hardware to create the JSON manifest files
that RobotAPI.replay_skill() reads. No arm connection needed — this only
writes JSON to disk.

Usage:
    python create_manifests.py                        # writes to ./skillpatch_data/
    python create_manifests.py --storage-dir /path    # custom location
    python create_manifests.py --dry-run              # print manifests, don't write

After running this, the robot needs to record each step dataset before replay
will work. Dataset IDs are printed at the end — give them to Diya.

To record a step dataset (run separately for each):
    RobotAPI.record_skill(skill_name, description=..., steps=[...])
Or use the lerobot-record CLI directly with the dataset_repo_id shown below.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))

from tinyvla_debugger.robot_api import RobotAPI

# ---------------------------------------------------------------------------
# Local dataset paths — pointing to the recorded data in ./data/
# Use the highest version of each (most recent recordings).
# For lerobot-replay, these are passed as dataset.repo_id with the parent
# dir as dataset.root so LeRobot loads from disk instead of HuggingFace.
# ---------------------------------------------------------------------------
DATA_DIR = "./Archive"

SKILL_MANIFESTS = {

    "pick_object": {
        "description": "Pick an object from the scene into the gripper.",
        "default_params": {
            "z_offset_mm": 0.0,
            "speed_scale": 1.0,
            "approach_angle_deg": 0.0,
            "gripper_close_force": 0.6,
            "retry_count": 2,
        },
        "steps": [
            {
                "name": "pick_object",
                "dataset_repo_id": f"{DATA_DIR}/pick_object_v5",
                "verification_query": "Is an object held securely in the gripper?",
                "expected_result": True,
                "description": "Pick the object — best recording is v5.",
            },
        ],
    },

    "place_in_box": {
        "description": "Place the object currently held in the gripper into the target box.",
        "default_params": {
            "z_offset_mm": 0.0,
            "speed_scale": 1.0,
            "approach_angle_deg": 0.0,
            "gripper_close_force": 0.6,
            "retry_count": 2,
        },
        "steps": [
            {
                "name": "place_in_box",
                "dataset_repo_id": f"{DATA_DIR}/place_in_box_v4",
                "verification_query": "Is the object now placed inside the box?",
                "expected_result": True,
                "description": "Place into box — best recording is v4.",
            },
        ],
    },

    "full_pick_and_place": {
        "description": "Pick an object from the scene and place it into the target box.",
        "default_params": {
            "z_offset_mm": 0.0,
            "speed_scale": 1.0,
            "approach_angle_deg": 0.0,
            "gripper_close_force": 0.6,
            "retry_count": 2,
        },
        "steps": [
            {
                "name": "pick_object",
                "dataset_repo_id": f"{DATA_DIR}/pick_object_v5",
                "verification_query": "Is an object held securely in the gripper?",
                "expected_result": True,
                "description": "Pick the object.",
            },
            {
                "name": "place_in_box",
                "dataset_repo_id": f"{DATA_DIR}/place_in_box_v4",
                "verification_query": "Is the object now placed inside the box?",
                "expected_result": True,
                "description": "Place into box.",
            },
        ],
    },

    "box_in_shelf": {
        "description": "Pick up a box and place it upright in an empty shelf slot.",
        "default_params": {
            "z_offset_mm": 0.0,
            "speed_scale": 0.8,
            "approach_angle_deg": 0.0,
            "gripper_close_force": 0.7,
            "retry_count": 2,
        },
        "steps": [
            {
                "name": "pick_object",
                "dataset_repo_id": f"{DATA_DIR}/pick_object_v5",
                "verification_query": "Is a box held securely in the gripper?",
                "expected_result": True,
                "description": "Pick the box.",
            },
            {
                "name": "move_box_to_shelf",
                "dataset_repo_id": f"{DATA_DIR}/place_in_shelf",
                "verification_query": "Is a box standing upright in the shelf slot?",
                "expected_result": True,
                "description": "Place the box upright into the shelf slot.",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Create skill manifests for all four skills")
    parser.add_argument(
        "--storage-dir", default="./skillpatch_data",
        help="Directory to write manifests into (default: ./skillpatch_data)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print manifests to stdout without writing files",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no files written ===\n")
        for skill_name, spec in SKILL_MANIFESTS.items():
            print(f"── {skill_name} ──────────────────────────")
            print(json.dumps({
                "skill_name": skill_name,
                "description": spec["description"],
                "steps": [{"name": s["name"], "dataset_repo_id": s["dataset_repo_id"],
                            "verification_query": s["verification_query"]} for s in spec["steps"]],
                "default_params": spec["default_params"],
            }, indent=2))
            print()
        return

    # Use a dummy RobotAPI (ports don't matter — we only call create_skill_manifest)
    api = RobotAPI(
        robot_port="none",
        teleop_port="none",
        robot_id="none",
        teleop_id="none",
        storage_dir=args.storage_dir,
    )

    print(f"\nWriting skill manifests to: {Path(args.storage_dir).resolve()}/skills/\n")

    for skill_name, spec in SKILL_MANIFESTS.items():
        api.create_skill_manifest(
            skill_name,
            description=spec["description"],
            steps=spec["steps"],
            default_params=spec["default_params"],
        )
        path = Path(args.storage_dir) / "skills" / f"{skill_name}.json"
        print(f"  ✓ {skill_name:25s} → {path}")

    print(f"\nAll 4 manifests created.")
    print("\n── Datasets used ────────────────────────────────────────────────")
    seen = set()
    for spec in SKILL_MANIFESTS.values():
        for step in spec["steps"]:
            rid = step["dataset_repo_id"]
            if rid not in seen:
                seen.add(rid)
                print(f"  {rid}")
    print("\nRun 'python run.py --robot-port <port> --teleop-port <port>' to start.")


if __name__ == "__main__":
    main()
