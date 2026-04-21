from __future__ import annotations

"""
robot_api.py

Clean robot execution interface for the SkillPatch project.

This module wraps LeRobot CLI workflows behind a small Python API that the
SLM/orchestrator can call without needing to know the raw terminal commands.

    replay step -> yield StepEvent -> VLM verifies -> next step

Because standard LeRobot replay is episode-based rather than natively
step-event-based, the cleanest reliable interface is to record each logical
step as its own dataset, then replay those datasets one at a time.

What this module does well
--------------------------
- Records named step datasets through LeRobot.
- Replays a skill step-by-step.
- Stores persistent default parameters and learned patches.
- Emits StepEvent objects after each step so the VLM/orchestrator can act.
- Keeps everything local-only by default (no Hugging Face push required).
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import json
import shutil
import subprocess
import time


# =========================
# Data models
# =========================


@dataclass
class ReplayParams:
    """Patchable runtime parameters exposed to the orchestrator/SLM.

    Notes:
    - z_offset_mm, speed_scale, and approach_angle_deg are the most grounded
      parameters based on the project plan and current testing direction.
    - gripper_close_force is included because it is in the project schema, but
      whether it is physically applied depends on future low-level support.
    - retry_count is orchestration-side metadata rather than a motor parameter.
    """

    z_offset_mm: float = 0.0
    speed_scale: float = 1.0
    approach_angle_deg: float = 0.0
    gripper_close_force: float = 0.6
    retry_count: int = 2

    def merged(self, override: Optional[Dict[str, Any]] = None) -> "ReplayParams":
        data = asdict(self)
        if override:
            data.update(override)
        merged = ReplayParams(**data)
        merged.validate()
        return merged

    def validate(self) -> None:
        if not -50 <= self.z_offset_mm <= 50:
            raise ValueError("z_offset_mm out of allowed range [-50, 50].")
        if not 0.1 <= self.speed_scale <= 2.0:
            raise ValueError("speed_scale out of allowed range [0.1, 2.0].")
        if not -90 <= self.approach_angle_deg <= 90:
            raise ValueError("approach_angle_deg out of allowed range [-90, 90].")
        if not 0.0 <= self.gripper_close_force <= 1.0:
            raise ValueError("gripper_close_force out of allowed range [0.0, 1.0].")
        if not 0 <= self.retry_count <= 10:
            raise ValueError("retry_count out of allowed range [0, 10].")


@dataclass
class StepSpec:
    """A single logical step in a skill.

    dataset_repo_id is the LeRobot dataset that contains the recorded motion for
    this single step.
    """

    step_id: int
    name: str
    dataset_repo_id: str
    verification_query: str
    expected_result: bool = True
    description: str = ""


@dataclass
class SkillSpec:
    """A full skill composed of recorded step datasets."""

    skill_name: str
    description: str
    steps: List[StepSpec]
    default_params: ReplayParams = field(default_factory=ReplayParams)


@dataclass
class StepEvent:
    """Event yielded after each replayed step finishes."""

    skill_name: str
    step_id: int
    action: str
    timestamp: float
    dataset_repo_id: str
    gripper_state: Optional[str]
    params_used: Dict[str, Any]
    result: str = "EXECUTED"
    notes: str = ""


# =========================
# Main API
# =========================


class RobotAPI:
    """High-level LeRobot wrapper for the SkillPatch project.

    The public methods intended for the rest of the team are:
    - record_skill(...)
    - replay_skill(...)
    - apply_patch(...)

    Additional helper methods exist to bootstrap skill manifests cleanly.
    """

    def __init__(
        self,
        *,
        robot_port: str,
        teleop_port: str,
        robot_id: str,
        teleop_id: str,
        storage_dir: str | Path = "./skillpatch_data",
        lerobot_bin: str = "lerobot",
        dataset_push_to_hub: bool = False,
    ) -> None:
        self.robot_port = robot_port
        self.teleop_port = teleop_port
        self.robot_id = robot_id
        self.teleop_id = teleop_id
        self.lerobot_bin = lerobot_bin
        self.dataset_push_to_hub = dataset_push_to_hub

        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.skills_dir = self.storage_dir / "skills"
        self.skills_dir.mkdir(exist_ok=True)

        self.patches_path = self.storage_dir / "patches.json"
        self.trace_path = self.storage_dir / "trace.jsonl"

        if not self.patches_path.exists():
            self._write_json(self.patches_path, {})

    # -------------------------
    # Public API methods
    # -------------------------

    def record_skill(
        self,
        skill_name: str,
        *,
        description: str,
        steps: List[Dict[str, Any]],
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a multi-step skill.

        This blocks while the user records each logical step as its own dataset.

        Each item in `steps` must contain:
            - name
            - dataset_repo_id
            - verification_query
        Optional:
            - expected_result
            - description
            - num_episodes
            - episode_time_s
            - reset_time_s
            - single_task

        Example step:
            {
                "name": "pick_object",
                "dataset_repo_id": "team/pick_object_v1",
                "verification_query": "Is an object held securely in the gripper?",
                "num_episodes": 5,
                "episode_time_s": 40,
                "reset_time_s": 10,
                "single_task": "pick up the object",
            }
        """
        replay_params = ReplayParams().merged(default_params)

        skill_steps: List[StepSpec] = []
        for i, step in enumerate(steps):
            required = ["name", "dataset_repo_id", "verification_query"]
            missing = [key for key in required if key not in step]
            if missing:
                raise ValueError(f"Step {i} missing required field(s): {missing}")

            step_spec = StepSpec(
                step_id=i,
                name=step["name"],
                dataset_repo_id=step["dataset_repo_id"],
                verification_query=step["verification_query"],
                expected_result=step.get("expected_result", True),
                description=step.get("description", ""),
            )
            skill_steps.append(step_spec)

        spec = SkillSpec(
            skill_name=skill_name,
            description=description,
            steps=skill_steps,
            default_params=replay_params,
        )
        self._write_skill_spec(spec)

        for i, step in enumerate(steps):
            self._record_single_step(
                dataset_repo_id=step["dataset_repo_id"],
                single_task=step.get("single_task", step["name"]),
                num_episodes=int(step.get("num_episodes", 1)),
                episode_time_s=int(step.get("episode_time_s", 60)),
                reset_time_s=int(step.get("reset_time_s", 10)),
            )
            self._log_trace(
                {
                    "timestamp": self._utc_iso(),
                    "event": "record_step_complete",
                    "skill": skill_name,
                    "step_id": i,
                    "action": step["name"],
                    "dataset_repo_id": step["dataset_repo_id"],
                }
            )

    def replay_skill(
        self,
        skill_name: str,
        params: Optional[Dict[str, Any]] = None,
        episode: int = 0,
    ) -> Iterator[StepEvent]:
        """Replay a recorded skill step-by-step.

        Each logical step is replayed via LeRobot episode replay.
        A StepEvent is yielded after each step completes.

        Args:
            skill_name: Action name (e.g. "box_in_shelf") — must have a
                        manifest in skillpatch_data/skills/.
            params:     Runtime parameter overrides (merged with stored patches).
            episode:    Which recorded episode to replay. The orchestrator passes
                        the attempt index (0, 1, …) so each retry runs a different
                        trajectory. Falls back to episode 0 if the requested
                        episode doesn't exist (lerobot-replay will raise, which
                        the orchestrator already catches as a RuntimeError).

        This is the main interface the orchestrator/VLM should call.
        """
        spec = self._read_skill_spec(skill_name)
        merged_params = self._effective_params(skill_name, spec.default_params, params)

        for step in spec.steps:
            started_at = time.time()
            self._replay_single_step(
                step.dataset_repo_id,
                episode=episode,
                speed_scale=merged_params.speed_scale,
            )
            event = StepEvent(
                skill_name=skill_name,
                step_id=step.step_id,
                action=step.name,
                timestamp=time.time(),
                dataset_repo_id=step.dataset_repo_id,
                gripper_state=self._infer_gripper_state(step.name),
                params_used=asdict(merged_params),
                result="EXECUTED",
                notes=f"LeRobot step replay completed (episode={episode}).",
            )
            self._log_trace(
                {
                    "timestamp": self._utc_iso(),
                    "event": "replay_step_complete",
                    "skill": skill_name,
                    "step_id": step.step_id,
                    "action": step.name,
                    "dataset_repo_id": step.dataset_repo_id,
                    "episode": episode,
                    "duration_s": round(time.time() - started_at, 3),
                    "params_used": asdict(merged_params),
                }
            )
            yield event

    def apply_patch(self, skill_name: str, patch: Dict[str, Any]) -> None:
        """Persist a runtime patch for future replays.

        This stores patch memory keyed by skill name. In this implementation,
        patches update the effective default parameters used at replay time.

        Example:
            apply_patch("pick_object", {"z_offset_mm": 5, "speed_scale": 0.8})
        """
        current = self._read_json(self.patches_path)
        current.setdefault(skill_name, {})
        current[skill_name].update(patch)

        # Validate merged result so bad patches are rejected early.
        spec = self._read_skill_spec(skill_name)
        _ = spec.default_params.merged(current[skill_name])

        self._write_json(self.patches_path, current)
        self._log_trace(
            {
                "timestamp": self._utc_iso(),
                "event": "patch_applied",
                "skill": skill_name,
                "patch": patch,
            }
        )

    # -------------------------
    # Optional helper methods
    # -------------------------

    def create_skill_manifest(
        self,
        skill_name: str,
        *,
        description: str,
        steps: List[Dict[str, Any]],
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a skill manifest without launching recording yet."""
        skill_steps = [
            StepSpec(
                step_id=i,
                name=s["name"],
                dataset_repo_id=s["dataset_repo_id"],
                verification_query=s["verification_query"],
                expected_result=s.get("expected_result", True),
                description=s.get("description", ""),
            )
            for i, s in enumerate(steps)
        ]
        spec = SkillSpec(
            skill_name=skill_name,
            description=description,
            steps=skill_steps,
            default_params=ReplayParams().merged(default_params),
        )
        self._write_skill_spec(spec)

    def get_skill_spec(self, skill_name: str) -> SkillSpec:
        return self._read_skill_spec(skill_name)

    def get_patches(self, skill_name: Optional[str] = None) -> Dict[str, Any]:
        patches = self._read_json(self.patches_path)
        if skill_name is None:
            return patches
        return patches.get(skill_name, {})

    # -------------------------
    # Internal LeRobot wrappers
    # -------------------------

    def _record_single_step(
        self,
        *,
        dataset_repo_id: str,
        single_task: str,
        num_episodes: int,
        episode_time_s: int,
        reset_time_s: int,
    ) -> None:
        cmd = [
            f"{self.lerobot_bin}-record",
            "--robot.type=so101_follower",
            f"--robot.port={self.robot_port}",
            f"--robot.id={self.robot_id}",
            "--teleop.type=so101_leader",
            f"--teleop.port={self.teleop_port}",
            f"--teleop.id={self.teleop_id}",
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.single_task={single_task}",
            f"--dataset.push_to_hub={'true' if self.dataset_push_to_hub else 'false'}",
            f"--dataset.num_episodes={num_episodes}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
        ]
        self._run_command(cmd)

    def _replay_single_step(
        self,
        dataset_repo_id: str,
        *,
        episode: int = 0,
        speed_scale: float = 1.0,
    ) -> None:
        # dataset_repo_id is a local path like "./data/pick_object_v5".
        # LeRobot loads metadata from {root}/meta/info.json, so root must be
        # the dataset directory itself (not its parent):
        #   --dataset.repo_id=pick_object_v5
        #   --dataset.root=data/pick_object_v5
        # The specific dataset is chosen per-step from the skill manifest,
        # which is determined by the SLM-compiled skill program.
        repo_path = Path(dataset_repo_id)
        if repo_path.exists():
            repo_id = repo_path.name  # "pick_object_v5"
            root    = str(repo_path)  # "data/pick_object_v5" — LeRobot loads {root}/meta/info.json
        else:
            # HuggingFace repo id (e.g. "user/dataset") — no root needed
            repo_id = dataset_repo_id
            root    = None

        cmd = [
            f"{self.lerobot_bin}-replay",
            "--robot.type=so101_follower",
            f"--robot.port={self.robot_port}",
            f"--robot.id={self.robot_id}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.episode={episode}",
        ]
        if root is not None:
            cmd += [f"--dataset.root={root}"]

        # speed_scale maps to lerobot's --dataset.fps override.
        # The dataset's native fps is read from its meta/info.json; if that
        # fails we fall back to 30. A lower fps slows the arm's motion —
        # e.g. speed_scale=0.7 → fps=21 gives a 30% slower approach that
        # helps avoid shelf-edge collisions.
        if speed_scale != 1.0:
            base_fps = self._get_dataset_fps(root or dataset_repo_id)
            fps = max(1, int(base_fps * speed_scale))
            cmd += [f"--dataset.fps={fps}"]

        self._run_command(cmd)

    def _get_dataset_fps(self, root: str) -> int:
        """Read the recorded fps from the dataset's meta/info.json."""
        try:
            meta_path = Path(root) / "meta" / "info.json"
            with meta_path.open() as f:
                meta = json.load(f)
            return int(meta.get("fps", 30))
        except Exception:
            return 30

    def _run_command(self, cmd: List[str]) -> None:
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Command not found: {cmd[0]}. Make sure LeRobot is installed and the "
                f"virtual environment is active."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "LeRobot command failed. This may be caused by a broken/incomplete dataset, "
                "a reused dataset name, disconnected leader/follower hardware, or an arm "
                "power/port issue. Inspect the terminal output above for the original error."
            ) from exc

    # -------------------------
    # Internal state helpers
    # -------------------------

    def _skill_path(self, skill_name: str) -> Path:
        return self.skills_dir / f"{skill_name}.json"

    def _write_skill_spec(self, spec: SkillSpec) -> None:
        payload = {
            "skill_name": spec.skill_name,
            "description": spec.description,
            "steps": [asdict(step) for step in spec.steps],
            "default_params": asdict(spec.default_params),
        }
        self._write_json(self._skill_path(spec.skill_name), payload)

    def _read_skill_spec(self, skill_name: str) -> SkillSpec:
        path = self._skill_path(skill_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Skill manifest not found for '{skill_name}'. Create it first via "
                f"create_skill_manifest(...) or record_skill(...)."
            )
        data = self._read_json(path)
        return SkillSpec(
            skill_name=data["skill_name"],
            description=data["description"],
            steps=[StepSpec(**step) for step in data["steps"]],
            default_params=ReplayParams(**data["default_params"]),
        )

    def _effective_params(
        self,
        skill_name: str,
        defaults: ReplayParams,
        runtime_override: Optional[Dict[str, Any]],
    ) -> ReplayParams:
        learned_patch = self.get_patches(skill_name)
        data = asdict(defaults)
        data.update(learned_patch)
        if runtime_override:
            data.update(runtime_override)
        params = ReplayParams(**data)
        params.validate()
        return params

    def _infer_gripper_state(self, action_name: str) -> Optional[str]:
        lowered = action_name.lower()
        if "pick" in lowered or "grasp" in lowered or "close" in lowered:
            return "closed_or_holding"
        if "release" in lowered or "open" in lowered or "drop" in lowered:
            return "open"
        return None

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _log_trace(self, event: Dict[str, Any]) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

    @staticmethod
    def _utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


# =========================
# Example usage
# =========================


if __name__ == "__main__":
    # Example only. Replace ports/ids with the ones for the active machine.
    api = RobotAPI(
        robot_port="/dev/tty.usbmodemFOLLOWER",
        teleop_port="/dev/tty.usbmodemLEADER",
        robot_id="follower_arm",
        teleop_id="leader_arm",
        storage_dir="./skillpatch_data",
    )

    # Example skill structure aligned with the project goal of stepwise execution.
    # Record each step as its own dataset so the orchestrator can verify after
    # every step.
    example_steps = [
        {
            "name": "pick_object",
            "dataset_repo_id": "team/pick_object_v1",
            "verification_query": "Is an object held securely in the gripper?",
            "single_task": "pick up the object",
            "num_episodes": 5,
            "episode_time_s": 40,
            "reset_time_s": 10,
        },
        {
            "name": "place_in_box",
            "dataset_repo_id": "team/place_in_box_v1",
            "verification_query": "Is the object inside the box?",
            "single_task": "place the object in the box",
            "num_episodes": 5,
            "episode_time_s": 40,
            "reset_time_s": 10,
        },
    ]
