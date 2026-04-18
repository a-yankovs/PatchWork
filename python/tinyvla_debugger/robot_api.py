from __future__ import annotations

"""
robot_api.py

Clean robot execution interface for the SkillPatch project.

Wraps LeRobot CLI workflows behind a small Python API the orchestrator
can call without needing raw terminal commands.

Design: each skill is an ordered list of recorded step-datasets.
    replay step -> yield StepEvent -> VLM verifies -> next step

apply_patch() updates persistent runtime patch memory; low-level motion
transforms are tracked but not physically injected until Diya's LeRobot
implementation is complete.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import json
import subprocess
import time


# ── Data models ──────────────────────────────────────────────────────────────���

@dataclass
class ReplayParams:
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
            raise ValueError("z_offset_mm out of range [-50, 50].")
        if not 0.1 <= self.speed_scale <= 2.0:
            raise ValueError("speed_scale out of range [0.1, 2.0].")
        if not -90 <= self.approach_angle_deg <= 90:
            raise ValueError("approach_angle_deg out of range [-90, 90].")
        if not 0.0 <= self.gripper_close_force <= 1.0:
            raise ValueError("gripper_close_force out of range [0.0, 1.0].")
        if not 0 <= self.retry_count <= 10:
            raise ValueError("retry_count out of range [0, 10].")


@dataclass
class StepSpec:
    step_id: int
    name: str
    dataset_repo_id: str
    verification_query: str
    expected_result: bool = True
    description: str = ""


@dataclass
class SkillSpec:
    skill_name: str
    description: str
    steps: List[StepSpec]
    default_params: ReplayParams = field(default_factory=ReplayParams)


@dataclass
class StepEvent:
    """Yielded after each replayed step — consumed by the orchestrator."""
    skill_name: str
    step_id: int
    action: str
    timestamp: float
    dataset_repo_id: str
    gripper_state: Optional[str]
    params_used: Dict[str, Any]
    result: str = "EXECUTED"
    notes: str = ""


# ── Main API ──────────────────────────────────────────────────────────────────

class RobotAPI:
    """High-level LeRobot wrapper for SkillPatch.

    Public interface for the orchestrator:
        record_skill(...)
        replay_skill(...)  -> Iterator[StepEvent]
        apply_patch(...)
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
        self.trace_path   = self.storage_dir / "trace.jsonl"

        if not self.patches_path.exists():
            self._write_json(self.patches_path, {})

    # ── Public methods ────────────────────────────────────────────────────────

    def record_skill(
        self,
        skill_name: str,
        *,
        description: str,
        steps: List[Dict[str, Any]],
        default_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        replay_params = ReplayParams().merged(default_params)
        skill_steps: List[StepSpec] = []
        for i, step in enumerate(steps):
            missing = [k for k in ("name", "dataset_repo_id", "verification_query") if k not in step]
            if missing:
                raise ValueError(f"Step {i} missing: {missing}")
            skill_steps.append(StepSpec(
                step_id=i,
                name=step["name"],
                dataset_repo_id=step["dataset_repo_id"],
                verification_query=step["verification_query"],
                expected_result=step.get("expected_result", True),
                description=step.get("description", ""),
            ))

        spec = SkillSpec(skill_name=skill_name, description=description,
                         steps=skill_steps, default_params=replay_params)
        self._write_skill_spec(spec)

        for i, step in enumerate(steps):
            self._record_single_step(
                dataset_repo_id=step["dataset_repo_id"],
                single_task=step.get("single_task", step["name"]),
                num_episodes=int(step.get("num_episodes", 1)),
                episode_time_s=int(step.get("episode_time_s", 60)),
                reset_time_s=int(step.get("reset_time_s", 10)),
            )
            self._log_trace({"event": "record_step_complete", "skill": skill_name,
                             "step_id": i, "action": step["name"],
                             "dataset_repo_id": step["dataset_repo_id"]})

    def replay_skill(self, skill_name: str, params: Optional[Dict[str, Any]] = None) -> Iterator[StepEvent]:
        spec = self._read_skill_spec(skill_name)
        merged = self._effective_params(skill_name, spec.default_params, params)
        for step in spec.steps:
            self._replay_single_step(step.dataset_repo_id, episode=0)
            event = StepEvent(
                skill_name=skill_name,
                step_id=step.step_id,
                action=step.name,
                timestamp=time.time(),
                dataset_repo_id=step.dataset_repo_id,
                gripper_state=self._infer_gripper_state(step.name),
                params_used=asdict(merged),
            )
            self._log_trace({"event": "replay_step_complete", "skill": skill_name,
                             "step_id": step.step_id, "action": step.name,
                             "params_used": asdict(merged)})
            yield event

    def apply_patch(self, skill_name: str, patch: Dict[str, Any]) -> None:
        current = self._read_json(self.patches_path)
        current.setdefault(skill_name, {})
        current[skill_name].update(patch)
        spec = self._read_skill_spec(skill_name)
        spec.default_params.merged(current[skill_name])  # validates
        self._write_json(self.patches_path, current)
        self._log_trace({"event": "patch_applied", "skill": skill_name, "patch": patch})

    def create_skill_manifest(self, skill_name: str, *, description: str,
                               steps: List[Dict[str, Any]],
                               default_params: Optional[Dict[str, Any]] = None) -> None:
        skill_steps = [
            StepSpec(step_id=i, name=s["name"], dataset_repo_id=s["dataset_repo_id"],
                     verification_query=s["verification_query"],
                     expected_result=s.get("expected_result", True),
                     description=s.get("description", ""))
            for i, s in enumerate(steps)
        ]
        self._write_skill_spec(SkillSpec(
            skill_name=skill_name, description=description,
            steps=skill_steps, default_params=ReplayParams().merged(default_params),
        ))

    def get_skill_spec(self, skill_name: str) -> SkillSpec:
        return self._read_skill_spec(skill_name)

    def get_patches(self, skill_name: Optional[str] = None) -> Dict[str, Any]:
        patches = self._read_json(self.patches_path)
        return patches if skill_name is None else patches.get(skill_name, {})

    # ── LeRobot wrappers ──────────────────────────────────────────────────────

    def _record_single_step(self, *, dataset_repo_id: str, single_task: str,
                             num_episodes: int, episode_time_s: int, reset_time_s: int) -> None:
        self._run_command([
            f"{self.lerobot_bin}-record",
            "--robot.type=so101_follower", f"--robot.port={self.robot_port}",
            f"--robot.id={self.robot_id}", "--teleop.type=so101_leader",
            f"--teleop.port={self.teleop_port}", f"--teleop.id={self.teleop_id}",
            f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.single_task={single_task}",
            f"--dataset.push_to_hub={'true' if self.dataset_push_to_hub else 'false'}",
            f"--dataset.num_episodes={num_episodes}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
        ])

    def _replay_single_step(self, dataset_repo_id: str, *, episode: int = 0) -> None:
        cmd = [
            f"{self.lerobot_bin}-replay",
            "--robot.type=so101_follower", f"--robot.port={self.robot_port}",
            f"--robot.id={self.robot_id}", f"--dataset.repo_id={dataset_repo_id}",
            f"--dataset.episode={episode}",
        ]
        if Path(dataset_repo_id).exists():
            cmd += [f"--dataset.root={Path(dataset_repo_id).parent}"]
        self._run_command(cmd)

    def _run_command(self, cmd: List[str]) -> None:
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError(f"{cmd[0]} not found — is LeRobot installed and venv active?") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError("LeRobot command failed — check terminal output above.") from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _skill_path(self, skill_name: str) -> Path:
        return self.skills_dir / f"{skill_name}.json"

    def _write_skill_spec(self, spec: SkillSpec) -> None:
        self._write_json(self._skill_path(spec.skill_name), {
            "skill_name": spec.skill_name, "description": spec.description,
            "steps": [asdict(s) for s in spec.steps],
            "default_params": asdict(spec.default_params),
        })

    def _read_skill_spec(self, skill_name: str) -> SkillSpec:
        path = self._skill_path(skill_name)
        if not path.exists():
            raise FileNotFoundError(f"No manifest for '{skill_name}'. Call create_skill_manifest() first.")
        data = self._read_json(path)
        return SkillSpec(skill_name=data["skill_name"], description=data["description"],
                         steps=[StepSpec(**s) for s in data["steps"]],
                         default_params=ReplayParams(**data["default_params"]))

    def _effective_params(self, skill_name: str, defaults: ReplayParams,
                          runtime_override: Optional[Dict[str, Any]]) -> ReplayParams:
        data = asdict(defaults)
        data.update(self.get_patches(skill_name))
        if runtime_override:
            data.update(runtime_override)
        p = ReplayParams(**data)
        p.validate()
        return p

    def _infer_gripper_state(self, action_name: str) -> Optional[str]:
        n = action_name.lower()
        if any(w in n for w in ("pick", "grasp", "close")):
            return "closed_or_holding"
        if any(w in n for w in ("release", "open", "drop")):
            return "open"
        return None

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def _log_trace(self, event: Dict[str, Any]) -> None:
        event.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
