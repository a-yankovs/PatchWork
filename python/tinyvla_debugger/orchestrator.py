"""
orchestrator.py — Central control loop (SkillPatch state machine)
Owner: Vera

The orchestrator is the only component that calls all other layers.
It owns the state machine for skill execution:

  NL command
    → Layer 1: compiler.py     compile to skill program JSON
    → load patch library, pre-apply known patches to params
    → for each step:
        → Layer 2: robot_api   replay step with current params
        → Layer 3: vlm_api     verify step result
        → if PASS: log, continue
        → if FAIL:
            → Layer 4a: classifier   determine failure_type
            → Layer 4b: patch_library look up / create patch
            → apply patch to params
            → Layer 2: robot_api     replay step again (patched)
            → Layer 3: vlm_api       verify again
            → if PASS: log patched_success, store patch, continue
            → if FAIL: log abort, raise SkillAbortError (max retries hit)
    → log SkillComplete

During hours 9-14, robot_api and vlm_api are MOCKED. See MockRobotAPI
and MockVLMAPI below — swap them in via Orchestrator(robot=..., vlm=...).

Usage (with real hardware):
    orc = Orchestrator()
    asyncio.run(orc.run_skill("Put the canned goods on the middle shelf"))

Usage (with mocks — for testing orchestration logic):
    robot = MockRobotAPI(failure_on_step=1)
    vlm   = MockVLMAPI(fail_step_ids={1})
    orc   = Orchestrator(robot=robot, vlm=vlm, compiler_backend="mock")
    asyncio.run(orc.run_skill("Stock the shelf"))
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Protocol

import cv2  # webcam capture — pip install opencv-python-headless

from .audio_feedback import AudioFeedback
from .classifier import FailureClassifier, OBJECT_NOT_FOUND
from .compiler import SkillCompiler
from .patch_library import PatchLibrary
from . import trace_logger

logger = logging.getLogger(__name__)

MAX_RETRIES_PER_STEP = 2   # spec: "orchestrator enforces a max retry count of 2"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SkillAbortError(Exception):
    """
    Raised when a step fails twice (patch is insufficient).
    Caught by the caller / demo runner to surface to the dashboard.
    """
    def __init__(self, failure_type: str, step_id: int, skill_name: str) -> None:
        self.failure_type = failure_type
        self.step_id = step_id
        self.skill_name = skill_name
        super().__init__(
            f"PATCH_INSUFFICIENT: {failure_type} on step {step_id} "
            f"of skill '{skill_name}' — surfacing to dashboard."
        )


# ---------------------------------------------------------------------------
# Protocols — lets us swap in mocks without monkey-patching
# ---------------------------------------------------------------------------

class RobotProtocol(Protocol):
    def replay_skill(self, skill_name: str, params: dict) -> Iterator[Any]: ...
    def apply_patch(self, skill_name: str, patch: dict) -> None: ...


class VLMProtocol(Protocol):
    def verify(self, frame: Any, query: str) -> tuple[bool, float]: ...


# ---------------------------------------------------------------------------
# Skill execution result
# ---------------------------------------------------------------------------

@dataclass
class SkillResult:
    skill_name: str
    outcome: str               # "success" | "aborted"
    steps_executed: int = 0
    steps_patched: int = 0
    total_gpu_ms: float = 0.0
    failure_type: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Central control loop for SkillPatch.

    Args:
        robot:            RobotAPI implementation. Defaults to importing
                          robot_api module (real hardware).
        vlm:              VLM verifier implementation. Defaults to importing
                          vlm_api module (real ROCm GPU inference).
        compiler_backend: Passed to SkillCompiler — "auto", "llama_cpp",
                          "onnx_rocm", or "mock".
        patches_file:     Path to patches.json.
        trace_file:       Path to trace.jsonl.
        webcam_index:     OpenCV webcam device index (default 0).
    """

    def __init__(
        self,
        robot: Optional[RobotProtocol] = None,
        vlm: Optional[VLMProtocol] = None,
        compiler_backend: str = "auto",
        patches_file: Optional[str] = None,
        trace_file: Optional[str] = None,
        webcam_index: int = 0,
        audio: Optional[AudioFeedback] = None,
        frame_source: Optional[Callable[[], Any]] = None,
    ) -> None:
        # Optional external frame provider (e.g. WebcamStream.get_latest_frame).
        # When set, _capture_frame() delegates here instead of opening VideoCapture.
        self._frame_source = frame_source
        # Layer 1
        self.compiler = SkillCompiler(backend=compiler_backend)

        # Layer 2 — robot (real or mock)
        # Pass a RobotAPI instance explicitly for real hardware, or MockRobotAPI for tests.
        # Auto-instantiation from env vars is the fallback when no robot is provided.
        if robot is not None:
            self._robot = robot
        else:
            import os
            from .robot_api import RobotAPI
            self._robot = RobotAPI(
                robot_port=os.environ.get("ROBOT_PORT", "/dev/ttyUSB0"),
                teleop_port=os.environ.get("TELEOP_PORT", "/dev/ttyUSB1"),
                robot_id=os.environ.get("ROBOT_ID", "follower_arm"),
                teleop_id=os.environ.get("TELEOP_ID", "leader_arm"),
                storage_dir=os.environ.get("ROBOT_STORAGE_DIR", "./skillpatch_data"),
            )

        # Layer 3 — VLM (real or mock)
        if vlm is not None:
            self._vlm = vlm
        else:
            from . import vlm_api as _vlm_api
            self._vlm = _vlm_api  # type: ignore[assignment]

        # Layer 4
        self.classifier    = FailureClassifier()
        self.patch_library = PatchLibrary(patches_file=patches_file)

        # Layer 6 — trace
        if trace_file is not None:
            from pathlib import Path
            trace_logger.TRACE_FILE = Path(trace_file)

        self._webcam_index = webcam_index

        # Layer 7 — audio feedback (ElevenLabs TTS)
        # Reads ELEVENLABS_API_KEY from env automatically; silently mocks if not set.
        self._audio = audio if audio is not None else AudioFeedback()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run_skill(self, nl_command: str) -> SkillResult:
        """
        Full orchestration loop for one natural language command.

        This is the entry point for the demo. It runs the complete
        compile → execute → verify → patch cycle.

        Args:
            nl_command: Natural language shelf command, e.g.
                        "Put the canned goods on the middle shelf."

        Returns:
            SkillResult with outcome, step counts, and latency stats.

        Raises:
            SkillAbortError: If a step fails twice (PATCH_INSUFFICIENT).
                             The caller should log this and surface to dashboard.
        """
        logger.info("=== run_skill: %r ===", nl_command)

        # --- Layer 1: Compile ---
        skill = self.compiler.compile(nl_command)
        skill_name = skill["skill_name"]
        base_params = dict(skill["parameters"])

        # --- Pre-apply known patches ---
        preexisting = self.patch_library.get_preexisting_patches(skill_name)
        params = self.patch_library.apply_to_params(base_params, preexisting)
        if preexisting:
            logger.info("Pre-applied patches for %s: %s", skill_name, preexisting)

        result = SkillResult(skill_name=skill_name, outcome="success")

        # --- Step loop ---
        for step in skill["steps"]:
            step_id = step["step_id"]
            action  = step["action"]
            query   = step["verification_query"]

            logger.info("Step %d: action=%r query=%r", step_id, action, query)

            step_passed, patch_was_applied, latency_ms, failure_type = (
                await self._execute_step_with_retry(
                    skill_name, step_id, action, query, params
                )
            )

            result.steps_executed += 1
            if patch_was_applied:
                result.steps_patched += 1
            if latency_ms:
                result.total_gpu_ms += latency_ms

            if not step_passed:
                # _execute_step_with_retry already logged the abort event
                result.outcome      = "aborted"
                result.failure_type = failure_type
                self._audio.speak_error(
                    failure_type or "UNKNOWN_FAIL", step_id, skill_name
                )
                raise SkillAbortError(failure_type or "UNKNOWN_FAIL", step_id, skill_name)

        # --- Skill complete ---
        trace_logger.log_event(
            skill=skill_name,
            action="skill_complete",
            result="skill_complete",
            step_id=None,
            failure_type=None,
            patch_applied=None,
            gpu_latency_ms=None,
            retry=False,
        )
        self._audio.speak_success(
            skill_name, result.steps_executed, result.steps_patched
        )
        logger.info("=== Skill complete: %s (%d steps, %d patched) ===",
                    skill_name, result.steps_executed, result.steps_patched)
        return result

    async def run_from_webcam(self) -> SkillResult:
        """
        Capture a webcam frame, route to a skill via VLM scene queries,
        then execute that skill.

        This is the primary entry point for autonomous operation — no NL
        command needed. The skill router asks Moondream2 a short sequence
        of yes/no questions to determine which of the four canonical skills
        (pick_object, place_in_box, full_pick_and_place, box_in_shelf)
        best matches the current scene.

        Because the routed skill name is a key in compiler.SKILL_REGISTRY,
        run_skill() returns the static program immediately without calling
        the Phi-3 LLM.

        Returns:
            SkillResult — same as run_skill().

        Raises:
            SkillAbortError: If a step fails twice and patching is insufficient.
        """
        from .skill_router import route_skill

        logger.info("=== run_from_webcam: capturing scene ===")
        frame = await asyncio.get_event_loop().run_in_executor(
            None, self._capture_frame
        )
        skill_name = route_skill(frame, self._vlm.verify)
        logger.info("Webcam routed to skill: %s", skill_name)
        return await self.run_skill(skill_name)

    # ------------------------------------------------------------------
    # Internal step execution
    # ------------------------------------------------------------------

    async def _execute_step_with_retry(
        self,
        skill_name: str,
        step_id: int,
        action: str,
        query: str,
        params: dict,
    ) -> tuple[bool, bool, Optional[float], Optional[str]]:
        """
        Execute one step with up to MAX_RETRIES_PER_STEP attempts.

        Returns:
            (passed, patch_was_applied, last_gpu_latency_ms, failure_type)
        """
        patch_applied    = False
        last_latency_ms  = None
        current_params   = dict(params)
        failure_type     = None

        for attempt in range(MAX_RETRIES_PER_STEP):
            is_retry = attempt > 0

            # --- Execute step (run in executor to not block event loop) ---
            # Pass `action` (the individual sub-skill, e.g. "full_pick_and_place"),
            # NOT `skill_name` (the compiled task name, e.g. "refill_inventory").
            # robot_api.replay_skill looks up the manifest by sub-skill name —
            # the compiled skill_name never has a manifest of its own.
            await asyncio.get_event_loop().run_in_executor(
                None,
                self._replay_step,
                action, current_params,
            )

            trace_logger.log_event(
                skill=skill_name,
                step_id=step_id,
                action=action,
                result="running",
                gpu_latency_ms=None,
                retry=is_retry,
            )

            # --- Capture frame and verify ---
            frame = await asyncio.get_event_loop().run_in_executor(
                None, self._capture_frame
            )
            verified, latency_ms = self._vlm.verify(frame, query)
            last_latency_ms = latency_ms

            if verified:
                outcome = "patched_success" if patch_applied else "PASS"
                trace_logger.log_event(
                    skill=skill_name,
                    step_id=step_id,
                    action=action,
                    result=outcome,
                    gpu_latency_ms=latency_ms,
                    patch_applied=self.patch_library.get_patch(skill_name, failure_type)
                        if patch_applied and failure_type else None,
                    retry=is_retry,
                )

                if patch_applied and failure_type:
                    self._audio.speak_patch_success(failure_type, step_id)

                if patch_applied and failure_type:
                    # Store the patch that worked
                    successful_patch = self.patch_library.get_patch(skill_name, failure_type)
                    self.patch_library.store_patch(skill_name, failure_type, successful_patch)
                    logger.info("Patch worked — stored %s:%s", skill_name, failure_type)

                return True, patch_applied, last_latency_ms, failure_type

            # --- Verification failed ---
            classification = self.classifier.classify(
                action, query, verified,
                retry=is_retry,
            )
            failure_type = classification.failure_type

            logger.warning(
                "Step %d FAILED (attempt %d/%d): %s — %s",
                step_id, attempt + 1, MAX_RETRIES_PER_STEP,
                failure_type, classification.reasoning,
            )

            if attempt >= MAX_RETRIES_PER_STEP - 1:
                # Out of retries — abort
                trace_logger.log_event(
                    skill=skill_name,
                    step_id=step_id,
                    action=action,
                    result="abort",
                    failure_type=failure_type,
                    patch_applied=None,
                    gpu_latency_ms=latency_ms,
                    retry=is_retry,
                )
                return False, patch_applied, last_latency_ms, failure_type

            # --- Visual re-localisation (OBJECT_NOT_FOUND only) ---
            # Before consulting the patch library, capture a fresh frame and
            # ask the VLM where the object is. The returned deltas are applied
            # on top of any param patch so the arm searches in the right area.
            reloc_delta: dict = {}
            if failure_type == OBJECT_NOT_FOUND:
                reloc_delta = await self._relocalize_object()
                if reloc_delta:
                    current_params = self.patch_library.apply_to_params(
                        current_params, reloc_delta
                    )
                    logger.info(
                        "Step %d: relocalization delta applied: %s",
                        step_id, reloc_delta,
                    )

            # --- Apply patch and retry ---
            patch = self.patch_library.get_patch(skill_name, failure_type)

            trace_logger.log_event(
                skill=skill_name,
                step_id=step_id,
                action=action,
                result="FAIL",
                failure_type=failure_type,
                patch_applied={**patch, **reloc_delta} if reloc_delta else patch,
                gpu_latency_ms=latency_ms,
                retry=is_retry,
            )

            if patch or reloc_delta:
                if patch:
                    current_params = self.patch_library.apply_to_params(current_params, patch)
                    self._robot.apply_patch(skill_name, patch)
                patch_applied = True
                logger.info(
                    "Patch applied: %s (reloc: %s) — retrying step %d",
                    patch, reloc_delta, step_id,
                )
            else:
                logger.warning("No patch available for %s — retrying without patch", failure_type)

        # Should not reach here
        return False, patch_applied, last_latency_ms, failure_type

    def _replay_step(self, skill_name: str, params: dict) -> None:
        """Consume the replay_skill iterator (runs robot motion synchronously)."""
        for _event in self._robot.replay_skill(skill_name, params):
            pass  # events are logged inside robot_api; orchestrator gets them on yield

    def _capture_frame(self):
        """
        Return the latest webcam frame.

        If a frame_source callable was provided at construction (e.g. from
        WebcamStream.get_latest_frame), it is called directly — the camera
        stays open in its background thread and this is just a dict lookup.

        Otherwise, falls back to the original open-read-release pattern so
        the orchestrator remains usable without WebcamStream.
        """
        if self._frame_source is not None:
            return self._frame_source()

        cap = cv2.VideoCapture(self._webcam_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(
                f"Failed to capture frame from webcam (index {self._webcam_index}). "
                "Check that the USB webcam is connected and not in use by another process."
            )
        return frame

    async def _relocalize_object(self) -> dict:
        """
        Run positional VLM queries to locate a lost/missing object and return
        parameter deltas that shift the arm's approach toward it.

        Called exclusively from the OBJECT_NOT_FOUND retry path.

        Delegates to vlm_api.locate_object() which runs three yes/no queries
        (visible? left? high?) and returns a hints dict.

        Returns:
            Parameter delta dict suitable for patch_library.apply_to_params(),
            e.g. {"z_offset_mm": 10.0, "approach_angle_deg": -10.0}.
            Returns {} if the object is not visible at all (hard abort on next
            retry attempt is the right behaviour in that case).
        """
        frame = await asyncio.get_event_loop().run_in_executor(
            None, self._capture_frame
        )

        # vlm_api.locate_object may not be present on older stub versions —
        # fall back gracefully so the normal patch path still runs.
        locate_fn = getattr(self._vlm, "locate_object", None)
        if locate_fn is None:
            logger.warning("_relocalize_object: vlm has no locate_object(), skipping")
            return {}

        hints = await asyncio.get_event_loop().run_in_executor(None, locate_fn, frame)
        logger.info("Relocalization hints: %s", hints)

        if not hints.get("visible", False):
            logger.warning("Relocalization: object not visible — no delta applied")
            return {}

        delta: dict = {}
        # Horizontal offset → adjust wrist approach angle
        # left=True  → shift arm left  (negative angle)
        # left=False → shift arm right (positive angle)
        delta["approach_angle_deg"] = -10.0 if hints.get("left") else 10.0

        # Vertical offset → adjust z approach
        # high=True  → object is higher than expected, raise z
        # high=False → object is lower, lower z
        delta["z_offset_mm"] = 10.0 if hints.get("high") else -5.0

        logger.info("Relocalization delta: %s (latency=%.0fms)",
                    delta, hints.get("latency_ms", 0))
        return delta


# ---------------------------------------------------------------------------
# [MOCK] Test doubles — used when running with --mock or in unit tests.
#
# To switch to real hardware, pass the real modules to Orchestrator:
#   import robot_api, vlm_api
#   orc = Orchestrator(robot=robot_api, vlm=vlm_api)   # [REAL]
#   orc = Orchestrator(robot=MockRobotAPI(), vlm=MockVLMAPI())  # [MOCK]
# ---------------------------------------------------------------------------

class MockRobotAPI:
    """
    [MOCK] Simulates robot_api without hardware.

    Yields synthetic StepEvents and records applied patches in memory.
    Use failure_on_step to inject a gripper failure on a specific step,
    which exercises the classifier → patch → retry path in the orchestrator.

    [REAL] Replace by passing the real robot_api module:
        from tinyvla_debugger import robot_api
        orc = Orchestrator(robot=robot_api)

    Args:
        failure_on_step: Step index whose gripper_state is forced "open"
                         (simulates a grasp failure). Set to -1 for no failures.
        num_steps:       Steps emitted per replay_skill call.
    """

    def __init__(self, failure_on_step: int = -1, num_steps: int = 4) -> None:
        self.failure_on_step = failure_on_step
        self.num_steps = num_steps
        self._patches: dict[str, dict] = {}
        logger.debug("MockRobotAPI initialized (failure_on_step=%d)", failure_on_step)

    def replay_skill(self, skill_name: str, params: dict) -> Iterator[Any]:
        from .robot_api import StepEvent
        for i in range(self.num_steps):
            # [MOCK] gripper_state and action name are synthetic.
            # [REAL] RobotAPI yields StepEvents from the SO-100 arm
            #        with live gripper sensor readings and the actual
            #        trajectory action name.
            gripper = "open" if i == self.failure_on_step else "closed"
            yield StepEvent(
                skill_name=skill_name,
                step_id=i,
                action=skill_name,
                timestamp=time.time(),
                dataset_repo_id=f"mock/{skill_name}_step_{i}",
                gripper_state=gripper,
                params_used=dict(params),
                result="EXECUTED",
            )

    def apply_patch(self, skill_name: str, patch: dict) -> None:
        # [MOCK] Stores patch in memory only — nothing written to disk.
        # [REAL] robot_api.apply_patch() writes deltas to the trajectory config.
        self._patches.setdefault(skill_name, {}).update(patch)
        logger.debug("MockRobotAPI.apply_patch(%s, %s)", skill_name, patch)

    def record_skill(self, skill_name: str) -> None:
        # [MOCK] No-op.
        # [REAL] robot_api.record_skill() runs LeRobot teleoperation recording.
        logger.debug("MockRobotAPI.record_skill(%s) — no-op in mock", skill_name)


class MockVLMAPI:
    """
    [MOCK] Simulates vlm_api without Ollama / ROCm GPU.

    By default every verify() call returns True (success). Pass step IDs
    in fail_step_ids to make those steps fail on the first attempt and
    pass on retry — this exercises the classifier → patch → retry path.

    Note: when a real numpy frame is passed (e.g. from WebcamStream in mock
    mode), step_id detection returns -1 and all calls succeed. To test the
    failure path, unit tests should set frame._mock_step_id directly.

    [REAL] Replace by passing the real vlm_api module:
        from tinyvla_debugger import vlm_api
        orc = Orchestrator(vlm=vlm_api)
        # Requires: ollama pull moondream-verify && ollama serve

    Args:
        fail_step_ids: Step IDs that return False on attempt 1, True on retry.
        latency_ms:    Simulated inference latency in milliseconds.
    """

    def __init__(
        self,
        fail_step_ids: set[int] | None = None,
        latency_ms: float = 95.0,   # [MOCK] hardcoded; [REAL] measured ROCm wall-clock time
    ) -> None:
        self.fail_step_ids = fail_step_ids or set()
        self.latency_ms = latency_ms
        self._call_counts: dict[int, int] = {}

    def verify(self, frame: Any, query: str) -> tuple[bool, float]:
        # [MOCK] Step detection via synthetic frame attribute.
        # Real numpy frames (WebcamStream) have no _mock_step_id → always pass.
        step_id = getattr(frame, "_mock_step_id", -1)
        count = self._call_counts.get(step_id, 0)
        self._call_counts[step_id] = count + 1

        if step_id in self.fail_step_ids and count == 0:
            # [MOCK] Simulated first-attempt failure to exercise patch path.
            # [REAL] vlm_api.verify() POSTs the live frame to Moondream2 via
            #        Ollama and parses "yes"/"no" from the model response.
            logger.debug("MockVLMAPI.verify — step %d attempt 1 → False", step_id)
            return False, self.latency_ms

        # [MOCK] Simulated success.
        # [REAL] Returns actual Moondream2 inference result + measured latency.
        return True, self.latency_ms


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _main(command: str, mock: bool = False) -> None:
    if mock:
        logger.info("Running with MOCK hardware interfaces")
        robot = MockRobotAPI(failure_on_step=1)   # step 1 fails first time
        vlm   = MockVLMAPI(fail_step_ids={1})
        orc   = Orchestrator(
            robot=robot, vlm=vlm, compiler_backend="mock"
        )
    else:
        orc = Orchestrator()

    try:
        result = await orc.run_skill(command)
        print(f"✓ Skill complete: {result.skill_name}")
        print(f"  Steps executed : {result.steps_executed}")
        print(f"  Steps patched  : {result.steps_patched}")
        print(f"  Total GPU time : {result.total_gpu_ms:.1f}ms")
    except SkillAbortError as e:
        print(f"✗ Skill aborted: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SkillPatch orchestrator")
    parser.add_argument("command", nargs="?",
                        default="Put the canned goods on the middle shelf")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock robot and VLM interfaces (no hardware needed)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    asyncio.run(_main(args.command, mock=args.mock))
