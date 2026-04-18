"""
orchestrator.py — Central control loop (SkillPatch state machine)
Owner: Vera

The orchestrator is the only component that calls all other layers.
It owns the state machine for skill execution:

  NL command
    → Layer 1: compiler.py     compile to skill program JSON
    → load patch library, pre-apply known patches to params
    → for each step:
        → Layer 3: vlm_api     pre-check scene precondition (camera BEFORE replay)
        → if pre-check FAIL: skip replay, classify, patch, retry
        → Layer 2: robot_api   replay step with current params
        → Layer 3: vlm_api     post-verify step result
        → if PASS: log, continue
        → if FAIL:
            → Layer 4a: classifier   determine failure_type
            → Layer 4b: patch_library look up / create patch
            → apply patch to params
            → Layer 3: vlm_api       pre-check again before retry
            → Layer 2: robot_api     replay step again (patched)
            → Layer 3: vlm_api       verify again
            → if PASS: log patched_success, store patch, continue
            → if FAIL: log abort, raise SkillAbortError (max retries hit)
    → log SkillComplete

Usage:
    orc = Orchestrator()
    asyncio.run(orc.run_skill("Put the canned goods on the middle shelf"))
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

# Pre-conditions checked via VLM *before* each lerobot-replay call.
# If the scene doesn't satisfy the pre-condition the replay is skipped entirely
# and the step is counted as a failure (triggering the normal retry/patch path).
# Tuple: (query, expected_result)
_PRE_CHECK: dict[str, tuple[str, bool]] = {
    "pick_object":         ("Is there an object visible and accessible in the pick zone?", True),
    "place_in_box":        ("Is an object currently held in the gripper?", True),
    "full_pick_and_place": ("Is there an object visible and accessible in the pick zone?", True),
    "box_in_shelf":        ("Is a box held securely in the gripper?", True),
}


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

        # Layer 2 — robot
        # Pass a RobotAPI instance explicitly, or omit to auto-instantiate from env vars.
        if robot is not None:
            self._robot = robot
        else:
            import os
            from .robot_api import RobotAPI
            self._robot = RobotAPI(
                robot_port=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
                teleop_port=os.environ.get("TELEOP_PORT", "/dev/ttyACM2"),
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

        result = SkillResult(skill_name=skill_name, outcome="success")

        # --- Step loop ---
        for step in skill["steps"]:
            step_id = step["step_id"]
            action  = step["action"]
            query   = step["verification_query"]

            logger.info("Step %d: action=%r query=%r", step_id, action, query)

            step_passed, patch_was_applied, latency_ms, failure_type = (
                await self._execute_step_with_retry(
                    skill_name, step_id, action, query, base_params
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
        failure_type     = None

        # Pre-apply any patches that have worked for this action before.
        # Keyed by action (e.g. "box_in_shelf"), not compiled skill_name.
        preexisting = self.patch_library.get_preexisting_patches(action)
        current_params = self.patch_library.apply_to_params(dict(params), preexisting)
        if preexisting:
            logger.info("Pre-applied patches for action %s: %s", action, preexisting)

        for attempt in range(MAX_RETRIES_PER_STEP):
            is_retry = attempt > 0

            # --- Pre-check: verify scene precondition before moving the arm ---
            pre_check = _PRE_CHECK.get(action)
            if pre_check is not None:
                pre_query, pre_expected = pre_check
                pre_frame = await asyncio.get_event_loop().run_in_executor(
                    None, self._capture_frame
                )
                pre_ok, pre_latency = self._vlm.verify(pre_frame, pre_query)
                logger.info(
                    "Pre-check step %d action=%r: %r → %s (expected %s)",
                    step_id, action, pre_query, pre_ok, pre_expected,
                )
                if pre_ok != pre_expected:
                    # Scene isn't ready — skip the replay and treat as a step
                    # failure so the normal classifier/patch/retry path handles it.
                    logger.warning(
                        "Pre-check FAILED for step %d (%s): scene not ready — skipping replay",
                        step_id, action,
                    )
                    trace_logger.log_event(
                        skill=skill_name,
                        step_id=step_id,
                        action=action,
                        result="pre_check_fail",
                        gpu_latency_ms=pre_latency,
                        retry=is_retry,
                    )
                    # Jump straight to the failure-handling block below.
                    # We set verified=False and latency so the rest of the loop
                    # behaves identically to a post-check failure.
                    verified    = False
                    latency_ms  = pre_latency
                    last_latency_ms = latency_ms
                    # Skip the replay and post-check entirely for this attempt.
                    classification = self.classifier.classify(
                        action, pre_query, verified, retry=is_retry,
                    )
                    failure_type = classification.failure_type
                    if attempt >= MAX_RETRIES_PER_STEP - 1:
                        trace_logger.log_event(
                            skill=skill_name, step_id=step_id, action=action,
                            result="abort", failure_type=failure_type,
                            patch_applied=None, gpu_latency_ms=latency_ms,
                            retry=is_retry,
                        )
                        return False, patch_applied, last_latency_ms, failure_type
                    patch = self.patch_library.get_patch(action, failure_type)
                    if patch:
                        current_params = self.patch_library.apply_to_params(current_params, patch)
                        self._robot.apply_patch(action, patch)
                        patch_applied = True
                    continue  # retry the attempt loop from the top (pre-check again)

            # --- Execute step (run in executor to not block event loop) ---
            # Pass `action` (the individual sub-skill, e.g. "full_pick_and_place"),
            # NOT `skill_name` (the compiled task name, e.g. "refill_inventory").
            # robot_api.replay_skill looks up the manifest by sub-skill name —
            # the compiled skill_name never has a manifest of its own.
            replay_error: Optional[str] = None
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._replay_step,
                    action, current_params,
                )
            except RuntimeError as exc:
                # lerobot-replay exited non-zero (e.g. motor overload on disconnect).
                # Treat as a step failure so the patch/retry path can handle it
                # rather than crashing the entire pipeline.
                replay_error = str(exc)
                logger.warning(
                    "Step %d replay raised RuntimeError (attempt %d/%d): %s",
                    step_id, attempt + 1, MAX_RETRIES_PER_STEP, replay_error,
                )

            trace_logger.log_event(
                skill=skill_name,
                step_id=step_id,
                action=action,
                result="running" if not replay_error else "replay_error",
                gpu_latency_ms=None,
                retry=is_retry,
            )

            # --- Capture frame and post-verify ---
            # Even if replay errored, check the frame — the arm may have
            # partially completed the step (e.g. overload after motion finished).
            frame = await asyncio.get_event_loop().run_in_executor(
                None, self._capture_frame
            )
            verified, latency_ms = self._vlm.verify(frame, query)
            # If replay hard-failed and VLM also says no, mark as not verified.
            if replay_error and not verified:
                verified = False
            last_latency_ms = latency_ms

            if verified:
                outcome = "patched_success" if patch_applied else "PASS"
                trace_logger.log_event(
                    skill=skill_name,
                    step_id=step_id,
                    action=action,
                    result=outcome,
                    gpu_latency_ms=latency_ms,
                    patch_applied=self.patch_library.get_patch(action, failure_type)
                        if patch_applied and failure_type else None,
                    retry=is_retry,
                )

                if patch_applied and failure_type:
                    self._audio.speak_patch_success(failure_type, step_id)

                if patch_applied and failure_type:
                    # Store the patch that worked
                    successful_patch = self.patch_library.get_patch(action, failure_type)
                    self.patch_library.store_patch(action, failure_type, successful_patch)
                    logger.info("Patch worked — stored %s:%s", action, failure_type)

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
            patch = self.patch_library.get_patch(action, failure_type)

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
                    self._robot.apply_patch(action, patch)
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


