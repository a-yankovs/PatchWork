"""
main.py — SkillPatch unified pipeline entry point
Owner: Vera

Ties together the full workflow:
  1. STT (faster-whisper + sounddevice)       — voice command → text
  2. SLM compiler (Phi-3-mini via llama_cpp)  — text → skill JSON
  3. WebcamStream (threaded OpenCV)           — continuous frame feed
  4. VLM verifier (Moondream2 via Ollama)     — per-step scene verification
  5. SLM classifier (rule-based)             — failure type → patch
  6. Orchestrator                             — state machine, retry, patches
  7. AudioFeedback (ElevenLabs TTS)          — spoken outcome cues

Run directly:
    python -m tinyvla_debugger [options]

Or via the top-level shortcut:
    python run.py [options]

Key options:
    --mock          Mock mode — no GPU or camera needed; mic/STT still active
    --cmd TEXT      Skip STT, use this text command directly
    --loop          Keep listening for commands (Ctrl+C to exit)
    --verbose       Enable DEBUG logging
    --webcam N      Webcam device index (default 0 — laptop built-in; AMD USB cam is 2)
    --model SIZE    Whisper model: tiny|base|small|medium (default: base)
    --record-secs N Mic recording duration in seconds (default: 5)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

from .orchestrator import Orchestrator, SkillAbortError
from .audio_feedback import AudioFeedback
from .robot_api import RobotAPI
from .stt import SpeechListener
from .webcam_stream import WebcamStream

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

_BANNER = """
╔═══════════════════════════════════════════════════════════╗
║             SkillPatch — tinyVLA Robot Pipeline           ║
║                                                           ║
║  [STT] → [SLM compiler] → [VLM verify] → [SLM classify]  ║
╚═══════════════════════════════════════════════════════════╝"""


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------

def _build_orchestrator(
    webcam_index: int,
    frame_source,
    compiler_backend: str,
    robot_port: str,
    teleop_port: str,
    robot_id: str,
    teleop_id: str,
    storage_dir: str,
) -> Orchestrator:
    """Build the orchestrator with hardware-backed RobotAPI and Phi-3 compiler."""
    robot = RobotAPI(
        robot_port=robot_port,
        teleop_port=teleop_port,
        robot_id=robot_id,
        teleop_id=teleop_id,
        storage_dir=storage_dir,
    )
    import os
    return Orchestrator(
        robot=robot,
        compiler_backend=compiler_backend,
        webcam_index=webcam_index,
        frame_source=frame_source,
        # Point PatchLibrary at the same file RobotAPI uses so patches
        # learned in one run are loaded on the next run.
        patches_file=os.path.join(storage_dir, "patches.json"),
        trace_file=os.path.join(storage_dir, "trace.jsonl"),
        # AudioFeedback() default: reads ELEVENLABS_API_KEY from env
    )


# ---------------------------------------------------------------------------
# Keyword normalization
# ---------------------------------------------------------------------------

# Maps: any of the listed keywords appearing in the transcript → canonical command.
# Values are (list_of_trigger_words, canonical_command).
# Trigger words can be exact substrings OR common STT mishearings of the same word.
_COMMAND_SHORTCUTS: list[tuple[list[str], str]] = [
    (["inventory", "inventor", "inventori", "inventor."], "refill the inventory"),
]


def _normalize_command(text: str) -> str:
    """
    Normalize a raw transcript to a canonical skill command.

    Checks every trigger word for each shortcut (case-insensitive substring match).
    Returns the canonical command if any trigger matches, otherwise the original.
    Handles STT mishearings by listing alternate spellings as extra triggers.
    """
    lowered = text.lower().strip(".!, ")
    for triggers, canonical in _COMMAND_SHORTCUTS:
        for kw in triggers:
            if kw in lowered:
                if text.strip() != canonical:
                    print(f"   ↳ normalized {text!r} → {canonical!r}  (matched {kw!r})")
                return canonical
    return text


# ---------------------------------------------------------------------------
# Single skill run
# ---------------------------------------------------------------------------

async def _run_once(command: str, orc: Orchestrator) -> tuple[bool, Optional["SkillAbortError"]]:
    """
    Execute one command through the full pipeline.

    Prints a human-readable summary.
    Returns (success, abort_error_or_None).
    """
    print(f"\n▶  Command : {command!r}")
    print("   Compiling skill via SLM ...", flush=True)

    try:
        result = await orc.run_skill(command)
        print(f"\n✓  Skill complete  : {result.skill_name}")
        print(f"   Steps executed  : {result.steps_executed}")
        print(f"   Steps patched   : {result.steps_patched}")
        print(f"   Total GPU time  : {result.total_gpu_ms:.1f} ms")
        return True, None
    except SkillAbortError as exc:
        print(f"\n✗  Skill aborted   : {exc}")
        return False, exc


# ---------------------------------------------------------------------------
# Learn-mode: auto-record after abort, then retry
# ---------------------------------------------------------------------------

def _record_episode_for_action(
    action: str,
    robot_port: str,
    teleop_port: str,
    robot_id: str,
    teleop_id: str,
    storage_dir: str,
    num_episodes: int = 1,
    episode_time_s: int = 40,
    reset_time_s: int = 10,
) -> bool:
    """
    Launch lerobot-record to append one episode to the failing action's dataset.

    Reads the dataset path from the skill manifest.
    Returns True if recording succeeded, False if it failed or was skipped.
    """
    import json
    import subprocess
    from pathlib import Path

    skills_dir = Path(storage_dir) / "skills"
    manifest_path = skills_dir / f"{action}.json"

    if not manifest_path.exists():
        print(f"   ⚠  No manifest for action {action!r} — cannot auto-record.")
        return False

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    steps = manifest.get("steps", [])
    if not steps:
        print(f"   ⚠  Manifest for {action!r} has no steps — cannot auto-record.")
        return False

    dataset_repo_id = steps[0]["dataset_repo_id"]
    repo_path = Path(dataset_repo_id)
    if repo_path.exists():
        repo_id = repo_path.name
        root    = str(repo_path)
    else:
        repo_id = dataset_repo_id
        root    = None

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={robot_port}",
        f"--robot.id={robot_id}",
        "--teleop.type=so101_leader",
        f"--teleop.port={teleop_port}",
        f"--teleop.id={teleop_id}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.single_task={action}",
        "--dataset.push_to_hub=false",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
    ]
    if root is not None:
        cmd += [f"--dataset.root={root}"]

    print(f"\n   Recording {num_episodes} new episode(s) for '{action}'")
    print(f"   Dataset  : {dataset_repo_id}")
    print(f"   → Position the arm at START, then follow the lerobot prompts.\n")

    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print("   ✗  lerobot-record not found — is the lerobot venv active?")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"   ✗  lerobot-record exited with code {exc.returncode}")
        return False


async def _run_with_learn(
    command: str,
    orc: Orchestrator,
    args: argparse.Namespace,
    max_learn_cycles: int = 5,
) -> bool:
    """
    Run a command in learn mode: on abort, auto-record an episode and retry.

    The loop:
      1. Run the skill
      2. If success → done
      3. If abort → print which action failed and which dataset needs data
      4. Launch lerobot-record for that action (human teleops)
      5. Retry from step 1
      6. Give up after max_learn_cycles total attempts

    Returns True if skill eventually succeeded.
    """
    for cycle in range(1, max_learn_cycles + 1):
        print(f"\n{'─'*60}")
        if cycle > 1:
            print(f"  [learn] Retry #{cycle} after recording new episode")
        print(f"{'─'*60}")

        success, abort_err = await _run_once(command, orc)
        if success:
            return True

        if abort_err is None:
            return False  # shouldn't happen but guard anyway

        # --- Prompt and record ---
        failed_action = abort_err.failure_type  # failure_type logged; action in str
        # Parse action from the SkillAbortError string or fall back to skill_name
        # Format: "PATCH_INSUFFICIENT: FAILURE_TYPE on step N of skill 'SKILL'"
        # The action we need is what the orchestrator calls `action` — which is the
        # step's action field. We derive it from trace.jsonl (last abort entry).
        action = _last_aborted_action(args.storage_dir) or abort_err.skill_name

        if cycle >= max_learn_cycles:
            print(f"\n  [learn] Reached max cycles ({max_learn_cycles}). Giving up.")
            print(f"  Run: python record_episode.py --action {action}")
            return False

        print(f"\n  [learn] Abort detected — action='{action}'")
        print(f"  [learn] Recording a new episode to expand the dataset.")
        print(f"  [learn] Reset the arm to start position, then press Enter when ready.")
        try:
            input("  [learn] Press Enter to start recording (Ctrl+C to cancel) ... ")
        except (EOFError, KeyboardInterrupt):
            print("\n  [learn] Cancelled.")
            return False

        recorded = _record_episode_for_action(
            action=action,
            robot_port=args.robot_port,
            teleop_port=args.teleop_port,
            robot_id=args.robot_id,
            teleop_id=args.teleop_id,
            storage_dir=args.storage_dir,
        )
        if not recorded:
            print("  [learn] Recording failed — cannot retry.")
            return False

        print(f"\n  [learn] Episode recorded ✓  Retrying skill ...\n")

    return False


def _last_aborted_action(storage_dir: str) -> Optional[str]:
    """Read trace.jsonl and return the action from the most recent abort event."""
    import json
    from pathlib import Path

    trace_path = Path(storage_dir) / "trace.jsonl"
    if not trace_path.exists():
        return None
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("result") == "abort" and event.get("action"):
                return event["action"]
        except json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------

async def _main_async(args: argparse.Namespace) -> None:
    print(_BANNER)

    stt_status = (
        "skip (--cmd provided)"
        if args.cmd is not None
        else f"faster-whisper ({args.model}), {args.record_secs:.0f}s window"
    )
    print(f"\n  • STT        → {stt_status}")
    print(f"  • SLM        → Phi-3-mini ({args.compiler})")
    print(f"  • Webcam     → Overview/stationary (device 0) + Follower Side (device 2) — stitched")
    print("  • VLM        → Moondream2 via Ollama (localhost:11434)")
    print(f"  • Robot      → SO-100 follower ({args.robot_port}, id={args.robot_id})")
    print(f"  • Teleop     → SO-100 leader  ({args.teleop_port}, id={args.teleop_id})")
    print(f"  • Storage    → {args.storage_dir}")
    print("  • Audio      → ElevenLabs TTS (ELEVENLABS_API_KEY)")

    # ------------------------------------------------------------------
    # Start both camera streams and stitch into one frame for VLM.
    #
    # Dashboard layout:  0 = Follower Top (arm built-in), 2 = Follower Side (USB)
    # The VLM receives a single horizontally-stitched image so it can reason
    # about both the gripper state and the scene simultaneously.
    # If one camera fails to open it is replaced with a grey placeholder so
    # the other camera still reaches the VLM.
    # ------------------------------------------------------------------
    import numpy as np

    _CAMERA_INDICES = [0, 2]
    _CAMERA_LABELS  = ["Overview (stationary)", "Follower Side"]

    cams: list[WebcamStream] = []
    for idx, label in zip(_CAMERA_INDICES, _CAMERA_LABELS):
        c = WebcamStream(index=idx, mock=False)
        try:
            c.start()
            print(f"   ✓ {label} (device {idx}) opened")
        except RuntimeError as exc:
            print(f"   ⚠  {label} (device {idx}) failed: {exc} — using blank placeholder")
            c = WebcamStream(mock=True)   # grey fallback
            c.start()
        cams.append(c)

    def _stitched_frame() -> "np.ndarray":
        """Return both camera frames side by side as one image for VLM."""
        frames = []
        for c in cams:
            try:
                frames.append(c.get_latest_frame())
            except RuntimeError:
                frames.append(np.full((480, 640, 3), 80, dtype=np.uint8))
        # Resize both to the same height before hstack
        h = min(f.shape[0] for f in frames)
        resized = [f[:h, :] for f in frames]
        return np.hstack(resized)

    try:
        # ------------------------------------------------------------------
        # Build orchestrator — frame_source delivers the stitched dual-cam frame
        # ------------------------------------------------------------------
        orc = _build_orchestrator(
            webcam_index=_CAMERA_INDICES[1],   # kept for fallback inside orchestrator
            frame_source=_stitched_frame,
            compiler_backend=args.compiler,
            robot_port=args.robot_port,
            teleop_port=args.teleop_port,
            robot_id=args.robot_id,
            teleop_id=args.teleop_id,
            storage_dir=args.storage_dir,
        )

        # ------------------------------------------------------------------
        # Build STT listener (faster-whisper + sounddevice)
        # ------------------------------------------------------------------
        # If --cmd is provided, SpeechListener is built but listen() is never
        # called — the command comes from args.cmd directly.
        listener = SpeechListener(
            mock=(args.cmd is not None),
            model_size=args.model,
            record_secs=args.record_secs,
        )

        # ------------------------------------------------------------------
        # Run loop
        # ------------------------------------------------------------------
        learn = getattr(args, "learn", False)

        if args.loop:
            mode = "learn" if learn else "normal"
            print(f"\nLoop mode active [{mode}] — speak after each prompt. Ctrl+C to exit.\n")
            iteration = 0
            while True:
                iteration += 1
                try:
                    command = (
                        args.cmd if args.cmd else
                        listener.listen(f"[{iteration}] Ready. Speak your robot command.")
                    )
                    command = _normalize_command(command)
                    if learn:
                        await _run_with_learn(command, orc, args)
                    else:
                        await _run_once(command, orc)
                    print()  # blank line between runs
                except KeyboardInterrupt:
                    print("\nLoop interrupted — exiting.")
                    break
        else:
            command = (
                args.cmd if args.cmd else
                listener.listen("Ready. Speak your robot command.")
            )
            command = _normalize_command(command)
            if learn:
                success = await _run_with_learn(command, orc, args)
            else:
                success, _ = await _run_once(command, orc)
            sys.exit(0 if success else 1)

    finally:
        for c in cams:
            c.stop()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tinyvla",
        description="SkillPatch tinyVLA unified pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--cmd", default=None, metavar="TEXT",
        help="Skip STT; use this text command directly",
    )

    # --- Hardware: compiler ---
    # Cameras are always opened at indices 0 (Follower Top) and 2 (Follower Side)
    # to match the dashboard. No --webcam flag needed.
    parser.add_argument(
        "--compiler", default="auto",
        choices=["auto", "llama_cpp", "onnx_rocm", "mock"],
        help="SLM compiler backend (default: auto — tries llama_cpp then onnx_rocm)",
    )

    # --- Hardware: robot arm ---
    import os
    parser.add_argument(
        "--robot-port", default=os.environ.get("ROBOT_PORT", "/dev/ttyACM1"),
        help="Serial port for SO-100 follower arm (default: $ROBOT_PORT or /dev/ttyACM1)",
    )
    parser.add_argument(
        "--teleop-port", default=os.environ.get("TELEOP_PORT", "/dev/ttyACM2"),
        help="Serial port for SO-100 leader arm (default: $TELEOP_PORT or /dev/ttyACM2)",
    )
    parser.add_argument(
        "--robot-id", default=os.environ.get("ROBOT_ID", "follower_arm"),
        help="Robot arm identifier (default: $ROBOT_ID or follower_arm)",
    )
    parser.add_argument(
        "--teleop-id", default=os.environ.get("TELEOP_ID", "leader_arm"),
        help="Teleop arm identifier (default: $TELEOP_ID or leader_arm)",
    )
    parser.add_argument(
        "--storage-dir", default=os.environ.get("ROBOT_STORAGE_DIR", "./skillpatch_data"),
        help="Directory for skill manifests and patch storage (default: ./skillpatch_data)",
    )

    # --- STT ---
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium"],
        help="Whisper model size for STT (default: base)",
    )
    parser.add_argument(
        "--record-secs", type=float, default=5.0, metavar="N",
        help="Mic recording window in seconds (default: 5.0)",
    )

    # --- Behaviour ---
    parser.add_argument(
        "--loop", action="store_true",
        help="Keep listening for commands after each skill run (Ctrl+C to stop)",
    )
    parser.add_argument(
        "--learn", action="store_true",
        help=(
            "Learn mode: on abort, automatically launch lerobot-record to add a "
            "new episode to the failing dataset, then retry the skill. "
            "Combine with --loop to keep the improvement loop running hands-free. "
            "You still physically teleop each new demonstration — the system handles "
            "everything else (detecting the abort, launching recording, retrying)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG-level logging to stderr",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    asyncio.run(_main_async(args))


# Allow: python -m tinyvla_debugger
if __name__ == "__main__":
    main()
