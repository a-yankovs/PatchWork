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
    --webcam N      Webcam device index (default 0)
    --model SIZE    Whisper model: tiny|base|small|medium (default: base)
    --record-secs N Mic recording duration in seconds (default: 5)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from typing import Optional

from .orchestrator import Orchestrator, MockRobotAPI, MockVLMAPI, SkillAbortError
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
    mock: bool,
    webcam_index: int,
    frame_source,
    compiler_backend: str,
) -> Orchestrator:
    """
    Build the orchestrator with the right backends.

    Mock mode:   MockRobotAPI + MockVLMAPI + mock compiler
    Real mode:   Real robot_api + real vlm_api + auto-detect compiler backend
    """
    if mock:
        robot = MockRobotAPI(failure_on_step=1)  # step 1 fails on first attempt → exercises patch path
        vlm   = MockVLMAPI(fail_step_ids={1})
        return Orchestrator(
            robot=robot,
            vlm=vlm,
            compiler_backend="mock",
            webcam_index=webcam_index,
            frame_source=frame_source,
        )

    return Orchestrator(
        compiler_backend=compiler_backend,
        webcam_index=webcam_index,
        frame_source=frame_source,
    )


# ---------------------------------------------------------------------------
# Single skill run
# ---------------------------------------------------------------------------

async def _run_once(command: str, orc: Orchestrator) -> bool:
    """
    Execute one command through the full pipeline.

    Prints a human-readable summary.
    Returns True on success, False on abort.
    """
    print(f"\n▶  Command : {command!r}")
    print("   Compiling skill via SLM ...", flush=True)

    try:
        result = await orc.run_skill(command)
        print(f"\n✓  Skill complete  : {result.skill_name}")
        print(f"   Steps executed  : {result.steps_executed}")
        print(f"   Steps patched   : {result.steps_patched}")
        print(f"   Total GPU time  : {result.total_gpu_ms:.1f} ms")
        return True
    except SkillAbortError as exc:
        print(f"\n✗  Skill aborted   : {exc}")
        return False


# ---------------------------------------------------------------------------
# Async main
# ---------------------------------------------------------------------------

async def _main_async(args: argparse.Namespace) -> None:
    print(_BANNER)

    mode_tag = "[MOCK]" if args.mock else "[REAL]"
    print(f"\nMode: {mode_tag}")
    # STT status is independent of --mock: mic works unless --cmd skips it entirely
    stt_status = (
        "skip (--cmd provided)"
        if args.cmd is not None
        else f"faster-whisper ({args.model}), {args.record_secs:.0f}s window"
    )
    print(f"  • STT        → {stt_status}")
    if args.mock:
        print("  • SLM        → mock compiler (no Phi-3 needed)")
        print("  • Webcam     → synthetic blank frames")
        print("  • VLM        → MockVLMAPI (step 1 fails once, then auto-patches)")
        print("  • Robot      → MockRobotAPI (motion simulated)")
        print("  • Audio      → print-only (no ElevenLabs)")
    else:
        print(f"  • SLM        → Phi-3-mini ({args.compiler})")
        print(f"  • Webcam     → device {args.webcam}")
        print("  • VLM        → Moondream2 via Ollama (localhost:11434)")
        print("  • Audio      → ElevenLabs TTS (ELEVENLABS_API_KEY)")

    # ------------------------------------------------------------------
    # Start webcam stream
    # ------------------------------------------------------------------
    cam = WebcamStream(index=args.webcam, mock=args.mock)
    try:
        cam.start()
    except RuntimeError as exc:
        if not args.mock:
            print(f"\n⚠  Webcam error: {exc}")
            print("   Switching to mock webcam (blank frames).")
            cam = WebcamStream(mock=True)
            cam.start()

    try:
        # ------------------------------------------------------------------
        # Build orchestrator — frame_source wires in the streaming webcam
        # ------------------------------------------------------------------
        orc = _build_orchestrator(
            mock=args.mock,
            webcam_index=args.webcam,
            frame_source=cam.get_latest_frame,
            compiler_backend=args.compiler,
        )

        # ------------------------------------------------------------------
        # Build STT listener
        # ------------------------------------------------------------------
        # STT is independent of --mock: microphone + faster-whisper always runs
        # unless --cmd is given (which bypasses STT entirely).
        # [REAL] faster-whisper + sounddevice — loads regardless of --mock flag
        # [SKIP] if --cmd is provided, SpeechListener is still built but listen()
        #        is never called (command comes from args.cmd instead)
        listener = SpeechListener(
            mock=(args.cmd is not None),
            model_size=args.model,
            record_secs=args.record_secs,
        )

        # ------------------------------------------------------------------
        # Run loop
        # ------------------------------------------------------------------
        if args.loop:
            print("\nLoop mode active — speak after each prompt. Ctrl+C to exit.\n")
            iteration = 0
            while True:
                iteration += 1
                try:
                    command = (
                        args.cmd if args.cmd else
                        listener.listen(f"[{iteration}] Ready. Speak your robot command.")
                    )
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
            success = await _run_once(command, orc)
            sys.exit(0 if success else 1)

    finally:
        cam.stop()


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

    # --- Mode ---
    parser.add_argument(
        "--mock", action="store_true",
        help="Mock mode — no GPU or camera required; STT (mic) still runs unless --cmd is given",
    )
    parser.add_argument(
        "--cmd", default=None, metavar="TEXT",
        help="Skip STT; use this text command directly",
    )

    # --- Hardware ---
    parser.add_argument(
        "--webcam", type=int, default=0, metavar="N",
        help="OpenCV webcam device index (default: 0)",
    )
    parser.add_argument(
        "--compiler", default="auto",
        choices=["auto", "llama_cpp", "onnx_rocm", "mock"],
        help="SLM compiler backend (default: auto — tries llama_cpp then onnx_rocm)",
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
