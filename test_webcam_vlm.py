"""
test_webcam_vlm.py — Interactive webcam + VLM test
Usage:
    python test_webcam_vlm.py              # uses AMD USB cam (index 1)
    python test_webcam_vlm.py --webcam 0   # laptop built-in
    python test_webcam_vlm.py --mock       # no Ollama needed (VLM always returns False)

What it does:
    • Opens the webcam and saves a snapshot to /tmp/frame.jpg so you can see what
      the VLM is looking at
    • Runs the full skill_router query sequence and prints which skill it would pick
    • Then lets you type any free-text yes/no question to ask Moondream2 directly
    • Press Ctrl+C to exit
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "python"))

import cv2

from tinyvla_debugger.webcam_stream import WebcamStream
from tinyvla_debugger.vlm_api import verify, locate_object, QUERIES
from tinyvla_debugger.skill_router import route_skill

logging.basicConfig(level=logging.WARNING)

SNAPSHOT_PATH = Path("/tmp/vlm_test_frame.jpg")

# ---------------------------------------------------------------------------
# Mock VLM — used when --mock is passed (no Ollama needed)
# ---------------------------------------------------------------------------

def _mock_verify(frame, query: str):
    print(f"  [MOCK VLM] query: {query!r:.60} → False (0ms)")
    return False, 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_snapshot(frame) -> None:
    cv2.imwrite(str(SNAPSHOT_PATH), frame)
    print(f"  📸 Snapshot saved → {SNAPSHOT_PATH}  (open to see what VLM sees)")


def run_canned_queries(frame, vlm_fn) -> None:
    """Run every canned QUERIES key and print results in a table."""
    print("\n── Canned query results ──────────────────────────────────────")
    print(f"  {'Query key':<22}  {'Result':<6}  {'Latency':>8}  Question")
    print(f"  {'─'*22}  {'─'*6}  {'─'*8}  {'─'*45}")
    for key, question in QUERIES.items():
        result, ms = vlm_fn(frame, key)
        marker = "✓ yes" if result else "✗ no "
        print(f"  {key:<22}  {marker}  {ms:>6.0f}ms  {question}")


def run_skill_router(frame, vlm_fn) -> None:
    """Ask skill_router which skill it would dispatch."""
    print("\n── Skill router ──────────────────────────────────────────────")
    skill = route_skill(frame, vlm_fn)
    print(f"  → Would dispatch: {skill!r}")


def run_locate(frame, vlm_fn) -> None:
    """Run locate_object and print spatial hints."""
    print("\n── locate_object() ───────────────────────────────────────────")
    hints = locate_object(frame)
    print(f"  visible={hints['visible']}  left={hints['left']}  high={hints['high']}  "
          f"latency={hints['latency_ms']:.0f}ms")


def interactive_loop(cam: WebcamStream, vlm_fn) -> None:
    """Refresh frame + ask a free-text question on each iteration."""
    print("\n── Interactive mode ──────────────────────────────────────────")
    print("  Type a yes/no question for Moondream2, or one of:")
    print("    'snap'   — save a new snapshot")
    print("    'canned' — re-run all canned queries")
    print("    'route'  — re-run skill router")
    print("    'locate' — re-run locate_object")
    print("    'quit'   — exit")
    print()

    while True:
        try:
            q = input("  Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break

        frame = cam.get_latest_frame()

        if q.lower() == "snap":
            save_snapshot(frame)
            continue
        if q.lower() == "canned":
            run_canned_queries(frame, vlm_fn)
            continue
        if q.lower() == "route":
            run_skill_router(frame, vlm_fn)
            continue
        if q.lower() == "locate":
            run_locate(frame, vlm_fn)
            continue

        # Free-text VLM query
        result, ms = vlm_fn(frame, q)
        answer = "YES" if result else "NO"
        print(f"  → {answer}  ({ms:.0f}ms)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Webcam + VLM interactive test")
    parser.add_argument("--webcam", type=int, default=1,
                        help="Camera index (default 1 = AMD USB cam)")
    parser.add_argument("--mock", action="store_true",
                        help="Skip Ollama — VLM always returns False (tests webcam only)")
    args = parser.parse_args()

    vlm_fn = _mock_verify if args.mock else verify

    print(f"\nWebcam + VLM test")
    print(f"  Camera  : index {args.webcam}")
    print(f"  VLM     : {'[MOCK] always False' if args.mock else 'Moondream2 via Ollama (localhost:11434)'}")
    print()

    # Open webcam
    cam = WebcamStream(index=args.webcam, mock=False)
    try:
        cam.start()
    except RuntimeError as exc:
        print(f"⚠  Could not open camera {args.webcam}: {exc}")
        sys.exit(1)

    print(f"  Camera open — warming up ... ", end="", flush=True)
    time.sleep(0.5)   # let auto-exposure settle
    print("ready.")

    try:
        frame = cam.get_latest_frame()
        save_snapshot(frame)

        run_canned_queries(frame, vlm_fn)
        run_skill_router(frame, vlm_fn)
        run_locate(frame, vlm_fn)

        interactive_loop(cam, vlm_fn)

    finally:
        cam.stop()
        print("Camera closed.")


if __name__ == "__main__":
    main()
