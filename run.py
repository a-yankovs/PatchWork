#!/usr/bin/env python3
"""
run.py — One-command entry point for the PatchWork tinyVLA pipeline.

Full pipeline:
  Voice command (STT) → SLM compiler → webcam stream → VLM verifier → robot

Usage
-----
# Mock mode — no GPU, camera, or mic needed (great for testing):
    python run.py --mock

# Mock mode with a fixed text command (skip stdin prompt):
    python run.py --mock --cmd "Put the canned goods on the middle shelf"

# Mock mode, keep running in a loop:
    python run.py --mock --loop

# Real hardware — mic + webcam + AMD ROCm GPU:
    python run.py

# Real hardware, debug logging:
    python run.py --verbose

All options:
    --mock              Full mock hardware (compiler, robot, VLM, webcam, STT)
    --cmd TEXT          Skip STT; pass text command directly
    --loop              Keep listening after each skill run (Ctrl+C to exit)
    --webcam N          Webcam device index (default: 0)
    --compiler BACKEND  SLM backend: auto | llama_cpp | onnx_rocm | mock
    --model SIZE        Whisper model: tiny | base | small | medium (default: base)
    --record-secs N     Mic recording window in seconds (default: 5)
    --verbose / -v      Debug logging to stderr
"""

import os
import sys

# Allow running from the project root without installing the package.
# Adds python/ to sys.path so `import tinyvla_debugger` works.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "python"))

from tinyvla_debugger.main import main

if __name__ == "__main__":
    main()
