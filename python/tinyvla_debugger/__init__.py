"""
tinyvla_debugger — PatchWork: Self-Healing Robot Execution Layer

Pipeline entry point:
  run.py (project root) — one command to run everything

Module map  (owner — mock status)
─────────────────────────────────────────────────────────────────────────────
  stt.py            Input    — voice command → text          (Vera)
    [REAL] faster-whisper + sounddevice, local on-device transcription
    [MOCK] reads command from stdin; auto-fallback if packages missing

  compiler.py       Layer 1  — NL text → skill JSON          (Vera)
    [REAL] Phi-3-mini-4k-instruct via llama_cpp (ROCm) or onnx_rocm
    [MOCK] keyword-based deterministic routing; used when no GPU/model found

  robot_api.py      Layer 2  — skill execution interface      (Diya)
    [REAL] RobotAPI class wraps LeRobot CLI (record/replay/patch)
    Requires: lerobot installed, SO-100 arm connected, skill manifests recorded

  vlm_api.py        Layer 3  — on-device scene verifier       (Sasha)
    [REAL] Moondream2 via Ollama at localhost:11434 (AMD ROCm, 25/25 layers)
    [MOCK] returns (False, 0.0) on connection error — orchestrator retries

  classifier.py     Layer 4a — rule-based failure classifier  (Vera)
    [REAL] pure rule-based logic, no model — always active

  patch_library.py  Layer 4b — persistent patch store         (Vera)
    [REAL] reads/writes patches.json — always active

  orchestrator.py   Layer 5  — central control loop           (Vera)
    [REAL] always active; uses injected robot/VLM backends
    [MOCK] MockRobotAPI + MockVLMAPI available for testing without hardware

  trace_logger.py   Layer 6  — execution trace to trace.jsonl (Elias)
    [REAL] always active

  audio_feedback.py Layer 7  — ElevenLabs TTS cues            (Vera)
    [REAL] requires ELEVENLABS_API_KEY env var
    [MOCK] prints text to console when key is absent

  webcam_stream.py  Input    — threaded live frame buffer      (Vera)
    [REAL] OpenCV VideoCapture in background thread
    [MOCK] returns synthetic grey numpy frames; used when mock=True

  skill_router.py   Routing  — webcam scene → skill name      (Vera)
    [REAL] up to 4 Moondream2 queries to classify scene (~400ms)
─────────────────────────────────────────────────────────────────────────────
"""

from .compiler import SkillCompiler
from .classifier import FailureClassifier
from .patch_library import PatchLibrary
from .orchestrator import Orchestrator, SkillAbortError
from .robot_api import RobotAPI, ReplayParams, StepSpec, SkillSpec, StepEvent
from .stt import SpeechListener
from .webcam_stream import WebcamStream

__all__ = [
    "SkillCompiler",
    "FailureClassifier",
    "PatchLibrary",
    "Orchestrator",
    "SkillAbortError",
    "RobotAPI",
    "ReplayParams",
    "StepSpec",
    "SkillSpec",
    "StepEvent",
    "SpeechListener",
    "WebcamStream",
]
