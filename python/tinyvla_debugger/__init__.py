"""
tinyvla_debugger — SkillPatch: Self-Healing Robot Execution Layer

Package structure (one module per architectural layer):
  compiler.py       Layer 1 — NL → skill program JSON (Phi-3)
  robot_api.py      Layer 2 — skill execution engine interface (Diya)
  vlm_api.py        Layer 3 — on-device VLM verifier interface (Sasha)
  classifier.py     Layer 4a — rule-based failure classifier (Vera)
  patch_library.py  Layer 4b — persistent patch store (Vera)
  orchestrator.py   Layer 5 — central control loop (Vera)
  trace_logger.py   Layer 6 — structured execution trace logger (Elias)
"""

from .compiler import SkillCompiler
from .classifier import FailureClassifier
from .patch_library import PatchLibrary
from .orchestrator import Orchestrator, SkillAbortError

__all__ = [
    "SkillCompiler",
    "FailureClassifier",
    "PatchLibrary",
    "Orchestrator",
    "SkillAbortError",
]
