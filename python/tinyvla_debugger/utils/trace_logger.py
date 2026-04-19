# The purpose of this module is to record every execution event to trace.jsonl.
# The dashboard reads this file live. The patch library is derived from it.
# It is the single source of truth for what the system did and why.

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


TRACE_FILE = Path("trace.jsonl")


@dataclass
class TraceEvent:
    step_id: int
    action: str
    result: str  # PASS = first-try success | FAIL = VLM said no | PATCHED = failed then fixed | ABORT = gave up
    skill: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    npu_latency_ms: Optional[float] = None
    failure_type: Optional[str] = None
    patch_applied: Optional[dict] = None
    params_used: Optional[dict] = None
    retry: bool = False
    backend: str = "NPU"


class TraceLogger:
    def __init__(self, trace_file: Path = TRACE_FILE) -> None:
        self.trace_file = trace_file

    def log(self, event: TraceEvent) -> None:
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(asdict(event)) + "\n")

    def log_pass(self, skill: str, step_id: int, action: str,
                 npu_latency_ms: float, params_used: dict | None = None,
                 backend: str = "NPU") -> None:
        self.log(TraceEvent(skill=skill, step_id=step_id, action=action,
                            result="PASS", npu_latency_ms=npu_latency_ms,
                            params_used=params_used, backend=backend))

    def log_fail(self, skill: str, step_id: int, action: str, failure_type: str,
                 npu_latency_ms: float, params_used: dict | None = None,
                 backend: str = "NPU") -> None:
        self.log(TraceEvent(skill=skill, step_id=step_id, action=action,
                            result="FAIL", failure_type=failure_type,
                            npu_latency_ms=npu_latency_ms, params_used=params_used,
                            backend=backend))

    def log_patched(self, skill: str, step_id: int, action: str, failure_type: str,
                    patch_applied: dict, npu_latency_ms: float,
                    params_used: dict | None = None, backend: str = "NPU") -> None:
        self.log(TraceEvent(skill=skill, step_id=step_id, action=action,
                            result="PATCHED", failure_type=failure_type,
                            patch_applied=patch_applied, npu_latency_ms=npu_latency_ms,
                            params_used=params_used, retry=True, backend=backend))

    def log_abort(self, skill: str, step_id: int, action: str, failure_type: str,
                  npu_latency_ms: float, params_used: dict | None = None) -> None:
        self.log(TraceEvent(skill=skill, step_id=step_id, action=action,
                            result="ABORT", failure_type=failure_type,
                            npu_latency_ms=npu_latency_ms, params_used=params_used))

    def clear(self) -> None:
        self.trace_file.unlink(missing_ok=True)
