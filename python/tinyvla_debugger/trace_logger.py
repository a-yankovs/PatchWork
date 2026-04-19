"""
trace_logger.py — Module-level event logging API for the orchestrator.

Re-exports the core types from utils/trace_logger.py and adds
module-level log_event() / read_trace() / clear_trace() functions
that the orchestrator calls directly.

The orchestrator sets trace_logger.TRACE_FILE = Path(...) to redirect
output per-run; all module functions read that attribute at call time.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

# Re-export core types (keeps existing import sites working)
from tinyvla_debugger.utils.trace_logger import TraceEvent, TraceLogger, TRACE_FILE  # noqa: F401


# ---------------------------------------------------------------------------
# Module-level API used by orchestrator.py
# ---------------------------------------------------------------------------

def log_event(
    skill: str,
    action: str,
    result: str,
    *,
    step_id,
    failure_type: Optional[str] = None,
    patch_applied=None,
    gpu_latency_ms: Optional[float] = None,
    retry: bool = False,
) -> None:
    """
    Append one event to the active trace file.

    The active trace file is ``trace_logger.TRACE_FILE``; the orchestrator
    overrides this per-run via::

        trace_logger.TRACE_FILE = Path(trace_file)

    This function reads the module attribute at call time so it always
    writes to whatever path is currently configured.
    """
    import tinyvla_debugger.trace_logger as _self   # late import — picks up reassigned TRACE_FILE
    path: Path = _self.TRACE_FILE

    event = {
        "timestamp":     time.time(),
        "skill":         skill,
        "step_id":       step_id,
        "action":        action,
        "result":        result,
        "failure_type":  failure_type,
        "patch_applied": patch_applied,
        "gpu_latency_ms": gpu_latency_ms,
        "retry":         retry,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def read_trace(n: Optional[int] = None, trace_file=None) -> list[dict]:
    """
    Read trace events from *trace_file* (or the module-level TRACE_FILE).

    Returns a list of event dicts, **newest first**.
    Returns an empty list if the file does not exist.

    Args:
        n:          Maximum number of events to return (newest n).
        trace_file: Path override — a ``Path`` or ``str``.
                    Defaults to the current ``trace_logger.TRACE_FILE``.
    """
    import tinyvla_debugger.trace_logger as _self
    path = Path(trace_file) if trace_file is not None else _self.TRACE_FILE
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    events: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    events.reverse()   # newest first
    if n is not None:
        events = events[:n]
    return events


def clear_trace(trace_file=None) -> None:
    """Delete the trace file (or the module-level TRACE_FILE)."""
    import tinyvla_debugger.trace_logger as _self
    path = Path(trace_file) if trace_file is not None else _self.TRACE_FILE
    path.unlink(missing_ok=True)
