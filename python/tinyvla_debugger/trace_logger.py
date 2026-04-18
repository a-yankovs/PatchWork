"""
trace_logger.py — Structured execution trace logger
Owner: Elias (dashboard integration); basic implementation provided here
       so Vera's orchestrator tests work before Elias's version lands.

Every event in the system is appended as a JSON line to trace.jsonl.
The Streamlit dashboard polls this file on a 500ms refresh cycle.
The patch library is derived from it. The pitch slides are built from it.

trace.jsonl schema (one JSON object per line):
  {
    "timestamp":     float,        # Unix time, e.g. 1713361391.44
    "skill":         str,          # e.g. "stock_middle_shelf"
    "step_id":       int | null,   # null for skill-level events
    "action":        str,          # e.g. "pick_from_box"
    "result":        str,          # "PASS" | "FAIL" | "patched_success" |
                                   # "abort" | "running" | "skill_complete"
    "failure_type":  str | null,   # e.g. "GRASP_FAIL" or null
    "patch_applied": dict | null,  # e.g. {"z_offset_mm": 5} or null
    "npu_latency_ms":float | null, # VLM inference latency in ms
    "retry":         bool          # True if this is a retry attempt
  }
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

# Default trace file location — override via TRACE_FILE env var or direct assignment
TRACE_FILE = Path("trace.jsonl")


def log_event(
    skill: str,
    action: str,
    result: str,
    *,
    step_id: Optional[int] = None,
    failure_type: Optional[str] = None,
    patch_applied: Optional[dict] = None,
    npu_latency_ms: Optional[float] = None,
    retry: bool = False,
    timestamp: Optional[float] = None,
    trace_file: Optional[Path] = None,
) -> None:
    """
    Append one structured event to trace.jsonl.

    Args:
        skill:          Skill name, e.g. "stock_middle_shelf".
        action:         Step action string, e.g. "pick_from_box".
        result:         One of: "PASS", "FAIL", "patched_success",
                        "abort", "running", "skill_complete".
        step_id:        Step index (0-based). None for skill-level events.
        failure_type:   Failure class from classifier, or None.
        patch_applied:  Parameter delta dict that was applied, or None.
        npu_latency_ms: VLM inference latency in milliseconds, or None.
        retry:          True if this event is a retry attempt.
        timestamp:      Unix timestamp; defaults to time.time().
        trace_file:     Override default TRACE_FILE path (useful in tests).
    """
    event = {
        "timestamp": timestamp if timestamp is not None else time.time(),
        "skill": skill,
        "step_id": step_id,
        "action": action,
        "result": result,
        "failure_type": failure_type,
        "patch_applied": patch_applied,
        "npu_latency_ms": round(npu_latency_ms, 2) if npu_latency_ms is not None else None,
        "retry": retry,
    }

    target = trace_file if trace_file is not None else TRACE_FILE
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def read_trace(
    n: Optional[int] = None,
    trace_file: Optional[Path] = None,
) -> list[dict]:
    """
    Read events from trace.jsonl, newest first.

    Args:
        n:          If given, return only the last n events.
        trace_file: Override default TRACE_FILE path.

    Returns:
        List of event dicts, newest at index 0.
    """
    target = trace_file if trace_file is not None else TRACE_FILE
    if not target.exists():
        return []

    lines = target.read_text(encoding="utf-8").strip().splitlines()
    events = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    events.reverse()  # newest first
    return events[:n] if n is not None else events


def clear_trace(trace_file: Optional[Path] = None) -> None:
    """
    Truncate trace.jsonl. Useful for test teardown.

    Args:
        trace_file: Override default TRACE_FILE path.
    """
    target = trace_file if trace_file is not None else TRACE_FILE
    if target.exists():
        target.write_text("", encoding="utf-8")
