# The purpose of this module is to provide a live view of the SkillPatch execution state.
# It reads trace.jsonl and patches.json every 500ms and displays robot step status,
# the ReAct agentic loop phase, NPU vs CPU latency, execution history, and patch memory.

import json
import socket
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
cv2.setLogLevel(0)  # suppress WARN/ERROR spam from probing unavailable camera indices
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TRACE_FILE   = Path("trace.jsonl")
PATCHES_FILE = Path("patches.json")
ARCHIVE_DIR  = Path("Archive")

NPU_TARGET_MS   = 120.0   # AMD real-time robotics target; anything above this turns orange
CPU_BASELINE_MS = 820.0   # measured baseline: same Moondream2 query on CPU-only

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@st.cache_data(ttl=60)
def load_archive_stats() -> list[dict]:
    rows = []
    if not ARCHIVE_DIR.exists():
        return rows
    for skill_dir in sorted(ARCHIVE_DIR.iterdir()):
        if skill_dir.name.startswith("_") or not skill_dir.is_dir():
            continue
        info_path = skill_dir / "meta" / "info.json"
        stats_path = skill_dir / "meta" / "stats.json"
        if not info_path.exists():
            continue
        try:
            info  = json.loads(info_path.read_text())
            stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        except Exception:
            continue
        name = skill_dir.name                       # e.g. "pick_object_v3"
        parts = name.rsplit("_", 1)
        family = "_".join(name.split("_")[:-1]) if parts[-1].startswith("v") and parts[-1][1:].isdigit() else name
        version = parts[-1] if parts[-1].startswith("v") and parts[-1][1:].isdigit() else "v1"
        episodes = info.get("total_episodes", 0)
        frames   = info.get("total_frames",   0)
        fps      = info.get("fps", 30)
        duration_s = round(frames / fps) if fps else 0
        gripper_range = ""
        if stats.get("action"):
            mn = stats["action"].get("min", [])
            mx = stats["action"].get("max", [])
            if len(mn) >= 6 and len(mx) >= 6:
                gripper_range = f"{mn[5]:.1f}→{mx[5]:.1f}°"
        rows.append({
            "Skill":      family,
            "Ver":        version,
            "Episodes":   episodes,
            "Frames":     frames,
            "Duration":   f"{duration_s}s",
            "Gripper":    gripper_range,
        })
    return rows


def derive_skill_steps(events: list[dict]) -> list[dict]:
    seen: dict[int, str] = {}
    for e in events:
        sid = e.get("step_id")
        action = e.get("action", "")
        if sid is not None and sid >= 0 and action:
            seen.setdefault(sid, action)
    if not seen:
        return []
    return [
        {"step_id": sid, "action": act, "label": act.replace("_", " ").title()}
        for sid, act in sorted(seen.items())
    ]

CAMERA_LABELS = ["Follower — Top", "Follower — Side"]

def _get_slot_map() -> dict[int, int]:
    # index 1 → slot 0 (Follower — Top), index 2 → slot 1 (Follower — Side); never touch 0 (laptop)
    return {1: 0, 2: 1}

# Indices 1, 2 = external USB webcams (index 0 = laptop built-in, skipped).
CAMERA_INDICES = [1, 2]
CAMERA_LABELS  = ["Follower — Top", "Follower — Side"]

RESULT_COLOR = {
    "PASS":           "#00D4AA",
    "PATCHED":        "#FF8C00",
    "patched_success":"#FF8C00",  # orchestrator emits this before the shim normalises it
    "FAIL":           "#FF4444",
    "ABORT":          "#FF0000",
}

# the five phases of the ReAct agentic loop — shown in order across the UI
REACT_PHASES = [
    {"key": "PLAN",    "label": "Plan",    "desc": "Phi-3 compiles command → skill JSON"},
    {"key": "ACT",     "label": "Act",     "desc": "Arm executes motion step"},
    {"key": "OBSERVE", "label": "Observe", "desc": "VLM checks webcam frame"},
    {"key": "REFLECT", "label": "Reflect", "desc": "Classifier diagnoses failure"},
    {"key": "PATCH",   "label": "Patch",   "desc": "Orchestrator applies fix + retries"},
]


MJPEG_PORT = 8765


class _CameraServer:
    """MJPEG server — streams camera frames directly to the browser, bypassing Streamlit."""

    def __init__(self) -> None:
        self._lock              = threading.Lock()
        self._handles: list[cv2.VideoCapture | None] = []
        self._streaming_enabled = False
        self._health_check_thread = None
        self._health_check_stop = False
        self._start_http()
        self._start_health_check()

    # ── HTTP server ────────────────────────────────────────────────────────────
    def _start_http(self) -> None:
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                path = self.path.split("?")[0]
                if not (path.startswith("/cam") and path[4:].isdigit()):
                    self.send_response(404)
                    self.end_headers()
                    return
                idx = int(path[4:])
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while srv._streaming_enabled:
                    data = srv._next_jpeg(idx)
                    try:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                        )
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break
                    time.sleep(0.033)  # cap at ~30fps; also prevents flooding during blank-frame phase

        httpd = ThreadingHTTPServer(("127.0.0.1", MJPEG_PORT), _Handler)
        # allow fast restart without "address already in use" on page refresh
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # ── camera lifecycle ───────────────────────────────────────────────────────
    def open(self) -> None:
        with self._lock:
            self._streaming_enabled = True
            self._health_check_stop = False
        threading.Thread(target=self._open_sync, daemon=True).start()
        t = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread = t
        t.start()

    def _open_one(self, idx: int) -> cv2.VideoCapture | None:
        # try DSHOW (most reliable on Windows for USB webcams); never call while holding self._lock
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()  # verify frames actually flow before returning
            if ret:
                return cap
            cap.release()
        return None

    def _open_sync(self) -> None:
        slot_map = _get_slot_map()
        new_handles: list[cv2.VideoCapture | None] = [None, None]
        for dev_idx, slot in slot_map.items():
            cap = self._open_one(dev_idx)
            if cap:
                new_handles[slot] = cap
        with self._lock:
            if not self._streaming_enabled:  # close() fired while we were opening
                for cap in new_handles:
                    if cap:
                        cap.release()
                return
            for cap in self._handles:
                if cap:
                    cap.release()
            self._handles = new_handles

    def close(self) -> None:
        with self._lock:
            self._streaming_enabled = False
            self._health_check_stop = True
            for cap in self._handles:
                if cap:
                    cap.release()
            self._handles = []

    def _health_check_loop(self) -> None:
        while not self._health_check_stop:
            time.sleep(0.5)
            if not self._streaming_enabled or self._health_check_stop:
                continue
            with self._lock:
                empty_slots = {i for i, c in enumerate(self._handles) if c is None}
            if not empty_slots:
                continue
            slot_map = _get_slot_map()
            for dev_idx, slot in slot_map.items():
                if slot not in empty_slots or not self._streaming_enabled:
                    continue
                new_cap = self._open_one(dev_idx)
                with self._lock:
                    if (self._streaming_enabled
                            and slot < len(self._handles)
                            and self._handles[slot] is None):
                        self._handles[slot] = new_cap
                    elif new_cap:
                        new_cap.release()

    def _start_health_check(self) -> None:
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()

    # ── frame production ───────────────────────────────────────────────────────
    def _next_jpeg(self, slot: int) -> bytes:
        # hold lock for full read so close() cannot release the cap mid-read
        with self._lock:
            if not self._streaming_enabled or slot >= len(self._handles):
                return self._placeholder()
            cap = self._handles[slot]
            if cap is None:
                return self._placeholder()
            ret, frame_bgr = cap.read()
            if not ret:
                cap.release()
                self._handles[slot] = None
                return self._placeholder()
        frame_small = cv2.resize(frame_bgr, (480, 360), interpolation=cv2.INTER_LINEAR)
        _, buf = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return buf.tobytes()

    def _placeholder(self) -> bytes:
        if not hasattr(self, "_placeholder_cache"):
            img = np.full((360, 480, 3), 28, dtype=np.uint8)
            text, font, scale, thick = "Not connected", cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.putText(img, text, ((480 - tw) // 2, (360 + th) // 2),
                        font, scale, (80, 80, 80), thick, cv2.LINE_AA)
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            self._placeholder_cache = buf.tobytes()
        return self._placeholder_cache


@st.cache_resource
def _camera_server() -> _CameraServer:
    return _CameraServer()


def load_trace() -> list[dict]:
    if not TRACE_FILE.exists():
        return []
    events = []
    with open(TRACE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


def load_patches() -> dict:
    if not PATCHES_FILE.exists():
        return {}
    with open(PATCHES_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def get_active_react_phases(events: list[dict]) -> set[str]:
    # determines which ReAct phases to highlight based on the most recent event
    if not events:
        return {"PLAN"}
    last = events[-1]
    result = last.get("result", "")
    if result == "FAIL":
        return {"OBSERVE", "REFLECT"}
    if result in ("PATCHED",):
        return {"REFLECT", "PATCH"}
    if result == "PASS" and last.get("retry"):
        return {"PATCH"}
    if result == "PASS":
        return {"ACT", "OBSERVE"}
    if result == "ABORT":
        return {"REFLECT"}
    return {"PLAN"}


st.set_page_config(page_title="SkillPatch", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif !important;
    }
    code, pre, .stCode { font-family: 'JetBrains Mono', monospace !important; }

    .stDeployButton { display: none !important; }
    #MainMenu       { display: none !important; }
    footer          { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }
    /* tighten all st.markdown vertical rhythm */
    .element-container { margin-bottom: 0 !important; }
    /* reduce plotly chart top whitespace */
    .js-plotly-plot { margin-top: -6px !important; }

    /* ── cards ── */
    .step-card {
        background: #161B27; border: 1px solid #2D3748;
        border-radius: 8px; padding: 8px 6px;
        text-align: center; font-size: 0.76rem; line-height: 1.3;
        transition: border-color 0.25s, box-shadow 0.25s;
    }
    .step-card.active {
        border-color: #ED1C24; box-shadow: 0 0 10px #ED1C2455;
        animation: pulse-border 1.4s ease-in-out infinite;
    }

    .react-card {
        background: #161B27; border: 1px solid #2D3748;
        border-radius: 8px; padding: 7px 5px; text-align: center; font-size: 0.73rem;
        transition: border-color 0.25s, background 0.25s;
    }
    .react-card.active {
        border-color: #ED1C24; background: #1f0f12;
        box-shadow: 0 0 8px #ED1C2444;
        animation: pulse-border 1.4s ease-in-out infinite;
    }

    .metric-card {
        background: #161B27; border: 1px solid #2D3748;
        border-radius: 8px; padding: 10px 14px;
    }

    /* ── pulsing border animation for active cards ── */
    @keyframes pulse-border {
        0%, 100% { box-shadow: 0 0 6px #ED1C2433; }
        50%       { box-shadow: 0 0 14px #ED1C24AA; }
    }

    /* ── compact streamlit metric widget ── */
    [data-testid="stMetric"] { padding: 6px 0 !important; }
    [data-testid="stMetricLabel"]  { font-size: 0.72rem !important; color: #8892A4 !important; }
    [data-testid="stMetricValue"]  { font-size: 1.05rem !important; font-weight: 700 !important; }

    /* ── dataframe compact ── */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    .dvn-scroller { max-height: 200px !important; }
</style>
""", unsafe_allow_html=True)

# ── header ─────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("""
<h1 style='margin:0; font-size:1.9rem; letter-spacing:-0.5px; font-family: Space Grotesk, sans-serif;'>
    <span style='color:#ED1C24;'>Skill</span>Patch
</h1>
<p style='color:#8892A4; margin:1px 0 0 0; font-size:0.85rem;'>
    Self-Healing Robot Execution Layer
</p>""", unsafe_allow_html=True)
with col_badge:
    st.markdown("""
<div style='text-align:right; padding-top:6px;'>
    <span style='background:#ED1C24; color:white; padding:3px 10px;
                 border-radius:4px; font-size:0.75rem; font-weight:600;'>
        AMD Ryzen AI NPU
    </span><br/>
    <span style='color:#8892A4; font-size:0.72rem;'>Phi-3-mini · LeRobot SO-100</span>
</div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2D3748; margin:0.2rem 0;'>", unsafe_allow_html=True)


# polls trace.jsonl + patches.json every 500ms and redraws only this fragment, not the full page
@st.fragment(run_every=0.5)
def live_dashboard() -> None:
    events  = load_trace()
    patches = load_patches()

    # ── simulation complete banner ─────────────────────────────────────────────
    # only show if the SIMULATION_COMPLETE event was written after this page session started —
    # prevents stale banner from appearing when the tab is refreshed between runs
    complete = next((e for e in events if e.get("type") == "SIMULATION_COMPLETE"), None)
    if complete and complete.get("timestamp", 0) > st.session_state.get("session_start", 0):
        runs      = complete.get("total_runs", "?")
        learned   = complete.get("patches_learned", len(patches))
        prevented = sum(1 for e in events if e.get("result") == "PASS" and e.get("patch_applied"))
        st.markdown(f"""
<div style='background:#0d2e1f; border:1.5px solid #00D4AA; border-radius:8px;
            padding:12px 18px; margin-bottom:10px;'>
    <span style='color:#00D4AA; font-weight:700; font-size:1rem;'>&#10003; Simulation Complete</span>
    &nbsp;&nbsp;
    <span style='color:#8892A4; font-size:0.85rem;'>
        {runs} runs &nbsp;·&nbsp; {learned} fixes learned &nbsp;·&nbsp; {prevented} failures prevented autonomously
    </span>
</div>""", unsafe_allow_html=True)

    # ── active skill ───────────────────────────────────────────────────────────
    current_skill = events[-1].get("skill", "stock_middle_shelf") if events else "stock_middle_shelf"
    st.markdown(f"""
<div style='background:#161B27; border:1px solid #2D3748; border-radius:8px;
            padding:10px 16px; margin-bottom:10px;'>
    <span style='color:#8892A4; font-size:0.75rem;'>ACTIVE SKILL</span>
    &nbsp;&nbsp;
    <code style='color:#F0F2F6; font-size:0.95rem;'>{current_skill}</code>
    &nbsp;&nbsp;
    <span style='color:#4A5568; font-size:0.78rem;'>compiled from natural language via Phi-3-mini</span>
</div>""", unsafe_allow_html=True)

    # ── ReAct agentic loop ────────────────────────────────────────────────────
    st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:0 0 4px 0;'>REACT AGENTIC LOOP</p>",
                unsafe_allow_html=True)
    active_phases = get_active_react_phases(events)
    phase_cols = st.columns(len(REACT_PHASES))
    for i, phase in enumerate(REACT_PHASES):
        is_active  = phase["key"] in active_phases
        card_class = "react-card active" if is_active else "react-card"
        color      = "#ED1C24" if is_active else "#4A5568"
        with phase_cols[i]:
            st.markdown(f"""
<div class="{card_class}">
    <div style='color:{color}; font-weight:700; font-size:0.85rem;'>{phase["label"]}</div>
    <div style='color:#8892A4; font-size:0.68rem; margin-top:2px;'>{phase["desc"]}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── execution pipeline ────────────────────────────────────────────────────
    st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:0 0 4px 0;'>EXECUTION PIPELINE</p>",
                unsafe_allow_html=True)

    latest_by_step: dict[int, dict] = {}
    for e in events:
        sid = e.get("step_id")
        if sid is not None:
            latest_by_step[sid] = e
    current_step = events[-1].get("step_id", -1) if events else -1

    skill_steps = derive_skill_steps(events)
    if not skill_steps:
        st.markdown("<div style='color:#4A5568; font-size:0.85rem; padding:8px 0;'>No steps yet.</div>",
                    unsafe_allow_html=True)
        skill_steps = []
    step_cols = st.columns(max(len(skill_steps), 1))
    for i, step in enumerate(skill_steps):
        sid = step["step_id"]
        ev  = latest_by_step.get(sid)
        if ev is None:
            icon, color, status = "○", "#4A5568", "waiting"
        else:
            r      = ev.get("result", "")
            icon   = {"PASS": "✓", "PATCHED": "⚡", "FAIL": "✗", "ABORT": "⊘"}.get(r, "…")
            color  = RESULT_COLOR.get(r, "#8892A4")
            status = r.lower()
        is_active  = sid == current_step and bool(events)
        card_class = "step-card active" if is_active else "step-card"
        with step_cols[i]:
            st.markdown(f"""
<div class="{card_class}">
    <div style='font-size:1.3rem; color:{color}; font-weight:700;'>{icon}</div>
    <div style='color:#F0F2F6; font-weight:600; margin-top:2px;'>{step["label"]}</div>
    <div style='color:{color}; font-size:0.7rem;'>{status}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── metrics row ───────────────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        if not events:
            label, note, color = "Idle", "run simulator to start", "#4A5568"
        else:
            latest = events[-1]
            result = latest.get("result", "—")
            ft     = latest.get("failure_type")
            label  = result + (f" → {ft}" if ft else "")
            note   = str(latest.get("patch_applied") or "no patch applied")
            color  = RESULT_COLOR.get(result, "#8892A4")
        st.markdown(f"""
<div class="metric-card">
    <div style='color:#8892A4; font-size:0.75rem;'>LAST EVENT</div>
    <div style='color:{color}; font-size:1.2rem; font-weight:700; margin-top:4px;'>{label}</div>
    <div style='color:#8892A4; font-size:0.72rem; margin-top:2px;'>{note}</div>
</div>""", unsafe_allow_html=True)

    with col_b:
        latency_events = [e for e in events if e.get("npu_latency_ms") is not None]
        if not latency_events:
            ms_str, speedup_str, color, backend = "— ms", "—× vs CPU", "#4A5568", "NPU"
        else:
            ms      = latency_events[-1]["npu_latency_ms"]
            backend = latency_events[-1].get("backend", "NPU")
            speedup = CPU_BASELINE_MS / ms
            color   = "#00D4AA" if ms < NPU_TARGET_MS else "#FF8C00"
            ms_str      = f"{ms:.1f} ms"
            speedup_str = f"{speedup:.1f}× faster than CPU"
        st.markdown(f"""
<div class="metric-card">
    <div style='color:#8892A4; font-size:0.75rem;'>{backend} INFERENCE LATENCY</div>
    <div style='color:{color}; font-size:1.2rem; font-weight:700; margin-top:4px;'>{ms_str}</div>
    <div style='color:#ED1C24; font-size:0.78rem; font-weight:600; margin-top:2px;'>{speedup_str}</div>
</div>""", unsafe_allow_html=True)

    with col_c:
        passes  = sum(1 for e in events if e.get("result") == "PASS")
        patched = sum(1 for e in events if e.get("result") == "PATCHED")
        fails   = sum(1 for e in events if e.get("result") == "FAIL")
        st.markdown(f"""
<div class="metric-card">
    <div style='color:#8892A4; font-size:0.75rem;'>RUN SUMMARY</div>
    <div style='margin-top:6px;'>
        <span style='color:#00D4AA; font-weight:700;'>{passes} pass</span> &nbsp;·&nbsp;
        <span style='color:#FF8C00; font-weight:700;'>{patched} patched</span> &nbsp;·&nbsp;
        <span style='color:#FF4444; font-weight:700;'>{fails} fail</span>
    </div>
    <div style='color:#8892A4; font-size:0.72rem; margin-top:2px;'>{len(events)} total events</div>
</div>""", unsafe_allow_html=True)

    with col_d:
        # PASS events with a patch_applied means a known failure was pre-empted this run
        prevented = sum(1 for e in events if e.get("result") == "PASS" and e.get("patch_applied"))
        cost_saved = prevented * 75  # estimated $75 per avoided engineer intervention (30min @ $150/hr)
        patch_count = len(patches)
        st.markdown(f"""
<div class="metric-card">
    <div style='color:#8892A4; font-size:0.75rem;'>PATCH MEMORY</div>
    <div style='color:#00D4AA; font-size:1.2rem; font-weight:700; margin-top:4px;'>
        {prevented} failures prevented
    </div>
    <div style='color:#8892A4; font-size:0.72rem; margin-top:2px;'>
        {patch_count} fixes stored &nbsp;·&nbsp; est. ${cost_saved} saved
    </div>
</div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)

    # ── NPU latency charts ────────────────────────────────────────────────────
    latency_events = [e for e in events if e.get("npu_latency_ms") is not None]
    col_bar, col_trend = st.columns(2)

    with col_bar:
        live_ms   = latency_events[-1]["npu_latency_ms"] if latency_events else NPU_TARGET_MS
        bar_color = "#00D4AA" if live_ms < NPU_TARGET_MS else "#FF8C00"
        fig = go.Figure(go.Bar(
            x=[live_ms, CPU_BASELINE_MS],
            y=["NPU (live)", "CPU only"],
            orientation="h",
            marker_color=[bar_color, "#4A5568"],
            text=[f"{live_ms:.1f} ms", f"{CPU_BASELINE_MS:.0f} ms"],
            textposition="inside",
            textfont=dict(color="#FFFFFF", size=12),
        ))
        fig.update_layout(
            title=dict(text="NPU vs CPU — real-time threshold is 120 ms", font=dict(color="#8892A4", size=12)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, CPU_BASELINE_MS * 1.1], color="#8892A4",
                       showgrid=True, gridcolor="#2D3748"),
            yaxis=dict(color="#F0F2F6"),
            height=140, margin=dict(l=0, r=10, t=28, b=0),
        )
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})

    with col_trend:
        if len(latency_events) > 1:
            recent = [e["npu_latency_ms"] for e in latency_events[-40:]]
            fig2 = go.Figure(go.Scatter(
                y=recent, mode="lines",
                line=dict(color="#ED1C24", width=2),
                fill="tozeroy", fillcolor="rgba(237,28,36,0.1)",
            ))
            fig2.add_hline(y=NPU_TARGET_MS,
                           line=dict(color="#FF8C00", dash="dash", width=1),
                           annotation_text="120ms target",
                           annotation_font_color="#FF8C00")
            fig2.update_layout(
                title=dict(text="NPU latency trend (last 40 inferences)", font=dict(color="#8892A4", size=12)),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#8892A4", showgrid=False),
                yaxis=dict(color="#8892A4", showgrid=True, gridcolor="#2D3748"),
                height=140, margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
            )
            st.plotly_chart(fig2, width='stretch', config={"displayModeBar": False})
        else:
            st.markdown(
                "<div style='color:#4A5568; font-size:0.85rem; padding:20px 0; text-align:center;'>"
                "Trend appears after first inference</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2D3748; margin:0.2rem 0;'>", unsafe_allow_html=True)

    # ── execution trace ───────────────────────────────────────────────────────
    st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:0 0 3px 0;'>EXECUTION TRACES</p>", unsafe_allow_html=True)
    skill_events = [e for e in events if e.get("type") != "SIMULATION_COMPLETE"]
    if not skill_events:
        st.markdown("<div style='color:#4A5568; padding:16px 0;'>No events yet — "
                    "run <code>python scripts/simulate_failure.py</code></div>",
                    unsafe_allow_html=True)
    else:
        rows = []
        for e in reversed(skill_events[-60:]):
            rows.append({
                "Timestamp": datetime.fromtimestamp(e.get("timestamp", 0)).strftime("%I:%M:%S %p") + " EST",
                "Step":    e.get("step_id", "—"),
                "Action":  e.get("action", "—"),
                "Result":  e.get("result", "—"),
                "Failure": e.get("failure_type") or "—",
                "Patch":   str(e.get("patch_applied") or "—"),
                "NPU ms":  f"{e['npu_latency_ms']:.1f}" if e.get("npu_latency_ms") else "—",
            })

        def _color(val: str) -> str:
            return {"PASS": "color: #00D4AA", "PATCHED": "color: #FF8C00",
                    "patched_success": "color: #FF8C00",
                    "FAIL": "color: #FF4444", "ABORT": "color: #FF0000"}.get(val, "")

        st.dataframe(pd.DataFrame(rows).style.map(_color, subset=["Result"]),
                     use_container_width=True, hide_index=True, height=190)

    st.markdown("<hr style='border-color:#2D3748; margin:0.2rem 0;'>", unsafe_allow_html=True)

    # ── patch memory ──────────────────────────────────────────────────────────
    st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:0 0 3px 0;'>PATCH MEMORY</p>", unsafe_allow_html=True)
    if not patches:
        st.markdown("<div style='color:#4A5568; padding:10px 0;'>No patches yet. "
                    "The first failure will populate this table.</div>", unsafe_allow_html=True)
    else:
        rows = []
        for key, val in patches.items():
            skill, failure = key.split(":", 1) if ":" in key else (key, "—")
            delta = {k: v for k, v in val.items() if k not in ("applied_count", "last_applied")}
            rows.append({
                "Skill":           skill,
                "Failure Type":    failure,
                "Parameter Fix":   str(delta),
                "Times Applied":   val.get("applied_count", 0),
                "Last Applied":    val.get("last_applied", "—"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<hr style='border-color:#2D3748; margin:0.2rem 0;'>", unsafe_allow_html=True)

    # ── training archive ──────────────────────────────────────────────────────
    st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:0 0 3px 0;'>TRAINING ARCHIVE</p>", unsafe_allow_html=True)
    archive_rows = load_archive_stats()
    if not archive_rows:
        st.markdown("<div style='color:#4A5568; padding:10px 0;'>Archive/ not found.</div>",
                    unsafe_allow_html=True)
    else:
        total_eps    = sum(r["Episodes"] for r in archive_rows)
        total_frames = sum(r["Frames"]   for r in archive_rows)
        col_e, col_f, col_s = st.columns(3)
        col_e.metric("Total Episodes",  total_eps)
        col_f.metric("Total Frames",    f"{total_frames:,}")
        col_s.metric("Skills Recorded", len({r["Skill"] for r in archive_rows}))
        st.dataframe(pd.DataFrame(archive_rows), use_container_width=True, hide_index=True, height=175)


@st.fragment(run_every=0.5)
def camera_feeds() -> None:
    srv = _camera_server()

    cam_header, cam_toggle = st.columns([5, 1])
    with cam_header:
        st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:6px 0 4px 0;'>LIVE CAMERA FEEDS</p>",
                    unsafe_allow_html=True)
    with cam_toggle:
        cameras_on = st.toggle("Enable Camera", key="cameras_enabled", value=False)

    # open/close only on toggle edge; cache-bust token forces browser fresh HTTP connection on re-enable
    was_on = st.session_state.get("_cam_was_on", False)
    if cameras_on and not was_on:
        srv.open()
        st.session_state["_cam_token"] = int(time.time())
    elif not cameras_on and was_on:
        srv.close()
    st.session_state["_cam_was_on"] = cameras_on

    if cameras_on:
        token = st.session_state.get("_cam_token", 0)
        cam_cols = st.columns(2)
        for i, label in enumerate(CAMERA_LABELS):
            with cam_cols[i]:
                st.markdown(
                    f"<img src='http://localhost:{MJPEG_PORT}/cam{i}?t={token}' "
                    f"style='width:100%;border-radius:8px;display:block;' alt='{label}'>"
                    f"<p style='color:#8892A4;font-size:0.75rem;text-align:center;margin:4px 0 0;'>"
                    f"{label}</p>",
                    unsafe_allow_html=True,
                )
    else:
        st.markdown(
            "<div style='background:#161B27;border:1px dashed #2D3748;border-radius:8px;"
            "padding:16px;text-align:center;color:#4A5568;font-size:0.8rem;'>"
            "Toggle <b style='color:#8892A4;'>Enable Camera</b> above to activate AMD webcams"
            "</div>",
            unsafe_allow_html=True,
        )


if "session_start" not in st.session_state:
    st.session_state.session_start = datetime.now().timestamp()

camera_feeds()
live_dashboard()
