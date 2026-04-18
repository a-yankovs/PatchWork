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
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

TRACE_FILE   = Path("trace.jsonl")
PATCHES_FILE = Path("patches.json")

NPU_TARGET_MS   = 120.0
CPU_BASELINE_MS = 820.0

SKILL_STEPS = [
    {"step_id": 0, "action": "scan_shelf",      "label": "Scan Shelf"},
    {"step_id": 1, "action": "pick_from_box",   "label": "Pick"},
    {"step_id": 2, "action": "place_slot_1",    "label": "Place Slot 1"},
    {"step_id": 3, "action": "place_slot_2",    "label": "Place Slot 2"},
    {"step_id": 4, "action": "check_box_empty", "label": "Check Done"},
]

# Index 0 = laptop built-in (testing). Swap to [1, 2] when AMD USB cams are connected.
CAMERA_INDICES = [0, 1]
CAMERA_LABELS  = ["Follower — Top", "Follower — Side"]

RESULT_COLOR = {
    "PASS":    "#00D4AA",
    "PATCHED": "#FF8C00",
    "FAIL":    "#FF4444",
    "ABORT":   "#FF0000",
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
        self._lock       = threading.Lock()
        self._handles:    list[cv2.VideoCapture | None] = []
        self._prev_grays: list[np.ndarray | None]       = []
        self._bg_subs:    list                          = []
        self._flow_on    = False
        self._start_http()

    # ── HTTP server ────────────────────────────────────────────────────────────
    def _start_http(self) -> None:
        srv = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_GET(self) -> None:
                if not (self.path.startswith("/cam") and self.path[4:].isdigit()):
                    self.send_response(404)
                    self.end_headers()
                    return
                idx = int(self.path[4:])
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while True:
                    data = srv._next_jpeg(idx)
                    if data is None:
                        time.sleep(0.033)
                        continue
                    try:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                        )
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        break

        httpd = ThreadingHTTPServer(("127.0.0.1", MJPEG_PORT), _Handler)
        # allow fast restart without "address already in use" on page refresh
        httpd.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()

    # ── camera lifecycle ───────────────────────────────────────────────────────
    def open(self) -> None:
        with self._lock:
            for cap in self._handles:
                if cap:
                    cap.release()
            handles, prev_grays, bg_subs = [], [], []
            for idx in CAMERA_INDICES:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    bg = cv2.createBackgroundSubtractorMOG2(
                        history=50, varThreshold=30, detectShadows=False
                    )
                    # pre-seed background with first still frame so arm detection is instant
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        for _ in range(50):
                            bg.apply(gray)
                    handles.append(cap)
                    bg_subs.append(bg)
                else:
                    cap.release()
                    handles.append(None)
                    bg_subs.append(None)
                prev_grays.append(None)
            self._handles    = handles
            self._prev_grays = prev_grays
            self._bg_subs    = bg_subs

    def close(self) -> None:
        with self._lock:
            for cap in self._handles:
                if cap:
                    cap.release()
            self._handles, self._prev_grays, self._bg_subs = [], [], []

    def set_flow(self, on: bool) -> None:
        self._flow_on = on

    # ── frame production ───────────────────────────────────────────────────────
    def _next_jpeg(self, idx: int) -> bytes | None:
        with self._lock:
            if idx >= len(self._handles):
                return None
            cap  = self._handles[idx]
            bg   = self._bg_subs[idx]    if idx < len(self._bg_subs)    else None
            prev = self._prev_grays[idx] if idx < len(self._prev_grays) else None
        if cap is None or not cap.isOpened():
            return None
        ret, frame_bgr = cap.read()
        if not ret:
            return None
        if self._flow_on and bg is not None:
            frame_rgb           = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb, new_prev = _overlay_flow(frame_rgb, prev, bg)
            with self._lock:
                if idx < len(self._prev_grays):
                    self._prev_grays[idx] = new_prev
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buf.tobytes()


@st.cache_resource
def _camera_server() -> _CameraServer:
    return _CameraServer()


# MediaPipe selfie segmentation — used to EXCLUDE human pixels from flow
# Falls back gracefully if mediapipe isn't installed
try:
    import mediapipe as mp
    _mp_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
except Exception:
    _mp_selfie = None


def _person_mask(frame_rgb: np.ndarray) -> np.ndarray:
    """Returns 255 where a person is detected (pixels to exclude from arm flow)."""
    if _mp_selfie is None:
        return np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    result = _mp_selfie.process(frame_rgb)
    if result.segmentation_mask is None:
        return np.zeros(frame_rgb.shape[:2], dtype=np.uint8)
    return (result.segmentation_mask > 0.5).astype(np.uint8) * 255


# SO-ARM100 color profile in HSV — white/cream plastic body + black servo joints
_ARM_RANGES = [
    (np.array([0,   0, 170]), np.array([180, 55, 255])),  # white / off-white body
    (np.array([0,   0,   0]), np.array([180, 70,  70])),  # black servo joints
]


def _arm_color_mask(frame_rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in _ARM_RANGES:
        mask |= cv2.inRange(hsv, lo, hi)
    # dilate so objects held inside the gripper are included
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    return cv2.dilate(mask, kernel, iterations=2)


def _overlay_flow(
    frame_rgb: np.ndarray,
    prev_gray: np.ndarray | None,
    bg_sub: cv2.BackgroundSubtractorMOG2,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    out  = frame_rgb.copy()

    # motion mask: MOG2 detects what's moving
    fg_raw  = bg_sub.apply(gray)
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_raw, cv2.MORPH_OPEN, kernel5)

    # color mask: only SO-ARM100 colors (white body + black joints)
    arm_mask = _arm_color_mask(frame_rgb)

    # person mask: exclude any pixels MediaPipe identifies as human
    not_person = cv2.bitwise_not(_person_mask(frame_rgb))

    # combined: moving + arm-colored + not a person
    combined = cv2.bitwise_and(cv2.bitwise_and(fg_mask, arm_mask), not_person)

    if prev_gray is not None and prev_gray.shape == gray.shape:
        half_prev = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
        half_curr = cv2.resize(gray,      (0, 0), fx=0.5, fy=0.5)
        flow = cv2.calcOpticalFlowFarneback(
            half_prev, half_curr, None,
            pyr_scale=0.5, levels=2, winsize=12,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        step = 10
        h_half, w_half = half_curr.shape
        for y in range(step, h_half - step, step):
            for x in range(step, w_half - step, step):
                x0, y0 = x * 2, y * 2
                if combined[y0, x0] == 0:
                    continue
                fx, fy = flow[y, x]
                mag = fx * fx + fy * fy
                if mag > 2.25:
                    x1, y1    = int(x0 + fx * 5), int(y0 + fy * 5)
                    thickness = 2 if mag > 9.0 else 1
                    cv2.line(out, (x0, y0), (x1, y1), (0, 212, 170), thickness)

    return out, gray


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
<style>
    .stDeployButton { display: none !important; }
    #MainMenu       { display: none !important; }
    footer          { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .block-container { padding-top: 1.2rem !important; }

    .step-card {
        background: #161B27; border: 1px solid #2D3748;
        border-radius: 8px; padding: 10px 8px;
        text-align: center; font-size: 0.78rem; line-height: 1.4;
    }
    .step-card.active { border-color: #ED1C24; box-shadow: 0 0 8px #ED1C2466; }

    .react-card {
        background: #161B27; border: 1px solid #2D3748;
        border-radius: 8px; padding: 8px 6px; text-align: center; font-size: 0.75rem;
    }
    .react-card.active { border-color: #ED1C24; background: #1f0f12; box-shadow: 0 0 6px #ED1C2444; }

    .metric-card {
        background: #161B27; border: 1px solid #2D3748;
        border-radius: 8px; padding: 14px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── header ─────────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown("""
<h1 style='margin:0; font-size:2.2rem; letter-spacing:-0.5px;'>
    <span style='color:#ED1C24;'>Skill</span>Patch
</h1>
<p style='color:#8892A4; margin:2px 0 0 0; font-size:0.95rem;'>
    Self-Healing Robot Execution Layer &nbsp;·&nbsp; On-device · No cloud · No engineer
</p>""", unsafe_allow_html=True)
with col_badge:
    st.markdown("""
<div style='text-align:right; padding-top:6px;'>
    <span style='background:#ED1C24; color:white; padding:3px 10px;
                 border-radius:4px; font-size:0.75rem; font-weight:600;'>
        AMD Ryzen AI NPU
    </span><br/>
    <span style='color:#8892A4; font-size:0.72rem;'>Moondream2 INT4 · Phi-3-mini · LeRobot SO-100</span>
</div>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2D3748; margin:0.5rem 0;'>", unsafe_allow_html=True)


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

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── execution pipeline ────────────────────────────────────────────────────
    st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:0 0 4px 0;'>EXECUTION PIPELINE</p>",
                unsafe_allow_html=True)

    latest_by_step: dict[int, dict] = {}
    for e in events:
        sid = e.get("step_id")
        if sid is not None:
            latest_by_step[sid] = e
    current_step = events[-1].get("step_id", -1) if events else -1

    step_cols = st.columns(len(SKILL_STEPS))
    for i, step in enumerate(SKILL_STEPS):
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

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

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

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

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
            height=160, margin=dict(l=0, r=10, t=28, b=0),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
                height=160, margin=dict(l=0, r=0, t=28, b=0), showlegend=False,
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(
                "<div style='color:#4A5568; font-size:0.85rem; padding:55px 0; text-align:center;'>"
                "Trend appears after first inference</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#2D3748; margin:0.4rem 0;'>", unsafe_allow_html=True)

    # ── execution trace ───────────────────────────────────────────────────────
    st.markdown("#### Execution Trace")
    skill_events = [e for e in events if e.get("type") != "SIMULATION_COMPLETE"]
    if not skill_events:
        st.markdown("<div style='color:#4A5568; padding:16px 0;'>No events yet — "
                    "run <code>python scripts/simulate_failure.py</code></div>",
                    unsafe_allow_html=True)
    else:
        rows = []
        for e in reversed(skill_events[-60:]):
            rows.append({
                "Time":    datetime.fromtimestamp(e.get("timestamp", 0)).strftime("%H:%M:%S"),
                "Step":    e.get("step_id", "—"),
                "Action":  e.get("action", "—"),
                "Result":  e.get("result", "—"),
                "Failure": e.get("failure_type") or "—",
                "Patch":   str(e.get("patch_applied") or "—"),
                "NPU ms":  f"{e['npu_latency_ms']:.1f}" if e.get("npu_latency_ms") else "—",
            })

        def _color(val: str) -> str:
            return {"PASS": "color: #00D4AA", "PATCHED": "color: #FF8C00",
                    "FAIL": "color: #FF4444", "ABORT": "color: #FF0000"}.get(val, "")

        st.dataframe(pd.DataFrame(rows).style.map(_color, subset=["Result"]),
                     use_container_width=True, hide_index=True, height=250)

    st.markdown("<hr style='border-color:#2D3748; margin:0.4rem 0;'>", unsafe_allow_html=True)

    # ── patch memory ──────────────────────────────────────────────────────────
    st.markdown("#### Patch Memory")
    st.markdown("<p style='color:#8892A4; font-size:0.82rem; margin-top:-6px;'>"
                "Every row is a failure the system has diagnosed and will never repeat — "
                "the data flywheel that makes each deployment more reliable over time.</p>",
                unsafe_allow_html=True)
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


@st.fragment(run_every=0.5)
def camera_feeds() -> None:
    srv = _camera_server()

    cam_header, col_flow, cam_toggle = st.columns([4, 1, 1])
    with cam_header:
        st.markdown("<p style='color:#8892A4; font-size:0.78rem; margin:6px 0 4px 0;'>LIVE CAMERA FEEDS</p>",
                    unsafe_allow_html=True)
    with cam_toggle:
        cameras_on = st.toggle("Enable Camera", key="cameras_enabled", value=False)
    with col_flow:
        flow_on = st.toggle("Motion", key="flow_enabled", value=False, disabled=not cameras_on)
    if not cameras_on:
        flow_on = False

    # open/close cameras only on toggle transitions
    was_on = st.session_state.get("_cam_was_on", False)
    if cameras_on and not was_on:
        srv.open()
    elif not cameras_on and was_on:
        srv.close()
    st.session_state["_cam_was_on"] = cameras_on
    srv.set_flow(flow_on)

    if cameras_on:
        cam_cols = st.columns(2)
        for i, label in enumerate(CAMERA_LABELS):
            with cam_cols[i]:
                st.markdown(
                    f"<img src='http://localhost:{MJPEG_PORT}/cam{i}' "
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
