# PatchWork -- Self-Healing Robot Execution Layer

**PatchWork** is a real-time failure detection and skill correction system for the WOWROBO SO-ARM101 robotic arm, built at StarkHacks 2026. It wraps LeRobot skill execution with a 7-layer pipeline that detects failures mid-task, classifies the cause, applies a learned patch, and retries. Operators monitor the system through a live dashboard and can intervene or re-trigger skills at any point.

---

## Hardware

- **Robot:** WOWROBO SO-ARM101 dual-arm system -- one leader arm for teleoperation and data collection, one follower arm for autonomous execution (6-DOF each, 30 fps telemetry)
- **Cameras:** Two AMD Blue USB webcams (640x480, MJPEG, 30 fps)
- **Computing:** AMD Ryzen AI laptop with NPU and ROCm GPU for on-device VLM and LLM inference

---

## Architecture -- 7 Layers

| Layer | Module | Role | Backend |
|---|---|---|---|
| Input | `stt.py` | Voice command to text | faster-whisper (local) |
| 1 | `compiler.py` | Natural language to skill JSON | Phi-3-mini via llama-cpp |
| 2 | `robot_api.py` | Skill execution on follower arm | LeRobot CLI |
| 3 | `vlm_api.py` | Scene verification after each step | Moondream2 via Ollama (ROCm) |
| 4a | `classifier.py` | Rule-based failure classification | Always active |
| 4b | `patch_library.py` | Persistent patch store | patches.json |
| 5 | `orchestrator.py` | Central retry and correction loop | Always active |
| 6 | `trace_logger.py` | Execution trace to trace.jsonl | Always active |
| 7 | `audio_feedback.py` | TTS spoken status cues | ElevenLabs API |

Every layer has a real backend and a mock fallback for running end-to-end without hardware. Layer 1 uses **Phi-3-mini** (Microsoft SLM) running fully on-device via AMD ROCm.

---

## Dashboard

`dashboard.py` is a Streamlit app at `localhost:8501` showing live camera feeds from both AMD webcams, robot connection status, skill execution controls, and real-time joint telemetry.

`ArmDetector` (`python/tinyvla_debugger/arm_detector.py`) runs on every frame using OpenCV. It uses three-cue fusion to confirm the SO-ARM101 is in frame: a white HSV body mask, rectangular black servo blob detection, and tri-color wire adjacency (red pixels touching black pixels, confirming the servo cable bundle). All three cues must pass or the detector returns None and no overlay is drawn.

---

## Training Data (Archive/)

51 episodes across 9 skill datasets -- 34,047 frames and approximately 19 minutes of real SO-ARM101 telemetry at 30 fps, collected via leader-arm teleoperation and stored as float32 Parquet (6-DOF joint positions).

| Skill | Episodes | Frames |
|---|---|---|
| full_pick_and_place_v1 | 3 | 5,381 |
| pick_object_v1/v2/v3/v5 | 7 each | ~4,176 each |
| place_in_box_v2/v3/v4 | 5 each | ~2,990 each |
| place_in_shelf | 5 | 2,993 |

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run dashboard.py          # live operator dashboard
python run.py                       # full pipeline (hardware required)
python scripts/simulate_failure.py  # end-to-end demo without hardware
```

Set `ELEVENLABS_API_KEY` for audio feedback. For AMD ROCm GPU inference, install `onnxruntime-rocm` and `llama-cpp-python` with HIP support separately. See `requirements.txt` for full notes.

---

## Team

Built at StarkHacks 2026, targeting AMD Pervasive AI, Microsoft Azure, robotics, and startup prize tracks.
