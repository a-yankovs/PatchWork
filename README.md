# PatchWork: Self-Healing Robot Execution Layer

**PatchWork** is a real-time failure detection and skill correction system for the LeRobot SO-101 robotic arm, built at StarkHacks 2026. It e that detects failures mid-task, classifies the cause, applies a learned patch, and retries, entirely on-device, with no cloud dependency.

🏆 Honorable Mention — Best Use of AMD Technology, StarkHacks 2026


## Hardware
- **Robot:** LeRobot SO-101 dual-arm system — one leader arm for teleoperation and demonstration, one follower arm for autonomous execution (6-DOF each)
- **Camera:** USB webcam (640x480, MJPEG) mounted to observe gripper and shelf simultaneously
- **Computing:** AMD Ryzen AI 9 HX 370 — Radeon 890M GPU for VLM inference via ROCm, XDNA 2 NPU (driver stack installed and validated)


## Architecture

| Layer | Module | Role | Backend |
|---|---|---|---|
| Input | `stt.py` | Voice command to text | faster-whisper (local) |
| 1 | `compiler.py` | Natural language to skill JSON | Phi-3-mini via llama-cpp |
| 2 | `robot_api.py` | Skill execution on follower arm | LeRobot record/replay + ACT policy |
| 3 | `vlm_api.py` | Scene verification after each step | Moondream2 via Ollama (ROCm) |
| 4a | `classifier.py` | Rule-based failure classification | Always active |
| 4b | `patch_library.py` | Persistent patch store | patches.json |
| 5 | `orchestrator.py` | Central retry and correction loop | asyncio |
| 6 | `trace_logger.py` | Execution trace to trace.jsonl | Always active |
| 7 | `audio_feedback.py` | TTS spoken status cues |

The low-level motion execution layer is implemented in C++ for deterministic 50Hz control. The Python orchestrator calls into this stack via the `robot_api.py` interface.

| File | Role |
|---|---|
| `common.cpp` | Shared data types — 3D vectors, poses, joint states, gripper states, error codes |
| `kinematics.cpp` | Forward and inverse kinematics using the SO-100's DH parameters |
| `trajectory.cpp` | Smooth trajectory generation between waypoints using trapezoidal velocity profiles and cubic spline interpolation |
| `motion_planner.cpp` | Skill dispatcher — maps skill names and patch parameters from the Python orchestrator to executable trajectories |
| `controller.cpp` | 50Hz PID control loop — tracks joint positions, handles gripper commands, supports emergency stop |
| `robot_interface.cpp` | Serial communication with the SO-100 servos via Dynamixel Protocol 2.0 — reads joint states, sends position commands, streams telemetry in a background thread |
 
The patch parameters from `patch_library.py` (`z_offset_mm`, `speed_scale`, `approach_angle_deg`, `gripper_close_force`) flow directly into `motion_planner.cpp`, which adjusts the trajectory before `controller.cpp` executes it. This is the bridge between the Python self-healing loop and the physical arm.

**VLM inference:** Moondream2 runs fully on the AMD Radeon 890M via ROCm — all 25 layers offloaded to GPU, no CPU fallback. Grammar-constrained to return deterministic (yes/no) JSON. Optimized from ~2400ms to ~200ms warm through a custom Ollama Modelfile, 3-token output limit, and image preprocessing (320x240, JPEG quality 50).

**Skill compiler:** Phi-3-mini (Microsoft) runs fully on-device via llama-cpp.

## Failure Types

| Failure Type | Description | Default Patch |
|---|---|---|
| `GRASP_FAIL` | Object not detected in gripper after pick | `z_offset_mm: +5` |
| `PLACEMENT_MISS` | Item not detected in target slot after place | `approach_angle_deg: +10` |
| `PLACEMENT_COLLISION` | Collision detected during placement | `speed_scale: 0.7` |
| `DROP_DURING_TRANSIT` | Item lost between pick and place | `gripper_close_force: +0.1` |

Patches are stored in `patches.json` keyed by `skill_name:failure_type` and pre-applied on subsequent runs.


## Dashboard

`dashboard.py` is a Streamlit app at `localhost:8501` showing:
- Live camera feed via dedicated MJPEG server (port 8765)
- ReAct agentic loop phase indicator (Plan → Act → Observe → Reflect → Patch)
- Step-by-step execution pipeline with pass/fail/patched state
- Live VLM inference latency vs CPU baseline
- Execution trace table (last 60 events)
- Patch memory table showing stored fixes and application count
- Camera scanner sidebar for device index identification

## Training Data

20 high-quality episodes of shelf-stocking teleoperation collected via leader-arm demonstration, stored as Parquet (6-DOF joint positions) with single-camera visual observations at 30fps. An ACT (Action Chunking Transformer) policy was trained on this data for learned motion execution alongside the record/replay fallback.

## NPU Status

Full AMD XDNA 2 NPU driver stack installed and validated on Ubuntu 24.04:
- XRT 2.21.75 installed via `.deb` packages
- `/dev/accel0` persistent via custom udev rule
- VitisAI Execution Provider confirmed in ONNX Runtime 1.23.3
- CNN model validated running on NPU


## Quickstart

```bash
pip install -r requirements.txt

# Ollama must be running with moondream-verify model
ollama serve &
ollama pull moondream
ollama create moondream-verify -f Modelfile

streamlit run dashboard.py                  # live operator dashboard
python run.py                               # full pipeline (hardware required)
python run.py --mock                        # full pipeline without hardware
python run.py --cmd "stock the shelf"       # skip voice input
python scripts/simulate_failure.py          # end-to-end demo without hardware
```

For AMD ROCm GPU inference, install PyTorch with ROCm 6.3 support separately. See `requirements.txt` for full notes.

## Repo Structure

```
python/          # core pipeline modules (compiler, orchestrator, vlm_api, etc.)
scripts/         # simulation and utility scripts
skillpatch_data/ # skill manifests and patch storage
configs/         # hardware and model configuration
data/            # training episodes
tests/           # unit and integration tests
dashboard.py     # Streamlit operator dashboard
run.py           # pipeline entry point
```


## Team

**Alexandra Yankovskaya, Vera Chuang, Diya Bengani, Elias Assalif**
Built at StarkHacks 2026
