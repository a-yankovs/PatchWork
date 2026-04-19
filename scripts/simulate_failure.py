# The purpose of this script is to generate a realistic execution trace for dashboard testing
# without requiring the physical robot or VLM. It simulates the full PatchWork story:
# failure on first run, autonomous patch applied, clean runs on all subsequent attempts.

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

TRACE_FILE   = ROOT / "trace.jsonl"
PATCHES_FILE = ROOT / "patches.json"

SKILL = "stock_middle_shelf"

STEPS = [
    {"step_id": 0, "action": "scan_shelf"},
    {"step_id": 1, "action": "pick_from_box"},
    {"step_id": 2, "action": "place_slot_1"},
    {"step_id": 3, "action": "place_slot_2"},
    {"step_id": 4, "action": "check_box_empty"},
]

FAILURE_TYPES = {
    "pick_from_box": "GRASP_FAIL",
    "place_slot_1":  "PLACEMENT_MISS",
    "place_slot_2":  "PLACEMENT_MISS",
}

DEFAULT_PATCHES = {
    "GRASP_FAIL":          {"z_offset_mm": 5},          # lower the arm before gripping
    "PLACEMENT_MISS":      {"approach_angle_deg": 10},   # rotate wrist to realign with slot
    "PLACEMENT_COLLISION": {"speed_scale": 0.7},         # slow down to avoid knocking items
    "DROP_DURING_TRANSIT": {"gripper_close_force": 0.1}, # grip harder so item doesn't fall
}

DEFAULT_PARAMS = {
    "z_offset_mm": 0,
    "speed_scale": 1.0,
    "approach_angle_deg": 0,
    "gripper_close_force": 0.6,
}


def _npu_latency() -> float:
    # simulates realistic NPU inference time matching real hardware target (~94ms mean)
    return round(random.gauss(94, 7), 1)


def _write(event: dict) -> None:
    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    icon      = {"PASS": "  OK", "FAIL": "FAIL", "PATCHED": " FIX", "ABORT": "STOP"}.get(event["result"], "    ")
    patch_str = f"  patch={event['patch_applied']}" if event.get("patch_applied") else ""
    fail_str  = f"  cause={event['failure_type']}"  if event.get("failure_type")  else ""
    print(f"  [{icon}] step {event['step_id']} {event['action']:<20}"
          f"  {event['npu_latency_ms']:>6.1f}ms{fail_str}{patch_str}")


def _load_patches() -> dict:
    if not PATCHES_FILE.exists():
        return {}
    with open(PATCHES_FILE) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _save_patch(failure_type: str, patch: dict) -> None:
    patches = _load_patches()
    key = f"{SKILL}:{failure_type}"
    if key in patches:
        patches[key]["applied_count"] += 1
        patches[key]["last_applied"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    else:
        patches[key] = {**patch, "applied_count": 1,
                        "last_applied": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(PATCHES_FILE, "w") as f:
        json.dump(patches, f, indent=2)


def run_scenario(inject_fail_at: int = -1, step_delay: float = 1.5, run_num: int = 1) -> None:
    # inject_fail_at is the step_id where a failure is forced; -1 means the run completes cleanly
    patches = _load_patches()
    params  = DEFAULT_PARAMS.copy()

    # bake in fixes from previous runs so the system doesn't repeat known failures
    for key, patch_data in patches.items():
        if key.startswith(f"{SKILL}:"):
            delta = {k: v for k, v in patch_data.items()
                     if k not in ("applied_count", "last_applied")}
            params.update(delta)

    pre_applied = {k: v for k, v in params.items() if v != DEFAULT_PARAMS.get(k)}
    print(f"\n{'─'*55}")
    print(f"  Run {run_num} — {SKILL}")
    print(f"  pre-applied: {pre_applied}" if pre_applied else "  no pre-applied patches")
    print(f"{'─'*55}")

    for step in STEPS:
        time.sleep(step_delay)
        sid       = step["step_id"]
        action    = step["action"]
        fail_type = FAILURE_TYPES.get(action)

        if sid == inject_fail_at and fail_type:
            _write({"timestamp": time.time(), "skill": SKILL, "step_id": sid,
                    "action": action, "result": "FAIL", "failure_type": fail_type,
                    "patch_applied": None, "npu_latency_ms": _npu_latency(),
                    "retry": False, "params_used": params.copy(), "backend": "NPU"})

            time.sleep(step_delay * 0.6)

            patch = DEFAULT_PATCHES[fail_type]
            params.update(patch)
            _save_patch(fail_type, patch)

            _write({"timestamp": time.time(), "skill": SKILL, "step_id": sid,
                    "action": action, "result": "PATCHED", "failure_type": fail_type,
                    "patch_applied": patch, "npu_latency_ms": _npu_latency(),
                    "retry": True, "params_used": params.copy(), "backend": "NPU"})
        else:
            _write({"timestamp": time.time(), "skill": SKILL, "step_id": sid,
                    "action": action, "result": "PASS", "failure_type": None,
                    "patch_applied": pre_applied if pre_applied else None,
                    "npu_latency_ms": _npu_latency(), "retry": False,
                    "params_used": params.copy(), "backend": "NPU"})

    print(f"\n  Skill complete.\n")


def run_demo(runs: int = 5, step_delay: float = 1.5) -> None:
    TRACE_FILE.unlink(missing_ok=True)
    PATCHES_FILE.unlink(missing_ok=True)

    fail_schedule = [1, 2] + [-1] * max(0, runs - 2)

    print("\nPatchWork trace simulator")
    print(f"Runs: {runs}   Step delay: {step_delay}s")
    print("Dashboard: python -m streamlit run dashboard.py\n")

    for i, fail_step in enumerate(fail_schedule[:runs]):
        run_scenario(inject_fail_at=fail_step, step_delay=step_delay, run_num=i + 1)
        if i < runs - 1:
            time.sleep(2)

    # write a completion marker so the dashboard can show the "Simulation Complete" banner
    learned = len(_load_patches())
    with open(TRACE_FILE, "a") as f:
        f.write(json.dumps({
            "type": "SIMULATION_COMPLETE",
            "timestamp": time.time(),
            "total_runs": runs,
            "patches_learned": learned,
        }) + "\n")

    print("Done. Events written to trace.jsonl.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatchWork trace simulator")
    parser.add_argument("--runs",   type=int,   default=5,   help="number of skill runs")
    parser.add_argument("--delay",  type=float, default=1.5, help="seconds between steps")
    parser.add_argument("--single", action="store_true",     help="one run, failure at step 1")
    args = parser.parse_args()

    if args.single:
        TRACE_FILE.unlink(missing_ok=True)
        PATCHES_FILE.unlink(missing_ok=True)
        run_scenario(inject_fail_at=1, step_delay=args.delay, run_num=1)
    else:
        run_demo(runs=args.runs, step_delay=args.delay)
