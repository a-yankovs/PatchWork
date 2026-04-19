"""
record_episode.py — Add training episodes to datasets that keep aborting.

After a skill run aborts, the orchestrator needs at least 2 recorded episodes
per dataset to use episode-based retry (attempt 0 → episode 0, attempt 1 →
episode 1). This script closes that loop by reading trace.jsonl for recent
aborts and launching lerobot-record to append one more good demonstration to
the failing dataset.

Usage — record episode for the last aborted action automatically:
    python record_episode.py

Usage — record episode for a specific action:
    python record_episode.py --action place_in_shelf
    python record_episode.py --action pick_object --num-episodes 2

Usage — just show recent aborts without recording:
    python record_episode.py --dry-run

Usage — show how many episodes each dataset currently has:
    python record_episode.py --status
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — must match robot_api.py and main.py defaults
# ---------------------------------------------------------------------------

STORAGE_DIR  = os.environ.get("ROBOT_STORAGE_DIR", "./skillpatch_data")
ROBOT_PORT   = os.environ.get("ROBOT_PORT",  "/dev/ttyACM1")
TELEOP_PORT  = os.environ.get("TELEOP_PORT", "/dev/ttyACM2")
ROBOT_ID     = os.environ.get("ROBOT_ID",    "follower_arm")
TELEOP_ID    = os.environ.get("TELEOP_ID",   "leader_arm")

# Camera config — indices must match your physical setup (see webcam_stream.py)
CAM_HIGH_INDEX = int(os.environ.get("CAM_HIGH_INDEX", "2"))   # overview / stationary
CAM_LOW_INDEX  = int(os.environ.get("CAM_LOW_INDEX",  "3"))   # follower side
CAM_WIDTH      = int(os.environ.get("CAM_WIDTH",  "640"))
CAM_HEIGHT     = int(os.environ.get("CAM_HEIGHT", "480"))
CAM_FPS        = int(os.environ.get("CAM_FPS",    "30"))

TRACE_FILE   = Path(STORAGE_DIR) / "trace.jsonl"
SKILLS_DIR   = Path(STORAGE_DIR) / "skills"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_trace(n: int = 200) -> list[dict]:
    if not TRACE_FILE.exists():
        return []
    events = []
    for line in TRACE_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return list(reversed(events))[:n]


def _find_recent_aborts(events: list[dict]) -> list[dict]:
    """Return abort events, newest first, deduplicated by action."""
    seen = set()
    aborts = []
    for e in events:
        if e.get("result") == "abort" and e.get("action"):
            action = e["action"]
            if action not in seen:
                seen.add(action)
                aborts.append(e)
    return aborts


def _load_manifest(action: str) -> dict | None:
    path = SKILLS_DIR / f"{action}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _episode_count(dataset_repo_id: str) -> int:
    """Count how many episodes are recorded in a local dataset."""
    repo_path = Path(dataset_repo_id)
    if not repo_path.exists():
        return 0
    # LeRobot stores episodes in data/chunk-*/episode_*.parquet
    # Count unique episode indices from file names.
    episodes = set()
    for f in repo_path.rglob("episode_*.parquet"):
        # e.g. episode_000000.parquet → 0
        try:
            idx = int(f.stem.split("_")[-1])
            episodes.add(idx)
        except ValueError:
            pass
    # Fallback: check meta/episodes.jsonl
    if not episodes:
        meta = repo_path / "meta" / "episodes.jsonl"
        if meta.exists():
            for line in meta.read_text(encoding="utf-8").splitlines():
                try:
                    episodes.add(json.loads(line)["episode_index"])
                except Exception:
                    pass
    return len(episodes)


def _dataset_path_for_action(action: str) -> str | None:
    """Return the dataset_repo_id for the first step of an action's manifest."""
    manifest = _load_manifest(action)
    if not manifest or not manifest.get("steps"):
        return None
    return manifest["steps"][0]["dataset_repo_id"]


def _record_episode(
    dataset_repo_id: str,
    action: str,
    num_episodes: int,
    episode_time_s: int,
    reset_time_s: int,
    dry_run: bool,
) -> None:
    """Launch lerobot-record to append `num_episodes` to the dataset."""
    repo_path = Path(dataset_repo_id)
    if repo_path.exists():
        repo_id = repo_path.name
        root    = str(repo_path)
    else:
        repo_id = dataset_repo_id
        root    = None

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={ROBOT_PORT}",
        f"--robot.id={ROBOT_ID}",
        "--teleop.type=so101_leader",
        f"--teleop.port={TELEOP_PORT}",
        f"--teleop.id={TELEOP_ID}",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.single_task={action}",
        "--dataset.push_to_hub=false",
        f"--dataset.num_episodes={num_episodes}",
        f"--dataset.episode_time_s={episode_time_s}",
        f"--dataset.reset_time_s={reset_time_s}",
        # Camera config — records MP4 alongside parquet (Dict format required)
        "--robot.cameras",
        (
            f'{{"cam_high": {{"type": "opencv", "index_or_path": {CAM_HIGH_INDEX}, '
            f'"width": {CAM_WIDTH}, "height": {CAM_HEIGHT}, "fps": {CAM_FPS}}}, '
            f'"cam_low": {{"type": "opencv", "index_or_path": {CAM_LOW_INDEX}, '
            f'"width": {CAM_WIDTH}, "height": {CAM_HEIGHT}, "fps": {CAM_FPS}}}}}'
        ),
    ]
    if root is not None:
        cmd += [f"--dataset.root={root}"]

    print(f"\n   Command: {' '.join(cmd)}\n")

    if dry_run:
        print("   [dry-run] Skipping lerobot-record — pass without --dry-run to record.")
        return

    try:
        subprocess.run(cmd, check=True)
        new_count = _episode_count(dataset_repo_id)
        print(f"\n✓  Done — {dataset_repo_id} now has {new_count} episode(s).")
        print("   The next retry run will use the new episode automatically.")
    except FileNotFoundError:
        print("✗  lerobot-record not found. Is the lerobot venv active?", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"✗  lerobot-record failed (exit {exc.returncode}).", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_status() -> None:
    """Print episode counts for all known datasets."""
    print(f"\nDataset status (reading from {SKILLS_DIR}):\n")
    seen = {}
    for manifest_path in sorted(SKILLS_DIR.glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for step in manifest.get("steps", []):
            rid = step["dataset_repo_id"]
            if rid in seen:
                continue
            count = _episode_count(rid)
            seen[rid] = count
            retries_ok = count >= 2
            icon = "✓" if retries_ok else "⚠ "
            note = "" if retries_ok else "  ← needs ≥2 for retry to work"
            print(f"  {icon}  {rid:40s}  {count} episode(s){note}")
    if not seen:
        print("  (no manifests found)")
    print()


def cmd_record(
    action: str | None,
    num_episodes: int,
    episode_time_s: int,
    reset_time_s: int,
    dry_run: bool,
) -> None:
    """Record episodes for the specified action (or the last aborted one)."""
    if action is None:
        # Auto-detect from trace
        events = _read_trace()
        aborts = _find_recent_aborts(events)
        if not aborts:
            print("No recent aborts found in trace.jsonl.")
            print("Either the last run succeeded, or trace.jsonl doesn't exist yet.")
            print("Use --action <name> to specify the action manually.")
            sys.exit(0)
        abort = aborts[0]
        action = abort["action"]
        failure_type = abort.get("failure_type", "unknown")
        import datetime
        ts = datetime.datetime.fromtimestamp(abort["timestamp"]).strftime("%H:%M:%S")
        print(f"\nLast abort: action={action!r}  failure={failure_type}  at {ts}")
    else:
        print(f"\nAction specified: {action!r}")

    dataset_repo_id = _dataset_path_for_action(action)
    if dataset_repo_id is None:
        print(f"✗  No manifest found for action {action!r} in {SKILLS_DIR}.", file=sys.stderr)
        print("   Check --action matches a file in skillpatch_data/skills/", file=sys.stderr)
        sys.exit(1)

    current_count = _episode_count(dataset_repo_id)
    target_count  = current_count + num_episodes

    print(f"   Dataset  : {dataset_repo_id}")
    print(f"   Episodes : {current_count} now → {target_count} after recording")
    print(f"   Duration : {episode_time_s}s per episode, {reset_time_s}s reset between")

    if current_count >= 2 and not dry_run:
        print(f"\n   ℹ  Dataset already has {current_count} episodes (retry is already possible).")
        ans = input("   Record anyway? [y/N] ").strip().lower()
        if ans != "y":
            print("   Skipped.")
            return

    print(f"\n▶  Recording {num_episodes} episode(s) for '{action}'...")
    print("   Position the arm at the START position, then press Enter in the lerobot prompt.")

    _record_episode(
        dataset_repo_id=dataset_repo_id,
        action=action,
        num_episodes=num_episodes,
        episode_time_s=episode_time_s,
        reset_time_s=reset_time_s,
        dry_run=dry_run,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="record_episode",
        description="Add training episodes to datasets that keep aborting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--action", default=None,
        help="Action name to record for (e.g. 'place_in_shelf'). "
             "Defaults to the most recently aborted action in trace.jsonl.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=1, metavar="N",
        help="Number of new episodes to record (default: 1)",
    )
    parser.add_argument(
        "--episode-time", type=int, default=40, metavar="SECS",
        help="Seconds to record each episode (default: 40)",
    )
    parser.add_argument(
        "--reset-time", type=int, default=10, metavar="SECS",
        help="Seconds between episodes for arm reset (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be recorded without actually running lerobot-record",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show episode counts for all datasets and exit",
    )
    parser.add_argument(
        "--storage-dir", default=STORAGE_DIR,
        help=f"skillpatch_data directory (default: {STORAGE_DIR})",
    )

    args = parser.parse_args()

    # Allow overriding storage dir from CLI
    global TRACE_FILE, SKILLS_DIR
    TRACE_FILE = Path(args.storage_dir) / "trace.jsonl"
    SKILLS_DIR = Path(args.storage_dir) / "skills"

    if args.status:
        cmd_status()
        return

    cmd_record(
        action=args.action,
        num_episodes=args.num_episodes,
        episode_time_s=args.episode_time,
        reset_time_s=args.reset_time,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
