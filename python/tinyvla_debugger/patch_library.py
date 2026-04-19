"""
patch_library.py — Persistent failure patch store
Owner: Vera

The patch library is the system's long-term memory. Every time a failure
is diagnosed and a parameter fix is found to work, it is stored here so
the NEXT run of that skill pre-applies the fix automatically — preventing
the failure rather than correcting it.

Storage format: patches.json, a flat JSON object keyed by
  "skill_name:failure_type"  →  {param_deltas..., applied_count, last_applied}

Example patches.json:
{
  "box_in_shelf:PLACEMENT_COLLISION": {
    "speed_scale": 0.65,
    "applied_count": 2,
    "last_applied": "2026-04-18T16:00:38"
  },
  "pick_object:GRASP_FAIL": {
    "speed_scale": 0.85,
    "applied_count": 1,
    "last_applied": "2026-04-18T16:01:04"
  }
}

Note: speed_scale is the only parameter that lerobot-replay physically
honours (via --dataset.fps). All other parameters are tracked for
observability but do not change the arm's motion in open-loop replay.

Default patches are seeded on first encounter of a failure type.
On a second consecutive failure with the same patch applied, the
orchestrator raises PATCH_INSUFFICIENT and surfaces to the dashboard.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default patches (per spec table)
# ---------------------------------------------------------------------------

# Note: PLACEMENT_COLLISION uses an absolute value (0.7), not a delta.
# The orchestrator handles this via apply_to_params — see _is_absolute.

DEFAULT_PATCHES: dict[str, dict] = {
    # speed_scale is the only parameter that lerobot-replay actually accepts
    # (maps to --dataset.fps=int(base_fps * speed_scale)).
    # z_offset_mm / approach_angle_deg / gripper_close_force are tracked as
    # metadata but NOT passed to lerobot — open-loop replay cannot modify the
    # recorded trajectory. Only speed_scale changes the physical motion.
    "GRASP_FAIL":          {"speed_scale": 0.85},       # slower close → better grip
    "PLACEMENT_MISS":      {"speed_scale": 0.80},       # slower approach → more accurate placement
    "PLACEMENT_COLLISION": {"speed_scale": 0.65},       # significantly slower → clear shelf edge
    "DROP_DURING_TRANSIT": {"speed_scale": 0.75},       # slower transit → less inertia-induced drop
    "OBJECT_NOT_FOUND":    {"speed_scale": 0.90},       # slight slow-down while searching
}

# Parameters whose patches are absolute values (replace, don't add)
_ABSOLUTE_PARAMS = {"speed_scale"}

# Metadata keys that are not parameter values
_META_KEYS = {"applied_count", "last_applied"}

# Default patches.json path — override via PATCHES_FILE env var
_DEFAULT_PATCHES_FILE = os.environ.get("PATCHES_FILE", "patches.json")


# ---------------------------------------------------------------------------
# PatchLibrary
# ---------------------------------------------------------------------------

class PatchLibrary:
    """
    Persistent store of parameter patches for known failure types.

    Args:
        patches_file: Path to patches.json. Defaults to "patches.json"
                      in the current working directory (or PATCHES_FILE env var).
        auto_seed:    If True, pre-populate the file with DEFAULT_PATCHES
                      entries for any new failure type on first encounter.
    """

    def __init__(
        self,
        patches_file: str | Path | None = None,
        auto_seed: bool = True,
    ) -> None:
        self.path = Path(patches_file or _DEFAULT_PATCHES_FILE)
        self.auto_seed = auto_seed
        self._patches: dict[str, dict] = self._load()
        logger.info("PatchLibrary loaded from %s (%d entries)", self.path, len(self._patches))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_patch(self, skill_name: str, failure_type: str) -> dict:
        """
        Return the parameter delta dict for this skill+failure combination.

        Checks skill-specific patches first; falls back to DEFAULT_PATCHES.
        Returns an empty dict if no patch is known.

        Args:
            skill_name:   e.g. "stock_middle_shelf"
            failure_type: e.g. "GRASP_FAIL"

        Returns:
            Dict of parameter adjustments, e.g. {"z_offset_mm": 5}.
            Excludes metadata keys (applied_count, last_applied).
        """
        key = self._key(skill_name, failure_type)
        if key in self._patches:
            return self._param_only(self._patches[key])

        # Fall back to global default for this failure type
        default = DEFAULT_PATCHES.get(failure_type, {})
        if default:
            logger.debug(
                "No skill-specific patch for %s, using default: %s",
                key, default,
            )
        return dict(default)

    def store_patch(
        self,
        skill_name: str,
        failure_type: str,
        delta: dict,
    ) -> None:
        """
        Persist a successful parameter fix.

        If an entry already exists for this key, increments applied_count
        and updates last_applied. Otherwise creates a new entry.

        Args:
            skill_name:   e.g. "stock_middle_shelf"
            failure_type: e.g. "GRASP_FAIL"
            delta:        Parameter adjustments that were applied, e.g.
                          {"z_offset_mm": 5}. Do NOT include metadata keys.
        """
        key = self._key(skill_name, failure_type)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

        if key in self._patches:
            entry = self._patches[key]
            entry["applied_count"] = entry.get("applied_count", 0) + 1
            entry["last_applied"] = now
            # Update parameter values (keep existing ones not in delta)
            for k, v in delta.items():
                if k not in _META_KEYS:
                    entry[k] = v
        else:
            entry = {k: v for k, v in delta.items() if k not in _META_KEYS}
            entry["applied_count"] = 1
            entry["last_applied"] = now
            self._patches[key] = entry

        logger.info("Stored patch %s: %s (count=%d)", key, delta, entry["applied_count"])
        self._save()

    def get_preexisting_patches(self, skill_name: str) -> dict:
        """
        Return merged parameter adjustments for ALL known failures of a skill.

        Called by the orchestrator before execution begins, so pre-known
        failures are corrected before they can happen again.

        If multiple failure types patch the same parameter, the last one wins
        (iteration order of patches.json). This is intentional — keep
        patches.json small and non-conflicting.

        Args:
            skill_name: e.g. "stock_middle_shelf"

        Returns:
            Merged dict of parameter adjustments for pre-application.
        """
        prefix = f"{skill_name}:"
        merged: dict = {}
        for key, entry in self._patches.items():
            if key.startswith(prefix):
                for k, v in entry.items():
                    if k not in _META_KEYS:
                        merged[k] = v
        return merged

    def apply_to_params(self, params: dict, patch: dict) -> dict:
        """
        Return a new params dict with patch applied.

        For most parameters: result[k] = params[k] + patch[k]  (delta)
        For absolute parameters (speed_scale): result[k] = patch[k]  (replace)

        Args:
            params: Base parameter dict (from skill program or previous apply).
            patch:  Parameter adjustments from get_patch().

        Returns:
            New dict — does not mutate the input dicts.
        """
        result = dict(params)
        for k, v in patch.items():
            if k in _META_KEYS:
                continue
            if k in _ABSOLUTE_PARAMS:
                result[k] = v  # absolute replacement
            elif k in result and isinstance(result[k], (int, float)):
                result[k] = result[k] + v  # delta
            else:
                result[k] = v  # new key

        logger.debug("apply_to_params: %s + %s → %s", params, patch, result)
        return result

    def all_entries(self) -> dict[str, dict]:
        """Return a copy of all patch entries (for dashboard display)."""
        return dict(self._patches)

    def entry_count(self) -> int:
        return len(self._patches)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _key(skill_name: str, failure_type: str) -> str:
        return f"{skill_name}:{failure_type}"

    @staticmethod
    def _param_only(entry: dict) -> dict:
        return {k: v for k, v in entry.items() if k not in _META_KEYS}

    def _load(self) -> dict:
        if not self.path.exists():
            logger.debug("patches.json not found, starting fresh")
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                logger.warning("patches.json root is not an object, resetting")
                return {}
            return data
        except json.JSONDecodeError as e:
            logger.error("patches.json is corrupt: %s — starting fresh", e)
            return {}

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._patches, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# CLI — inspect or reset the patch library
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect SkillPatch patch library")
    parser.add_argument("--file", default=_DEFAULT_PATCHES_FILE, help="patches.json path")
    parser.add_argument("--clear", action="store_true", help="Clear all entries")
    parser.add_argument("--seed", action="store_true", help="Show default patches")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.seed:
        print("Default patches:")
        print(json.dumps(DEFAULT_PATCHES, indent=2))
        raise SystemExit(0)

    lib = PatchLibrary(patches_file=args.file)

    if args.clear:
        lib._patches = {}
        lib._save()
        print("Cleared all patch entries.")
        raise SystemExit(0)

    entries = lib.all_entries()
    if not entries:
        print("Patch library is empty.")
    else:
        print(f"Patch library ({len(entries)} entries):")
        print(json.dumps(entries, indent=2))
