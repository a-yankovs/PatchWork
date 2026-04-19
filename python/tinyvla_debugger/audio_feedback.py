"""
audio_feedback.py — ElevenLabs TTS audio cues for skill events
Owner: Vera

Speaks aloud when the robot skill succeeds, self-corrects via patch,
or fails unrecoverably. Designed for live demo use — the audio cue
lands at the same moment the trace.jsonl event fires.

Setup:
    export ELEVENLABS_API_KEY="sk_..."
    pip install elevenlabs --break-system-packages

Playback (Linux node):
    sudo apt-get install -y mpg123   # preferred, lowest latency
    # fallback: ffplay (ffmpeg) or aplay (wav only, needs conversion)

Usage:
    audio = AudioFeedback()                         # reads env for API key
    audio = AudioFeedback(enabled=False)            # silence (tests / CI)
    audio.speak_success("stock_middle_shelf", steps_executed=4, steps_patched=1)
    audio.speak_patch_success("GRASP_FAIL", step_id=1)
    audio.speak_error("GRASP_FAIL", step_id=1, skill_name="stock_middle_shelf")
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — override via env vars or constructor args
# ---------------------------------------------------------------------------

_DEFAULT_VOICE_ID = os.environ.get(
    "ELEVENLABS_VOICE_ID",
    "21m00Tcm4TlvDq8ikWAM",  # ElevenLabs built-in "Rachel" — clear, neutral
)
_DEFAULT_MODEL_ID = os.environ.get(
    "ELEVENLABS_MODEL_ID",
    "eleven_turbo_v2",        # lowest latency model; swap to eleven_multilingual_v2
)                             # if you need non-English output

# ---------------------------------------------------------------------------
# Message templates
# ---------------------------------------------------------------------------

def _success_text(skill_name: str, steps_executed: int, steps_patched: int) -> str:
    readable = skill_name.replace("_", " ")
    base = f"Skill complete. {readable} finished in {steps_executed} step{'s' if steps_executed != 1 else ''}."
    if steps_patched > 0:
        base += f" {steps_patched} step{'s' if steps_patched != 1 else ''} self-corrected automatically."
    return base


def _patch_success_text(failure_type: str, step_id: int) -> str:
    readable = failure_type.replace("_", " ").lower()
    return f"Step {step_id} auto-corrected. {readable} patched. Continuing."


def _error_text(failure_type: str, step_id: int, skill_name: str) -> str:
    readable_skill   = skill_name.replace("_", " ")
    readable_failure = failure_type.replace("_", " ").lower()
    return (
        f"Error. {readable_failure} on step {step_id} of {readable_skill}. "
        "Patch insufficient. Escalating to dashboard."
    )


# ---------------------------------------------------------------------------
# AudioFeedback class
# ---------------------------------------------------------------------------

class AudioFeedback:
    """
    ElevenLabs TTS wrapper for PatchWork event audio cues.

    Falls back to a console-only mock if:
      - `enabled=False` is passed
      - `ELEVENLABS_API_KEY` is not set
      - the `elevenlabs` package is not installed

    The fallback never raises — audio is always best-effort so it
    can't crash the orchestrator loop.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = _DEFAULT_VOICE_ID,
        model_id: str = _DEFAULT_MODEL_ID,
        enabled: bool = True,
        audio_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            api_key:   ElevenLabs API key. Defaults to ELEVENLABS_API_KEY env var.
            voice_id:  ElevenLabs voice ID. Defaults to Rachel (ELEVENLABS_VOICE_ID env).
            model_id:  ElevenLabs model. Defaults to eleven_turbo_v2 (lowest latency).
            enabled:   Set False to silence all audio (useful in tests and CI).
            audio_dir: Directory to save generated .mp3 files. Defaults to a
                       temp dir. Set a persistent path if you want to keep the files.
        """
        self.voice_id  = voice_id
        self.model_id  = model_id
        self.enabled   = enabled
        self._client   = None
        self._audio_dir = Path(audio_dir) if audio_dir else Path(tempfile.mkdtemp(prefix="skillpatch_audio_"))

        if not enabled:
            logger.info("AudioFeedback disabled — running in silent mode")
            return

        resolved_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        if not resolved_key:
            logger.warning(
                "ELEVENLABS_API_KEY not set — AudioFeedback running in mock (print-only) mode. "
                "Set the env var to enable real TTS."
            )
            return

        try:
            from elevenlabs.client import ElevenLabs  # type: ignore
            self._client = ElevenLabs(api_key=resolved_key)
            logger.info("AudioFeedback ready (voice=%s, model=%s)", voice_id, model_id)
        except ImportError:
            logger.warning(
                "elevenlabs package not installed — AudioFeedback in mock mode. "
                "Run: pip install elevenlabs --break-system-packages"
            )

    # ------------------------------------------------------------------
    # Public speak methods — call these from the orchestrator
    # ------------------------------------------------------------------

    def speak_success(
        self,
        skill_name: str,
        steps_executed: int,
        steps_patched: int,
    ) -> None:
        """Speak a success cue when the skill completes all steps."""
        text = _success_text(skill_name, steps_executed, steps_patched)
        self._speak(text, event="success")

    def speak_patch_success(self, failure_type: str, step_id: int) -> None:
        """Speak when a failed step is recovered via patch (mid-skill)."""
        text = _patch_success_text(failure_type, step_id)
        self._speak(text, event="patch_success")

    def speak_error(
        self,
        failure_type: str,
        step_id: int,
        skill_name: str,
    ) -> None:
        """Speak an error cue when the skill aborts (patch insufficient)."""
        text = _error_text(failure_type, step_id, skill_name)
        self._speak(text, event="error")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _speak(self, text: str, event: str = "event") -> None:
        """Generate TTS audio and play it, or print if in mock mode."""
        logger.info("[audio_feedback:%s] %s", event, text)

        if not self.enabled:
            return

        if self._client is None:
            # Mock mode — just print so the orchestrator log shows intent
            print(f"[AudioFeedback MOCK] {text}")
            return

        try:
            audio_bytes = self._generate(text)
            mp3_path    = self._save(audio_bytes, event)
            self._play(mp3_path)
        except Exception as exc:  # noqa: BLE001
            # Never let audio errors crash the orchestrator
            logger.warning("AudioFeedback._speak failed (non-fatal): %s", exc)

    def _generate(self, text: str) -> bytes:
        """Call ElevenLabs API and return raw mp3 bytes."""
        chunks = self._client.text_to_speech.convert(
            text=text,
            voice_id=self.voice_id,
            model_id=self.model_id,
            output_format="mp3_44100_128",
        )
        # convert generator → bytes
        return b"".join(chunks)

    def _save(self, audio_bytes: bytes, label: str) -> Path:
        """Write mp3 bytes to a file in audio_dir and return the path."""
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        path = self._audio_dir / f"{label}.mp3"
        path.write_bytes(audio_bytes)
        logger.debug("Audio saved: %s", path)
        return path

    def _play(self, path: Path) -> None:
        """
        Play an mp3 file using the first available CLI player on the node.

        Tries (in order): mpg123 → ffplay → aplay (via ffmpeg pipe) → afplay.
        Install mpg123 for lowest latency: sudo apt-get install -y mpg123
        """
        players = [
            ["mpg123", "-q", str(path)],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
            ["afplay", str(path)],  # macOS fallback
        ]

        for cmd in players:
            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    timeout=30,
                )
                logger.debug("Played via %s", cmd[0])
                return
            except FileNotFoundError:
                continue  # player not installed, try next
            except subprocess.CalledProcessError as exc:
                logger.warning("%s returned non-zero: %s", cmd[0], exc)
                return

        logger.warning(
            "No audio player found. Install mpg123: sudo apt-get install -y mpg123\n"
            "Audio saved at: %s", path
        )
