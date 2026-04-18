"""
stt.py — Speech-to-text listener for robot voice commands
Owner: Vera

Captures microphone audio and transcribes it using faster-whisper
(local, on-device — no cloud). Falls back to stdin in mock mode.

Real mode (requires hardware + packages):
    pip install faster-whisper sounddevice --break-system-packages
    listener = SpeechListener()
    command = listener.listen()   # records N seconds, transcribes, returns text

Mock mode (for testing without microphone):
    listener = SpeechListener(mock=True)
    command = listener.listen()   # reads from stdin

Env overrides:
    WHISPER_MODEL=tiny|base|small|medium   (default: base)
    WHISPER_RECORD_SECS=N                  (default: 5)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16_000          # Whisper expects 16 kHz mono
DEFAULT_MODEL_SIZE: str = os.environ.get("WHISPER_MODEL", "base")
DEFAULT_RECORD_SECS: float = float(os.environ.get("WHISPER_RECORD_SECS", "5"))


# ---------------------------------------------------------------------------
# SpeechListener
# ---------------------------------------------------------------------------

class SpeechListener:
    """
    Listen for a spoken robot command and return the transcribed text.

    Args:
        mock:         Use stdin input instead of microphone + Whisper.
                      Automatically falls back to True if faster-whisper or
                      sounddevice are not installed.
        model_size:   Whisper model size: "tiny" | "base" | "small" | "medium".
                      Smaller = faster but less accurate. "base" is a good default
                      for short robot commands in English.
        record_secs:  How many seconds to record per utterance (real mode).
                      Increase if commands are long.
        device:       sounddevice input device index (None = system default).
        compute_type: faster-whisper quantisation: "int8" (fastest, CPU-friendly)
                      or "float16" (GPU, better accuracy).
    """

    def __init__(
        self,
        mock: bool = False,
        model_size: str = DEFAULT_MODEL_SIZE,
        record_secs: float = DEFAULT_RECORD_SECS,
        device: Optional[int] = None,
        compute_type: str = "int8",
    ) -> None:
        self.mock = mock
        self.record_secs = record_secs
        self.device = device
        self._model = None

        if not mock:
            self._load_model(model_size, compute_type)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def listen(self, prompt: str = "Listening for command...") -> str:
        """
        Block until a command is captured and return the transcribed text.

        In mock mode  → prints prompt, reads one line from stdin.
        In real mode  → records from mic, transcribes with Whisper, returns text.

        Returns:
            Non-empty transcribed command string (stripped).
        """
        if self.mock:
            return self._listen_stdin(prompt)
        return self._listen_mic(prompt)

    # ------------------------------------------------------------------
    # Internal — real mic path
    # ------------------------------------------------------------------

    def _load_model(self, model_size: str, compute_type: str) -> None:
        """
        [REAL] Load faster-whisper model for local on-device transcription.
        [MOCK] On ImportError, sets self.mock=True and falls back to stdin.
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore
            logger.info("Loading faster-whisper (%s, %s)...", model_size, compute_type)
            t0 = time.perf_counter()
            # [REAL] WhisperModel runs entirely on-device (CPU int8 or GPU float16).
            self._model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
            logger.info("faster-whisper ready (%.1fs)", time.perf_counter() - t0)
        except ImportError:
            # [MOCK] Package missing — silently downgrade to stdin input.
            logger.warning(
                "faster-whisper not installed — STT falling back to [MOCK] stdin.\n"
                "  Install with: pip install faster-whisper sounddevice "
                "--break-system-packages"
            )
            self.mock = True

    def _listen_mic(self, prompt: str) -> str:
        """
        [REAL] Record from microphone via sounddevice, transcribe with faster-whisper.
        [MOCK] Falls back to _listen_stdin if sounddevice is missing or record fails.
        """
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            # [MOCK] sounddevice missing — downgrade to stdin.
            logger.warning(
                "sounddevice not installed — STT falling back to [MOCK] stdin.\n"
                "  Install with: pip install sounddevice --break-system-packages"
            )
            self.mock = True
            return self._listen_stdin(prompt)

        print(f"\n🎙  {prompt}")
        print(f"   Recording for {self.record_secs:.0f}s ... ", end="", flush=True)

        try:
            audio = sd.rec(
                int(self.record_secs * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self.device,
            )
            sd.wait()
        except Exception as exc:
            logger.warning("sounddevice record failed: %s — falling back to stdin", exc)
            self.mock = True
            return self._listen_stdin(prompt)

        print("done.")

        # flatten to 1-D (sounddevice returns (N, 1) for mono)
        audio_1d: np.ndarray = audio[:, 0] if audio.ndim == 2 else audio

        # Silence check — RMS below threshold means no speech was detected.
        # Whisper hallucinates plausible text from background noise if we skip this.
        rms = float(np.sqrt(np.mean(audio_1d ** 2)))
        if rms < 0.01:
            print("(silence detected)")
            return self._listen_stdin("No speech detected. Type your command instead:")

        print("   Transcribing ... ", end="", flush=True)

        try:
            segments, info = self._model.transcribe(
                audio_1d,
                beam_size=1,
                language="en",
                condition_on_previous_text=False,
                no_speech_threshold=0.6,   # discard segments Whisper itself flags as non-speech
                log_prob_threshold=-1.0,   # discard low-confidence transcriptions
            )
            text = " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            logger.warning("Whisper transcription failed: %s", exc)
            text = ""

        if text:
            print(f"→ {text!r}")
            return text

        logger.warning("Whisper returned empty transcription — falling back to stdin")
        return self._listen_stdin("No speech detected. Type your command instead:")

    # ------------------------------------------------------------------
    # Internal — mock stdin path
    # ------------------------------------------------------------------

    @staticmethod
    def _listen_stdin(prompt: str) -> str:
        """
        [MOCK] Read one line from stdin.
        Active when mock=True, or as an automatic fallback when faster-whisper
        or sounddevice are not installed, or when mic recording fails.
        [REAL] This method is never called in production — _listen_mic handles input.
        """
        print(f"\n⌨   {prompt}")
        print("   [STT MOCK] Command: ", end="", flush=True)
        try:
            line = sys.stdin.readline().strip()
        except (EOFError, KeyboardInterrupt):
            line = ""

        if not line:
            default = "Put the canned goods on the middle shelf"
            print(f"   (empty — using default: {default!r})")
            return default

        return line
