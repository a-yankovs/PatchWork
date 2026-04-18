"""
webcam_stream.py — Threaded webcam frame buffer for continuous capture
Owner: Vera

Keeps the camera open and continuously reads frames in a background thread.
The orchestrator calls get_latest_frame() whenever it needs to verify a step —
it always gets the most recent frame without camera open/close overhead.

Usage (as context manager — recommended):
    with WebcamStream(index=0) as cam:
        frame = cam.get_latest_frame()   # latest BGR numpy array

Usage (manual start/stop):
    cam = WebcamStream(index=0)
    cam.start()
    frame = cam.get_latest_frame()
    cam.stop()

Mock mode (no webcam required — for testing):
    cam = WebcamStream(mock=True)
    cam.start()
    frame = cam.get_latest_frame()   # returns a 480×640 grey numpy array
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Timeout waiting for first frame after start()
_FRAME_TIMEOUT_S = 3.0


class WebcamStream:
    """
    Background-threaded webcam capture.

    The capture loop runs in a daemon thread so it never blocks the
    orchestrator's asyncio event loop. get_latest_frame() returns a
    copy of the most recent frame (thread-safe).

    Args:
        index:   OpenCV VideoCapture device index (default 0).
        mock:    Return synthetic blank frames — no webcam needed.
        width:   Requested capture width (hint to driver; may be ignored).
        height:  Requested capture height (hint to driver; may be ignored).
    """

    def __init__(
        self,
        index: int = 0,
        mock: bool = False,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.index = index
        self.mock = mock
        self._width = width
        self._height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "WebcamStream":
        """Open the camera and start the capture thread. Returns self."""
        if self.mock:
            # Synthetic grey frame fallback — used when the real camera fails to open.
            self._frame = np.full(
                (self._height, self._width, 3), fill_value=100, dtype=np.uint8
            )
            logger.info("WebcamStream: [MOCK] returning %dx%d synthetic frames",
                        self._width, self._height)
            return self

        # [REAL] Open physical camera via OpenCV VideoCapture.
        # Suppress V4L2 "can't open camera by index" warnings that OpenCV
        # prints to stderr even before isOpened() returns False — they are
        # expected when device 0 has a permissions issue and we handle it below.
        os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
        self._cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            self._cap.release()
            raise RuntimeError(
                f"WebcamStream: cannot open camera at index {self.index} via V4L2. "
                "Check the webcam is plugged in and not in use by another process."
            )

        # Request resolution (driver may ignore)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # Flush a few warm-up frames so the background thread starts from
        # a valid frame rather than whatever stale buffer the driver has.
        for _ in range(5):
            ret, frame = self._cap.read()
            if ret:
                self._frame = frame
        if self._frame is not None:
            logger.debug("WebcamStream: warm-up frames flushed (device %d)", self.index)

        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="WebcamStream",
            daemon=True,
        )
        self._thread.start()
        logger.info("WebcamStream started (device %d)", self.index)
        return self

    def stop(self) -> None:
        """Stop the capture thread and release the camera."""
        if self.mock:
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        logger.info("WebcamStream stopped")

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> np.ndarray:
        """
        Return the most recent captured frame as a BGR numpy array.

        Thread-safe. Blocks up to _FRAME_TIMEOUT_S seconds on the very
        first call if no frame has been captured yet (rare edge case).

        Returns:
            BGR uint8 numpy array, shape (H, W, 3).

        Raises:
            RuntimeError: If no frame is available within the timeout.
        """
        if self.mock:
            # [MOCK] Returns a copy of the synthetic frame with a timestamp burned in.
            # [REAL] This branch never runs — the background thread populates self._frame
            #        from the live camera feed and this method returns that instead.
            frame = self._frame.copy()
            ts = f"{time.time():.3f}"
            cv2.putText(frame, ts, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 200, 200), 1)
            return frame

        # [REAL] Return the latest frame captured by the background thread.
        # Wait for first real frame
        deadline = time.monotonic() + _FRAME_TIMEOUT_S
        while self._frame is None and time.monotonic() < deadline:
            time.sleep(0.01)

        if self._frame is None:
            raise RuntimeError(
                f"WebcamStream: no frame received within {_FRAME_TIMEOUT_S}s. "
                "Camera may be faulty or blocked."
            )

        with self._lock:
            return self._frame.copy()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "WebcamStream":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Continuously read frames and store the latest one."""
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._frame = frame
            else:
                # Transient read failure — brief back-off before retrying
                time.sleep(0.005)
