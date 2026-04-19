"""
arm_detector.py — AMD SO-100 robotic arm detector.

Three-cue fusion that requires ALL cues simultaneously:
  1. White HSV mask    — 3D-printed white plastic body
  2. Rectangular black blobs — servo motors at every joint (box-shaped, not circular)
  3. Tri-color wire adjacency — red pixels physically touching black pixels,
                                confirming the red+black+white servo cable bundle

The wire-adjacency check is the hardest discriminator: a red cup, red clothing,
or red marker will NOT have black wire immediately next to it the way the SO-100's
servo cable always does. This gate alone eliminates most false positives.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


# ── HSV ranges (tuned from 6 real SO-100 arm photos, hackathon lighting) ─────
# White 3D-printed plastic: very bright, almost no saturation
HSV_WHITE_LOW  = np.array([0,   0, 155], dtype=np.uint8)
HSV_WHITE_HIGH = np.array([180, 60, 255], dtype=np.uint8)

# Red servo wiring: vivid red at both ends of the hue circle
HSV_RED_LOW1  = np.array([0,   110, 70], dtype=np.uint8)
HSV_RED_HIGH1 = np.array([15,  255, 255], dtype=np.uint8)
HSV_RED_LOW2  = np.array([162, 110, 70], dtype=np.uint8)
HSV_RED_HIGH2 = np.array([180, 255, 255], dtype=np.uint8)

# Black servo motors: matte dark plastic, may look slightly lighter under harsh lights
HSV_BLACK_LOW  = np.array([0,   0,   0], dtype=np.uint8)
HSV_BLACK_HIGH = np.array([180, 255, 75], dtype=np.uint8)

# ── Size / scoring thresholds ─────────────────────────────────────────────────
MIN_ARM_AREA_PX        = 1500  # white blob floor
MIN_SERVO_AREA_PX      = 60    # minimum area for a servo blob to count
MIN_SERVOS_INSIDE      = 3     # all 6 photos show ≥3 rectangular servo motors
MIN_WIRE_ADJACENT_PX   = 30    # red pixels that are directly touching black pixels
                               # (confirms tri-color cable, not a standalone red object)
CONFIDENCE_THRESH      = 0.55  # only reached after ALL hard gates pass

MORPH_K = 7  # morphology kernel size


@dataclass
class ArmDetection:
    joints:     list[tuple[int, int]]           # 5 pts: base → gripper
    segments:   list[tuple[tuple, tuple]]        # 4 connecting segments
    confidence: float
    mask:       np.ndarray = field(repr=False)   # white body mask (for debug)


class ArmDetector:
    """
    Detects the white AMD SO-100 robotic arm using three-cue fusion:
    white body + black servos + red wiring.

    Returns None when the frame does not match the arm signature,
    preventing the overlay from appearing on non-arm objects.

    Usage:
        d = ArmDetector()
        result = d.detect(frame_bgr)   # ArmDetection | None
    """

    def __init__(self) -> None:
        k = MORPH_K | 1
        self._k_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self._k_small   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ── public ────────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[ArmDetection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        white_mask  = self._white_mask(hsv)
        black_mask  = cv2.inRange(hsv, HSV_BLACK_LOW, HSV_BLACK_HIGH)
        servo_mask  = self._servo_mask(black_mask, white_mask)
        red_mask    = self._red_mask(hsv)

        arm_contour, confidence = self._best_arm_contour(
            white_mask, servo_mask, red_mask, black_mask, frame.shape
        )
        if arm_contour is None or confidence < CONFIDENCE_THRESH:
            return None

        joints   = self._joints_from_servos(arm_contour, servo_mask, frame.shape)
        segments = list(zip(joints[:-1], joints[1:]))
        arm_mask = np.zeros_like(white_mask)
        cv2.drawContours(arm_mask, [arm_contour], -1, 255, -1)
        return ArmDetection(joints=joints, segments=segments,
                            confidence=confidence, mask=arm_mask)

    # ── mask builders ─────────────────────────────────────────────────────────

    def _white_mask(self, hsv: np.ndarray) -> np.ndarray:
        m = cv2.inRange(hsv, HSV_WHITE_LOW, HSV_WHITE_HIGH)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self._k_ellipse, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  self._k_ellipse, iterations=1)
        return m

    def _servo_mask(self, black_mask: np.ndarray, arm_mask: np.ndarray) -> np.ndarray:
        """Black servo motors — only inside the dilated arm region."""
        dilated = cv2.dilate(arm_mask, self._k_ellipse, iterations=1)
        return cv2.bitwise_and(black_mask, dilated)

    def _red_mask(self, hsv: np.ndarray) -> np.ndarray:
        r1 = cv2.inRange(hsv, HSV_RED_LOW1, HSV_RED_HIGH1)
        r2 = cv2.inRange(hsv, HSV_RED_LOW2, HSV_RED_HIGH2)
        return cv2.bitwise_or(r1, r2)

    # ── contour selection ─────────────────────────────────────────────────────

    def _best_arm_contour(
        self,
        white_mask: np.ndarray,
        servo_mask: np.ndarray,
        red_mask: np.ndarray,
        black_mask: np.ndarray,
        shape: tuple,
    ) -> tuple[Optional[np.ndarray], float]:
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if cv2.contourArea(c) >= MIN_ARM_AREA_PX]
        if not valid:
            return None, 0.0

        best_contour: Optional[np.ndarray] = None
        best_score = 0.0
        for contour in valid:
            score = self._score(contour, servo_mask, red_mask, black_mask, shape)
            if score > best_score:
                best_contour = contour
                best_score = score
        return best_contour, best_score

    # ── confidence scoring ────────────────────────────────────────────────────

    def _score(self, contour: np.ndarray, servo_mask: np.ndarray,
               red_mask: np.ndarray, black_mask: np.ndarray, shape: tuple) -> float:
        frame_h, frame_w = shape[:2]
        contour_area = cv2.contourArea(contour)
        if contour_area <= 0:
            return 0.0

        contour_mask = np.zeros(servo_mask.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        # ── Gate A: rectangular servo blobs inside the white region ──────────
        # Servo motors are box-shaped. Filter blobs by area AND rectangularity
        # (circularity < 0.80) to exclude shadow smears or circular holes.
        servo_inside = cv2.bitwise_and(servo_mask, contour_mask)
        servo_cnts, _ = cv2.findContours(servo_inside, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        n_servos = 0
        for c in servo_cnts:
            area = cv2.contourArea(c)
            if area < MIN_SERVO_AREA_PX:
                continue
            perim = cv2.arcLength(c, True)
            if perim == 0:
                continue
            circularity = 4 * math.pi * area / (perim * perim)
            # servo motors: low circularity (rectangular), not round blobs
            if circularity < 0.82:
                n_servos += 1
        if n_servos < MIN_SERVOS_INSIDE:
            return 0.0

        # ── Gate B: tri-color wire adjacency (red pixels touching black) ─────
        # The SO-100 servo cable is always red+black+white running together.
        # Dilate the red mask by ~6px and AND with the full black mask to find
        # red pixels that have black wire immediately adjacent to them.
        # A standalone red object (cup, clothing, marker) won't have this.
        red_expanded = cv2.dilate(red_mask, self._k_small, iterations=6)
        arm_zone = cv2.dilate(contour_mask, self._k_ellipse, iterations=2)
        wire_adjacent = cv2.bitwise_and(red_expanded, cv2.bitwise_and(black_mask, arm_zone))
        adjacent_px = cv2.countNonZero(wire_adjacent)
        if adjacent_px < MIN_WIRE_ADJACENT_PX:
            return 0.0

        # ── Gate C: reasonable size in frame ─────────────────────────────────
        frame_ratio = contour_area / float(frame_h * frame_w)
        if not (0.006 <= frame_ratio <= 0.55):
            return 0.0

        # ── Graduated score — only reached when ALL gates pass ────────────────
        score = 0.0

        # Servo count: more rectangular boxes = more certain
        if n_servos >= 5:
            score += 0.45
        elif n_servos >= 4:
            score += 0.38
        elif n_servos == 3:
            score += 0.28

        # Wire adjacency quality: stronger signal = more certain
        if adjacent_px >= MIN_WIRE_ADJACENT_PX * 4:
            score += 0.35
        elif adjacent_px >= MIN_WIRE_ADJACENT_PX * 2:
            score += 0.28
        else:
            score += 0.18

        # Size quality
        if 0.01 <= frame_ratio <= 0.40:
            score += 0.12

        return min(score, 1.0)

    # ── joint estimation ──────────────────────────────────────────────────────

    def _joints_from_servos(self, arm_contour: np.ndarray,
                            servo_mask: np.ndarray,
                            shape: tuple) -> list[tuple[int, int]]:
        """
        Use the detected servo (black) blobs as joint anchors when available.
        Fall back to ellipse-axis sampling if too few servos are found.
        """
        servo_cnts, _ = cv2.findContours(servo_mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        servo_centers = []
        for c in servo_cnts:
            if cv2.contourArea(c) < MIN_SERVO_AREA_PX:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            servo_centers.append((cx, cy))

        # sort top-to-bottom (gripper is usually highest / lowest y depending on pose)
        servo_centers.sort(key=lambda p: p[1])

        if len(servo_centers) >= 5:
            return servo_centers[:5]

        if len(servo_centers) >= 2:
            # fill gaps by interpolating between detected servos
            return self._interpolate_joints(servo_centers, 5)

        # fallback: fit ellipse to white body and sample along major axis
        return self._ellipse_joints(arm_contour)

    def _interpolate_joints(self, pts: list[tuple[int, int]],
                            n: int) -> list[tuple[int, int]]:
        """Linearly interpolate n evenly spaced points between detected servos."""
        if len(pts) < 2:
            return pts
        result = []
        steps = n - 1
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        for i in range(n):
            t = i / steps
            result.append((int(x0 + t * (x1 - x0)), int(y0 + t * (y1 - y0))))
        return result

    def _ellipse_joints(self, contour: np.ndarray) -> list[tuple[int, int]]:
        if len(contour) >= 5:
            (cx, cy), (ma, mi), angle_deg = cv2.fitEllipse(contour)
            rad  = math.radians(angle_deg - 90)
            half = max(ma, mi) / 2.0
            pts = []
            for frac in (-0.50, -0.25, 0.00, 0.25, 0.50):
                px = int(cx + frac * half * 2 * math.cos(rad))
                py = int(cy + frac * half * 2 * math.sin(rad))
                pts.append((px, py))
            # base = lowest point
            if pts[0][1] < pts[-1][1]:
                pts = pts[::-1]
            return pts
        x, y, w, h = cv2.boundingRect(contour)
        cx = x + w // 2
        return [(cx, y + int(h * f)) for f in (1.0, 0.75, 0.50, 0.25, 0.0)]
