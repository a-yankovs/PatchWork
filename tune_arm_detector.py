"""
tune_arm_detector.py — Live three-cue tuning for the SO-100 arm detector.

Shows four windows simultaneously:
  • Main     — camera feed with overlay when arm is detected
  • White    — white body mask
  • Servos   — black servo mask (inside the arm region)
  • RedWire  — red wiring mask

Controls:
    S — print current threshold values → paste into arm_detector.py
    Q — quit

Run with the arm in frame under the actual demo lighting conditions.
The arm is confirmed when:
  • White mask covers the arm body cleanly
  • Servos mask shows dark blobs at joint locations
  • RedWire mask shows the wiring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "python"))

import cv2
import numpy as np
from tinyvla_debugger.arm_detector import (
    ArmDetector, JOINT_LABELS,
    HSV_WHITE_LOW, HSV_WHITE_HIGH,
    HSV_RED_LOW1, HSV_RED_HIGH1,
    HSV_RED_LOW2, HSV_RED_HIGH2,
    HSV_BLACK_LOW, HSV_BLACK_HIGH,
    CONFIDENCE_THRESH,
)

W_MAIN   = "Main (S=save Q=quit)"
W_WHITE  = "White body mask"
W_SERVO  = "Servo mask (black joints)"
W_RED    = "Red wire mask"

def nothing(_): pass

def make_windows():
    for name in (W_MAIN, W_WHITE, W_SERVO, W_RED):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(W_MAIN,  640, 480)
    cv2.resizeWindow(W_WHITE, 320, 240)
    cv2.resizeWindow(W_SERVO, 320, 240)
    cv2.resizeWindow(W_RED,   320, 240)
    cv2.moveWindow(W_MAIN,  0,   0)
    cv2.moveWindow(W_WHITE, 650, 0)
    cv2.moveWindow(W_SERVO, 650, 260)
    cv2.moveWindow(W_RED,   980, 0)

def add_sliders():
    # White body
    cv2.createTrackbar("W V_low",  W_WHITE, int(HSV_WHITE_LOW[2]),  255, nothing)
    cv2.createTrackbar("W S_high", W_WHITE, int(HSV_WHITE_HIGH[1]), 100, nothing)
    # Black servos
    cv2.createTrackbar("B V_high", W_SERVO, int(HSV_BLACK_HIGH[2]), 120, nothing)
    # Red wire
    cv2.createTrackbar("R S_low",  W_RED,   int(HSV_RED_LOW1[1]),   255, nothing)
    cv2.createTrackbar("R V_low",  W_RED,   int(HSV_RED_LOW1[2]),   255, nothing)
    # Confidence
    cv2.createTrackbar("Conf x100", W_MAIN, int(CONFIDENCE_THRESH * 100), 100, nothing)

def get_thresholds():
    w_v_low  = cv2.getTrackbarPos("W V_low",   W_WHITE)
    w_s_high = cv2.getTrackbarPos("W S_high",  W_WHITE)
    b_v_high = cv2.getTrackbarPos("B V_high",  W_SERVO)
    r_s_low  = cv2.getTrackbarPos("R S_low",   W_RED)
    r_v_low  = cv2.getTrackbarPos("R V_low",   W_RED)
    conf     = cv2.getTrackbarPos("Conf x100", W_MAIN) / 100.0
    return w_v_low, w_s_high, b_v_high, r_s_low, r_v_low, conf

def apply_thresholds(detector, w_v_low, w_s_high, b_v_high, r_s_low, r_v_low, conf):
    detector.confidence_thresh = conf
    # patch module-level arrays directly on the instance via monkey-patch
    import tinyvla_debugger.arm_detector as _mod
    _mod.HSV_WHITE_LOW[2]  = w_v_low
    _mod.HSV_WHITE_HIGH[1] = w_s_high
    _mod.HSV_BLACK_HIGH[2] = b_v_high
    _mod.HSV_RED_LOW1[1]   = r_s_low
    _mod.HSV_RED_LOW1[2]   = r_v_low
    _mod.HSV_RED_LOW2[1]   = r_s_low
    _mod.HSV_RED_LOW2[2]   = r_v_low

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Camera 0 not available — check connection.")
        return

    make_windows()
    add_sliders()
    detector = ArmDetector()

    print("Point camera at the AMD SO-100 arm. Adjust sliders per mask windows.")
    print("Goal: White=arm body only  |  Servos=joint blobs  |  Red=wiring")
    print("Press S to save, Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        thresholds = get_thresholds()
        apply_thresholds(detector, *thresholds)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = detector._white_mask(hsv)
        servo_mask = detector._servo_mask(hsv, white_mask)
        red_mask   = detector._red_mask(hsv)

        result  = detector.detect(frame)
        display = frame.copy()

        if result:
            for p1, p2 in result.segments:
                cv2.line(display, p1, p2, (0, 255, 0), 3)
            for pt, lbl in zip(result.joints, JOINT_LABELS):
                cv2.circle(display, pt, 7, (0, 0, 255), -1)
                cv2.putText(display, lbl, (pt[0]+8, pt[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
            cv2.putText(display,
                        f"ARM DETECTED  conf={result.confidence:.2f}  servos=OK  wire=OK",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        else:
            cv2.putText(display, "no arm — adjust masks",
                        (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)

        # colour the servo mask green for clarity
        servo_vis = cv2.cvtColor(servo_mask, cv2.COLOR_GRAY2BGR)
        servo_vis[:, :, 1] = servo_mask      # green channel = servo blobs
        servo_vis[:, :, 2] = 0
        servo_vis[:, :, 0] = 0

        red_vis = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        red_vis[:, :, 2] = red_mask
        red_vis[:, :, 0] = 0
        red_vis[:, :, 1] = 0

        cv2.imshow(W_MAIN,  display)
        cv2.imshow(W_WHITE, white_mask)
        cv2.imshow(W_SERVO, servo_vis)
        cv2.imshow(W_RED,   red_vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            w_v, w_s, b_v, r_s, r_v, conf = thresholds
            print("── Paste into arm_detector.py ───────────────────────────────")
            print(f"HSV_WHITE_LOW  = np.array([0,    0, {w_v}], dtype=np.uint8)")
            print(f"HSV_WHITE_HIGH = np.array([180, {w_s}, 255], dtype=np.uint8)")
            print(f"HSV_BLACK_LOW  = np.array([0,   0,   0], dtype=np.uint8)")
            print(f"HSV_BLACK_HIGH = np.array([180, 255, {b_v}], dtype=np.uint8)")
            print(f"HSV_RED_LOW1   = np.array([0,   {r_s}, {r_v}], dtype=np.uint8)")
            print(f"HSV_RED_HIGH1  = np.array([10,  255, 255], dtype=np.uint8)")
            print(f"HSV_RED_LOW2   = np.array([165, {r_s}, {r_v}], dtype=np.uint8)")
            print(f"HSV_RED_HIGH2  = np.array([180, 255, 255], dtype=np.uint8)")
            print(f"CONFIDENCE_THRESH = {conf}")
            print()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
