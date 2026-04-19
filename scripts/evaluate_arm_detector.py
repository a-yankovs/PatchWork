"""
Evaluate the AMD arm detector on a folder of positive / negative photos.

Expected layout:
    images/
      positive/
        arm_01.jpg
        arm_02.png
      negative/
        desk_01.jpg
        box_01.jpg

Usage:
    python scripts/evaluate_arm_detector.py --images ./images
    python scripts/evaluate_arm_detector.py --images ./images --save-debug ./debug

This is intentionally simple: for this project, "training correctly" starts
with measuring whether the overlay appears only on the AMD SO-100 arm and not
on random white objects. That is usually a better first step than jumping
straight into Colab fine-tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from tinyvla_debugger.arm_detector import ArmDetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _draw_overlay(frame, detection):
    out = frame.copy()
    if detection is None:
        return out

    for p1, p2 in detection.segments:
        cv2.line(out, p1, p2, (0, 255, 0), 3)
    for pt in detection.joints:
        cv2.circle(out, pt, 6, (0, 0, 255), -1)
    cv2.putText(
        out,
        f"arm {detection.confidence:.2f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return out


def evaluate_split(detector: ArmDetector, split_dir: Path, expect_arm: bool, save_debug: Path | None) -> tuple[int, int]:
    total = 0
    correct = 0

    for img_path in _iter_images(split_dir):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"skip unreadable: {img_path}")
            continue

        total += 1
        detection = detector.detect(frame)
        predicted_arm = detection is not None
        ok = predicted_arm == expect_arm
        if ok:
            correct += 1

        conf = detection.confidence if detection is not None else 0.0
        status = "OK" if ok else "MISS"
        label = "arm" if predicted_arm else "no-arm"
        print(f"[{status}] {img_path.name:<30} pred={label:<6} conf={conf:.2f}")

        if save_debug is not None:
            out_dir = save_debug / split_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            overlay = _draw_overlay(frame, detection)
            cv2.imwrite(str(out_dir / img_path.name), overlay)

    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AMD arm detector on labeled image folders.")
    parser.add_argument("--images", required=True, help="Folder containing positive/ and negative/ subfolders.")
    parser.add_argument("--save-debug", default=None, help="Optional folder to save overlay previews.")
    args = parser.parse_args()

    root = Path(args.images)
    positive = root / "positive"
    negative = root / "negative"

    if not positive.exists() or not negative.exists():
        raise SystemExit("Expected both 'positive/' and 'negative/' under --images.")

    save_debug = Path(args.save_debug) if args.save_debug else None
    detector = ArmDetector()

    print("=== positive (should detect arm) ===")
    pos_correct, pos_total = evaluate_split(detector, positive, expect_arm=True, save_debug=save_debug)
    print("\n=== negative (should suppress overlay) ===")
    neg_correct, neg_total = evaluate_split(detector, negative, expect_arm=False, save_debug=save_debug)

    total = pos_total + neg_total
    correct = pos_correct + neg_correct
    accuracy = (correct / total) if total else 0.0

    print("\n=== summary ===")
    print(f"positive: {pos_correct}/{pos_total}")
    print(f"negative: {neg_correct}/{neg_total}")
    print(f"overall : {correct}/{total} ({accuracy:.1%})")


if __name__ == "__main__":
    main()
