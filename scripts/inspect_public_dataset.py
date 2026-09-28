#!/usr/bin/env python3
"""Validate public test clips and render ground-truth contact sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ball_annotations(path: Path) -> dict[int, tuple[int, float, float]]:
    rows: dict[int, tuple[int, float, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            rows[int(row["Frame"])] = (
                int(row["Visibility"]),
                float(row["X"]),
                float(row["Y"]),
            )
    return rows


def nearest_annotation(
    annotations: dict[int, tuple[int, float, float]], frame_index: int
) -> tuple[int, tuple[int, float, float]] | None:
    if not annotations:
        return None
    nearest_frame = min(annotations, key=lambda item: abs(item - frame_index))
    return nearest_frame, annotations[nearest_frame]


def inspect_clip(video_path: Path, annotation_path: Path, output_dir: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps if fps > 0 else None
    annotations = load_ball_annotations(annotation_path)

    sample_indices = np.linspace(0, max(frame_count - 1, 0), num=6, dtype=int)
    panels: list[np.ndarray] = []
    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue

        annotation = nearest_annotation(annotations, int(frame_index))
        if annotation is not None:
            annotation_frame, (visible, x, y) = annotation
            if visible:
                cv2.circle(frame, (round(x), round(y)), 16, (0, 255, 255), 3)
                cv2.circle(frame, (round(x), round(y)), 3, (0, 0, 255), -1)
            label = f"frame {frame_index} | nearest GT {annotation_frame}"
        else:
            label = f"frame {frame_index} | no GT"

        cv2.rectangle(frame, (0, 0), (width, 46), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            (18, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        panel_width = 640
        panel_height = round(height * panel_width / width)
        panels.append(cv2.resize(frame, (panel_width, panel_height)))

    capture.release()
    if panels:
        rows = [np.hstack(panels[index : index + 2]) for index in range(0, len(panels), 2)]
        sheet = np.vstack(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"{video_path.stem}_contact_sheet.jpg"), sheet)

    visible_annotations = sum(1 for visible, _, _ in annotations.values() if visible)
    return {
        "video": str(video_path),
        "annotation": str(annotation_path),
        "sha256": sha256(video_path),
        "codec": "decoded-by-opencv",
        "width": width,
        "height": height,
        "fps": fps,
        "frameCount": frame_count,
        "durationSeconds": duration_seconds,
        "annotationCount": len(annotations),
        "visibleAnnotationCount": visible_annotations,
        "firstAnnotatedFrame": min(annotations) if annotations else None,
        "lastAnnotatedFrame": max(annotations) if annotations else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/public/racketvision"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dataset_inspection"))
    args = parser.parse_args()

    reports = []
    for video_path in sorted(args.data_dir.glob("*.mp4")):
        annotation_path = args.data_dir / "annotations" / f"{video_path.stem}_ball.csv"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Missing annotation for {video_path.name}")
        reports.append(inspect_clip(video_path, annotation_path, args.output_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "dataset_report.json"
    report_path.write_text(json.dumps({"clips": reports}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clips": reports}, indent=2))


if __name__ == "__main__":
    main()

