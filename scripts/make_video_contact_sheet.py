#!/usr/bin/env python3
"""Create a compact six-frame contact sheet for one or more videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_sheet(video_path: Path, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, max(frame_count - 1, 0), num=6, dtype=int)
    panels: list[np.ndarray] = []

    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            continue
        cv2.rectangle(frame, (0, 0), (width, 46), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"{video_path.parent.name} | frame {frame_index}",
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
    if not panels:
        raise RuntimeError(f"No frames decoded from {video_path}")

    rows = [np.hstack(panels[index : index + 2]) for index in range(0, len(panels), 2)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for video_path in args.videos:
        output_path = args.output_dir / f"{video_path.parent.name}_contact_sheet.jpg"
        make_sheet(video_path, output_path)
        print(output_path)


if __name__ == "__main__":
    main()
