#!/usr/bin/env python3
"""Refresh court-space player positions after pose/player selection changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from run_court_calibration import ROOT, CourtMapper, project_players, render_overlay


def main() -> None:
    output_dir = ROOT / "outputs/court_calibration"
    report_path = output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    by_clip = {item["clip"]: item for item in report["clips"]}

    for video_path in sorted((ROOT / "data/public/racketvision").glob("*.mp4")):
        clip_output = output_dir / video_path.stem
        calibration = json.loads(
            (clip_output / "court_calibration.json").read_text(encoding="utf-8")
        )
        corners = np.asarray(calibration["calibration"]["imageCornersPx"], dtype=np.float32)
        mapper = CourtMapper(corners)
        positions_path = clip_output / "player_court_positions.json"
        projection = project_players(
            ROOT / "outputs/rtmpose_players" / video_path.stem / "pose_observations.json",
            mapper,
            positions_path,
        )
        render_overlay(
            video_path,
            clip_output / "overlay.mp4",
            mapper,
            positions_path,
            float(calibration["calibration"]["medianDetectorConfidence"]),
        )
        clip_report = by_clip[video_path.name]
        for key in ("projectedPlayerObservations", "withinExtendedCourtRate", "groundPointMethods"):
            clip_report[key] = projection[key]
        print(f"refreshed {video_path.stem}: {projection}")

    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
