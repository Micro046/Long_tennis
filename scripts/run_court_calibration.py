#!/usr/bin/env python3
"""Build a consensus court calibration and project player feet to court metres."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GOOD_TENNIS_ROOT = ROOT / "third_party/Good-Tennis"
sys.path.insert(0, str(GOOD_TENNIS_ROOT))

from tennis_analysis.court.auto_detector import CourtLineAutoDetector  # noqa: E402
from tennis_analysis.court.mapper import (  # noqa: E402
    TENNIS_COURT_LENGTH,
    TENNIS_DOUBLES_WIDTH,
    CourtMapper,
)


def sample_indices(frame_count: int, count: int) -> list[int]:
    return sorted(set(np.linspace(0, max(0, frame_count - 1), count, dtype=int).tolist()))


def detect_samples(video_path: Path, count: int) -> tuple[list[dict], dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    results = []
    for frame_index in sample_indices(frame_count, count):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            results.append({"frameIndex": frame_index, "status": "decode_failed"})
            continue
        detector = CourtLineAutoDetector()
        started = time.perf_counter()
        detected = detector.detect(frame)
        elapsed = time.perf_counter() - started
        if detected is None:
            results.append(
                {
                    "frameIndex": frame_index,
                    "status": "failed",
                    "runtimeSeconds": elapsed,
                    "diagnostics": detector.last_diagnostics,
                }
            )
        else:
            results.append(
                {
                    "frameIndex": frame_index,
                    "status": "accepted",
                    "runtimeSeconds": elapsed,
                    "corners": detected["corners"],
                    "confidence": detected["confidence"],
                    "diagnostics": detected["diagnostics"],
                }
            )
    capture.release()
    return results, {
        "fileName": video_path.name,
        "widthPx": width,
        "heightPx": height,
        "fps": fps,
        "frameCount": frame_count,
    }


def consensus_corners(sample_results: list[dict]) -> tuple[np.ndarray, dict]:
    accepted = [item for item in sample_results if item["status"] == "accepted"]
    if len(accepted) < 3:
        raise RuntimeError(f"Need at least 3 accepted court samples; got {len(accepted)}")
    stack = np.asarray([item["corners"] for item in accepted], dtype=np.float64)
    first_median = np.median(stack, axis=0)
    errors = np.sqrt(np.mean(np.sum((stack - first_median) ** 2, axis=2), axis=1))
    median_error = float(np.median(errors))
    mad = float(np.median(np.abs(errors - median_error)))
    threshold = max(12.0, median_error + 3.0 * max(mad, 1.0))
    keep = errors <= threshold
    inliers = stack[keep]
    if len(inliers) < 3:
        inliers = stack
        keep = np.ones(len(stack), dtype=bool)
    consensus = np.median(inliers, axis=0).astype(np.float32)
    final_errors = np.sqrt(np.mean(np.sum((stack - consensus) ** 2, axis=2), axis=1))
    for item, error, is_inlier in zip(accepted, final_errors, keep):
        item["cornerRmsErrorPxFromConsensus"] = round(float(error), 3)
        item["consensusInlier"] = bool(is_inlier)
    return consensus, {
        "acceptedSamples": len(accepted),
        "consensusInliers": int(np.sum(keep)),
        "outlierThresholdPx": round(threshold, 3),
        "medianCornerRmsErrorPx": round(float(np.median(final_errors)), 3),
        "p95CornerRmsErrorPx": round(float(np.percentile(final_errors, 95)), 3),
        "maxCornerRmsErrorPx": round(float(np.max(final_errors)), 3),
    }


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[round(float(value), 10) for value in row] for row in matrix]


def normalized_corners(corners: np.ndarray, width: int, height: int) -> list[list[float]]:
    return [[round(float(x / width), 8), round(float(y / height), 8)] for x, y in corners]


def validate_consensus(video_path: Path, samples: list[dict], corners: np.ndarray) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    detector = CourtLineAutoDetector(min_confidence=0.0, min_reference_score=0.0)
    reference_scores = []
    coverages = []
    supported_lines = []
    for item in samples:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["frameIndex"]))
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = detector._line_mask(gray)
        support = detector.court_reference.prepare_line_support(mask, frame.shape[:2])
        score, details = detector.court_reference.score_line_support(
            corners, support, frame.shape[:2]
        )
        item["consensusReferenceScore"] = round(float(score), 4)
        item["consensusReferenceCoverage"] = details["reference_coverage"]
        item["consensusSupportedLines"] = details["reference_supported_lines"]
        reference_scores.append(score)
        coverages.append(details["reference_coverage"])
        supported_lines.append(details["reference_supported_lines"])
    capture.release()
    return {
        "medianReferenceScore": round(float(np.median(reference_scores)), 4),
        "minReferenceScore": round(float(np.min(reference_scores)), 4),
        "medianReferenceCoverage": round(float(np.median(coverages)), 4),
        "minSupportedLines": int(np.min(supported_lines)),
    }


def ground_point(player: dict) -> tuple[float, float, str]:
    joints = {joint["name"]: joint for joint in player["joints"]}
    ankles = [
        joints[name]
        for name in ("left_ankle", "right_ankle")
        if joints[name]["confidence"] >= 0.3
    ]
    if ankles:
        return (
            float(np.mean([joint["xPx"] for joint in ankles])),
            float(np.mean([joint["yPx"] for joint in ankles])),
            "ankle_mean",
        )
    box = player["bbox"]
    return (
        (float(box["x1Px"]) + float(box["x2Px"])) / 2.0,
        float(box["y2Px"]),
        "bbox_bottom_center",
    )


def project_players(pose_path: Path, mapper: CourtMapper, output_path: Path) -> dict:
    pose = json.loads(pose_path.read_text(encoding="utf-8"))
    projected_frames = []
    total = 0
    inside_extended = 0
    method_counts = {"ankle_mean": 0, "bbox_bottom_center": 0}
    for frame in pose["observations"]:
        players = []
        for player in frame["players"]:
            x_px, y_px, method = ground_point(player)
            x_m, y_m = [float(value) for value in mapper.image_to_court((x_px, y_px))]
            within_extended = (
                -1.5 <= x_m <= TENNIS_DOUBLES_WIDTH + 1.5
                and -5.0 <= y_m <= TENNIS_COURT_LENGTH + 5.0
            )
            players.append(
                {
                    "id": f"court:{frame['frameIndex']}:{player['role']}",
                    "type": "player_court_observation",
                    "role": player["role"],
                    "frameIndex": frame["frameIndex"],
                    "timestampUs": frame["timestampUs"],
                    "sourcePoseId": player["id"],
                    "imageGroundPoint": {
                        "xPx": round(x_px, 3),
                        "yPx": round(y_px, 3),
                        "method": method,
                    },
                    "court": {
                        "xMetres": round(x_m, 3),
                        "yMetres": round(y_m, 3),
                        "withinExtendedCourt": within_extended,
                    },
                }
            )
            total += 1
            inside_extended += int(within_extended)
            method_counts[method] += 1
        projected_frames.append(
            {
                "frameIndex": frame["frameIndex"],
                "timestampUs": frame["timestampUs"],
                "players": players,
            }
        )
    document = {
        "schemaVersion": "0.1",
        "sessionId": pose["sessionId"],
        "coordinateSystem": {
            "origin": "far-left doubles baseline as seen by camera",
            "xAxis": "left-to-right across court",
            "yAxis": "far-to-near along court",
            "units": "metres",
            "doublesWidthMetres": TENNIS_DOUBLES_WIDTH,
            "courtLengthMetres": TENNIS_COURT_LENGTH,
        },
        "observations": projected_frames,
    }
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return {
        "projectedPlayerObservations": total,
        "withinExtendedCourtRate": inside_extended / total if total else 0.0,
        "groundPointMethods": method_counts,
    }


def render_overlay(
    video_path: Path,
    output_path: Path,
    mapper: CourtMapper,
    player_positions_path: Path,
    confidence: float,
) -> None:
    positions = json.loads(player_positions_path.read_text(encoding="utf-8"))
    by_frame = {int(item["frameIndex"]): item["players"] for item in positions["observations"]}
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame, _ = mapper.draw_court_overlay(frame)
        for player in by_frame.get(frame_index, []):
            point = player["imageGroundPoint"]
            xy = (round(point["xPx"]), round(point["yPx"]))
            color = (90, 230, 90) if player["role"].startswith("near") else (0, 170, 255)
            cv2.circle(frame, xy, 8, color, -1, cv2.LINE_AA)
            court = player["court"]
            cv2.putText(
                frame,
                f"{player['role']} ({court['xMetres']:.1f}m,{court['yMetres']:.1f}m)",
                (xy[0] + 10, xy[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            f"consensus court calibration | confidence {confidence:.2f}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()


def process_clip(video_path: Path, output_dir: Path, sample_count: int) -> dict:
    started = time.perf_counter()
    samples, video = detect_samples(video_path, sample_count)
    corners, stability = consensus_corners(samples)
    validation = validate_consensus(video_path, samples, corners)
    mapper = CourtMapper(corners)
    accepted_confidences = [
        item["confidence"] for item in samples if item["status"] == "accepted"
    ]
    confidence = float(np.median(accepted_confidences))
    calibration = {
        "schemaVersion": "0.1",
        "sessionId": video_path.stem,
        "video": video,
        "court": {
            "type": "ITF doubles court",
            "widthMetres": TENNIS_DOUBLES_WIDTH,
            "lengthMetres": TENNIS_COURT_LENGTH,
            "cornerOrder": ["far_left", "far_right", "near_right", "near_left"],
        },
        "producer": {
            "component": "Good-Tennis CourtLineAutoDetector + consensus adapter",
            "sourceCommit": "62759624f8aaf3fd1abd5a6a03695150e90277e2",
            "license": "Apache-2.0",
            "modelWeights": None,
        },
        "calibration": {
            "imageCornersPx": [[round(float(x), 3), round(float(y), 3)] for x, y in corners],
            "imageCornersNorm": normalized_corners(corners, video["widthPx"], video["heightPx"]),
            "imageToCourtHomography": matrix_list(mapper.matrix),
            "courtToImageHomography": matrix_list(mapper.inv_matrix),
            "medianDetectorConfidence": round(confidence, 4),
            "stability": stability,
            "referenceValidation": validation,
        },
        "samples": samples,
    }
    clip_output = output_dir / video_path.stem
    clip_output.mkdir(parents=True, exist_ok=True)
    calibration_path = clip_output / "court_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    player_positions_path = clip_output / "player_court_positions.json"
    projection = project_players(
        ROOT / "outputs/rtmpose_players" / video_path.stem / "pose_observations.json",
        mapper,
        player_positions_path,
    )
    render_overlay(
        video_path,
        clip_output / "overlay.mp4",
        mapper,
        player_positions_path,
        confidence,
    )
    elapsed = time.perf_counter() - started
    return {
        "clip": video_path.name,
        "sampleCount": len(samples),
        "acceptedSamples": stability["acceptedSamples"],
        "sampleAcceptanceRate": stability["acceptedSamples"] / len(samples),
        "consensusCornersPx": calibration["calibration"]["imageCornersPx"],
        "medianDetectorConfidence": calibration["calibration"]["medianDetectorConfidence"],
        **stability,
        **validation,
        **projection,
        "runtimeSecondsIncludingRendering": elapsed,
        "warning": "No independent court-corner ground truth is available; reference support and temporal stability are proxy metrics.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/public/racketvision")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/court_calibration")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--clip", action="append", help="Video stem to process; repeat as needed")
    args = parser.parse_args()

    reports = []
    for video_path in sorted(args.data_dir.glob("*.mp4")):
        if args.clip and video_path.stem not in set(args.clip):
            continue
        reports.append(process_clip(video_path, args.output_dir, args.samples))
        print(f"completed {video_path.stem}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"component": "Good-Tennis court detector + consensus", "clips": reports}
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
