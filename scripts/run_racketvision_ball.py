#!/usr/bin/env python3
"""Run RacketVision BallTrack on the selected public clips and evaluate sparse GT."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BALLTRACK_ROOT = ROOT / "third_party" / "RacketVision" / "source" / "BallTrack"
sys.path.insert(0, str(BALLTRACK_ROOT))

from inference import BallInferencer  # noqa: E402


def read_annotations(path: Path) -> dict[int, dict[str, float | int]]:
    result: dict[int, dict[str, float | int]] = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            result[int(row["Frame"])] = {
                "visibility": int(row["Visibility"]),
                "x": float(row["X"]),
                "y": float(row["Y"]),
            }
    return result


def prepare_frames(video_path: Path, cache_dir: Path, model_width: int, model_height: int) -> dict:
    metadata_path = cache_dir / "metadata.json"
    frame_dir = cache_dir / "frames"
    median_path = cache_dir / "median.npz"
    if metadata_path.exists() and median_path.exists() and frame_dir.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if len(list(frame_dir.glob("*.jpg"))) == metadata["frameCount"]:
            return metadata

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    frame_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    resized_frames: list[np.ndarray] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        resized = cv2.resize(frame, (model_width, model_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(frame_dir / f"{frame_index:06d}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
        resized_frames.append(resized)
        frame_index += 1
    capture.release()

    median = np.median(np.stack(resized_frames, axis=0), axis=0).astype(np.uint8)
    np.savez_compressed(median_path, median=median)
    metadata = {
        "video": str(video_path),
        "originalWidth": original_width,
        "originalHeight": original_height,
        "modelWidth": model_width,
        "modelHeight": model_height,
        "fps": fps,
        "frameCount": frame_index,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def evaluate(
    predictions: list[dict],
    annotations: dict[int, dict[str, float | int]],
    scale_x: float,
    scale_y: float,
    tolerances_px: tuple[float, ...] = (15.0, 25.0, 40.0),
) -> dict:
    by_frame = {int(item["Frame"]): item for item in predictions}
    presence_tp = presence_tn = presence_fp = presence_fn = 0
    distances: list[float] = []
    evaluated: list[dict] = []

    for frame_index, truth in sorted(annotations.items()):
        prediction = by_frame.get(frame_index, {"Visibility": 0, "X": 0, "Y": 0, "Confidence": 0.0})
        gt_visible = int(truth["visibility"]) > 0
        pred_visible = int(prediction["Visibility"]) > 0
        pred_x = float(prediction["X"]) * scale_x
        pred_y = float(prediction["Y"]) * scale_y
        distance = None

        if not gt_visible and not pred_visible:
            outcome = "TN"
            presence_tn += 1
        elif not gt_visible and pred_visible:
            outcome = "FP_presence"
            presence_fp += 1
        elif gt_visible and not pred_visible:
            outcome = "FN_presence"
            presence_fn += 1
        else:
            distance = math.dist((pred_x, pred_y), (float(truth["x"]), float(truth["y"])))
            distances.append(distance)
            outcome = "both_visible"
            presence_tp += 1

        evaluated.append(
            {
                "frame": frame_index,
                "outcome": outcome,
                "prediction": {
                    "visible": pred_visible,
                    "x": round(pred_x, 2),
                    "y": round(pred_y, 2),
                    "confidence": prediction.get("Confidence", 0.0),
                },
                "truth": truth,
                "distancePx": None if distance is None else round(distance, 2),
            }
        )

    predicted_positive = presence_tp + presence_fp
    actual_positive = presence_tp + presence_fn
    precision = presence_tp / predicted_positive if predicted_positive else 0.0
    recall = presence_tp / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    within_tolerance = {}
    for tolerance in tolerances_px:
        count = sum(distance <= tolerance for distance in distances)
        within_tolerance[str(int(tolerance))] = {
            "count": count,
            "rateAmongFramesWhereBothVisible": count / len(distances) if distances else 0.0,
        }
    return {
        "annotatedFrames": len(annotations),
        "presence": {
            "TP": presence_tp,
            "TN": presence_tn,
            "FP": presence_fp,
            "FN": presence_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "localization": {
            "bothVisibleFrames": len(distances),
            "meanDistancePx": float(np.mean(distances)) if distances else None,
            "medianDistancePx": float(np.median(distances)) if distances else None,
            "withinTolerancePx": within_tolerance,
        },
        "evaluatedFrames": evaluated,
    }


def render_overlay(
    video_path: Path,
    output_path: Path,
    predictions: list[dict],
    annotations: dict[int, dict[str, float | int]],
    model_width: int,
    model_height: int,
) -> None:
    prediction_by_frame = {int(item["Frame"]): item for item in predictions}
    capture = cv2.VideoCapture(str(video_path))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    scale_x = width / model_width
    scale_y = height / model_height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        prediction = prediction_by_frame.get(frame_index)
        if prediction and int(prediction["Visibility"]) > 0:
            point = (round(float(prediction["X"]) * scale_x), round(float(prediction["Y"]) * scale_y))
            cv2.circle(frame, point, 13, (255, 0, 255), 3)
            cv2.putText(frame, "prediction", (point[0] + 16, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        truth = annotations.get(frame_index)
        if truth and int(truth["visibility"]) > 0:
            point = (round(float(truth["x"])), round(float(truth["y"])))
            cv2.circle(frame, point, 18, (0, 255, 255), 3)
            cv2.putText(frame, "ground truth", (point[0] + 20, point[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        writer.write(frame)
        frame_index += 1
    capture.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/public/racketvision")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "work/models/racketvision/balltrack_best.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/racketvision_ball")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--clip", action="append", help="Video stem to process; repeat as needed")
    args = parser.parse_args()

    config_path = BALLTRACK_ROOT / "configs/tracknetv3_base.py"
    inferencer = BallInferencer(
        str(config_path), str(args.checkpoint), device="cpu", thre=args.threshold, batchsize=args.batch_size
    )
    use_mps = args.device == "mps" or (args.device == "auto" and torch.backends.mps.is_available())
    if use_mps:
        inferencer.device = torch.device("mps")
        inferencer.model.to(inferencer.device)
    reports = []
    for video_path in sorted(args.data_dir.glob("*.mp4")):
        if args.clip and video_path.stem not in set(args.clip):
            continue
        cache_dir = ROOT / "work/racketvision_inputs" / video_path.stem
        metadata = prepare_frames(video_path, cache_dir, inferencer.width, inferencer.height)
        frame_paths = sorted((cache_dir / "frames").glob("*.jpg"))
        started = time.perf_counter()
        predictions = inferencer([str(item) for item in frame_paths], str(cache_dir / "median.npz"))
        elapsed = time.perf_counter() - started
        annotation_path = args.data_dir / "annotations" / f"{video_path.stem}_ball.csv"
        annotations = read_annotations(annotation_path)
        scale_x = metadata["originalWidth"] / inferencer.width
        scale_y = metadata["originalHeight"] / inferencer.height
        metrics = evaluate(predictions, annotations, scale_x, scale_y)

        clip_output = args.output_dir / video_path.stem
        clip_output.mkdir(parents=True, exist_ok=True)
        (clip_output / "predictions.json").write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
        render_overlay(
            video_path,
            clip_output / "overlay.mp4",
            predictions,
            annotations,
            inferencer.width,
            inferencer.height,
        )
        reports.append(
            {
                "clip": video_path.name,
                "split": "test",
                "metadata": metadata,
                "runtimeSeconds": elapsed,
                "analysisFps": metadata["frameCount"] / elapsed,
                "metrics": metrics,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "RacketVision/BallTrack",
        "threshold": args.threshold,
        "device": str(inferencer.device),
        "clips": reports,
    }
    (args.output_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    compact = {
        "model": summary["model"],
        "device": summary["device"],
        "clips": [
            {
                "clip": item["clip"],
                "runtimeSeconds": round(item["runtimeSeconds"], 2),
                "analysisFps": round(item["analysisFps"], 2),
                "presenceF1": round(item["metrics"]["presence"]["f1"], 4),
                "medianDistancePx": round(item["metrics"]["localization"]["medianDistancePx"], 2),
                "within25PxRate": round(
                    item["metrics"]["localization"]["withinTolerancePx"]["25"][
                        "rateAmongFramesWhereBothVisible"
                    ],
                    4,
                ),
            }
            for item in reports
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
