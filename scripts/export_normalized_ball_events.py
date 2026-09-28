#!/usr/bin/env python3
"""Export upstream BallTrack predictions into the project interchange schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RACKETVISION_COMMIT = "c44af2a08524d3cb54d818f19686f4cdea4d2793"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", type=Path, default=ROOT / "outputs/racketvision_ball/report.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "work/models/racketvision/balltrack_best.pth",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "outputs/normalized"
    )
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    model_sha256 = sha256(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for clip in report["clips"]:
        metadata = clip["metadata"]
        session_id = Path(clip["clip"]).stem
        predictions_path = args.report.parent / session_id / "predictions.json"
        predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
        width = int(metadata["originalWidth"])
        height = int(metadata["originalHeight"])
        model_width = int(metadata["modelWidth"])
        model_height = int(metadata["modelHeight"])
        fps = float(metadata["fps"])

        observations = []
        for raw in predictions:
            frame_index = int(raw["Frame"])
            visible = int(raw["Visibility"]) > 0
            image = None
            if visible:
                x_px = float(raw["X"]) * width / model_width
                y_px = float(raw["Y"]) * height / model_height
                image = {
                    "xPx": round(x_px, 3),
                    "yPx": round(y_px, 3),
                    "xNorm": round(x_px / width, 8),
                    "yNorm": round(y_px / height, 8),
                }
            observations.append(
                {
                    "id": f"ball:{frame_index}",
                    "type": "ball_observation",
                    "frameIndex": frame_index,
                    "timestampUs": round(frame_index * 1_000_000 / fps),
                    "visible": visible,
                    "confidence": float(raw.get("Confidence", 0.0)),
                    "image": image,
                    "provenance": {
                        "adapter": "racketvision_balltrack_v1",
                        "rawModelX": float(raw["X"]),
                        "rawModelY": float(raw["Y"]),
                    },
                }
            )

        document = {
            "schemaVersion": "0.1",
            "sessionId": session_id,
            "video": {
                "fileName": clip["clip"],
                "widthPx": width,
                "heightPx": height,
                "fps": fps,
                "frameCount": int(metadata["frameCount"]),
            },
            "producer": {
                "component": report["model"],
                "sourceCommit": RACKETVISION_COMMIT,
                "modelSha256": model_sha256,
                "threshold": float(report["threshold"]),
            },
            "observations": observations,
        }
        output_path = args.output_dir / f"{session_id}_ball_observations.json"
        output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(f"{output_path}: {len(observations)} observations")


if __name__ == "__main__":
    main()
