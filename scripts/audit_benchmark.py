#!/usr/bin/env python3
"""Audit a candidate tennis-video corpus without treating missing labels as truth."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from benchmark.manifest import validate_manifest
from benchmark.video import probe_video
ROOT = Path(__file__).resolve().parents[1]
FRAME = re.compile(r"^frame_(\d+)$")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_csv(path, frame_count, width, height, kind):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Empty annotation file: {path}")
    indices = []
    for row in rows:
        match = FRAME.fullmatch(row["frame"])
        if match is None:
            raise ValueError(f"Malformed annotation frame: {row['frame']}")
        indices.append(int(match[1]))
    if len(set(indices)) != len(indices):
        raise ValueError(f"Duplicate annotation frames: {path}")
    if any(i < 0 or i >= frame_count for i in indices):
        raise ValueError(f"Annotation frame outside video: {path}")
    result = {"rows": len(rows), "frameCoverage": round(len(rows)/frame_count, 4),
              "missingFrameCount": frame_count-len(rows), "sha256": file_hash(path)}
    if kind == "ball":
        points = [(float(row["ball_x"]), float(row["ball_y"])) for row in rows]
        result["topRightCornerCoordinateCount"] = sum(x >= .98*width and y < .03*height for x,y in points)
        result["topRightCornerCoordinateFraction"] = round(result["topRightCornerCoordinateCount"]/len(points), 4)
        result["outOfImageCoordinateCount"] = sum(not (0 <= x < width and 0 <= y < height) for x,y in points)
        result["qualityFlag"] = "top_right_cluster_needs_label_review" if result["topRightCornerCoordinateFraction"] > .2 else None
    return result


def audit(manifest_path):
    manifest = read_json(manifest_path)
    summary = validate_manifest(manifest)
    sessions = []
    seen_video_hashes = {}
    for entry in manifest["sessions"]:
        video_path = ROOT / entry["videoPath"]
        video = probe_video(video_path)
        digest = file_hash(video_path)
        if digest in seen_video_hashes:
            raise ValueError(f"Duplicate video bytes: {entry['sessionId']} and {seen_video_hashes[digest]}")
        seen_video_hashes[digest] = entry["sessionId"]
        timing_path = ROOT / "outputs/milestone2/timestamps" / f"{entry['sessionId']}.json"
        write_json(timing_path, {"sessionId": entry["sessionId"], "videoSha256": digest,
                                 "relativePresentationTimestampsUs": video.pop("relativePresentationTimestampsUs")})
        row = {"sessionId": entry["sessionId"], "captureMode": entry["captureMode"],
               "split": entry["split"], "videoSha256": digest,
               "video": video, "timestampSidecar": str(timing_path.relative_to(ROOT)),
               "annotations": {}}
        for kind, relative in entry.get("annotations", {}).items():
            row["annotations"][kind] = audit_csv(ROOT/relative, video["frameCount"],
                                                  video["widthPx"], video["heightPx"], kind)
        sessions.append(row)
        print(entry["sessionId"], video["frameCount"], video["variableFrameRateObserved"], flush=True)
    return {"schemaVersion": "0.1", "manifestSha256": file_hash(manifest_path),
            "summary": summary, "sessions": sessions,
            "limitations": ["Diagnostic-only public clips are not fixed-phone product validation",
                            "A dataset CC0 label does not independently establish rights to embedded third-party video",
                            "Ball corner clusters may be placeholders; treat as unverified until manually checked",
                            "No contacts, bounces, changeovers, strokes, or speed ground truth in this dataset"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT/"data/public/tenis_backview/benchmark_manifest.json")
    p.add_argument("--output", type=Path, default=ROOT/"outputs/milestone2/backview_audit.json")
    args = p.parse_args()
    result = audit(args.manifest)
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
