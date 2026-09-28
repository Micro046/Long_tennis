#!/usr/bin/env python3
"""Convert an exported browser review into partial, clearly qualified landing statistics."""
import argparse
from pathlib import Path

from milestone1.core import read_json, write_json, file_hash
from milestone1.review import validate_review, reviewed_results

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analysis-dir", type=Path, default=ROOT / "outputs/milestone1")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/public/racketvision")
    p.add_argument("--court-dir", type=Path, default=ROOT / "outputs/court_calibration")
    a = p.parse_args()
    review = read_json(a.review)
    session = review["sessionId"]
    identity = read_json(a.analysis_dir / session / "identities.json")
    video = identity["video"]
    video_path = a.data_dir / f"{session}.mp4"
    validate_review(review, session, file_hash(video_path), video["frameCount"],
                    video["widthPx"], video["heightPx"])
    calibration = read_json(a.court_dir / session / "court_calibration.json")
    result = reviewed_results(review, calibration, video["fps"])
    result["sessionId"] = session
    result["provenance"] = {"reviewSha256": file_hash(a.review),
                            "videoSha256": file_hash(video_path),
                            "calibrationSha256": file_hash(a.court_dir / session / "court_calibration.json")}
    write_json(a.output, result)
    print(a.output)


if __name__ == "__main__":
    main()
