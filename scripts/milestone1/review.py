"""Apply human decisions to candidate events without claiming exhaustive labels."""
from __future__ import annotations

from collections import Counter
import math

from .core import landing_statistics
from .events import build_shots


def validate_review(review, session_id, video_sha256, frame_count, width, height):
    if review.get("sessionId") != session_id or review.get("videoSha256") != video_sha256:
        raise ValueError("Review source video does not match the analysis session")
    if review.get("annotationMode") != "candidate_review":
        raise ValueError("Expected a candidate review, not independent ground truth")
    if not isinstance(review.get("events"), list):
        raise ValueError("Review events must be a list")
    seen = set()
    for event in review["events"]:
        eid = event.get("id")
        frame = event.get("frameIndex")
        if not isinstance(eid, str) or not eid or eid in seen:
            raise ValueError("Review event IDs must be unique strings")
        seen.add(eid)
        if event.get("type") not in ("hit", "bounce") or type(frame) is not int or not 0 <= frame < frame_count:
            raise ValueError(f"Invalid type or frame for {eid}")
        if event.get("status") not in ("accepted", "rejected", "unreviewed_candidate"):
            raise ValueError(f"Invalid review status for {eid}")
        point = event.get("imagePx")
        if point is not None and (not isinstance(point, list) or len(point) != 2 or
                                  not all(isinstance(v, (float, int)) and math.isfinite(v) for v in point) or
                                  not 0 <= point[0] < width or not 0 <= point[1] < height):
            raise ValueError(f"Invalid image point for {eid}")
        if event["status"] == "accepted":
            if event["type"] == "bounce" and point is None:
                raise ValueError(f"Accepted bounce needs a reviewed ground-contact point: {eid}")
            if event["type"] == "hit" and (event.get("playerId") not in (None, "A", "B") or
                                           event.get("hitterSide") not in (None, "near", "far")):
                raise ValueError(f"Invalid hitter identity or side for {eid}")


def reviewed_results(review, calibration, fps):
    accepted = [e for e in review["events"] if e["status"] == "accepted"]
    unreviewed = [e for e in review["events"] if e["status"] == "unreviewed_candidate"]
    shots = build_shots(accepted, calibration, fps)
    for shot in shots:
        start = shot["frameIndex"]
        end = next((e["frameIndex"] for e in accepted if e["id"] == shot["bounceEventId"]), None)
        if end is None:
            end = start + 3 * fps
        if any(start < e["frameIndex"] < end for e in unreviewed):
            shot["landing"] = {"status": "unreviewed_event_in_shot_window", "region": None}
            shot["bounceEventId"] = None
            shot["inputEventIds"] = [shot["hitEventId"]]
        shot["status"] = "reviewed_candidates_partial_coverage"
    return {"status": "partial_review_only", "annotationMode": "candidate_review",
            "accuracyEvaluated": False, "completeShotCoverage": False,
            "eventDecisions": dict(Counter(e["status"] for e in review["events"])),
            "shots": shots, "landingStatistics": landing_statistics(shots),
            "caveat": "Counts use reviewed candidates only. Missed events can change every total and percentage."}
