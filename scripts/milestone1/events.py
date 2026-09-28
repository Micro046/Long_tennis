"""Isolated adapters for pinned MIT tennis-vision event heuristics.

The upstream 'physics bounce' function is a 2D y-direction-change heuristic.
Perspective, racket contacts and tracking noise can imitate that signature.
Every output here remains a candidate requiring evaluation or review.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
import numpy as np

from .core import (file_hash, project, calibration_usable, region_for_landing,
                   NET)

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/tennis-vision/src/tennis_vision/events"
HASHES = {"bounce_physics": "595becd16b30ba5742c387e0b0d1f89fa067e3828867cd6b260d55850556f915",
          "hit_detector": "da0a4ecfb9b01caaa08f2356bdf03fa071a750b13ccebfa556bc277c2f43dde7"}


def load_component(name):
    path = UPSTREAM / f"{name}.py"
    if file_hash(path) != HASHES[name]:
        raise ValueError(f"Pinned upstream file changed: {path}")
    spec = importlib.util.spec_from_file_location(f"isolated_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def candidate_events(ball, identities, calibration):
    n, fps = ball["video"]["frameCount"], ball["video"]["fps"]
    xy = np.full((n, 2), np.nan)
    for o in ball["observations"]:
        if o["visible"]:
            xy[o["frameIndex"]] = [o["image"]["xPx"], o["image"]["yPx"]]
    # Normalize spatial and temporal scale to the upstream 1080p / 25fps baseline.
    scaled = xy * 1080 / ball["video"]["heightPx"]
    bounce_fn = load_component("bounce_physics").detect_physics_bounces
    bounces = sorted(bounce_fn(scaled, fps, 4*25/fps, 3*25/fps))
    result = []

    def event(frame, event_type, **extra):
        lo, hi = max(0, frame-4), min(n, frame+5)
        image = xy[frame].tolist() if np.isfinite(xy[frame]).all() else None
        return {"id": f"{event_type}:{frame}", "type": event_type, "frameIndex": frame,
                "timestampUs": round(frame*1e6/fps), "imagePx": image,
                "status": "unreviewed_candidate", "confidence": None,
                "inputObservationIds": [f"ball:{i}" for i in range(lo, hi)], **extra}

    for f in bounces:
        result.append(event(f, "bounce", method="tennis_vision_2d_y_reversal",
                            caveat="An image-space direction change does not establish ground contact."))

    direction = np.full(n, np.nan)
    radius = max(2, round(fps*.12))
    for f in range(radius, n-radius):
        a, b = xy[f]-xy[f-radius], xy[f+radius]-xy[f]
        if np.isfinite([a, b]).all() and np.linalg.norm(a)*np.linalg.norm(b) > 1e-6:
            direction[f] = np.degrees(np.arccos(np.clip(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1)))
    hit_module = load_component("hit_detector")
    for player_id in ("A", "B"):
        poses = {f["frameIndex"]: p for f in identities for p in f["players"] if p["playerId"] == player_id}
        best_per_frame = {}
        for wrist in ("left_wrist", "right_wrist"):
            wrist_xy, heights, distance = np.full((n, 2), np.nan), np.full(n, np.nan), np.full(n, np.nan)
            for f, p in poses.items():
                joint = next(j for j in p["joints"] if j["name"] == wrist)
                if joint["confidence"] >= .3:
                    wrist_xy[f] = [joint["xPx"], joint["yPx"]]
                    heights[f] = max(10, p["bbox"]["y2Px"]-p["bbox"]["y1Px"])
                    distance[f] = np.linalg.norm(wrist_xy[f]-xy[f]) / heights[f] * 100
            speeds = np.full(n, np.nan)
            speeds[1:-1] = np.linalg.norm(wrist_xy[2:]-wrist_xy[:-2], axis=1)*fps/(2*heights[1:-1])*100
            if np.isfinite(speeds).sum() < 5:
                continue
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                hits = hit_module.find_candidates(speeds, distance, direction,
                    min_spacing_frames=max(1, round(.28*fps)), proximity_scale_px=80,
                    rolling_window=max(5, round(2*fps)))
            for hit in hits:
                f = hit.frame
                # No invented direction/proximity evidence from the upstream fallback.
                if f not in poses or not np.isfinite([distance[f], direction[f]]).all():
                    continue
                if distance[f] > 130 or hit.score < .08:
                    continue
                candidate = event(f, "hit", playerId=player_id,
                    hitterSide=poses[f]["side"], heuristicScore=hit.score,
                    method="tennis_vision_wrist_speed_proximity_direction",
                    wrist=wrist, ballWristDistanceBodyHeights=float(distance[f]/100),
                    ballDirectionChangeDegrees=float(direction[f]))
                candidate["inputObservationIds"] += [poses[i]["sourcePoseId"] for i in (f-1, f, f+1) if i in poses]
                if f not in best_per_frame or hit.score > best_per_frame[f]["heuristicScore"]:
                    best_per_frame[f] = candidate
        kept = []
        for candidate in sorted(best_per_frame.values(), key=lambda c: c["heuristicScore"], reverse=True):
            if all(abs(candidate["frameIndex"]-other["frameIndex"]) > round(.2*fps) for other in kept):
                kept.append(candidate)
        result.extend(kept)
    return sorted(result, key=lambda e: (e["frameIndex"], e["type"]))


def build_shots(events, calibration, fps, uncertainty_m=.35):
    """Associate a hit with at most one bounce before the next hit, never reuse it."""
    accepted = sorted([e for e in events if e.get("status") != "rejected"], key=lambda e: e["frameIndex"])
    hits = [e for e in accepted if e["type"] == "hit"]
    bounces = [e for e in accepted if e["type"] == "bounce"]
    shots = []
    for idx, hit in enumerate(hits):
        frame = hit["frameIndex"]
        next_frame = hits[idx+1]["frameIndex"] if idx+1 < len(hits) else float("inf")
        limit = min(next_frame, frame+3*fps)
        possible = [e for e in bounces if frame < e["frameIndex"] < limit]
        shot = {"id": f"shot:{hit['id']}", "hitEventId": hit["id"],
                "playerId": hit.get("playerId"), "hitterSide": hit.get("hitterSide"),
                "frameIndex": frame, "bounceEventId": None,
                "landing": {"status": "no_observed_first_bounce", "region": None},
                "status": "provisional", "inputEventIds": [hit["id"]]}
        # Competing player hits near the same moment make attribution unresolved.
        competing = any(other["id"] != hit["id"] and abs(other["frameIndex"]-frame) <= .12*fps for other in hits)
        contact_conflict = any(abs(b["frameIndex"]-frame) <= .12*fps for b in bounces)
        if competing or contact_conflict:
            shot["landing"] = {"status": "contact_type_or_hitter_ambiguous", "region": None}
        elif possible:
            bounce = possible[0]
            shot["bounceEventId"] = bounce["id"]
            shot["inputEventIds"].append(bounce["id"])
            if not calibration_usable(calibration, bounce["frameIndex"]):
                shot["landing"] = {"status": "calibration_unstable", "region": None}
            elif bounce.get("imagePx") is None:
                shot["landing"] = {"status": "missing_contact_position", "region": None}
            else:
                point = project(bounce["imagePx"], calibration["calibration"]["imageToCourtHomography"])
                shot["landing"] = {**region_for_landing(point, hit.get("hitterSide"), uncertainty_m),
                                   "courtMetres": point, "boundaryBufferMetres": uncertainty_m,
                                   "positionAssumption": "ball point is on court at the proposed bounce"}
        shots.append(shot)
    return shots
