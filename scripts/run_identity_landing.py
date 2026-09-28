#!/usr/bin/env python3
"""Run milestone-one adapters on saved model outputs; no new model inference."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import time
import cv2
import numpy as np

from milestone1.core import (read_json, write_json, file_hash, audit_inputs, IdentityTracker,
    shirt_descriptor, ground_point, project, calibration_usable, landing_statistics, NET)
from milestone1.events import candidate_events, build_shots, HASHES

ROOT = Path(__file__).resolve().parents[1]


def process(args, session):
    paths = {"pose": args.pose_dir/session/"pose_observations.json",
             "ball": args.ball_dir/f"{session}_ball_observations.json",
             "court": args.court_dir/session/"court_calibration.json",
             "video": args.data_dir/f"{session}.mp4"}
    pose, ball, court = [read_json(paths[k]) for k in ("pose", "ball", "court")]
    if pose["sessionConfig"]["format"] != "singles":
        raise ValueError("This milestone evaluates singles only; doubles identities are not implemented")
    audit = audit_inputs(pose, ball, court)
    if audit["errors"]:
        raise ValueError(audit["errors"])
    if not all("candidates" in f for f in pose["observations"]):
        raise ValueError("Re-run pose export: all candidate detections are required for changeovers")
    out = args.output_dir/session
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(paths["video"]))
    if not capture.isOpened():
        raise ValueError("Cannot decode source video")
    fps, n = ball["video"]["fps"], ball["video"]["frameCount"]
    width, height = ball["video"]["widthPx"], ball["video"]["heightPx"]
    trackers = {"appearance_motion": IdentityTracker(), "motion_only": IdentityTracker(False)}
    observations = {key: [] for key in trackers}
    seed_info = None
    seeds_config = read_json(args.seeds) if args.seeds else None
    if seeds_config and seeds_config["sessionId"] != session:
        raise ValueError("Seed file session mismatch")
    if seeds_config and (type(seeds_config.get("frameIndex")) is not int or
                         not 0 <= seeds_config["frameIndex"] < n or
                         set(seeds_config.get("playerDetectionIndices", {})) != {"A", "B"}):
        raise ValueError("Seed file needs an in-range frame and A/B detection indices")
    if seeds_config and seeds_config["playerDetectionIndices"]["A"] == seeds_config["playerDetectionIndices"]["B"]:
        raise ValueError("A and B seeds must be distinct detections")
    started = time.perf_counter()
    writer = cv2.VideoWriter(str(out/"identity_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot create identity overlay")
    for f, frame_obs in enumerate(pose["observations"]):
        ok, image = capture.read()
        if not ok:
            raise ValueError(f"Source video ended before expected frame {f}")
        ts = frame_obs["timestampUs"]
        stable = calibration_usable(court, f)
        candidates = []
        # Identity follows image-space motion even when court calibration fails.
        # Use 30 * normalized image coordinates as association units, not metres.
        for person in frame_obs["candidates"]:
            foot = ground_point(person)
            court_point = project(foot, court["calibration"]["imageToCourtHomography"])
            if person["confidentJointCount"] < 6:
                continue
            if stable and (court_point is None or not (-2 <= court_point[0] <= 12.97 and -7 <= court_point[1] <= 30.77)):
                continue
            candidates.append({"point": [30*foot[0]/width,30*foot[1]/height],
                               "courtPoint":court_point if stable else None,
                               "descriptor": shirt_descriptor(image, person["bbox"]),
                               "pose": person})
        just_seeded = False
        if seed_info is None and stable and len(candidates) >= 2:
            chosen = []
            if seeds_config and f == seeds_config["frameIndex"]:
                for pid, raw in seeds_config["playerDetectionIndices"].items():
                    matching = [c for c in candidates if c["pose"]["provenance"]["rawDetectionIndex"] == raw]
                    if not matching:
                        raise ValueError("Requested seed was filtered or missing")
                    chosen.append((pid, matching[0]))
            elif not seeds_config:
                for pid, role in (("A", "near"), ("B", "far")):
                    selected = next((p for p in frame_obs["players"] if p["role"] == role), None)
                    if selected:
                        matching = [c for c in candidates if c["pose"]["provenance"]["rawDetectionIndex"] == selected["provenance"]["rawDetectionIndex"]]
                        if matching:
                            chosen.append((pid, matching[0]))
            if len(chosen) == 2:
                if {pid for pid, _ in chosen} != {"A", "B"} or chosen[0][1] is chosen[1][1]:
                    raise ValueError("Seeds must identify two distinct players A and B")
                for tracker in trackers.values():
                    for pid, candidate in chosen:
                        tracker.seed(pid, candidate["point"], candidate["descriptor"], ts)
                seed_info = {"frameIndex": f, "status": "user_supplied" if seeds_config else "automatic_proposal_unconfirmed",
                    "players": [{"playerId": pid, "sourcePoseId": c["pose"]["id"],
                                 "rawDetectionIndex": c["pose"]["provenance"]["rawDetectionIndex"]} for pid, c in chosen]}
                just_seeded = True
        for name, tracker in trackers.items():
            if just_seeded:
                matches = [{"candidateIndex": next(i for i,c in enumerate(candidates) if c is candidate),
                            "playerId": pid, "method": "seed", "associationScore": None}
                           for pid,candidate in chosen]
            elif seed_info is not None:
                matches = tracker.update(candidates, ts)
            else:
                matches = []
            players = []
            for match in matches:
                candidate = candidates[match["candidateIndex"]]
                person = candidate["pose"]
                side = ("near" if candidate["courtPoint"][1] > NET else "far") if candidate["courtPoint"] else None
                players.append({"playerId": match["playerId"], "side": side,
                    "sourcePoseId": person["id"], "bbox": person["bbox"], "joints": person["joints"],
                    "courtGroundPointMetres": candidate["courtPoint"],
                    "imageGroundPointNormalized":[v/30 for v in candidate["point"]],
                    "association": {k:v for k,v in match.items() if k != "candidateIndex"}})
            observations[name].append({"frameIndex": f, "timestampUs": ts, "players": players,
                "unassignedCandidateCount": len(candidates)-len(players), "calibrationUsable": stable})
        for p in observations["appearance_motion"][-1]["players"]:
            b = p["bbox"]
            x1,y1,x2,y2 = [round(b[k]) for k in ("x1Px","y1Px","x2Px","y2Px")]
            color = (80,220,100) if p["playerId"] == "A" else (255,190,60)
            cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            cv2.putText(image,f"{p['playerId']} / {p['side']}",(x1,max(24,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        cv2.putText(image,"RESEARCH: proposed identities" if stable else "COURT MAPPING UNVERIFIED - landings withheld",
            (20,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,210,255),2)
        writer.write(image)
    extra, _ = capture.read()
    capture.release()
    writer.release()
    if extra:
        raise ValueError("Video contains more frames than analysis inputs")
    if seed_info is None:
        raise ValueError("Could not initialize identities; supply a valid seed frame")
    candidates = candidate_events(ball, observations["appearance_motion"], court)
    shots = build_shots(candidates, court, fps)
    provenance = {key: {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                        "sha256": file_hash(path)} for key,path in paths.items()}
    identity_doc = {"schemaVersion": "0.2-research", "sessionId": session, "video": ball["video"],
                    "seed": seed_info, "observations": observations["appearance_motion"], "provenance": provenance,
                    "limitations": ["Shirt appearance is not a learned re-ID model", "No real changeover in this clip",
                                    "Identity scores are uncalibrated", "Unverified court intervals withhold side and ground coordinates"],
                    "trackingCoordinateSystem":"30 times normalized image ground point; not physical metres"}
    write_json(out/"identities.json", identity_doc)
    write_json(out/"motion_only_identities.json", {**identity_doc, "observations": observations["motion_only"]})
    write_json(out/"events.json", {"schemaVersion":"0.2-research", "sessionId":session,
        "events":candidates, "upstreamFileSha256": HASHES, "status":"unvalidated_candidates",
        "provenance": provenance})
    write_json(out/"shots.json", {"sessionId":session,"shots":shots,"status":"provisional"})
    summary = {"sessionId":session, "audit":audit, "seed":seed_info,
        "candidateEvents":dict(Counter(e["type"] for e in candidates)),
        "provisionalLandingStatistics":landing_statistics(shots),
        "identityCoverage":{name:dict(Counter(p["playerId"] for f in frames for p in f["players"])) for name,frames in observations.items()},
        "evaluatedAccuracy":None, "realChangeoverTested":False,"phoneFootageTested":False,
        "adapterRuntimeSeconds":round(time.perf_counter()-started,3),
        "runtimeIncludes":"decode, association, overlay encoding and event proposals; excludes model inference",
        "status":"research_output_requires_review", "provenance":provenance}
    write_json(out/"summary.json", summary)
    print(session, summary["candidateEvents"], summary["identityCoverage"], flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", default=None)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/public/racketvision")
    parser.add_argument("--pose-dir",type=Path,default=ROOT/"outputs/milestone1/poses")
    parser.add_argument("--ball-dir",type=Path,default=ROOT/"outputs/normalized")
    parser.add_argument("--court-dir",type=Path,default=ROOT/"outputs/court_calibration")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/milestone1")
    parser.add_argument("--seeds",type=Path,help="Optional single-session reviewed A/B seed file")
    args = parser.parse_args()
    sessions = args.clip or ["match100_000", "match102_000"]
    if args.seeds and len(sessions) != 1:
        parser.error("Use --clip once with --seeds")
    write_json(args.output_dir/"report.json", {"milestone":"identity_and_landing", "complete":False,
        "remainingGate":"Licensed phone singles recording with real changeover and independent event labels",
        "sessions":[process(args,s) for s in sessions]})


if __name__ == "__main__":
    main()
