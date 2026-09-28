#!/usr/bin/env python3
"""Frozen identity association baseline on external UVY tennis annotations.

Ground-truth person boxes replace the detector (oracle-box experiment). Known
athlete IDs initialize A/B once; future GT IDs are used ONLY to score results.
This cannot establish end-to-end detection or phone/changeover performance.
"""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import cv2
from milestone1.core import IdentityTracker, shirt_descriptor, read_json, write_json, file_hash

ROOT=Path(__file__).resolve().parents[1]

def evaluate(folder):
    rows=[]
    for r in csv.reader((folder/"gt.txt").open()):
        rows.append({"frame":int(r[0])-1,"trackId":int(r[1]),"left":float(r[2]),"top":float(r[3]),
            "width":float(r[4]),"height":float(r[5]),"class":int(r[7]),"visibility":float(r[8])})
    if len({(r["frame"],r["trackId"]) for r in rows})!=len(rows):
        raise ValueError("Duplicate frame/track keys in external truth")
    player_counts=Counter(r["trackId"] for r in rows if r["class"]==1)
    athlete_ids=[p for p,_ in player_counts.most_common(2)]
    label_map={gt:pid for gt,pid in zip(sorted(athlete_ids),("A","B"))}
    by_frame=defaultdict(list)
    for r in rows:
        if r["class"] in (1,4,5,7):by_frame[r["frame"]].append(r)
    video=next(folder.glob('*.mp4'));cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError("Cannot open UVY video")
    fps=cap.get(cv2.CAP_PROP_FPS);width=cap.get(cv2.CAP_PROP_FRAME_WIDTH);height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    trackers={"appearance_motion":IdentityTracker(),"motion_only":IdentityTracker(False)}
    metrics={k:{"correctAssignments":0,"wrongAssignments":0,"emittedAssignments":0,
                "eligibleVisibleAthleteObservations":0,"identitySwitches":0,"diagnosticFrames":[]} for k in trackers}
    last_label={k:{} for k in trackers}
    seed_frame=None;frame=0
    while True:
        ok,img=cap.read()
        if not ok:break
        candidates=[]
        for r in by_frame[frame]:
            if r["width"]<=0 or r["height"]<=0:raise ValueError("Nonpositive truth box")
            box={"x1Px":r["left"],"y1Px":r["top"],"x2Px":r["left"]+r["width"],"y2Px":r["top"]+r["height"]}
            candidates.append({"point":[30*(r["left"]+r["width"]/2)/width,30*(r["top"]+r["height"])/height],
                "descriptor":shirt_descriptor(img,box),"truthId":r["trackId"]})
        if frame%2==0:candidates.reverse()
        ts=round(frame*1e6/fps)
        if seed_frame is None:
            lookup={c["truthId"]:c for c in candidates}
            if all(gt in lookup for gt in athlete_ids):
                seed_frame=frame
                for t in trackers.values():
                    for gt,pid in label_map.items():
                        c=lookup[gt];t.seed(pid,c["point"],c["descriptor"],ts)
            frame+=1;continue
        for name,tracker in trackers.items():
            # Strip ground-truth IDs before passing detections to the tracker.
            match=tracker.update([{k:v for k,v in c.items() if k!="truthId"} for c in candidates],ts)
            m=metrics[name]
            m["eligibleVisibleAthleteObservations"]+=sum(c["truthId"] in label_map for c in candidates)
            for out in match:
                gt=candidates[out["candidateIndex"]]["truthId"]
                correct=label_map.get(gt)==out["playerId"]
                m["emittedAssignments"]+=1
                m["correctAssignments" if correct else "wrongAssignments"]+=1
                if gt in label_map:
                    if gt in last_label[name] and last_label[name][gt]!=out["playerId"]:m["identitySwitches"]+=1
                    last_label[name][gt]=out["playerId"]
                if not correct and len(m["diagnosticFrames"])<12:
                    m["diagnosticFrames"].append({"frameIndex":frame,"predicted":out["playerId"],"truthTrackId":gt,
                        "expected":label_map.get(gt),"associationScore":out["associationScore"]})
        frame+=1
    cap.release()
    if seed_frame is None:raise ValueError("Two athletes never jointly visible")
    for m in metrics.values():
        m["assignmentPrecision"]=m["correctAssignments"]/m["emittedAssignments"] if m["emittedAssignments"] else None
        m["correctIdentityCoverage"]=m["correctAssignments"]/m["eligibleVisibleAthleteObservations"] if m["eligibleVisibleAthleteObservations"] else None
    return {"sessionId":folder.name,"experiment":"oracle_person_boxes_with_ground_truth_initialization",
        "videoFrames":frame,"videoFps":fps,"declaredAnnotationFps":30,"gtFrameRangeZeroBased":[min(r["frame"] for r in rows),max(r["frame"] for r in rows)],
        "seedFrame":seed_frame,"identityMap":label_map,"otherPlayerTrackIds":sorted(set(player_counts)-set(athlete_ids)),
        "metrics":metrics,"sourceVideoSha256":file_hash(video),"sourceLabelsSha256":file_hash(folder/"gt.txt"),
        "limitations":["Oracle person boxes and labelled initialization bypass detection and athlete selection",
            "No verified changeover or phone camera model", "Publisher annotations include semi-automated labels; not independently audited exhaustively",
            "Frame-based scoring; container FPS differs from the declared 30fps",
            "A third short player track, if present, is treated as a distractor and needs label review"]}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--data-dir",type=Path,default=ROOT/"data/public/uvy")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/milestone1/uvy_identity_evaluation.json");a=p.parse_args()
    result={"source":"https://zenodo.org/records/21303900", "configuration":"Frozen before evaluation; no tuning on UVY results",
            "sessions":[]}
    for folder in sorted(a.data_dir.glob('tennis*')):
        row=evaluate(folder);result["sessions"].append(row)
        print(row["sessionId"], {k:{m:v for m,v in x.items() if m in ('assignmentPrecision','correctIdentityCoverage','identitySwitches')} for k,x in row["metrics"].items()},flush=True)
    write_json(a.output,result)

if __name__=='__main__':main()
