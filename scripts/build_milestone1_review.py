#!/usr/bin/env python3
"""Build a portable frame-accurate research review workspace from raw video."""
import argparse
import json
from pathlib import Path
import cv2
from milestone1.core import read_json, file_hash

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs/milestone1/review")
    p.add_argument("--analysis-dir",type=Path,default=ROOT/"outputs/milestone1")
    p.add_argument("--data-dir",type=Path,default=ROOT/"data/public/racketvision")
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    data=[]
    for summary in read_json(a.analysis_dir/"report.json")["sessions"]:
        sid=summary["sessionId"]
        ids=read_json(a.analysis_dir/sid/"identities.json")
        ball=read_json(ROOT/"outputs/normalized"/f"{sid}_ball_observations.json")
        events=read_json(a.analysis_dir/sid/"events.json")["events"]
        frame_dir=a.output_dir/"frames"/sid;frame_dir.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(a.data_dir/f"{sid}.mp4"));i=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            image=cv2.resize(frame,(960,round(frame.shape[0]*960/frame.shape[1])))
            if not cv2.imwrite(str(frame_dir/f"{i:06d}.jpg"),image,[cv2.IMWRITE_JPEG_QUALITY,88]):
                raise RuntimeError("Failed to write review frame")
            i+=1
        cap.release()
        if i!=ids["video"]["frameCount"]:raise ValueError("Review frame count mismatch")
        data.append({"sessionId":sid,"video":ids["video"],"summary":summary,"events":events,
            "identities":[{"players":[{k:v for k,v in player.items() if k in ("playerId","bbox","side")}
                for player in f["players"]],"calibrationUsable":f["calibrationUsable"]} for f in ids["observations"]],
            "ball":[o["image"] for o in ball["observations"]],
            "videoSha256":file_hash(a.data_dir/f"{sid}.mp4")})
    html=(ROOT/"scripts/milestone1/review_template.html").read_text()
    payload=json.dumps(data,allow_nan=False).replace("<","\\u003c")
    (a.output_dir/"index.html").write_text(html.replace("__REVIEW_DATA__",payload))
    print(a.output_dir/"index.html")

if __name__=="__main__":main()
