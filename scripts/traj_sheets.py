#!/usr/bin/env python3
"""Labelling aid: raw ball-tracker positions drawn on the video per 3-s segment (no event predictions).
Usage: traj_sheets.py VIDEO BALL_JSON OUT_DIR --windows a:b,c:d  |  --random N --seed S --max-frame F"""
import argparse, json
from pathlib import Path
import cv2, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("ball"); ap.add_argument("out")
ap.add_argument("--windows"); ap.add_argument("--random", type=int); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-frame", type=int); ap.add_argument("--length", type=int, default=360)
a = ap.parse_args()
b = json.loads(Path(a.ball).read_text())["frames"]
W = a.length
if a.windows:
    wins = [[int(x) for x in w.split(":")] for w in a.windows.split(",")]
else:
    top = min(len(b), a.max_frame or len(b))
    pres = [np.mean([f is not None for f in b[s:s + W]]) for s in range(0, top - W, W)]
    cands = [i for i, p in enumerate(pres) if p > 0.5]
    pick = sorted(np.random.default_rng(a.seed).choice(cands, a.random, replace=False).tolist())
    wins = [[i * W, i * W + W] for i in pick]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "windows.json").write_text(json.dumps({"windows": wins}))
cap = cv2.VideoCapture(a.video)
for wi, (s0, z) in enumerate(wins):
    for s in range(s0, z, 90):
        cap.set(1, s + 45); ok, img = cap.read()
        img = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
        pts = [(f, b[f]) for f in range(s, min(z, s + 90)) if b[f]]
        for (f1, p1), (f2, p2) in zip(pts, pts[1:]):
            if f2 - f1 <= 3:
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 200, 255), 1)
        for f, p in pts:
            c = (0, 255, 0) if (f - s) < 30 else (0, 255, 255) if (f - s) < 60 else (255, 100, 255)
            cv2.circle(img, (int(p[0]), int(p[1])), 4, c, -1)
            if f % 3 == 0:
                cv2.putText(img, str(f), (int(p[0]) + 5, int(p[1]) - 5), 0, .45, (255, 255, 255), 1)
        cv2.putText(img, f"frames {s}-{min(z, s + 90) - 1}", (10, 40), 0, 1, (255, 255, 255), 2)
        cv2.imwrite(str(out / f"traj_w{wi}_{s}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
print(wins)
