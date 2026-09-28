#!/usr/bin/env python3
"""Contact sheets for manual hit/bounce labelling: 1:1 crops centred on the ball position.

Only ball *positions* are used (to frame the crop); no event predictions are drawn.
Usage: label_sheets.py <frames_dir> <ball_json> <out_prefix> [--step 3] [--start F --stop F]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("frames"); ap.add_argument("ball"); ap.add_argument("out")
ap.add_argument("--step", type=int, default=3); ap.add_argument("--start", type=int); ap.add_argument("--stop", type=int)
ap.add_argument("--cols", type=int, default=5); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--cw", type=int, default=480); ap.add_argument("--ch", type=int, default=270)
a = ap.parse_args()

files = sorted(Path(a.frames).glob("*.jpg"))
first = int(files[0].stem)
ball = json.loads(Path(a.ball).read_text())["frames"]
start = a.start if a.start is not None else first
stop = a.stop if a.stop is not None else first + len(files)
last = None
tiles = []
for f in range(start, stop, a.step):
    img = cv2.imread(str(Path(a.frames) / f"{f:06d}.jpg"))
    b = ball[f - first]
    if b:
        last = (b[0], b[1])
    c = last or (img.shape[1] / 2, img.shape[0] / 2)
    x1 = int(np.clip(c[0] - a.cw / 2, 0, img.shape[1] - a.cw)); y1 = int(np.clip(c[1] - a.ch / 2, 0, img.shape[0] - a.ch))
    t = img[y1:y1 + a.ch, x1:x1 + a.cw].copy()
    cv2.putText(t, f"{f}", (5, 22), 0, 0.7, (0, 255, 255), 2)
    cv2.putText(t, f"{x1},{y1}", (5, a.ch - 8), 0, 0.45, (200, 200, 200), 1)
    tiles.append(t)
per = a.cols * a.rows
for s in range(0, len(tiles), per):
    g = tiles[s:s + per]
    while len(g) < per:
        g.append(np.zeros_like(tiles[0]))
    sheet = np.vstack([np.hstack(g[r * a.cols:(r + 1) * a.cols]) for r in range(a.rows)])
    cv2.imwrite(f"{a.out}_{s // per:02d}.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(tiles), "tiles")
