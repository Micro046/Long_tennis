#!/usr/bin/env python3
"""Frame-by-frame crops around candidate contacts, for exact hit/bounce labelling.

  label_grid.py SESSION_DIR OUT.jpg F1[:near|far] [F2 ...] [--span 12]

For each candidate frame F: frames F-span..F+span (step 2 when span > 8, else 1), cropped around the
player of the given side (from pose.json + report identity) or around the ball, ball marked in magenta.
One row per candidate; the tracked ball y (image px) is printed on each tile to spot the turn-around.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def main(d, out, cands, span=12):
    d = Path(d)
    ball = json.loads((d / "ball.json").read_text())["frames"]
    pose = json.loads((d / "pose.json").read_text())["frames"]
    net_y = None
    cal = json.loads((d / "calibration.json").read_text())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipeline import court
    net_y = float(court.project(np.array(cal["imageFromCourt"]), [(court.CX, court.NET_Y)])[0][1])
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    rows = []
    for c in cands:
        f, side = (c.split(":") + [None])[:2]
        f = int(f)
        # crop centre: the chosen side's person nearest the ball, else the ball
        centre = None
        for g in sorted(range(f - 3, f + 4), key=lambda g: abs(g - f)):
            if 0 <= g < len(pose) and pose[g]:
                ppl = [p for p in pose[g] if side is None or ((p[6] > court.NET_Y) == (side == "near"))
                       and -1.5 < p[5] < 12.5]
                if ppl:
                    b = ball[f] or ball[g] or [ (ppl[0][0] + ppl[0][2]) / 2, (ppl[0][1] + ppl[0][3]) / 2]
                    p = min(ppl, key=lambda p: np.hypot((p[0] + p[2]) / 2 - b[0], (p[1] + p[3]) / 2 - b[1]))
                    centre = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2); ph = p[3] - p[1]
                    break
        if centre is None:
            centre, ph = (ball[f][:2] if ball[f] else (960, 540)), 100
        cw = int(np.clip(ph * 3.2, 220, 520)); ch = int(cw * 0.75)
        step = 2 if span > 8 else 1
        tiles = []
        for g in range(f - span, f + span + 1, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, g); ok, img = cap.read()
            if not ok:
                continue
            H, W = img.shape[:2]
            x1 = int(np.clip(centre[0] - cw / 2, 0, W - cw)); y1 = int(np.clip(centre[1] - ch * 0.55, 0, H - ch))
            t = img[y1:y1 + ch, x1:x1 + cw].copy()
            lab = f"f{g}"
            if 0 <= g < len(ball) and ball[g]:
                bx, by = ball[g][0] - x1, ball[g][1] - y1
                if 0 <= bx < cw and 0 <= by < ch:
                    cv2.circle(t, (int(bx), int(by)), max(8, cw // 30), (255, 0, 255), 1)
                lab += f" y{int(ball[g][1])}"
            t = cv2.resize(t, (240, 180))
            cv2.rectangle(t, (0, 0), (130, 20), (0, 0, 0), -1)
            cv2.putText(t, lab, (3, 15), 0, .45, (0, 255, 255) if g == f else (255, 255, 255), 1)
            tiles.append(t)
        rows.append(np.hstack(tiles))
    w = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.zeros((r.shape[0], w - r.shape[1], 3), np.uint8)]) for r in rows]
    cv2.imwrite(out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    args = [a for a in sys.argv[3:] if not a.startswith("--")]
    span = int(sys.argv[sys.argv.index("--span") + 1]) if "--span" in sys.argv else 12
    if "--span" in sys.argv:
        args.remove(str(span))
    main(sys.argv[1], sys.argv[2], args, span)
