#!/usr/bin/env python3
"""Figures for the workflow guide (docs/workflow_guide/guide.html), made from a real analysed session.

  make_figures.py SESSION_DIR      -> docs/workflow_guide/fig/*.jpg|png
"""
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipeline import court, events as ev  # noqa: E402

OUT = Path(__file__).parent / "fig"
SK = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def frame(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    return cap.read()[1]


def save(name, img, q=88):
    cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, q])


def main(d):
    d = Path(d); OUT.mkdir(exist_ok=True)
    cal = json.loads((d / "calibration.json").read_text())
    pose = json.loads((d / "pose.json").read_text())
    ball = json.loads((d / "ball.json").read_text())["frames"]
    rep = json.loads((d / "report.json").read_text())
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    H_img = court.image_from_court(cal)
    bg = cv2.imread(str(d / "background.jpg"))
    h, w = bg.shape[:2]

    # 1. calibration: markers the user placed + snapped court lines on the median background
    img = court.draw(bg, H_img, (0, 255, 255), 3)
    for name, (x, y) in cal["markers"].items():
        cv2.circle(img, (int(x * w), int(y * h)), 14, (255, 0, 255), 3)
    save("calibration.jpg", cv2.resize(img, (1280, 720)))

    # 2. median background vs a raw frame
    save("background.jpg", cv2.resize(bg, (960, 540)))
    raw = frame(cap, 544)
    save("raw_frame.jpg", cv2.resize(raw, (960, 540)))

    # 3. players: detections, far tiles, pose skeletons, identity, ball
    f = 543  # pose frame (stride 3)
    img = frame(cap, f)
    for x1, y1, x2, y2 in pose["farTiles"]:
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 200, 0), 2)
    ident = rep["perFrameIdentity"][f]
    names = {"A": "Orange (A)", "B": "Blue (B)"}
    det_to = {v["det"]: k for k, v in ident.items()}
    for di, p in enumerate(pose["frames"][f] or []):
        x1, y1, x2, y2 = map(int, p[:4]); k = np.array(p[7:]).reshape(17, 3)
        col = (0, 140, 255) if det_to.get(di) == "A" else (255, 120, 0) if det_to.get(di) == "B" else (160, 160, 160)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        for a, b in SK:
            if k[a, 2] > .3 and k[b, 2] > .3:
                cv2.line(img, tuple(k[a, :2].astype(int)), tuple(k[b, :2].astype(int)), (0, 255, 0), 2)
        lab = names.get(det_to.get(di), "not a player")
        cv2.putText(img, f"{lab}  court ({p[5]:.1f}, {p[6]:.1f}) m", (x1, max(20, y1 - 8)), 0, 0.7, col, 2)
    if ball[f]:
        cv2.circle(img, (int(ball[f][0]), int(ball[f][1])), 12, (255, 0, 255), 3)
    save("players.jpg", cv2.resize(img, (1280, 720)))

    # 4. ball track: 3 s of detections on one frame
    a, b = 985, 1080
    img = frame(cap, 1030)
    pts = [(i, ball[i]) for i in range(a, b) if ball[i]]
    for (i1, p1), (i2, p2) in zip(pts, pts[1:]):
        if i2 - i1 <= 3:
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 200, 255), 2)
    for i, p in pts:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (255, 0, 255), -1)
        if i % 6 == 0:
            cv2.putText(img, str(i), (int(p[0]) + 6, int(p[1]) - 6), 0, .6, (255, 255, 255), 2)
    for e in rep["events"]["hits"]:
        if a <= e < b and ball[e]:
            cv2.putText(img, "HIT", (int(ball[e][0]) - 20, int(ball[e][1]) - 20), 0, 1.0, (0, 0, 255), 3)
    for e in rep["events"]["bounces"]:
        if a <= e < b and ball[e]:
            cv2.putText(img, "BOUNCE", (int(ball[e][0]) - 40, int(ball[e][1]) + 40), 0, 1.0, (0, 255, 0), 3)
    save("ball_track.jpg", cv2.resize(img, (1280, 720)))

    # 5. tracker input: background + 4 frames at 512x288
    tiles = [cv2.resize(bg, (512, 288))] + [cv2.resize(frame(cap, i), (512, 288)) for i in range(561, 565)]
    for t, lab in zip(tiles, ["background", "frame t-3", "frame t-2", "frame t-1", "frame t"]):
        cv2.rectangle(t, (0, 0), (190, 30), (0, 0, 0), -1); cv2.putText(t, lab, (6, 22), 0, .7, (255, 255, 255), 2)
    save("tracker_input.jpg", np.hstack(tiles))

    # 6. event probabilities over time (the classifier output) with detected events
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = pickle.load(open(ROOT / "models/event_model.pkl", "rb"))
    H = ball_h = rep["video"]["heightPx"]
    xy = np.array([[p[0], p[1]] if p else [np.nan, np.nan] for p in ball], float) / H
    xyc, obs = ev.clean_track(xy)
    X, _ = ev.trajectory_features(xyc, obs, 30.0)
    prob = m["model"].predict_proba(X)
    a, b = 420, 720
    t = np.arange(a, b)
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1]})
    ax[0].plot(t, xyc[a:b, 1] * H, color="#444"); ax[0].invert_yaxis(); ax[0].set_ylabel("ball y (px)\nup = far court")
    ax[1].plot(t, prob[a:b, 1], color="#d33"); ax[1].axhline(0.5, ls="--", color="#999"); ax[1].set_ylabel("P(hit)")
    ax[2].plot(t, prob[a:b, 2], color="#2a2"); ax[2].axhline(0.5, ls="--", color="#999"); ax[2].set_ylabel("P(bounce)")
    for e in rep["events"]["hits"]:
        if a <= e < b:
            for k in ax: k.axvline(e, color="#d33", alpha=.35)
    for e in rep["events"]["bounces"]:
        if a <= e < b:
            for k in ax: k.axvline(e, color="#2a2", alpha=.35)
    ax[2].set_xlabel("frame (30 per second)")
    fig.tight_layout(); fig.savefig(OUT / "event_probs.png", dpi=110); plt.close(fig)

    # 7. an event strip from the review tool
    rv = ROOT / "work/review_test1/event_20.jpg"
    if rv.exists():
        save("event_strip.jpg", cv2.imread(str(rv)))
    print("figures in", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
