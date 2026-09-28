#!/usr/bin/env python3
"""Leave-one-game-out evaluation of hit/bounce detection on the TrackNet tennis dataset.

Ball input:  'gt'   = annotated ball positions (oracle upper bound for the event logic)
             'pred' = RacketVision TrackNetV3 ball track (outputs/v2/ball/tracknet), which
                      was NOT trained on this dataset -> realistic end-to-end number.
Events are exhaustively labelled per frame (status 1 = hit, 2 = bounce).
Also trains + saves the final model on all games for use on new footage.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import events as ev  # noqa: E402

DATA = ROOT / "data/public/tracknet/Dataset"
H, W, FPS = 720, 1280, 30.0
TOL = 3  # frames (+-100 ms)


def load_clips(source):
    clips = []
    for lab in sorted(DATA.glob("game*/Clip*/Label.csv")):
        game, clip = lab.parent.parent.name, lab.parent.name
        df = pd.read_csv(lab)
        n = len(df)
        if source == "gt":
            xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
            bad = (df["visibility"].to_numpy() == 0) | ~np.isfinite(xy).all(1) | (xy[:, 0] <= 0)
            xy[bad] = np.nan
        else:
            sub = "ball_student" if source == "student" else "ball"
            p = ROOT / f"outputs/v2/{sub}/tracknet/{game}_{clip}_ball.json"
            if not p.exists():
                continue
            fr = json.loads(p.read_text())["frames"]
            xy = np.array([[f[0], f[1]] if f else [np.nan, np.nan] for f in fr[:n]], float)
        status = df["status"].fillna(0).to_numpy(int)
        clips.append({"game": game, "clip": clip, "xy": xy / H, "status": status})
    return clips


def build(clips):
    Xs, ys, meta = [], [], []
    for c in clips:
        xy, obs = ev.clean_track(c["xy"])
        X, names = ev.trajectory_features(xy, obs, FPS)
        y = np.zeros(len(X), int)
        for cls in (1, 2):
            for f in np.where(c["status"] == cls)[0]:
                y[max(0, f - 1):f + 2] = cls  # tolerate +-1 frame label jitter
        Xs.append(X); ys.append(y)
        meta += [(c["game"], c["clip"], i) for i in range(len(X))]
    return np.concatenate(Xs), np.concatenate(ys), meta, names


def fit(X, y):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                       min_samples_leaf=20, l2_regularization=1.0,
                                       class_weight={0: 1, 1: 6, 2: 6}, random_state=0)
    return m.fit(X, y)


def detect(prob, cls, thr):
    return ev.peaks(prob[:, cls], thr, radius=6)


def evaluate(source, thr_grid=(0.3, 0.4, 0.5, 0.6)):
    clips = load_clips(source)
    games = sorted({c["game"] for c in clips})
    results = {t: {1: [0, 0, 0, []], 2: [0, 0, 0, []]} for t in thr_grid}
    per_game = {}
    for g in games:
        train = [c for c in clips if c["game"] != g]
        test = [c for c in clips if c["game"] == g]
        X, y, _, _ = build(train)
        m = fit(X, y)
        per_game[g] = {}
        for c in test:
            xy, obs = ev.clean_track(c["xy"])
            Xc, _ = ev.trajectory_features(xy, obs, FPS)
            prob = m.predict_proba(Xc)
            for t in thr_grid:
                for cls in (1, 2):
                    tp, fp, fn, off = ev.match_events(detect(prob, cls, t), np.where(c["status"] == cls)[0].tolist(), TOL)
                    r = results[t][cls]
                    r[0] += tp; r[1] += fp; r[2] += fn; r[3] += off
    summary = {}
    for t, r in results.items():
        summary[t] = {}
        for cls, name in ((1, "hit"), (2, "bounce")):
            tp, fp, fn, off = r[cls]
            p, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
            summary[t][name] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(p, 3), "recall": round(rc, 3),
                                "f1": round(2 * p * rc / max(1e-9, p + rc), 3),
                                "meanAbsOffsetFrames": round(float(np.mean(np.abs(off))), 2) if off else None}
    return summary, len(clips), games


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("gt", "pred", "student"), default="gt")
    ap.add_argument("--save-model", action="store_true")
    a = ap.parse_args()
    summary, nclips, games = evaluate(a.source)
    print(json.dumps({"source": a.source, "clips": nclips, "games": games, "toleranceFrames": TOL,
                      "byThreshold": summary}, indent=1))
    out = ROOT / "outputs/v2/events"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"tracknet_logo_{a.source}.json").write_text(json.dumps(
        {"protocol": "leave-one-game-out, 10 broadcast games, exhaustive per-frame labels",
         "source": a.source, "clips": nclips, "toleranceFrames": TOL, "byThreshold": summary}, indent=1))
    if a.save_model:
        X, y, _, names = build(load_clips(a.source))
        m = fit(X, y)
        pickle.dump({"model": m, "features": names}, open(out / f"event_model_{a.source}.pkl", "wb"))
