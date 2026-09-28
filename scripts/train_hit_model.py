#!/usr/bin/env python3
"""Hit classifier on ball-trajectory + body features (pipeline/hitfeat.py).

Training data: calibrated TrackNet broadcast clips with pose (outputs/v2/pose/tracknet), real tracker
ball tracks, exhaustive per-frame hit labels. Leave-one-game-out evaluation with --logo.
  train_hit_model.py --logo            # broadcast LOGO scores
  train_hit_model.py --out models/hit_model.pkl
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import events as ev, hitfeat  # noqa: E402


def clip_data():
    data = []
    for pp in sorted((ROOT / "outputs/v2/pose/tracknet").glob("*_pose.json")):
        name = pp.stem.replace("_pose", "")
        game, clip = name.split("_")
        lab = pd.read_csv(ROOT / f"data/public/tracknet/Dataset/{game}/{clip}/Label.csv")
        ball = json.loads((ROOT / f"outputs/v2/ball/tracknet/{name}_ball.json").read_text())["frames"]
        pose = json.loads(pp.read_text())
        n = min(len(lab), len(ball), len(pose["frames"]))
        H = pose["video"]["heightPx"]
        xy = np.array([[f[0], f[1]] if f else [np.nan, np.nan] for f in ball[:n]], float)
        data.append({"game": game, "xy": xy, "H": H, "pose": pose["frames"][:n],
                     "status": lab["status"].fillna(0).to_numpy(int)[:n]})
    return data


def features(c, fps=30.0):
    xyc, obs = ev.clean_track(c["xy"] / c["H"])
    Xb, _ = ev.trajectory_features(xyc, obs, fps)
    Xs = hitfeat.side_features(xyc * c["H"], c["pose"], c["H"])
    return np.concatenate([Xb, Xs], 1)


def labels(status):
    y = np.zeros(len(status), int)
    for f in np.where(status == 1)[0]:
        y[max(0, f - 1):f + 2] = 1
    return y


def fit(X, y):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20,
                                          l2_regularization=1.0, class_weight={0: 1, 1: 8}, random_state=0).fit(X, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logo", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "models/hit_model.pkl"))
    a = ap.parse_args()
    data = clip_data()
    for c in data:
        c["X"] = features(c); c["y"] = labels(c["status"])
    games = sorted({c["game"] for c in data})
    print("clips", len(data), "games", games, "hits", sum(int((c["status"] == 1).sum()) for c in data))
    if a.logo:
        tot = {"ball": [0, 0, 0], "ball+body": [0, 0, 0]}
        nb = data[0]["X"].shape[1] - len(hitfeat.NAMES)
        for g in games:
            tr = [c for c in data if c["game"] != g]; te = [c for c in data if c["game"] == g]
            for key, cols in (("ball", slice(0, nb)), ("ball+body", slice(None))):
                m = fit(np.concatenate([c["X"][:, cols] for c in tr]), np.concatenate([c["y"] for c in tr]))
                for c in te:
                    p = m.predict_proba(c["X"][:, cols])[:, 1]
                    tp, fp, fn, _ = ev.match_events(ev.peaks(p, 0.5, 6), np.where(c["status"] == 1)[0].tolist(), 3)
                    tot[key][0] += tp; tot[key][1] += fp; tot[key][2] += fn
        for k, (tp, fp, fn) in tot.items():
            P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
            print(f"LOGO {k}: P {P:.3f} R {R:.3f} F1 {2 * P * R / max(1e-9, P + R):.3f} (tp {tp} fp {fp} fn {fn})")
    m = fit(np.concatenate([c["X"] for c in data]), np.concatenate([c["y"] for c in data]))
    pickle.dump({"model": m, "features": "events.trajectory_features + hitfeat.NAMES"}, open(a.out, "wb"))
    print("saved", a.out)


if __name__ == "__main__":
    main()
