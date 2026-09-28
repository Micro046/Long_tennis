#!/usr/bin/env python3
"""Train the hit/bounce classifier on real TrackNet tracks (RacketVision tracker output, broadcast)
plus simulated phone-view rallies (pipeline/synth.py).

  train_event_model.py --synth 3000 --out models/event_model_synth.pkl [--holdout-game game10]

With --holdout-game the TrackNet game is excluded from training and scored (broadcast sanity check).
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "scripts"))
from pipeline import events as ev, synth  # noqa: E402
import eval_tracknet_events as E  # noqa: E402


def features_labels(clips):
    Xs, ys = [], []
    for c in clips:
        xy, obs = ev.clean_track(c["xy"])
        X, names = ev.trajectory_features(xy, obs, c.get("fps", 30.0))
        y = np.zeros(len(X), int)
        r = max(1, int(round(c.get("fps", 30.0) / 30)))
        for cls in (1, 2):
            for f in np.where(c["status"] == cls)[0]:
                y[max(0, f - r):f + r + 1] = cls
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs), np.concatenate(ys), names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=int, default=3000)
    ap.add_argument("--synth-weight", type=float, default=1.0)
    ap.add_argument("--holdout-game")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "models/event_model_synth.pkl"))
    a = ap.parse_args()
    t0 = time.time()
    real = E.load_clips("pred")
    for c in real:
        c["fps"] = 30.0
    test = [c for c in real if c["game"] == a.holdout_game] if a.holdout_game else []
    train_real = [c for c in real if c["game"] != a.holdout_game]
    sim = synth.generate(a.synth, seed=a.seed) if a.synth else []
    Xr, yr, names = features_labels(train_real)
    parts = [(Xr, yr, np.ones(len(yr)))]
    if sim:
        Xs, ys, _ = features_labels(sim)
        parts.append((Xs, ys, np.full(len(ys), a.synth_weight)))
    X = np.concatenate([p[0] for p in parts]); y = np.concatenate([p[1] for p in parts]); w = np.concatenate([p[2] for p in parts])
    print(f"train frames real {len(yr)} synth {len(y) - len(yr)}; built in {time.time() - t0:.0f}s", flush=True)
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=48, min_samples_leaf=40,
                                       l2_regularization=1.0, class_weight={0: 1, 1: 6, 2: 6}, random_state=0)
    m.fit(X, y, sample_weight=w)
    print(f"trained in {time.time() - t0:.0f}s", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"model": m, "features": names, "trainedOn": {"tracknetGames": sorted({c['game'] for c in train_real}),
                                                               "synthRallies": a.synth}}, open(a.out, "wb"))
    if test:
        res = {1: [0, 0, 0], 2: [0, 0, 0]}
        for c in test:
            xy, obs = ev.clean_track(c["xy"])
            Xc, _ = ev.trajectory_features(xy, obs, 30.0)
            p = m.predict_proba(Xc)
            for cls in (1, 2):
                tp, fp, fn, _ = ev.match_events(ev.peaks(p[:, cls], 0.5, 6), np.where(c["status"] == cls)[0].tolist(), 3)
                res[cls][0] += tp; res[cls][1] += fp; res[cls][2] += fn
        for cls, name in ((1, "hit"), (2, "bounce")):
            tp, fp, fn = res[cls]
            P, R = tp / max(1, tp + fp), tp / max(1, tp + fn)
            print(f"{a.holdout_game} {name}: P {P:.3f} R {R:.3f} F1 {2 * P * R / max(1e-9, P + R):.3f}")


if __name__ == "__main__":
    main()
