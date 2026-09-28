#!/usr/bin/env python3
"""Export the sklearn HistGradientBoosting event model to a compact JSON the app evaluates.

Also writes a parity fixture (features + expected probabilities for one track) used by the
Android JVM unit test, so the Kotlin feature code and tree evaluator are verified exactly.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import events as ev  # noqa: E402


def export(model_pkl, out_json):
    d = pickle.load(open(model_pkl, "rb"))
    m = d["model"]
    trees = []
    for it in m._predictors:
        per_class = []
        for p in it:
            n = p.nodes
            per_class.append({
                "f": n["feature_idx"].astype(int).tolist(),
                "t": [float(x) for x in n["num_threshold"]],
                "l": n["left"].astype(int).tolist(),
                "r": n["right"].astype(int).tolist(),
                "leaf": n["is_leaf"].astype(int).tolist(),
                "v": [float(x) for x in n["value"]],
                "ml": n["missing_go_to_left"].astype(int).tolist(),
            })
        trees.append(per_class)
    doc = {"type": "hist_gbm_multiclass", "classes": ["none", "hit", "bounce"], "features": d["features"],
           "baseline": [float(x) for x in np.ravel(m._baseline_prediction)], "trees": trees,
           "featureScales": list(ev.SCALES), "maxGap": ev.MAX_GAP}
    Path(out_json).write_text(json.dumps(doc, separators=(",", ":")))
    return m


def fixture(m, ball_json, out, n=600):
    ball = json.loads(Path(ball_json).read_text())
    H, fps = ball["video"]["heightPx"], ball["video"]["fps"]
    fr = ball["frames"][:n]
    xy = np.array([[f[0], f[1]] if f else [np.nan, np.nan] for f in fr], float) / H
    xyc, obs = ev.clean_track(xy)
    X, _ = ev.trajectory_features(xyc, obs, fps)
    prob = m.predict_proba(X)
    Path(out).write_text(json.dumps({"fps": fps, "height": H, "frames": fr, "features": X.tolist(),
                                     "prob": prob.tolist()}, separators=(",", ":")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "outputs/v2/events/event_model_pred.pkl"))
    ap.add_argument("--out", default=str(ROOT / "android/app/src/main/assets/models/event_gbm.json"))
    ap.add_argument("--fixture-ball")
    ap.add_argument("--fixture-out", default=str(ROOT / "android/app/src/test/resources/event_fixture.json"))
    a = ap.parse_args()
    m = export(a.model, a.out)
    if a.fixture_ball:
        Path(a.fixture_out).parent.mkdir(parents=True, exist_ok=True)
        fixture(m, a.fixture_ball, a.fixture_out)
    print("exported", a.out)
