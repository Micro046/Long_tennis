#!/usr/bin/env python3
"""Print the rally decoder's candidates and decisions for frame ranges of an analysed session.

  rally_debug.py SESSION_DIR 300-450 [900-1000 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import court, identity as idm, rally, session as S  # noqa: E402

d = Path(sys.argv[1]); RANGES = [tuple(map(int, r.split("-"))) for r in sys.argv[2:]]
meta = json.loads((d / "meta.json").read_text())
ball, pose, cal = S.load_inputs(d / "ball.json", d / "pose.json", d / "calibration.json")
n = min(len(ball["frames"]), len(pose["frames"])); ball["frames"] = ball["frames"][:n]; pose["frames"] = pose["frames"][:n]
fps = ball["video"]["fps"]
stride = 1 if meta.get("quality") == "best" else 3
frames = [S.person_list(f) if i % stride == 0 else None for i, f in enumerate(pose["frames"])]
S.compute_appearance(d / "video.mp4", frames, stride=stride)
ident, tracks, seed = idm.assign_identities(frames, pose.get("sceneCuts", []), fps, seed_a_side=meta.get("seedASide", "near"))
Hc, Hi = court.court_from_image(cal), court.image_from_court(cal)
raw = np.array([[f[0], f[1]] if f else [np.nan, np.nan] for f in ball["frames"]], float)
cl, st = rally.suppress_static(raw, fps)
_, _, prob, px = S.detect_events(ball, str(ROOT / "models/event_model.pkl"), ball_px=cl)
R = rally.run(prob, px, frames, ident, fps, Hi, Hc)
print("static ball detections removed:", len(st))
for c in R["candidates"]:
    if not any(a <= c["t"] <= b for a, b in RANGES):
        continue
    if c["type"] == "hit":
        ex = " ".join(f"{k}:{v[0]:.1f}/{v[2]}" for k, v in c["sides"].items())
        ex += f" lv={c.get('leaves')} toss={c.get('toss')} rs={c.get('restruck')} arr={c.get('arrives')}"
    else:
        ex = f"court={tuple(round(v, 1) for v in c['court']) if 'court' in c else None} in={c.get('in')}"
    print(f"{c['t']:6d} {c['type']:6s} p={c['p']:.2f} dead={int(c['dead_evidence'])} sp={c['speed'] and round(c['speed'], 1)} "
          f"{ex} -> {c['role']} {c.get('virtual', '')}")
print("points", [p for p in R["points"] if any(a <= p[0] <= b for a, b in RANGES)])
