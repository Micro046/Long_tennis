#!/usr/bin/env python3
"""Check the forehand/backhand side rule on THETIS (55 players, camera in front of the player).

THETIS has no ball, so 'contact' is approximated by the frame of peak wrist speed of the
faster wrist. The app rule: lateral offset of the racket wrist from the hip centre; for a
player facing the camera (our far player), image-left = player's right = forehand side for
a right-hander. Also reports a within-player handedness estimate (majority side).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "outputs/v2/thetis_pose.json").read_text())
LW, RW, LH, RH = 9, 10, 11, 12


def lateral_at_peak(frames):
    K = [np.array(f[4:], float).reshape(17, 3) if f else None for f in frames]
    boxes = [f[:4] if f else None for f in frames]
    best = None
    for w in (LW, RW):
        for i in range(1, len(K) - 1):
            if K[i - 1] is None or K[i + 1] is None or K[i] is None:
                continue
            if min(K[i - 1][w, 2], K[i + 1][w, 2]) < 0.3:
                continue
            h = boxes[i][3] - boxes[i][1]
            sp = np.linalg.norm(K[i + 1][w, :2] - K[i - 1][w, :2]) / h
            if best is None or sp > best[0]:
                best = (sp, i, w)
    if best is None:
        return None
    _, i, w = best
    k = K[i]; h = boxes[i][3] - boxes[i][1]
    cx = (k[LH, 0] + k[RH, 0]) / 2 if min(k[LH, 2], k[RH, 2]) > 0.3 else (boxes[i][0] + boxes[i][2]) / 2
    return (k[w, 0] - cx) / h


rows = []
for key, v in d.items():
    cls, name = key.split("/")
    lat = lateral_at_peak(v["frames"])
    if lat is None:
        continue
    truth = "forehand" if cls.startswith("forehand") else "backhand"
    player = name.split("_")[0]
    rows.append((player, truth, lat))

# rule for a player facing the camera: lateral < 0 -> player's right side -> forehand (right-hander)
res = defaultdict(int)
per_player = defaultdict(list)
for p, t, lat in rows:
    if abs(lat) < 0.08:
        res["abstain"] += 1; continue
    pred = "forehand" if lat < 0 else "backhand"
    res["correct" if pred == t else "wrong"] += 1
    per_player[p].append((t, lat))
# per-player handedness: if the player's forehands mostly sit on image-right, the player is left-handed
lefty = {p: np.mean([np.sign(l) for t, l in v if t == "forehand"] or [0]) > 0 for p, v in per_player.items()}
res2 = defaultdict(int)
for p, t, lat in rows:
    if abs(lat) < 0.08:
        continue
    right_side = lat < 0
    pred = ("forehand" if right_side else "backhand") if not lefty.get(p) else ("backhand" if right_side else "forehand")
    res2["correct" if pred == t else "wrong"] += 1
n = res["correct"] + res["wrong"]
out = {"clips": len(rows), "players": len(per_player), "abstainNearBody": res["abstain"],
       "accuracyAssumingRightHanded": round(res["correct"] / max(1, n), 3),
       "playersEstimatedLeftHanded": sorted(p for p, l in lefty.items() if l),
       "accuracyWithPerPlayerHandedness(oracle-fit)": round(res2["correct"] / max(1, res2["correct"] + res2["wrong"]), 3),
       "note": "contact approximated by peak wrist speed (no ball in THETIS); shadow swings, front view"}
print(json.dumps(out, indent=1))
(ROOT / "outputs/v2/thetis_stroke_rule.json").write_text(json.dumps(out, indent=1))
