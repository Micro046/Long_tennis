"""Ball + body features for the hit classifier (identity-free, so broadcast training data works).

For every frame and each court side (near/far) we take the person of that side nearest the ball
(pose runs every few frames; the closest sample within +-2 frames is used) and describe:
  reach      ball distance to the nearest wrist in units of 0.45 x body height
  reachMin   minimum reach over +-3 frames
  wristSpd   fastest wrist speed (body heights / frame) between the pose samples around the frame
  relX,relY  ball position relative to hip centre / head, in body heights
  boxH       body height / image height
  has        1 if a person of that side exists
Concatenated with the ball-trajectory features of pipeline/events.py.
"""
from __future__ import annotations

import numpy as np

LWR, RWR, LHIP, RHIP = 9, 10, 11, 12
NET_Y = 23.77 / 2
NAMES = [f"{s}_{k}" for s in ("near", "far") for k in ("reach", "reachMin", "wristSpd", "relX", "relY", "boxH", "has")]


def _people(p):
    """pose JSON person row -> dict (box, court, kps)."""
    return {"box": p[:4], "court": (p[5], p[6]), "kps": np.array(p[7:], float).reshape(17, 3)}


def _sample(frames, f, radius=2):
    for off in sorted(range(-radius, radius + 1), key=abs):
        g = f + off
        if 0 <= g < len(frames) and frames[g] is not None:
            return g, frames[g]
    return None, None


def _reach(b, d):
    x1, y1, x2, y2 = d["box"]; h = max(20.0, y2 - y1)
    wr = [np.linalg.norm(b - d["kps"][j, :2]) for j in (LWR, RWR) if d["kps"][j, 2] > 0.3]
    return (min(wr) / (0.45 * h)) if wr else np.linalg.norm(b - [(x1 + x2) / 2, (y1 + y2) / 2]) / (0.6 * h)


def side_features(ball_px, pose_frames, img_h):
    """ball_px (n,2) with NaN; pose_frames list of lists of raw person rows or None. Returns (n, 14)."""
    frames = [None if f is None else [_people(p) for p in f] for f in pose_frames]
    n = len(ball_px)
    out = np.zeros((n, len(NAMES)), np.float32)
    out[:, :] = [5, 5, 0, 0, 0, 0, 0] * 2
    reach_series = np.full((n, 2), 5.0)
    for f in range(n):
        b = ball_px[f]
        g, ppl = _sample(frames, f)
        if ppl is None or not np.isfinite(b).all():
            continue
        for si, side in enumerate(("near", "far")):
            cands = [d for d in ppl if (d["court"][1] > NET_Y) == (side == "near") and -2 < d["court"][0] < 13]
            if not cands:
                continue
            d = min(cands, key=lambda d: _reach(b, d))
            k, (x1, y1, x2, y2) = d["kps"], d["box"]; h = max(20.0, y2 - y1)
            r = _reach(b, d)
            reach_series[f, si] = r
            hx = (k[LHIP, 0] + k[RHIP, 0]) / 2 if min(k[LHIP, 2], k[RHIP, 2]) > 0.3 else (x1 + x2) / 2
            head = min([k[j, 1] for j in range(5) if k[j, 2] > 0.3] or [y1 + 0.1 * h])
            # wrist speed between neighbouring pose samples of the same side-nearest person
            spd = 0.0
            g2, ppl2 = _sample(frames, g + 3 if g + 3 < n else g - 3, radius=1)
            if ppl2:
                c2 = [e for e in ppl2 if (e["court"][1] > NET_Y) == (side == "near")]
                if c2:
                    e = min(c2, key=lambda e: np.linalg.norm(np.array(e["court"]) - d["court"]))
                    for j in (LWR, RWR):
                        if k[j, 2] > 0.3 and e["kps"][j, 2] > 0.3:
                            spd = max(spd, np.linalg.norm(e["kps"][j, :2] - k[j, :2]) / h / max(1, abs(g2 - g)))
            base = si * 7
            out[f, base:base + 7] = [r, r, spd, (b[0] - hx) / h, (b[1] - head) / h, h / img_h, 1]
    for si in range(2):  # reachMin over +-3 frames
        rs = reach_series[:, si]
        for f in range(n):
            out[f, si * 7 + 1] = rs[max(0, f - 3):f + 4].min()
    return out
