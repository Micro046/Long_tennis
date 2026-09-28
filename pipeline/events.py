"""Hit / bounce event detection from a ball track (+ optional player poses).

Per-frame 3-class classifier (none / hit / bounce) over scale-normalised trajectory
features, followed by peak picking. The feature code is deliberately plain numpy so it
can be ported line-by-line to Kotlin for the Android app.
"""
from __future__ import annotations

import numpy as np

SCALES = (1, 2, 3, 5, 8)
MAX_GAP = 6  # frames interpolated across when the ball is briefly missed


def clean_track(xy: np.ndarray, conf=None, max_jump=0.12):
    """xy: (n,2) image px normalised by image height (NaN = missing).

    Removes isolated outliers (a point far from both neighbours that are close to each
    other) and linearly fills gaps up to MAX_GAP frames. Returns (xy_filled, observed_mask).
    """
    xy = xy.copy()
    n = len(xy)
    for i in range(1, n - 1):
        a, b, c = xy[i - 1], xy[i], xy[i + 1]
        if np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all():
            if np.linalg.norm(b - a) > max_jump and np.linalg.norm(b - c) > max_jump and np.linalg.norm(a - c) < max_jump:
                xy[i] = np.nan
    # a point with no finite neighbour within 2 frames on either side is dropped too
    fin = np.isfinite(xy).all(1)
    for i in range(n):
        if fin[i]:
            lo, hi = max(0, i - 2), min(n, i + 3)
            if fin[lo:hi].sum() == 1:
                xy[i] = np.nan
    observed = np.isfinite(xy).all(1)
    idx = np.where(observed)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= MAX_GAP + 1:
            for j in range(a + 1, b):
                t = (j - a) / (b - a)
                xy[j] = xy[a] * (1 - t) + xy[b] * t
    return xy, observed


def _safe(v):
    return np.where(np.isfinite(v), v, 0.0)


def trajectory_features(xy: np.ndarray, observed: np.ndarray, fps: float):
    """Per-frame features. xy normalised by image height, time normalised to 30 fps."""
    n = len(xy)
    r = fps / 30.0
    feats, names = [], []
    fin = np.isfinite(xy).all(1).astype(float)

    def shift(a, k):
        out = np.full_like(a, np.nan)
        if k > 0:
            out[k:] = a[:-k]
        elif k < 0:
            out[:k] = a[-k:]
        else:
            out[:] = a
        return out

    for s in SCALES:
        k = max(1, int(round(s * r)))
        before = (xy - shift(xy, k)) / s   # velocity arriving at t (per 30fps frame)
        after = (shift(xy, -k) - xy) / s   # velocity leaving t
        nb, na = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
        cos = np.sum(before * after, 1) / (nb * na + 1e-9)
        acc = after - before
        feats += [before[:, 0], before[:, 1], after[:, 0], after[:, 1], nb, na,
                  cos, np.log((na + 1e-4) / (nb + 1e-4)), acc[:, 0], acc[:, 1], np.linalg.norm(acc, axis=1),
                  np.sign(before[:, 1]) * np.sign(after[:, 1]),
                  np.abs(after[:, 1]) - np.abs(before[:, 1])]
        names += [f"vbx{s}", f"vby{s}", f"vax{s}", f"vay{s}", f"sb{s}", f"sa{s}", f"cos{s}", f"lsr{s}",
                  f"ax{s}", f"ay{s}", f"an{s}", f"flipy{s}", f"dvy{s}"]
    # kink: residual of t from the line joining t-3 and t+3
    k = max(1, int(round(3 * r)))
    mid = (shift(xy, k) + shift(xy, -k)) / 2
    feats += [np.linalg.norm(xy - mid, axis=1), (xy - mid)[:, 1]]
    names += ["kink3", "kink3y"]
    # observation support
    for w in (3, 8):
        kk = max(1, int(round(w * r)))
        c = np.convolve(observed.astype(float), np.ones(2 * kk + 1), "same") / (2 * kk + 1)
        feats.append(c); names.append(f"obs{w}")
    feats.append(observed.astype(float)); names.append("obs0")
    X = np.stack([_safe(f) for f in feats], 1).astype(np.float32)
    return X, names


def player_features(xy_px: np.ndarray, people_per_frame, img_h: float):
    """Ball-to-player geometry. people_per_frame[i] = list of (box[4], kps (17,3)) in px."""
    n = len(xy_px)
    out = np.zeros((n, 6), np.float32)
    out[:, :] = [5, 5, 5, 0, 0, 0]
    for i in range(n):
        b = xy_px[i]
        ppl = people_per_frame[i] if i < len(people_per_frame) else None
        if not ppl or not np.isfinite(b).all():
            continue
        best = None
        for box, kps in ppl:
            hgt = max(20.0, box[3] - box[1])
            wr = [kps[j, :2] for j in (9, 10) if kps[j, 2] > 0.3]
            dw = min([np.linalg.norm(b - w) for w in wr]) / hgt if wr else 5.0
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            dc = np.linalg.norm(b - [cx, cy]) / hgt
            rel_y = (b[1] - box[1]) / hgt   # 0 = top of head, 1 = feet
            inside = float(box[0] - 0.3 * hgt <= b[0] <= box[2] + 0.3 * hgt and box[1] - 0.3 * hgt <= b[1] <= box[3])
            cand = (dw, dc, rel_y, inside, hgt / img_h)
            if best is None or cand[0] < best[0]:
                best = cand
        if best:
            out[i] = [best[0], best[1], best[2], best[3], best[4], 1]
    return out, ["dWrist", "dCentre", "relY", "nearBox", "boxH", "hasPlayer"]


def peaks(prob: np.ndarray, thr: float, radius: int):
    """Local maxima above thr, at least `radius` frames apart."""
    # deterministic across platforms: 6-decimal probabilities, ties -> earlier frame
    order = np.argsort(-np.round(prob, 6), kind="stable")
    taken = np.zeros(len(prob), bool)
    out = []
    for i in order:
        if prob[i] < thr:
            break
        if taken[max(0, i - radius):i + radius + 1].any():
            continue
        taken[i] = True
        out.append(int(i))
    return sorted(out)


def match_events(pred, truth, tol):
    """One-to-one greedy matching by time distance. Returns tp, fp, fn, offsets."""
    pred, truth = sorted(pred), sorted(truth)
    pairs = sorted((abs(p - t), p, t) for p in pred for t in truth if abs(p - t) <= tol)
    used_p, used_t, offs = set(), set(), []
    for d, p, t in pairs:
        if p in used_p or t in used_t:
            continue
        used_p.add(p); used_t.add(t); offs.append(p - t)
    tp = len(used_p)
    return tp, len(pred) - tp, len(truth) - tp, offs
