"""Persistent player identity (A/B) for a post-session singles recording.

Offline, whole-session algorithm (the phone analyses after recording, so it may look
at the entire session):
  1. tracklets   - link per-frame player detections by court-metre continuity; break on
                   scene cuts, gaps, or ambiguity.
  2. appearance  - per detection: pose-guided shirt + shorts HSV histograms; per tracklet
                   the median descriptor over clean (non-overlapping) frames.
  3. assignment  - seed A/B prototypes from the setup frame (A = the player the user tapped,
                   default: the near player), then iterate: label every tracklet by nearest
                   prototype with the constraint that time-overlapping tracklets get different
                   labels; recompute prototypes from confident tracklets. Low-margin tracklets
                   are left 'unknown' instead of guessed.
Identity is therefore independent of court side: `side` is derived per frame.
"""
from __future__ import annotations

import cv2
import numpy as np

NET_Y = 23.77 / 2
HB, SB, VB = 12, 4, 4  # histogram bins


def region_hist(img, pts, pad=0.0):
    """HSV histogram of the convex polygon spanned by pts (px)."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 3:
        return None
    if pad:
        c = pts.mean(0)
        pts = c + (pts - c) * (1 + pad)
    x1, y1 = np.floor(pts.min(0)).astype(int)
    x2, y2 = np.ceil(pts.max(0)).astype(int)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    crop = img[y1:y2, x1:x2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.round(pts - [x1, y1]).astype(np.int32), 1)
    if mask.sum() < 12:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], mask, [HB, SB], [0, 180, 0, 256]).ravel()
    v = cv2.calcHist([hsv], [2], mask, [VB], [0, 256]).ravel()
    tot = hs.sum()
    return np.concatenate([0.75 * hs / tot, 0.25 * v / v.sum()]).astype(np.float32)


def appearance(img, box, kps):
    """Shirt (shoulders+hips polygon) and shorts (hips+mid-thigh) descriptors; box fallback."""
    k = np.asarray(kps).reshape(17, 3)
    ok = k[:, 2] > 0.3
    parts = []
    if ok[[5, 6, 11, 12]].all():
        torso = k[[5, 6, 12, 11], :2]
        parts.append(region_hist(img, torso, pad=-0.1))
    else:
        x1, y1, x2, y2 = box
        hh = y2 - y1
        parts.append(region_hist(img, [(x1 + .3 * (x2 - x1), y1 + .2 * hh), (x2 - .3 * (x2 - x1), y1 + .2 * hh),
                                       (x2 - .3 * (x2 - x1), y1 + .5 * hh), (x1 + .3 * (x2 - x1), y1 + .5 * hh)]))
    if ok[[11, 12, 13, 14]].all():
        hip = k[[11, 12], :2]
        knee = k[[13, 14], :2]
        mid = hip + 0.5 * (knee - hip)
        parts.append(region_hist(img, [hip[0], hip[1], mid[1], mid[0]]))
    else:
        parts.append(None)
    if parts[0] is None:
        return None
    lower = parts[1] if parts[1] is not None else np.full_like(parts[0], np.nan)
    return np.concatenate([parts[0], lower])


def desc_distance(a, b):
    """Mean Bhattacharyya distance over the available parts (shirt, shorts)."""
    n = len(a) // 2
    ds = []
    for sl in (slice(0, n), slice(n, 2 * n)):
        x, y = a[sl], b[sl]
        if np.isnan(x).any() or np.isnan(y).any():
            continue
        bc = np.sum(np.sqrt(np.clip(x, 0, None) * np.clip(y, 0, None))) / np.sqrt(max(1e-9, x.sum() * y.sum()))
        ds.append(np.sqrt(max(0.0, 1 - bc)))
    return float(np.mean(ds)) if ds else 1.0


def build_tracklets(frames, cuts, fps, gate_m=1.2, max_gap=8):
    """frames[i] = list of dets {court:(x,y), box, kps, desc}. Returns tracklets with frame->det idx."""
    cut_set = set(cuts)
    active, done = [], []
    for i, dets in enumerate(frames):
        if dets is None:
            continue
        if i in cut_set:
            done += active; active = []
        # expire
        still = []
        for t in active:
            (done if i - t["last"] > max_gap else still).append(t)
        active = still
        # greedy association by court distance (with velocity prediction)
        pairs = []
        for ti, t in enumerate(active):
            dt = i - t["last"]
            pred = np.array(t["pos"]) + np.array(t["vel"]) * dt
            for di, d in enumerate(dets):
                dist = np.linalg.norm(np.array(d["court"]) - pred)
                if dist < gate_m + 0.25 * dt:
                    pairs.append((dist, ti, di))
        pairs.sort()
        used_t, used_d = set(), set()
        for dist, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            # ambiguity: another detection almost as close -> end the tracklet instead of guessing
            rivals = [p for p in pairs if p[1] == ti and p[2] != di and p[2] not in used_d and p[0] < dist + 0.3]
            if rivals:
                continue
            used_t.add(ti); used_d.add(di)
            t = active[ti]
            dt = i - t["last"]
            newpos = np.array(dets[di]["court"])
            t["vel"] = (0.6 * np.array(t["vel"]) + 0.4 * (newpos - np.array(t["pos"])) / dt).tolist()
            t["pos"], t["last"] = newpos.tolist(), i
            t["dets"][i] = di
        for di, d in enumerate(dets):
            if di not in used_d:
                active.append({"pos": list(d["court"]), "vel": [0.0, 0.0], "last": i, "start": i, "dets": {i: di}})
    done += active
    for k, t in enumerate(done):
        t["id"] = k
        t["start"], t["end"] = min(t["dets"]), max(t["dets"])
    return done


def tracklet_descriptor(t, frames):
    ds = [frames[f][di]["desc"] for f, di in t["dets"].items()
          if frames[f][di].get("desc") is not None and not frames[f][di].get("overlap")]
    if not ds:
        return None
    ds = np.stack(ds)
    return np.nanmedian(ds, 0) if not np.isnan(ds).all(0).any() else np.nanmedian(np.where(np.isnan(ds), 0, ds), 0)


def is_player_tracklet(t, frames, fps):
    """Players spend time inside the singles court area; officials/ball kids do not."""
    if len(t["dets"]) < max(5, int(0.3 * fps)):
        return False
    xs = np.array([frames[f][di]["court"] for f, di in t["dets"].items()])
    inside = (xs[:, 0] > -1.5) & (xs[:, 0] < 12.5) & (xs[:, 1] > -5.5) & (xs[:, 1] < 29.3)
    return inside.mean() > 0.5


def assign_identities(frames, cuts, fps, seed_frame=None, seed_a_side="near", iterations=4, min_margin=0.04):
    tracks = [t for t in build_tracklets(frames, cuts, fps) if is_player_tracklet(t, frames, fps)]
    for t in tracks:
        t["desc"] = tracklet_descriptor(t, frames)
        t["n"] = len(t["dets"])
    tracks = [t for t in tracks if t["desc"] is not None]
    # seed: first frame (>= seed_frame) where exactly one player tracklet on each half
    protoA = protoB = None
    order = sorted(tracks, key=lambda t: t["start"])
    start = seed_frame or 0
    for i in range(start, len(frames)):
        live = [t for t in order if t["start"] <= i <= t["end"] and i in t["dets"]]
        if len(live) != 2:
            continue
        sides = ["near" if frames[i][t["dets"][i]]["court"][1] > NET_Y else "far" for t in live]
        if sorted(sides) != ["far", "near"]:
            continue
        a = live[sides.index(seed_a_side)]
        b = live[1 - sides.index(seed_a_side)]
        protoA, protoB, seed_at = a["desc"].copy(), b["desc"].copy(), i
        a["seed"], b["seed"] = "A", "B"
        break
    if protoA is None:
        raise RuntimeError("no frame with one player per half to seed identities")
    # overlap graph
    for t in tracks:
        t["overlaps"] = [u["id"] for u in tracks if u is not t and u["start"] <= t["end"] and t["start"] <= u["end"]
                         and len(set(u["dets"]) & set(t["dets"])) > 0]
    by_id = {t["id"]: t for t in tracks}
    for _ in range(iterations):
        for t in tracks:
            da, db = desc_distance(t["desc"], protoA), desc_distance(t["desc"], protoB)
            t["cost"] = {"A": da, "B": db}
            t["label"] = t.get("seed") or ("A" if da < db else "B")
            t["margin"] = abs(da - db)
        # resolve conflicts: overlapping tracklets with the same label -> flip the less confident
        for t in sorted(tracks, key=lambda t: -t["margin"]):
            for oid in t["overlaps"]:
                o = by_id[oid]
                if o["label"] == t["label"] and o["label"] != "unknown":
                    loser = o if (o["margin"], o["n"]) < (t["margin"], t["n"]) else t
                    if loser.get("seed"):
                        loser = t if loser is o else o
                    loser["label"] = "B" if loser["label"] == "A" else "A"
                    loser["margin"] = -loser["margin"]
        conf = [t for t in tracks if t["margin"] > 2 * min_margin]
        for lab in ("A", "B"):
            sel = [t for t in conf if t["label"] == lab]
            if sel:
                w = np.array([t["n"] for t in sel], float)
                D = np.stack([np.nan_to_num(t["desc"]) for t in sel])
                proto = (D * w[:, None]).sum(0) / w.sum()
                if lab == "A":
                    protoA = proto
                else:
                    protoB = proto
    for t in tracks:
        if abs(t["margin"]) < min_margin and not t.get("seed"):
            t["label"] = "unknown"
    # per-frame output
    out = [dict() for _ in frames]
    for t in tracks:
        for f, di in t["dets"].items():
            if t["label"] in ("A", "B"):
                d = frames[f][di]
                prev = out[f].get(t["label"])
                if prev is None or t["margin"] > prev["margin"]:
                    out[f][t["label"]] = {"det": di, "tracklet": t["id"], "margin": float(t["margin"]),
                                          "side": "near" if d["court"][1] > NET_Y else "far"}
    # singles completion: one identified player + exactly one unassigned person in the playing area
    # on the opposite half -> that person is the other player
    for f, dets in enumerate(frames):
        if not dets or len(out[f]) != 1:
            continue
        (pid, a), = out[f].items()
        other = "B" if pid == "A" else "A"
        cands = [i for i, d in enumerate(dets) if i != a["det"] and -1.5 < d["court"][0] < 12.5
                 and -6.0 < d["court"][1] < 29.8 and (d["court"][1] > NET_Y) != (a["side"] == "near")]
        if len(cands) == 1:
            d = dets[cands[0]]
            out[f][other] = {"det": cands[0], "tracklet": -1, "margin": 0.0, "byExclusion": True,
                             "side": "near" if d["court"][1] > NET_Y else "far"}
    return out, tracks, {"seedFrame": seed_at, "protoA": protoA.tolist(), "protoB": protoB.tolist()}
