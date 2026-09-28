#!/usr/bin/env python3
"""Full post-session analysis: ball + pose + calibration -> per-player report JSON.

Stages (all portable to the phone):
  identity  -> A/B per frame, independent of court side
  events    -> hit / bounce candidates from the ball track (GBM, trained on TrackNet)
  shots     -> hitter attribution, first-bounce linking, stroke class, landing region, speed
  report    -> per-player totals, region counts + percentages with explicit denominators
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import court, events as ev, identity as idm  # noqa: E402

NOSE, LSH, RSH, LWR, RWR, LHIP, RHIP = 0, 5, 6, 9, 10, 11, 12


# ----------------------------------------------------------------------------- inputs
def load_inputs(ball_path, pose_path, calib_path):
    ball = json.loads(Path(ball_path).read_text())
    pose = json.loads(Path(pose_path).read_text())
    cal = json.loads(Path(calib_path).read_text())
    return ball, pose, cal


def person_list(frame_people):
    out = []
    for p in frame_people or []:
        out.append({"box": p[:4], "score": p[4], "court": (p[5], p[6]), "kps": np.array(p[7:], float).reshape(17, 3)})
    return out


def compute_appearance(video, frames, stride=2):
    """Adds 'desc' (+ 'overlap') to each detection; every `stride` frames (others: None)."""
    cap = cv2.VideoCapture(str(video))
    i = 0
    while i < len(frames):
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0 and frames[i]:
            ok, img = cap.retrieve()
            dets = frames[i]
            for a, d in enumerate(dets):
                d["overlap"] = any(_iou(d["box"], e["box"]) > 0.05 for b, e in enumerate(dets) if b != a)
                d["desc"] = idm.appearance(img, d["box"], d["kps"])
        i += 1
    cap.release()


def _iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


# ----------------------------------------------------------------------------- events
def detect_events(ball, model_path, thr_hit=0.5, thr_bounce=0.5):
    m = pickle.load(open(model_path, "rb"))
    W, H = ball["video"]["widthPx"], ball["video"]["heightPx"]
    fps = ball["video"]["fps"]
    xy = np.array([[f[0], f[1]] if f else [np.nan, np.nan] for f in ball["frames"]], float) / H
    xyc, obs = ev.clean_track(xy)
    X, _ = ev.trajectory_features(xyc, obs, fps)
    prob = m["model"].predict_proba(X)
    hits = ev.peaks(prob[:, 1], thr_hit, radius=int(round(6 * fps / 30)))
    bounces = ev.peaks(prob[:, 2], thr_bounce, radius=int(round(6 * fps / 30)))
    # an event cannot be both: keep the stronger class within +-2 frames
    hits = [h for h in hits if not any(abs(h - b) <= 2 and prob[b, 2] > prob[h, 1] for b in bounces)]
    bounces = [b for b in bounces if not any(abs(h - b) <= 2 and prob[h, 1] >= prob[b, 2] for h in hits)]
    return hits, bounces, prob, xyc * H


# ----------------------------------------------------------------------------- dead-ball filter
def ball_handling(frame, ball_px, frames, ident, fps, H, net_img_y=None):
    """True when, within +-0.5 s, the ball stays around one player's body (bouncing it before a
    serve, catching, picking up). Ball points far from every player (other balls, false
    detections) are ignored; >= 75% of the remaining points must be within that player's reach."""
    r = max(6, round(0.5 * fps))
    idx = [i for i in range(max(0, frame - r), min(len(ball_px), frame + r + 1)) if np.isfinite(ball_px[i]).all()]
    if len(idx) < 6:
        return False
    for pid in ("A", "B"):
        pf, a = nearest_pose(frame, pid, ident, radius=6)
        if not a:
            continue
        d = frames[pf][a["det"]]
        x1, y1, x2, y2 = d["box"]; h = y2 - y1
        near = [i for i in idx if np.hypot(ball_px[i][0] - (x1 + x2) / 2, ball_px[i][1] - (y1 + y2) / 2) < 3 * h]
        if len(near) < 6:
            continue
        inside = [i for i in near if x1 - 0.3 * h < ball_px[i][0] < x2 + 0.3 * h and y1 < ball_px[i][1] < y2 + 0.1 * h]
        if len(inside) / len(near) >= 0.75 and len(near) / len(idx) >= 0.6:
            if net_img_y is None:
                return True
            # the ball must also stay on this player's side of the net for +-1 s: a real shot always
            # has an incoming ball from, or an outgoing ball to, the other half
            w = max(8, round(1.0 * fps))
            ys = [ball_px[i][1] for i in range(max(0, frame - w), min(len(ball_px), frame + w + 1)) if np.isfinite(ball_px[i]).all()]
            if a["side"] == "far" and ys and max(ys) < net_img_y + 0.02 * H:
                return True
            if a["side"] == "near" and ys and min(ys) > net_img_y - 0.02 * H:
                return True
    return False


# ----------------------------------------------------------------------------- shots
def departure_side(frame, ball_px, fps):
    """Camera behind the near baseline: a ball leaving contact upward in the image was hit by the
    near player, downward by the far player. Returns ('near'|'far'|None, mean vy px/frame)."""
    k1, k2 = frame + max(1, round(2 * fps / 30)), frame + max(3, round(9 * fps / 30))
    seg = [ball_px[i] for i in range(k1, min(len(ball_px), k2 + 1)) if np.isfinite(ball_px[i]).all()]
    if len(seg) < 3:
        return None, None
    vy = (seg[-1][1] - seg[0][1]) / max(1, len(seg) - 1)
    if abs(vy) < 0.8:
        return None, float(vy)
    return ("near" if vy < 0 else "far"), float(vy)


def reach_distance(b, d):
    """Ball distance to the racket hand in units of the player's reach (0.45 x body height)."""
    k, (x1, y1, x2, y2) = d["kps"], d["box"]
    h = max(20.0, y2 - y1)
    wr = [np.linalg.norm(b - k[j, :2]) for j in (LWR, RWR) if k[j, 2] > 0.3]
    if wr:
        return min(wr) / (0.45 * h)
    return np.linalg.norm(b - [(x1 + x2) / 2, (y1 + y2) / 2]) / (0.6 * h)


def hitter_for(frame, ball_px, frames, ident, fps):
    """Hitter = the detected person whose racket hand is nearest the ball at contact, over ALL
    detected people (not only identified ones): if that person has no A/B identity the hitter is
    reported unknown instead of falling back to the other player. Searched over +-4 frames."""
    best = None
    for f in range(max(0, frame - 4), min(len(frames), frame + 5)):
        if not frames[f]:
            continue
        b = ball_px[frame] if np.isfinite(ball_px[frame]).all() else ball_px[f]
        if not np.isfinite(b).all():
            continue
        det_to_pid = {a["det"]: pid for pid, a in ident[f].items()}
        for di, d in enumerate(frames[f]):
            r = reach_distance(b, d)
            if best is None or r < best[0]:
                best = (r, det_to_pid.get(di, "unknown"))
    if best is None or best[0] > 1.5:
        return None, None
    return (None if best[1] == "unknown" else best[1]), round(float(best[0]), 3)


def nearest_pose(frame, pid, ident, radius=4):
    """Closest frame (<= radius) that has a pose for player pid -> (frame, assignment)."""
    for off in sorted(range(-radius, radius + 1), key=abs):
        f = frame + off
        if 0 <= f < len(ident) and pid in ident[f]:
            return f, ident[f][pid]
    return None, None


def head_y(k, box):
    """Top-of-head height from face/ear keypoints (visible from behind too), else shoulders - 0.12 h.
    Never from the box top: a raised arm defines the box top."""
    pts = [k[j, 1] for j in range(5) if k[j, 2] > 0.3]
    if pts:
        return min(pts)
    sh = [k[j, 1] for j in (5, 6) if k[j, 2] > 0.3]
    return (min(sh) - 0.12 * (box[3] - box[1])) if sh else None


def stroke_class(frame, pid, side, ball_px, frames, ident, prev_hit_gap_s, bounced_before, handed="right"):
    """Stroke class from pose around contact (validated on 27 hand-labelled shots of the amateur match).

    serve    : a wrist above the nose in the 4 frames before contact (or at contact) AND the hitter within 1.2 m of
               (or behind) their own baseline.
    overhead : wrist above the nose elsewhere on court.
    volley   : no bounce since the previous shot and within 7 m of the net.
    FH / BH  : follow-through side: mean position of the racket-hand wrist (COCO right_wrist for a
               right-hander) 4-12 frames after contact relative to the hip centre, in body heights.
               A forehand finishes across the body (player's non-racket side), a backhand on the
               racket side. Seen from behind (near player) the player's left is image-left; the far
               player faces the camera, so the sign flips.
    """
    pf, a = nearest_pose(frame, pid, ident)
    if not a:
        return "unknown", {}
    d = frames[pf][a["det"]]
    k, box = d["kps"], d["box"]
    hgt = max(20.0, box[3] - box[1])
    y = d["court"][1]
    at_baseline = (y > court.LENGTH - 1.2) if side == "near" else (y < 1.2)
    wrist_up = False
    for f in range(frame - 4, frame + 2):   # up to contact only: follow-throughs also lift the wrist
        if 0 <= f < len(ident) and pid in ident[f]:
            kk = frames[f][ident[f][pid]["det"]]["kps"]; bb = frames[f][ident[f][pid]["det"]]["box"]
            head = head_y(kk, bb)
            if head is not None and any(kk[j, 2] > 0.3 and kk[j, 1] < head for j in (LWR, RWR)):
                wrist_up = True
    if wrist_up:
        return ("serve" if at_baseline else "overhead"), {"wristAboveHead": True, "atBaseline": bool(at_baseline)}
    if not bounced_before:
        dist_net = abs(y - court.NET_Y)
        if dist_net < 7.0:
            return "volley", {"distanceToNetM": round(float(dist_net), 2)}
    racket = RWR if handed == "right" else LWR
    xs = []
    for f in range(frame + 4, frame + 13):
        if 0 <= f < len(ident) and pid in ident[f]:
            dd = frames[f][ident[f][pid]["det"]]; kk, bb = dd["kps"], dd["box"]
            if kk[racket, 2] <= 0.3:
                continue
            hx = (kk[LHIP, 0] + kk[RHIP, 0]) / 2 if min(kk[LHIP, 2], kk[RHIP, 2]) > 0.3 else (bb[0] + bb[2]) / 2
            xs.append((kk[racket, 0] - hx) / max(20.0, bb[3] - bb[1]))
    if not xs:
        return "unknown", {}
    ft = float(np.mean(xs))
    s = ft if side == "near" else -ft          # player's-right-positive
    if handed != "right":
        s = -s
    return ("forehand" if s < 0.1 else "backhand"), {"followThroughBodyHeights": round(ft, 3)}


def rally_valid(hits, bounces, ball_px, frames, ident, H_court, fps, window_s=2.5):
    """A shot must be answered within window_s by a hit of the opponent or a bounce in the opponent's
    half (or a bounce with unknown position). Returns the kept hits and the removed ones."""
    att = {h: hitter_for(h, ball_px, frames, ident, fps)[0] for h in hits}
    keep, removed = [], []
    for h in sorted(hits):
        pid = att[h]
        side = nearest_pose(h, pid, ident)[1]["side"] if pid and nearest_pose(h, pid, ident)[1] else None
        lim = h + window_s * fps
        ok = False
        for g in sorted([x for x in hits if h < x <= lim] + [x for x in bounces if h + 2 < x <= lim]):
            if g in att:  # a hit: answered unless both hitters are known and identical
                ok = not (att[g] is not None and pid is not None and att[g] == pid)
            else:        # a bounce
                if side is None or not np.isfinite(ball_px[g]).all():
                    ok = True
                else:
                    y = court.to_court(H_court, ball_px[g])[1]
                    ok = (y < court.NET_Y) if side == "near" else (y > court.NET_Y)
            break
        (keep if ok else removed).append(h)
    return keep, removed


def attribute_and_dedupe(hits, ball_px, frames, ident, fps, window_s=1.5):
    """Attribute every hit, then drop the earlier of two consecutive same-player hits within window_s
    (serve toss release, ball feed): a player cannot hit twice in a row."""
    att = [(h, *hitter_for(h, ball_px, frames, ident, fps)) for h in sorted(hits)]
    keep, dropped = [], []
    for i, (h, pid, d) in enumerate(att):
        nxt = att[i + 1] if i + 1 < len(att) else None
        if nxt and pid in ("A", "B") and nxt[1] == pid and (nxt[0] - h) / fps < window_s:
            dropped.append(h); continue
        keep.append(h)
    return keep, dropped


def build_shots(hits, bounces, ball_px, frames, ident, H_court, fps, handed=None):
    handed = handed or {}
    shots = []
    hits = sorted(hits)
    bounces = sorted(bounces)
    for n, h in enumerate(hits):
        pid, dist = hitter_for(h, ball_px, frames, ident, fps)
        side = nearest_pose(h, pid, ident)[1]["side"] if pid and nearest_pose(h, pid, ident)[1] else None
        nxt = hits[n + 1] if n + 1 < len(hits) else h + int(4 * fps)
        prv = hits[n - 1] if n else None
        gap = (h - prv) / fps if prv is not None else None
        bounced_before = prv is not None and any(prv < b < h for b in bounces)
        first_b = next((b for b in bounces if h + 2 < b < min(nxt, h + int(3.5 * fps))), None)
        shot = {"hitFrame": h, "timeS": round(h / fps, 3), "playerId": pid or "unknown", "side": side,
                "attributionDistanceBodyHeights": None if dist is None else round(float(dist), 3)}
        shot["stroke"], shot["strokeEvidence"] = (stroke_class(h, pid, side, ball_px, frames, ident, gap,
                                                             bounced_before or gap is None or gap > 2.5,
                                                             handed.get(pid, "right"))
                                                  if pid else ("unknown", {}))
        if first_b is None:
            shot["landing"] = {"status": "no_bounce_before_next_hit" if nxt - h < 3.5 * fps else "not_observed"}
        elif not np.isfinite(ball_px[first_b]).all():
            shot["landing"] = {"status": "bounce_position_missing", "bounceFrame": first_b}
        else:
            pt = court.to_court(H_court, ball_px[first_b])
            reg = region(pt, side)
            shot["landing"] = {"status": reg["status"], "region": reg.get("region"), "bounceFrame": first_b,
                               "courtM": [round(float(pt[0]), 2), round(float(pt[1]), 2)]}
            # average horizontal speed from the hitter's position to the bounce point
            pf, pa = nearest_pose(h, pid, ident) if pid else (None, None)
            if pa:
                src = np.array(frames[pf][pa["det"]]["court"], float)
                dt = (first_b - h) / fps
                dist_m = float(np.linalg.norm(pt - src))
                # plausibility: > 230 km/h average flight speed means a wrong hit/bounce pairing
                if dt > 0.15 and 3 < dist_m < 32 and dist_m / dt * 3.6 <= 230:
                    shot["speed"] = {"avgKmh": round(dist_m / dt * 3.6, 1), "distanceM": round(dist_m, 2),
                                     "flightS": round(dt, 3), "definition": "average ground-projected speed from hitter position to first bounce"}
        shots.append(shot)
    return shots


def drop_repeat_hits(shots, fps, window_s=1.5):
    """A player cannot legally hit twice in a row: the earlier of two consecutive same-player hits
    within window_s (serve toss release, ball feed, catch) is removed."""
    keep, dropped = [], []
    for i, s in enumerate(shots):
        nxt = shots[i + 1] if i + 1 < len(shots) else None
        if (nxt and s["playerId"] in ("A", "B") and nxt["playerId"] == s["playerId"]
                and (nxt["hitFrame"] - s["hitFrame"]) / fps < window_s):
            dropped.append(s["hitFrame"]); continue
        keep.append(s)
    return keep, dropped


def region(pt, hitter_side, buffer_m=0.15):
    """Six regions of the opponent's singles half, oriented from the hitter's view."""
    if hitter_side not in ("near", "far") or pt is None:
        return {"status": "unknown"}
    x, y = float(pt[0]), float(pt[1])
    # map into the target half with origin at the opponent's baseline, x from hitter's left
    if hitter_side == "near":        # target = far half (y 0..NET), hitter looks away from camera
        depth_from_net, lat = court.NET_Y - y, x - court.SL
    else:                            # target = near half, hitter faces camera (left/right flip)
        depth_from_net, lat = y - court.NET_Y, (court.WIDTH - court.SL) - x
    if depth_from_net < -buffer_m:
        return {"status": "hitter_half"}
    if lat < -buffer_m or lat > court.SINGLES + buffer_m or depth_from_net > court.NET_Y + buffer_m:
        return {"status": "out"}
    lateral = ("left", "centre", "right")[min(2, max(0, int(lat / (court.SINGLES / 3))))]
    depth = "short" if depth_from_net < court.SERVICE else "deep"
    near_line = (min(abs(lat), abs(lat - court.SINGLES), abs(depth_from_net - court.NET_Y)) < buffer_m)
    return {"status": "in" if not near_line else "in_near_line", "region": f"{depth}_{lateral}"}


def estimate_handedness(shots, frames, ident):
    """Serve arm = wrist highest at serve contact; majority vote per player."""
    votes = {"A": Counter(), "B": Counter()}
    for s in shots:
        if s["stroke"] != "serve" or s["playerId"] not in ("A", "B"):
            continue
        pf, a = nearest_pose(s["hitFrame"], s["playerId"], ident)
        if not a:
            continue
        k = frames[pf][a["det"]]["kps"]
        if k[LWR, 2] > .3 and k[RWR, 2] > .3:
            # image-left/right of the higher wrist, converted to the player's own left/right
            hi = LWR if k[LWR, 1] < k[RWR, 1] else RWR
            other = RWR if hi == LWR else LWR
            img_right = k[hi, 0] > k[other, 0]
            player_right = img_right if s["side"] == "near" else not img_right
            votes[s["playerId"]]["right" if player_right else "left"] += 1
    return {p: (v.most_common(1)[0][0] if sum(v.values()) >= 3 else "right") for p, v in votes.items()}, \
        {p: dict(v) for p, v in votes.items()}


def summarize(shots, names):
    players = {}
    for pid in ("A", "B"):
        sel = [s for s in shots if s["playerId"] == pid]
        strokes = Counter(s["stroke"] for s in sel)
        landed = [s for s in sel if s["landing"].get("region")]
        regions = Counter(s["landing"]["region"] for s in landed)
        out = sum(1 for s in sel if s["landing"].get("status") == "out")
        speeds = [s["speed"]["avgKmh"] for s in sel if "speed" in s]
        serve_speeds = [s["speed"]["avgKmh"] for s in sel if "speed" in s and s["stroke"] == "serve"]
        players[pid] = {
            "name": names.get(pid, pid), "shots": len(sel),
            "strokes": {k: strokes.get(k, 0) for k in ("serve", "forehand", "backhand", "volley", "overhead", "unknown")},
            "landing": {"inCourtFirstBounces": len(landed), "out": out,
                        "noObservedBounce": len(sel) - len(landed) - out,
                        "denominator": "shots with a detected first bounce inside the opponent's singles half",
                        "regions": {r: {"count": regions.get(r, 0),
                                        "percent": round(100 * regions.get(r, 0) / len(landed), 1) if landed else None}
                                    for r in ("deep_left", "deep_centre", "deep_right", "short_left", "short_centre", "short_right")}},
            "speed": {"shotsWithEstimate": len(speeds),
                      "medianAvgKmh": round(float(np.median(speeds)), 1) if speeds else None,
                      "maxAvgKmh": round(float(np.max(speeds)), 1) if speeds else None,
                      "serveMedianAvgKmh": round(float(np.median(serve_speeds)), 1) if serve_speeds else None,
                      "definition": "average ground-projected speed hitter->first bounce (not speed off the racket)"},
        }
    return players


def run(video, ball_path, pose_path, calib_path, model_path, out_path, names=None, seed_a_side="near", pose_stride=3,
        handed=None):
    ball, pose, cal = load_inputs(ball_path, pose_path, calib_path)
    fps = ball["video"]["fps"]
    H_court = np.array(cal["courtFromImage"])
    # the phone runs pose on every `pose_stride`-th frame; mimic that exactly
    frames = [person_list(f) if i % pose_stride == 0 else None for i, f in enumerate(pose["frames"])]
    compute_appearance(video, frames, stride=pose_stride)
    ident, tracks, seed = idm.assign_identities(frames, pose.get("sceneCuts", []), fps, seed_a_side=seed_a_side)
    hits, bounces, prob, ball_px = detect_events(ball, model_path)
    Himg = ball["video"]["heightPx"]
    net_img_y = float(court.project(np.array(cal["imageFromCourt"]), [(court.CX, court.NET_Y)])[0][1])
    dead = {"hits": [h for h in hits if ball_handling(h, ball_px, frames, ident, fps, Himg, net_img_y)],
            "bounces": [b for b in bounces if ball_handling(b, ball_px, frames, ident, fps, Himg, net_img_y)]}
    hits = [h for h in hits if h not in dead["hits"]]
    bounces = [b for b in bounces if b not in dead["bounces"]]
    hits, dead["sameHitterRepeats"] = attribute_and_dedupe(hits, ball_px, frames, ident, fps)
    hits, dead["unanswered"] = rally_valid(hits, bounces, ball_px, frames, ident, H_court, fps)
    handed = dict(handed or {"A": "right", "B": "right"})   # from setup (default right-handed)
    shots = build_shots(hits, bounces, ball_px, frames, ident, H_court, fps, handed)
    _, votes = estimate_handedness(shots, frames, ident)    # reported as a hint only
    report = {
        "schemaVersion": "2.0", "type": "session_report", "ballPath": str(ball_path),
        "video": ball["video"], "fps": fps,
        "players": summarize(shots, names or {}),
        "handedness": handed, "handednessSource": "setup (default right)", "handednessVotesHint": votes,
        "identity": {"seedFrame": seed["seedFrame"], "tracklets": len(tracks),
                     "framesWithA": sum(1 for f in ident if "A" in f), "framesWithB": sum(1 for f in ident if "B" in f)},
        "events": {"hits": hits, "bounces": bounces, "ballHandlingRemoved": dead},
        "shots": shots,
        "perFrameIdentity": [{k: {"det": v["det"], "side": v["side"]} for k, v in f.items()} for f in ident],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, separators=(",", ":")))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--ball", required=True); ap.add_argument("--pose", required=True)
    ap.add_argument("--calibration", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--names", default="A=Player A,B=Player B")
    a = ap.parse_args()
    names = dict(kv.split("=", 1) for kv in a.names.split(","))
    r = run(a.video, a.ball, a.pose, a.calibration, a.model, a.out, names)
    print(json.dumps({k: r[k] for k in ("players", "handedness", "identity")}, indent=1))
