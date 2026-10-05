"""Rally decoding: turn raw hit/bounce candidates into the events of real points.

The event classifier only sees the ball's motion. It cannot tell a rally bounce from a dribble before
a serve, a shot from a catch or a feed between points, or which player hit when the far player overlaps
the ball in the image. Tennis itself constrains all of that, identically for every video:

  * a point starts with a hit from a standing start (serve, drop-hit) -- or, if that hit was not
    detected, with its first bounce;
  * in play the ball alternates: hit by side S -> (bounce on the other side) -> hit by the other side,
    or a volley (hit without a bounce);
  * a point ends on a bounce outside the court, a net ball (bounce on the hitter's own side next to
    the net), a second bounce, or when nothing happens for a few seconds;
  * everything outside points is dead ball (catching, dribbling, feeding the ball back).

`decode` finds the best explanation of all candidates under these rules by dynamic programming. Every
candidate is explained as (a) an in-play event, (b) dead-ball activity (only between points), or (c) a
false detection. Scores are log-probabilities from the classifier plus physical evidence (a bounce
must lie on the court floor; a hitter must be within reach in metres; a dribble is a bounce next to a
standing player with the ball never leaving him). The hitter's side comes from the rally structure,
so the far player's small image size no longer decides who hit.

Also here: suppression of stationary balls (spare balls lying on court steal the tracker) and virtual
serve contacts when the ball is tossed out of the top of the frame.
"""
from __future__ import annotations

import numpy as np

from pipeline import court

NEAR, FAR = "near", "far"
OTHER = {NEAR: FAR, FAR: NEAR}

# timing limits (seconds)
HIT_TO_BOUNCE = (0.2, 3.0)
HIT_TO_VOLLEY = (0.3, 2.5)
BOUNCE_TO_HIT = (0.03, 1.0)
BOUNCE_TO_BOUNCE2 = (0.2, 2.0)
V_MAX_MS = 50.0               # no shot covers ground faster than this on average (fast serves ~45 m/s)
V_MAX_BOUNCED_MS = 35.0       # after a bounce the ball is slower
TIMING_SLACK_S = 0.12         # hit/bounce frames and far-court positions are uncertain
NET_BALL_ZONE_M = 7.0         # a bounce this close to the net on the hitter's own side = net ball
IN_MARGIN_M = 0.3             # tolerance on the singles lines for an "in" bounce
PLAUSIBLE = (-5.0, 16.0, -9.0, 33.0)   # x0, x1, y0, y1: a bounce must project inside this area
OUT_PENALTY = -0.7            # continuing a rally after a bounce judged out (players often play them)
DRIBBLE_M, DRIBBLE_S = 1.5, 1.3   # two bounces this close in space and time = dribbling
DRIBBLE_MAX_SPEED = 12.0      # ... if the ball is slow around both (a ball in play is faster)

# scores
C_START_HIT = -0.5            # starting a point with a hit
C_START_BOUNCE = -3.0         # starting a point with a bounce (its hit was missed)
Q_DEAD_EVIDENT = 0.9          # candidate with dead-ball evidence explained as dead ball
Q_DEAD_OTHER = 0.1            # candidate without such evidence explained as dead ball
REACH_OK_M = 2.0              # wrist-to-ball distance (metres) that is certainly within reach
REACH_MAX_PENALTY = -8.0      # a ball metres away from a player in the image cannot be his hit
NO_BALL = -1.0                # no ball observed around the candidate at all
ABSENT_PLAYER = -0.3          # no identified player on the hitter's side (often just not detected)
NO_FLIGHT = -2.5              # the ball does not fly away after the 'hit' (caught, held)
MIN_POINT_GAP_S = 3.0         # points are separated by at least this (fetch ball, walk, serve)
C_RESTRUCK_RALLY = -1.5       # a rally 'hit' struck again by the same side before landing (bounce-up, catch)
NO_ARRIVAL = -2.0             # a rally hit needs the ball seen flying in just before it (not held in a hand)
C_RESTRUCK = -2.5             # a point-start hit after which the same side strikes the ball again before it lands


# ---------------------------------------------------------------------------------------- track
def suppress_static(ball_px, fps, max_step_px=4.0, min_len=12, min_span_s=0.6, max_speed_px=2.0):
    max_speed_px *= 30.0 / fps      # thresholds are per frame at 30 fps
    """Detections that belong to a ball that barely moves (lying or rolling slowly on the court, held in
    a hand) are removed: a chain of detections, each within a few pixels of the previous one, lasting
    at least `min_span_s` with a mean speed below `max_speed_px` per frame. The real ball in play never
    moves that slowly for that long. The tracker often alternates between the real ball and such a
    ball; removing the still one leaves short gaps that the event features interpolate."""
    ball_px = np.array(ball_px, float)
    n = len(ball_px)
    chains = []           # each: [last_frame, last_xy, members]
    for f in range(n):
        p = ball_px[f]
        if not np.isfinite(p).all():
            continue
        best = None
        for ch in chains:
            gap = f - ch[0]
            if 0 < gap <= 6:
                d = np.linalg.norm(p - ch[1])
                if d <= max_step_px + 0.5 * gap and (best is None or d < best[0]):
                    best = (d, ch)
        if best:
            ch = best[1]; ch[0] = f; ch[1] = p; ch[2].append(f)
        else:
            chains.append([f, p, [f]])
        if len(chains) > 400:
            chains = [c for c in chains if f - c[0] <= 6]
    removed = []
    for _, _, m in chains:
        if len(m) < min_len or (m[-1] - m[0]) < min_span_s * fps:
            continue
        pts = ball_px[m]
        steps = np.linalg.norm(np.diff(pts, axis=0), axis=1) / np.maximum(1, np.diff(m))
        drift = np.linalg.norm(pts[-1] - pts[0]) / max(1, m[-1] - m[0])
        # detector jitter adds 2-4 px per frame; judge by the typical step and the overall drift
        if np.median(steps) <= 3.0 and drift <= max_speed_px:
            removed += m
    out = ball_px.copy()
    out[removed] = np.nan
    return out, sorted(removed)


def toss_serves(ball_px, fps, min_rise=60.0):
    """Serve contacts the tracker cannot see: the ball rises almost vertically (a toss, >= min_rise px),
    then is lost -- above the top of the frame or against the dark background -- and reappears within
    1.5 s at about the same height or higher, already flying fast. The contact is just before it
    reappears. Returns [(contact_frame_estimate, toss_last_seen_frame)]."""
    n = len(ball_px)
    fin = np.isfinite(ball_px).all(1)
    out = []
    f = 1
    while f < n - 1:
        # end of an observed run followed by a gap of >= 5 frames
        gap_min = max(3, int(round(0.17 * fps)))
        if fin[f] and not fin[f + 1] and not any(fin[f + 1:f + 1 + gap_min]):
            run = [f]                     # the rising part only: walk back while the ball was lower
            k = f - 1
            while k >= 0 and fin[k] and f - k < int(0.8 * fps) and ball_px[k][1] >= ball_px[run[0]][1] - 6:
                run.insert(0, k); k -= 1
            if len(run) >= 4:
                dy = ball_px[run[0]][1] - ball_px[f][1]
                dx = abs(ball_px[run[0]][0] - ball_px[f][0])
                ys = [ball_px[g][1] for g in run]
                rising = all(a_ >= b_ - 6 for a_, b_ in zip(ys, ys[1:]))
                if dy >= min_rise and dx <= 0.5 * dy and rising:
                    H_img = max(500.0, float(np.nanmax(ball_px[:, 1])) if np.isfinite(ball_px[:, 1]).any() else 1080.0)
                    step = max(3, int(round(4 * fps / 30)))
                    # first reappearance (within 1.5 s) that is already flying fast; isolated detections
                    # at the top of the toss are skipped
                    for g in range(f + 1, min(n - 2, f + int(1.5 * fps))):
                        if not fin[g]:
                            continue
                        nx = [k for k in range(g + 1, min(n, g + step)) if fin[k]]
                        if nx and ball_px[g][1] <= ball_px[f][1] + 0.15 * H_img and \
                                np.linalg.norm(ball_px[nx[0]] - ball_px[g]) / (nx[0] - g) >= 15 * 30.0 / fps:
                            out.append((g - 1, f))
                            f = g
                            break
        f += 1
    return out


def is_toss(h, ball_px, fps, next_hit=None, top=60):
    """A toss release (serve toss, bouncing the ball on the racket), not a shot: from the candidate the
    ball goes straight up and, before the next hit candidate, comes back down to about where it started
    (it is caught or struck again there), or it leaves the top of the frame going straight up. A lob
    also goes up but then keeps falling far past its starting height."""
    n = len(ball_px)
    if not np.isfinite(ball_px[h]).all():
        prev = [g for g in range(h - 2, h + 3) if 0 <= g < n and np.isfinite(ball_px[g]).all()]
        if not prev:
            return False
        h = prev[0]
    x0, y0 = ball_px[h]
    end = min(n, h + int(1.2 * fps), next_hit if next_hit is not None and next_hit > h + 3 else n)
    seg = [(g, ball_px[g]) for g in range(h + 1, end) if np.isfinite(ball_px[g]).all()]
    if len(seg) < 4:
        return False
    gmin, pmin = min(seg, key=lambda s: s[1][1])
    dy = y0 - pmin[1]
    if dy < 40 or abs(pmin[0] - x0) > max(40.0, 0.3 * dy):
        return False
    if pmin[1] < top:                  # left the frame at the top going straight up
        return True
    after = [q for g, q in seg if g > gmin]
    if not after:
        return False
    lowest = max(q[1] for q in after)
    came_back = lowest >= pmin[1] + 0.4 * dy
    stayed = lowest <= y0 + 0.5 * dy + 20 and all(abs(q[0] - x0) < max(60.0, 0.5 * dy) for q in after)
    return bool(came_back and stayed)


def ball_leaves(h, ball_px, fps, min_px=30.0):
    """After a real shot the ball keeps flying and stays visible; after a catch it disappears into the
    hand or barely moves. True / False, or None near the end of the video."""
    n = len(ball_px)
    a, z = h + 2, h + int(round(0.67 * fps))
    if z >= n:
        return None
    obs = [g for g in range(a, z + 1) if np.isfinite(ball_px[g]).all()]
    if len(obs) < 5:
        return False
    return bool(max(np.linalg.norm(ball_px[g] - ball_px[obs[0]]) for g in obs) >= min_px)


def serve_contact_after_toss(h, ball_px, fps):
    """For a toss released at h that stays in the frame: the contact is where the falling ball suddenly
    speeds up after the apex (the racket hits it downward-forward). Returns a frame or None."""
    n = len(ball_px)
    seg = [g for g in range(h, min(n, h + int(1.6 * fps))) if np.isfinite(ball_px[g]).all()]
    if len(seg) < 6:
        return None
    apex = min(seg, key=lambda g: ball_px[g][1])
    post = [g for g in seg if g >= apex]
    for a_, b_, c_ in zip(post, post[1:], post[2:]):
        v1 = np.linalg.norm(ball_px[b_] - ball_px[a_]) / (b_ - a_)
        v2 = np.linalg.norm(ball_px[c_] - ball_px[b_]) / (c_ - b_)
        k = 30.0 / fps
        if v2 > 2.2 * max(v1, 2.0 * k) and v2 > 8 * k:
            return b_
    return None


def toss_apexes(ball_px, fps, min_rise=60.0):
    """Serve tosses inside the frame: the ball rises almost vertically by >= min_rise px within 0.8 s to a
    top point and falls again. Returns [(apex_frame, release_frame)]."""
    n = len(ball_px)
    fin = np.isfinite(ball_px).all(1)
    out = []
    w = max(3, int(round(0.17 * fps)))
    for g in range(w, n - w):
        if not fin[g]:
            continue
        y = ball_px[g][1]
        nb = [k for k in range(g - w, g + w + 1) if fin[k] and k != g]
        if len(nb) < w or any(ball_px[k][1] < y for k in nb):
            continue
        # walk back along the rising part
        rise = [k for k in range(max(0, g - int(0.8 * fps)), g) if fin[k]]
        if len(rise) < 4:
            continue
        k0 = max(rise, key=lambda k: ball_px[k][1])
        dy = ball_px[k0][1] - y
        dx = abs(ball_px[k0][0] - ball_px[g][0])
        if dy >= min_rise and dx <= 0.3 * dy:
            ys = [ball_px[k][1] for k in rise if k >= k0]
            if all(a_ >= b_ - 6 for a_, b_ in zip(ys, ys[1:])):   # monotonic rise (jitter tolerated)
                out.append((g, k0))
    return out


# ---------------------------------------------------------------------------------------- evidence
def ground_scale(H_img, H_court, pt):
    g = np.asarray(court.to_court(H_court, pt), float)
    a = court.project(H_img, [g, g + [1, 0], g + [0, 1]])
    return (np.linalg.norm(a[1] - a[0]) + np.linalg.norm(a[2] - a[0])) / 2


def reach_m(b, det, H_img, H_court):
    """Ball-to-racket-hand distance in metres at the player's depth: horizontal pixels are converted
    with the court scale at the player's feet, vertical pixels with the body height (~1.75 m)."""
    x1, y1, x2, y2 = det["box"]
    feet = ((x1 + x2) / 2, y2)
    pxm_h = max(1e-3, ground_scale(H_img, H_court, feet))
    pxm_v = max(1e-3, (y2 - y1) / 1.75)
    k = det["kps"]
    pts = [k[j, :2] for j in (9, 10) if k[j, 2] > 0.3] or [np.array([(x1 + x2) / 2, (y1 + y2) / 2])]
    return float(min(np.hypot((b[0] - p[0]) / pxm_h, (b[1] - p[1]) / pxm_v) for p in pts))


def side_players(frames, ident, f, radius=4):
    """{side: det} for the identified players nearest in time to frame f."""
    out = {}
    for off in sorted(range(-radius, radius + 1), key=abs):
        g = f + off
        if 0 <= g < len(ident) and frames[g]:
            for pid, a in ident[g].items():
                out.setdefault(a["side"], (g, frames[g][a["det"]], pid))
    return out


def hit_evidence(f, ball_px, frames, ident, H_img, H_court, window=2):
    """Per side: (score, player id, reach metres, player court position). Reach = min over +-window frames."""
    res = {}
    for side in (NEAR, FAR):
        best = None
        for g in range(max(0, f - window), min(len(ball_px), f + window + 1)):
            b = ball_px[g]
            if not np.isfinite(b).all():
                continue
            sp = side_players(frames, ident, g, radius=2).get(side)
            if not sp:
                continue
            r = reach_m(b, sp[1], H_img, H_court)
            if best is None or r < best[0]:
                best = (r, sp[2], tuple(sp[1]["court"]))
        if best is None:
            pos = None
            sp = side_players(frames, ident, f, radius=8).get(side)
            if sp:
                pos = tuple(sp[1]["court"])
            seen = any(np.isfinite(ball_px[g]).all() for g in range(max(0, f - window), min(len(ball_px), f + window + 1)))
            res[side] = (ABSENT_PLAYER if seen else NO_BALL, sp[2] if sp else None, None, pos)
        else:
            r, pid, pos = best
            # image overlap can make a far-away ball look close (so a small reach proves little), but a
            # ball clearly away from a player in the image is never his hit: steep penalty above 2 m
            res[side] = (max(REACH_MAX_PENALTY, -1.5 * max(0.0, r - REACH_OK_M)), pid, round(r, 2), pos)
    return res


def handling(f, ball_px, frames, ident, fps, H_img, H_court):
    """Dead-ball evidence: within +-0.6 s the observed ball never leaves one standing player's reach
    (dribbling before a serve, catching, carrying). Returns True/False."""
    r = int(round(0.6 * fps))
    idx = [g for g in range(max(0, f - r), min(len(ball_px), f + r + 1)) if np.isfinite(ball_px[g]).all()]
    if len(idx) < max(6, r // 2):
        return False
    for side in (NEAR, FAR):
        near_cnt = 0
        for g in idx:
            sp = side_players(frames, ident, g, radius=3).get(side)
            if sp and reach_m(ball_px[g], sp[1], H_img, H_court) < 1.3:
                near_cnt += 1
        if near_cnt >= 0.85 * len(idx):
            return True
    return False


def ball_speed_ms(f, ball_px, fps, H_img, H_court, w=8):
    v = []
    for i in range(f - w, f + w):
        if 0 <= i < len(ball_px) - 1 and np.isfinite(ball_px[i]).all() and np.isfinite(ball_px[i + 1]).all():
            v.append(np.linalg.norm(ball_px[i + 1] - ball_px[i]) * fps / max(1e-6, ground_scale(H_img, H_court, ball_px[i])))
    return float(np.median(v)) if len(v) >= 4 else None


# ---------------------------------------------------------------------------------------- decoder
def build_candidates(prob, ball_px, frames, ident, fps, H_img, H_court, thr=0.2, extra_hits=()):
    from pipeline import events as ev
    rad = int(round(6 * fps / 30))
    hits = ev.peaks(prob[:, 1], thr, rad)
    bounces = ev.peaks(prob[:, 2], thr, rad)
    hits = [h for h in hits if not any(abs(h - b) <= 2 and prob[b, 2] > prob[h, 1] for b in bounces)]
    bounces = [b for b in bounces if not any(abs(h - b) <= 2 and prob[h, 1] >= prob[b, 2] for h in hits)]
    cands = []
    for h in hits:
        cands.append({"t": h, "type": "hit", "p": float(prob[h, 1])})
    for f, p, why in extra_hits:
        if not any(c["type"] == "hit" and abs(c["t"] - f) <= 3 for c in cands):
            cands.append({"t": f, "type": "hit", "p": p, "virtual": why})
    for b in bounces:
        c = {"t": b, "type": "bounce", "p": float(prob[b, 2])}
        # ground contact = the lowest observed ball image position around the detected frame
        ref = ball_px[b] if np.isfinite(ball_px[b]).all() else None
        near_f = [g for g in range(max(0, b - 3), min(len(ball_px), b + 4)) if np.isfinite(ball_px[g]).all()
                  and (ref is None or np.linalg.norm(ball_px[g] - ref) < 12 + 12 * abs(g - b))]
        if near_f:
            g = max(near_f, key=lambda g: ball_px[g][1])
            x, y = court.to_court(H_court, ball_px[g])
            c["court"] = (float(x), float(y))
            c["side"] = NEAR if y > court.NET_Y else FAR
            c["plausible"] = PLAUSIBLE[0] <= x <= PLAUSIBLE[1] and PLAUSIBLE[2] <= y <= PLAUSIBLE[3]
            # line tolerance: 0.3 m plus 3 image pixels at that depth (far court: ~0.5 m)
            m = IN_MARGIN_M + 3.0 / max(1e-3, ground_scale(H_img, H_court, ball_px[g]))
            c["in"] = (court.SL - m <= x <= court.WIDTH - court.SL + m and -m <= y <= court.LENGTH + m)
        else:
            c["plausible"] = False
        if "court" in c and not c["plausible"]:
            # a 'bounce' far outside the court is a ball in the air changing direction: a volley/overhead
            c = {"t": b, "type": "hit", "p": 0.8 * c["p"], "virtual": "air_contact"}
        cands.append(c)
    cands.sort(key=lambda c: c["t"])
    dup = set()
    bl = [c for c in cands if c["type"] == "bounce" and "court" in c]
    for a_, b_ in zip(bl, bl[1:]):
        if b_["t"] - a_["t"] < 0.3 * fps and np.hypot(a_["court"][0] - b_["court"][0], a_["court"][1] - b_["court"][1]) < 2.0:
            dup.add(id(a_) if a_["p"] < b_["p"] else id(b_))
    cands = [c for c in cands if id(c) not in dup]
    for c in cands:
        c["p"] = float(np.clip(c["p"], 0.02, 0.98))
        sp = ball_speed_ms(c["t"], ball_px, fps, H_img, H_court)
        c["speed"] = sp
        c["dead_evidence"] = bool(sp is not None and sp < 5.0)
        if c["type"] == "hit":
            c["sides"] = hit_evidence(c["t"], ball_px, frames, ident, H_img, H_court)
            c["leaves"] = ball_leaves(c["t"], ball_px, fps)
            w_ = max(4, int(round(0.33 * fps)))
            c["arrives"] = sum(np.isfinite(ball_px[g]).all() for g in range(max(0, c["t"] - w_), max(0, c["t"] - 1))) >= 3
    # toss: goes straight up and comes back near its origin, and the next contact (if any) is by the same
    # side -- a lob also goes up and down in the image but is answered by the opponent
    hitc = [c for c in cands if c["type"] == "hit"]
    best_side = lambda c: max(c["sides"], key=lambda k: c["sides"][k][0])
    for c in hitc:
        nxt = next((o for o in hitc if o["t"] > c["t"] + 3), None)
        tos = not c.get("virtual") and is_toss(c["t"], ball_px, fps, nxt["t"] if nxt else None)
        if tos and nxt is not None and nxt["t"] <= c["t"] + 1.8 * fps:
            tos = nxt["sides"][best_side(c)][0] > -1.0      # the same player strikes it next
        c["toss"] = bool(tos)
        if tos:
            c["dead_evidence"] = True
    # a hit after which the ball is struck again (same side) before any bounce: toss, drop, bounce on racket
    hitc = [c for c in cands if c["type"] == "hit"]
    bnc = [c for c in cands if c["type"] == "bounce" and c["p"] >= 0.5]
    for c in hitc:
        nb = next((o["t"] for o in bnc if o["t"] > c["t"]), None)
        c["restruck"] = any(o is not c and o["p"] >= 0.5 and c["t"] < o["t"] <= c["t"] + 1.5 * fps
                            and (nb is None or o["t"] < nb) and best_side(o) == best_side(c) for o in hitc)
    # dribbling: a bounce with another bounce at (almost) the same spot shortly before or after
    bl = [c for c in cands if c["type"] == "bounce" and "court" in c]
    for c in bl:
        for o in bl:
            if o is not c and 0.3 * fps <= abs(o["t"] - c["t"]) <= DRIBBLE_S * fps and \
                    np.hypot(o["court"][0] - c["court"][0], o["court"][1] - c["court"][1]) < DRIBBLE_M and \
                    all(x["speed"] is None or x["speed"] < DRIBBLE_MAX_SPEED for x in (c, o)):
                c["dead_evidence"] = True; c["dribble"] = True
    # dribbling with the racket/hand: a hit followed within 0.6 s by a bounce on the hitter's own side next
    # to him (pushing the ball down before a serve) -- both touches are dead ball. (A rally shot is never
    # followed by a bounce on the hitter's own side next to him; a net ball lands next to the net.)
    for c in cands:
        if c["type"] != "hit":
            continue
        side = best_side(c)
        pos = c["sides"][side][3]
        # only a real, slow touch next to a player can be a dribble (not an inferred serve contact, not a
        # ball without a player in reach)
        if pos is None or c.get("virtual") or c["sides"][side][0] < -1.0:
            continue
        for o in bl:
            if 0 < o["t"] - c["t"] <= 0.6 * fps and o["side"] == side and \
                    np.hypot(o["court"][0] - pos[0], o["court"][1] - pos[1]) < 3.0:
                c["dead_evidence"] = o["dead_evidence"] = True
                c["dribble"] = o["dribble"] = True
    return cands


def decode(cands, fps):
    """Viterbi over the candidates (see module doc). Returns per candidate: ("event", state, is_start),
    "dead" or "false". States after an accepted event: ("F", T, hitter_side, is_start) ball flying to
    side T; ("B", T) first bounce on T; ("E", reason) point over."""
    N = len(cands)
    NEG = -1e18
    t = [c["t"] for c in cands]
    lp = [np.log(c["p"]) for c in cands]
    s_false = [np.log(1 - c["p"] + 0.02) for c in cands]
    s_dead = [np.log(c["p"]) + np.log(Q_DEAD_EVIDENT if c["dead_evidence"] else Q_DEAD_OTHER) for c in cands]
    be = [max(a, b) for a, b in zip(s_false, s_dead)]
    PF = np.concatenate([[0.0], np.cumsum(s_false)])     # prefix sums: candidates explained as false
    PB = np.concatenate([[0.0], np.cumsum(be)])          # ... as dead ball (or false) between points
    gap = MIN_POINT_GAP_S * fps

    def acc_bounce(j):
        c = cands[j]
        return lp[j] - (3.0 if c["dead_evidence"] else 0.0) if c.get("plausible") else NEG

    def acc_hit(j, side, start):
        c = cands[j]
        sc = lp[j] + c["sides"][side][0] - (2.0 if c["dead_evidence"] else 0.0)
        if c.get("restruck"):
            sc += C_RESTRUCK if start else C_RESTRUCK_RALLY
        if c.get("leaves") is False:
            sc += NO_FLIGHT
        if not start and not c.get("arrives", True):
            sc += NO_ARRIVAL
        return sc

    A = [dict() for _ in range(N)]        # A[j][state] = (score, back-pointer)
    Aend = [NEG] * N                      # best score of a point that ends right after event j
    Aend_st = [None] * N
    D = [NEG] * (N + 1)                   # DEAD before k, a new point may start (gap respected)
    Dbp = [None] * (N + 1)
    D[0] = 0.0
    ptr, run_best, run_i = 0, NEG, None   # running max over ended points far enough in the past
    lookback = int(3.2 * fps)
    for j in range(N):
        # D[j]: previous DEAD chain, or a point that ended at i with t_i <= t_j - gap
        while ptr < j and t[ptr] <= t[j] - gap:
            v = Aend[ptr] - PB[ptr + 1]
            if v > run_best:
                run_best, run_i = v, ptr
            ptr += 1
        if j > 0:
            best, bp = D[j - 1] + be[j - 1], ("D", j - 1)
            if run_i is not None and run_best + PB[j] > best:
                best, bp = run_best + PB[j], ("A", run_i, Aend_st[run_i])
            D[j], Dbp[j] = best, bp
        c = cands[j]

        def put(state, score, bp):
            if score > A[j].get(state, (NEG,))[0]:
                A[j][state] = (score, bp)

        # (1) point start
        if c["type"] == "hit":
            for side in (NEAR, FAR):
                put(("F", OTHER[side], side, True), D[j] + C_START_HIT + acc_hit(j, side, True), ("D", j))
        elif c.get("plausible") and c["in"]:
            put(("B", c["side"]), D[j] + C_START_BOUNCE + acc_bounce(j), ("D", j))
        # (2) continuation from an in-play event i
        i = j - 1
        while i >= 0 and t[j] - t[i] <= lookback:
            dt = (t[j] - t[i]) / fps
            skip = PF[j] - PF[i + 1]
            for st, (sc, _) in A[i].items():
                base = sc + skip
                if st[0] == "F":
                    T, hs = st[1], st[2]
                    src = cands[i]["sides"][hs][3] if cands[i]["type"] == "hit" else None
                    if c["type"] == "bounce" and c.get("plausible") and HIT_TO_BOUNCE[0] <= dt <= HIT_TO_BOUNCE[1]:
                        if src is not None and dt < np.hypot(c["court"][0] - src[0], c["court"][1] - src[1]) / V_MAX_MS - TIMING_SLACK_S:
                            continue
                        if c["side"] == T:
                            put(("B", T), base + acc_bounce(j) + (0.0 if c["in"] else OUT_PENALTY), ("A", i, st))
                            if not c["in"]:
                                put(("E", "out"), base + acc_bounce(j), ("A", i, st))
                        elif abs(c["court"][1] - court.NET_Y) <= NET_BALL_ZONE_M:
                            put(("E", "net"), base + acc_bounce(j), ("A", i, st))
                    elif c["type"] == "hit" and HIT_TO_VOLLEY[0] <= dt <= HIT_TO_VOLLEY[1]:
                        dst = c["sides"][T][3]
                        if src is not None and dst is not None and dt < np.hypot(dst[0] - src[0], dst[1] - src[1]) / V_MAX_MS - TIMING_SLACK_S:
                            continue
                        put(("F", OTHER[T], T, False), base + acc_hit(j, T, False), ("A", i, st))
                elif st[0] == "B":
                    T = st[1]
                    if c["type"] == "hit" and BOUNCE_TO_HIT[0] <= dt <= BOUNCE_TO_HIT[1]:
                        dst = c["sides"][T][3]
                        q = cands[i].get("court")
                        if q is not None and dst is not None and dt < np.hypot(q[0] - dst[0], q[1] - dst[1]) / V_MAX_BOUNCED_MS - TIMING_SLACK_S:
                            continue
                        put(("F", OTHER[T], T, False), base + acc_hit(j, T, False), ("A", i, st))
                    elif c["type"] == "bounce" and c.get("plausible") and c["side"] == T \
                            and BOUNCE_TO_BOUNCE2[0] <= dt <= BOUNCE_TO_BOUNCE2[1]:
                        put(("E", "double"), base + acc_bounce(j), ("A", i, st))
            i -= 1
        # a point can end after j unless j is only its first hit
        for st, (sc, _) in A[j].items():
            if st[0] == "F" and st[3]:
                continue
            if sc > Aend[j]:
                Aend[j], Aend_st[j] = sc, st
    # final: DEAD chain to the end, or a point ending at i followed by dead/false candidates
    best, bp = (D[N - 1] + be[N - 1], ("D", N - 1)) if N else (0.0, None)
    for i in range(N):
        v = Aend[i] + PB[N] - PB[i + 1]
        if v > best:
            best, bp = v, ("A", i, Aend_st[i])
    D[N], Dbp[N] = best, bp
    role = [None] * N
    cur = ("D", N)
    while N:
        if cur[0] == "D":
            k = cur[1]
            if k == 0:
                break
            bp = Dbp[k]
            if bp[0] == "D":
                role[k - 1] = "dead" if s_dead[k - 1] >= s_false[k - 1] else "false"
                cur = ("D", k - 1)
            else:
                i = bp[1]
                for m in range(i + 1, k):
                    role[m] = "dead" if s_dead[m] >= s_false[m] else "false"
                cur = ("A", i, bp[2])
        else:
            _, j, st = cur
            bp = A[j][st][1]
            if bp[0] == "D":
                role[j] = ("event", st, True)
                cur = ("D", j)
            else:
                role[j] = ("event", st, False)
                for m in range(bp[1] + 1, j):
                    role[m] = "false"
                cur = ("A", bp[1], bp[2])
    return role


def run(prob, ball_px, frames, ident, fps, H_img, H_court, thr=0.2):
    """Full rally decoding. Returns dict with hits [(frame, side, pid)], bounces [(frame, side, status)],
    points [(start, end, end_reason)], dead [frames], false [frames], candidates."""
    extra = [(f, 0.55, "serve_above_frame") for f, _ in toss_serves(ball_px, fps)]
    cands = build_candidates(prob, ball_px, frames, ident, fps, H_img, H_court, thr, extra)
    more = []
    for c in cands:
        if c["type"] == "hit" and c.get("toss"):
            g = serve_contact_after_toss(c["t"], ball_px, fps)
            if g is not None:
                more.append((g, 0.55, "serve_after_toss"))
    for apex, release in toss_apexes(ball_px, fps):
        g = serve_contact_after_toss(release, ball_px, fps)
        if g is not None and g > apex and not any(abs(g - m[0]) <= 3 for m in more):
            more.append((g, 0.5, "serve_after_toss"))
    if more:
        cands = build_candidates(prob, ball_px, frames, ident, fps, H_img, H_court, thr, extra + more)
    role = decode(cands, fps)
    hits, bounces, dead, false, points = [], [], [], [], []
    cur = None
    for c, r in zip(cands, role):
        c["role"] = r if isinstance(r, str) else ("start" if r[2] else "event")
        if isinstance(r, tuple):
            st, start = r[1], r[2]
            if start or cur is None:
                if cur:
                    points.append(cur)
                cur = [c["t"], c["t"], None]
            cur[1] = c["t"]
            if c["type"] == "hit":
                side = st[2]
                hits.append((c["t"], side, c["sides"][side][1]))
            else:
                status = st[1] if st[0] == "E" else "in"
                bounces.append((c["t"], c.get("side"), status))
                if st[0] == "E":
                    cur[2] = status
        elif r == "dead":
            dead.append(c["t"])
        else:
            false.append(c["t"])
    if cur:
        points.append(cur)
    return {"hits": hits, "bounces": bounces, "points": [tuple(p) for p in points], "dead": dead, "false": false,
            "candidates": cands}
