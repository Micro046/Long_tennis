"""Synthetic labelled ball tracks as seen from a fixed phone behind the baseline.

Why: the hit/bounce classifier learned from TV broadcasts (high camera, long lens) and transfers
poorly to low phone views. Here we simulate rallies with ball physics (gravity, drag, topspin/slice
Magnus, bounce with restitution + friction), project them through randomly sampled phone cameras,
and corrupt them like the real tracker (jitter, dropouts that grow with blur, outliers).
Also simulates dead-ball activity (dribbling before a serve, rolling balls) as negatives.

Court frame (metres): x across 0..10.97, y far->near 0..23.77, z up. Camera behind y = 23.77.
Output per rally: {"xy": (n,2) image px / image height (NaN = missing), "status": (n,) 0/1/2, "fps"}.
"""
from __future__ import annotations

import numpy as np

W_C, L_C, SL = 10.97, 23.77, 1.37
NET_Y, NET_H = L_C / 2, 0.95
G = 9.81
K_DRAG = 0.0195          # 0.5 rho Cd A / m  (1/m)
K_MAGNUS = 0.0009        # lift per (rad/s * m/s)  -> order of 1 g for heavy topspin
E_REST = 0.76            # vertical restitution
MU_KEEP = (0.55, 0.85)   # horizontal speed kept at bounce


# --------------------------------------------------------------------------- physics
def fly(p, v, spin, dt, t_max, stop_at_bounce=True):
    """Integrate until the ball touches the ground (z<=r) or t_max. spin>0 topspin (downward force).
    Returns positions array, time of ground contact (or None)."""
    r = 0.033
    pts = [p.copy()]
    t = 0.0
    while t < t_max:
        sp = np.linalg.norm(v)
        a = np.array([0, 0, -G]) - K_DRAG * sp * v
        horiz = np.array([v[0], v[1], 0.0])
        hn = np.linalg.norm(horiz)
        if hn > 1e-6:  # Magnus: topspin pushes down, slice lifts
            a += np.array([0, 0, -K_MAGNUS * spin * hn])
        v = v + a * dt
        p = p + v * dt
        t += dt
        if p[2] <= r:
            p[2] = r
            pts.append(p.copy())
            return np.array(pts), t, v
        pts.append(p.copy())
    return np.array(pts), None, v


def bounce(v, spin, rng):
    keep = rng.uniform(*MU_KEEP) + (0.08 if spin > 0 else -0.05)
    return np.array([v[0] * keep, v[1] * keep, -v[2] * E_REST])


def aim_velocity(p, target, speed, spin, rng, dt):
    """Find elevation so a ball hit at `speed` from p lands near target (2D search)."""
    d = np.array(target) - p[:2]
    dist = np.linalg.norm(d)
    u = d / max(dist, 1e-6)
    best = None
    for elev in np.radians(np.linspace(-12, 45, 40)):
        v = np.array([u[0] * speed * np.cos(elev), u[1] * speed * np.cos(elev), speed * np.sin(elev)])
        pts, t, _ = fly(p.copy(), v.copy(), spin, dt, 6.0)
        if t is None:
            continue
        err = np.linalg.norm(pts[-1][:2] - target)
        # reject trajectories that do not clear the net when crossing it
        cross = np.where(np.diff(np.sign(pts[:, 1] - NET_Y)) != 0)[0]
        if len(cross) and pts[cross[0], 2] < NET_H + 0.05:
            continue
        if best is None or err < best[0]:
            best = (err, v)
    return None if best is None else best[1]


# --------------------------------------------------------------------------- rally generator
def rally(rng, dt):
    """Returns list of (t, pos) samples and event list [(t, 'hit'|'bounce')]."""
    samples, events = [], []
    t = 0.0
    server_far = rng.random() < 0.5
    sy = -0.3 if server_far else L_C + 0.3
    sx = rng.uniform(1.5, 9.5)

    # pre-serve dribble (dead ball): 1-4 vertical bounces at the server's feet, no event labels
    if rng.random() < 0.6:
        p = np.array([sx + rng.uniform(-0.4, 0.4), sy + rng.uniform(-0.3, 0.3), 1.0])
        for _ in range(rng.integers(1, 5)):
            v = np.array(rng.normal(0, 0.15, 2).tolist() + [-rng.uniform(1.5, 3.5)])
            pts, tb, v = fly(p, v, 0, dt, 1.0)
            for q in pts:
                samples.append((t, q)); t += dt
            p = pts[-1]; p[2] = 0.034
            v2 = np.array([v[0], v[1], abs(v[2]) * 0.8])
            pts, _, _ = fly(p, v2, 0, dt, 0.35, stop_at_bounce=False)
            for q in pts:
                samples.append((t, q)); t += dt
            p = pts[-1]
        t += rng.uniform(0.3, 1.5)

    # toss (no event), then serve contact
    p = np.array([sx, sy, 1.5])
    toss_v = np.array([0, 0, rng.uniform(4.0, 6.5)])
    pts, _, _ = fly(p, toss_v, 0, dt, rng.uniform(0.8, 1.2), stop_at_bounce=False)
    for q in pts:
        samples.append((t, q)); t += dt
    p = pts[-1].copy(); p[2] = max(p[2], 2.4)
    hitter_far = server_far
    box_y = (NET_Y + rng.uniform(0.5, 6.4)) if server_far else (NET_Y - rng.uniform(0.5, 6.4))
    box_x = rng.uniform(SL + 0.2, W_C - SL - 0.2)
    speed, spin = rng.uniform(22, 50), rng.uniform(-50, 150)
    n_shots = int(rng.integers(1, 12))
    for shot in range(n_shots):
        target = np.array([box_x, box_y]) if shot == 0 else np.array([
            rng.uniform(SL - 0.5, W_C - SL + 0.5),
            (rng.uniform(NET_Y + 1.0, L_C + 0.8) if hitter_far else rng.uniform(-0.8, NET_Y - 1.0))])
        v = aim_velocity(p, target, speed, spin, rng, dt * 2)
        if v is None:
            break
        events.append((t, "hit"))
        pts, tb, vend = fly(p, v, spin, dt, 6.0)
        for q in pts[1:]:
            t += dt; samples.append((t, q))
        if tb is None:
            break
        events.append((t, "bounce"))
        # after the first bounce
        vb = bounce(vend, spin, rng)
        p = pts[-1].copy()
        end = rng.random()
        wait = rng.uniform(0.25, 0.9)
        pts, tb2, vend2 = fly(p, vb, spin * 0.5, dt, wait if end > 0.15 else 3.0)
        for q in pts[1:]:
            t += dt; samples.append((t, q))
        if end < 0.15:  # not returned: second bounce, rolls away (dead)
            if tb2 is not None:
                events.append((t, "bounce"))
            break
        if tb2 is not None:  # waited too long -> ball bounced again: point over
            events.append((t, "bounce"))
            break
        p = pts[-1].copy()
        hitter_far = not hitter_far
        speed, spin = rng.uniform(12, 38), rng.uniform(-60, 200)
        if rng.random() < 0.1:  # lob
            speed, spin = rng.uniform(10, 18), rng.uniform(0, 80)
    # tail: ball rolling/slow after the point
    if rng.random() < 0.5 and samples:
        p = samples[-1][1].copy(); p[2] = 0.034
        v = np.array(list(rng.normal(0, 1.5, 2)) + [0.0])
        for _ in range(int(rng.uniform(0.5, 2.0) / dt)):
            p = p + v * dt; v *= 0.985; t += dt
            samples.append((t, p.copy()))
    return samples, events


# --------------------------------------------------------------------------- camera
def random_camera(rng, w=1920, h=1080):
    """Phone behind the near baseline looking at the court."""
    c = np.array([rng.uniform(1.5, 9.5), L_C + rng.uniform(1.0, 6.0), rng.uniform(0.9, 4.5)])
    look = np.array([W_C / 2 + rng.uniform(-1.2, 1.2), rng.uniform(7.0, 14.0), rng.uniform(-0.5, 0.8)])
    fwd = look - c; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0]); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    hfov = np.radians(rng.uniform(55, 85))
    f = (w / 2) / np.tan(hfov / 2)
    # the whole court must be visible; otherwise resample
    corners = np.array([[0, 0, 0], [W_C, 0, 0], [0, L_C, 0], [W_C, L_C, 0]], float)
    uv = project(corners, c, R, f, w, h)
    if not np.all(np.isfinite(uv)) or uv[:, 0].min() < -0.05 * w or uv[:, 0].max() > 1.05 * w \
            or uv[:, 1].min() < 0 or uv[:, 1].max() > 1.1 * h:
        return random_camera(rng, w, h)
    return c, R, f, w, h


def project(P, c, R, f, w, h):
    q = (np.atleast_2d(P) - c) @ R.T
    z = q[:, 2]
    uv = np.full((len(q), 2), np.nan)
    ok = z > 0.3
    uv[ok, 0] = f * q[ok, 0] / z[ok] + w / 2
    uv[ok, 1] = f * q[ok, 1] / z[ok] + h / 2
    return uv


# --------------------------------------------------------------------------- tracker corruption
def to_track(samples, events, cam, fps, rng):
    c, R, f, w, h = cam
    dt_sim = samples[1][0] - samples[0][0]
    T = samples[-1][0]
    n = int(T * fps)
    times = np.arange(n) / fps
    sim_t = np.array([s[0] for s in samples]); sim_p = np.array([s[1] for s in samples])
    idx = np.clip(np.searchsorted(sim_t, times), 0, len(sim_t) - 1)
    uv = project(sim_p[idx], c, R, f, w, h)
    status = np.zeros(n, int)
    for te, kind in events:
        k = int(round(te * fps))
        if 0 <= k < n:
            status[k] = 1 if kind == "hit" else 2
    # visibility: inside frame
    vis = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    # blur-dependent dropouts + random dropouts + bursts
    speed = np.r_[0, np.linalg.norm(np.diff(uv, axis=0), axis=1)]
    speed = np.nan_to_num(speed)
    p_drop = rng.uniform(0.02, 0.12) + np.clip(speed / 400, 0, 0.5)
    vis &= rng.random(n) > p_drop
    for _ in range(rng.integers(0, 4)):
        a = rng.integers(0, max(1, n)); vis[a:a + rng.integers(2, 12)] = False
    uv = uv + rng.normal(0, rng.uniform(0.8, 3.0), uv.shape)
    uv[~vis] = np.nan
    # outliers (false detections elsewhere)
    for k in np.where(rng.random(n) < rng.uniform(0.0, 0.02))[0]:
        uv[k] = [rng.uniform(0, w), rng.uniform(0, h)]
    return {"xy": uv / h, "status": status, "fps": fps}


def generate(n_rallies, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n_rallies:
        fps = float(rng.choice([25.0, 29.97, 30.0, 50.0, 59.94], p=[0.1, 0.35, 0.25, 0.1, 0.2]))
        samples, events = rally(rng, 1 / 240)
        if len(samples) < 60 or not events:
            continue
        out.append(to_track(samples, events, random_camera(rng), fps, rng))
    return out
