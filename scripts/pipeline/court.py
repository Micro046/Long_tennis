"""Court model + line-snapping calibration for a fixed camera.

Workflow (identical in the Android setup screen):
1. The user roughly places four reference points (or an automatic proposal does).
2. An initial homography projects every court line into the image.
3. Each projected line is snapped to nearby white-paint pixels (robust line fit).
4. Refined line pairs are intersected (even outside the frame) and a least-squares
   homography is re-estimated. Steps 2-4 repeat a few times.

Court coordinates: metres, origin far-left doubles corner as seen by the camera,
x across (0..10.97), y far->near (0..23.77). The camera is assumed to be behind
the near baseline; `side` near/far is relative to that.
"""
from __future__ import annotations

import cv2
import numpy as np

WIDTH, LENGTH, SINGLES = 10.97, 23.77, 8.23
SL = (WIDTH - SINGLES) / 2          # singles sideline offset 1.37
NET_Y = LENGTH / 2
SERVICE = 6.40
FAR_SERVICE_Y, NEAR_SERVICE_Y = NET_Y - SERVICE, NET_Y + SERVICE
CX = WIDTH / 2

# name -> (p1, p2) in court metres
LINES = {
    "far_baseline": ((0, 0), (WIDTH, 0)),
    "near_baseline": ((0, LENGTH), (WIDTH, LENGTH)),
    "left_doubles": ((0, 0), (0, LENGTH)),
    "right_doubles": ((WIDTH, 0), (WIDTH, LENGTH)),
    "left_singles": ((SL, 0), (SL, LENGTH)),
    "right_singles": ((WIDTH - SL, 0), (WIDTH - SL, LENGTH)),
    "far_service": ((SL, FAR_SERVICE_Y), (WIDTH - SL, FAR_SERVICE_Y)),
    "near_service": ((SL, NEAR_SERVICE_Y), (WIDTH - SL, NEAR_SERVICE_Y)),
    "centre_service": ((CX, FAR_SERVICE_Y), (CX, NEAR_SERVICE_Y)),
}
# Keypoints used for homography (intersections of named lines)
KEYPOINTS = {
    "far_left_doubles": ("far_baseline", "left_doubles"),
    "far_left_singles": ("far_baseline", "left_singles"),
    "far_right_singles": ("far_baseline", "right_singles"),
    "far_right_doubles": ("far_baseline", "right_doubles"),
    "near_left_doubles": ("near_baseline", "left_doubles"),
    "near_left_singles": ("near_baseline", "left_singles"),
    "near_right_singles": ("near_baseline", "right_singles"),
    "near_right_doubles": ("near_baseline", "right_doubles"),
    "far_service_left": ("far_service", "left_singles"),
    "far_service_right": ("far_service", "right_singles"),
    "far_service_centre": ("far_service", "centre_service"),
    "near_service_left": ("near_service", "left_singles"),
    "near_service_right": ("near_service", "right_singles"),
    "near_service_centre": ("near_service", "centre_service"),
}


def _intersect_model(a, b):
    (x1, y1), (x2, y2) = LINES[a]
    (x3, y3), (x4, y4) = LINES[b]
    if x1 == x2:
        return (x1, y3)
    return (x3, y1)


MODEL_KP = {k: _intersect_model(*v) for k, v in KEYPOINTS.items()}


def project(H, pts):
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def white_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    # white paint: bright + low saturation, and brighter than its local surrounding
    local = cv2.blur(v.astype(np.float32), (31, 31))
    return ((s < 70) & (v > 150) & (v > local + 25)).astype(np.uint8)


def snap_line(mask, p1, p2, band=10, min_pixels=40):
    """Robustly refit a line near the projected segment p1-p2 (image px). Returns (point, dir)."""
    h, w = mask.shape
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    # clip segment to image for sampling
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1:
        return None
    u = d / L
    nrm = np.array([-u[1], u[0]])
    ts = np.arange(0, L, 1.0)
    pts = []
    for off in range(-band, band + 1):
        q = p1 + np.outer(ts, u) + off * nrm
        xi, yi = np.round(q[:, 0]).astype(int), np.round(q[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = xi[ok], yi[ok]
        hit = mask[yi, xi] > 0
        pts.append(np.stack([xi[hit], yi[hit]], 1))
    pts = np.concatenate(pts, 0).astype(np.float32)
    if len(pts) < min_pixels:
        return None
    for _ in range(3):
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, .01, .01).ravel()
        dist = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        keep = dist < 2.5
        if keep.sum() < min_pixels:
            break
        pts = pts[keep]
    return np.array([x0, y0]), np.array([vx, vy]), int(len(pts))


def _line_intersect(l1, l2):
    (p, r), (q, s) = (l1[0], l1[1]), (l2[0], l2[1])
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-6:
        return None
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / cross
    return p + t * r


def line_support(mask, H, step=2.0, tol=2):
    """Fraction of in-image samples along all projected court lines that lie on white paint (+-tol px)."""
    h, w = mask.shape
    dil = cv2.dilate(mask, np.ones((2 * tol + 1, 2 * tol + 1), np.uint8))
    hit = tot = 0
    for a, b in LINES.values():
        p, q = project(H, [a, b])
        n = int(np.linalg.norm(q - p) / step)
        for t in np.linspace(0, 1, max(2, n)):
            x, y = p + t * (q - p)
            if 0 <= x < w and 0 <= y < h:
                tot += 1; hit += dil[int(y), int(x)] > 0
    return hit / max(1, tot)


def calibrate(img, init_image_pts: dict, iterations=3, band=12):
    """init_image_pts: {keypoint_name: (x, y)} with >=4 entries (rough user placement)."""
    names = list(init_image_pts)
    H, _ = cv2.findHomography(np.array([MODEL_KP[n] for n in names], np.float64),
                              np.array([init_image_pts[n] for n in names], np.float64))
    mask = white_mask(img)
    h, w = mask.shape
    fitted = {}
    for it in range(iterations):
        fitted = {}
        for name, (a, b) in LINES.items():
            seg = project(H, [a, b])
            # ignore lines whose projection is entirely outside the frame
            r = snap_line(mask, seg[0], seg[1], band=band if it == 0 else max(4, band // 2))
            if r is not None:
                fitted[name] = r
        src, dst, used = [], [], []
        for kp, (a, b) in KEYPOINTS.items():
            if a in fitted and b in fitted:
                p = _line_intersect(fitted[a], fitted[b])
                if p is not None and -0.5 * w < p[0] < 1.5 * w and -0.5 * h < p[1] < 1.5 * h:
                    src.append(MODEL_KP[kp]); dst.append(p); used.append(kp)
        if len(src) < 4:
            break
        # Candidate fits: all keypoints, near half only, far half only. The net band can masquerade as
        # the far baseline in low views, and off-frame near corners can be wrong in others; keep the fit
        # whose projected court lines land on the most white paint.
        cands = []
        for subset in (range(len(used)),
                       [i for i, kp in enumerate(used) if MODEL_KP[kp][1] > NET_Y],
                       [i for i, kp in enumerate(used) if MODEL_KP[kp][1] < NET_Y]):
            subset = list(subset)
            if len(subset) < 4:
                continue
            Hc, _ = cv2.findHomography(np.array([src[i] for i in subset]), np.array([dst[i] for i in subset]), cv2.RANSAC, 4.0)
            if Hc is not None:
                cands.append((line_support(mask, Hc), Hc, subset))
        if not cands:
            break
        _, H, subset = max(cands, key=lambda c: c[0])
        # keypoints consistent with the chosen fit (for the reported RMS)
        keep = [i for i in range(len(used)) if np.linalg.norm(project(H, [src[i]])[0] - dst[i]) < 4.0]
        src, dst, used = [src[i] for i in keep], [dst[i] for i in keep], [used[i] for i in keep]
    reproj = []
    for kp, p in zip(used, dst):
        q = project(H, [MODEL_KP[kp]])[0]
        reproj.append(float(np.linalg.norm(q - p)))
    return {"imageFromCourt": np.asarray(H).tolist(),
            "courtFromImage": np.linalg.inv(H).tolist(),
            "linesSnapped": sorted(fitted), "keypointsUsed": used,
            "reprojectionRmsPx": float(np.sqrt(np.mean(np.square(reproj)))) if reproj else None,
            "lineSupport": float(line_support(mask, H)),
            "imageSize": [w, h]}


def draw(img, H, color=(0, 255, 255), thickness=2):
    out = img.copy()
    for a, b in LINES.values():
        seg = project(H, [a, b]).astype(int)
        cv2.line(out, tuple(seg[0]), tuple(seg[1]), color, thickness)
    return out


def to_court(H_court_from_image, pt):
    return project(H_court_from_image, [pt])[0]
