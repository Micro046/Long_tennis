#!/usr/bin/env python3
"""Render an analysed session onto its video: player boxes with names, ball trail, court lines, hit and
bounce markers, stroke/landing/speed per shot, point boundaries, and a mini-court with every landing.

  render_overlay.py SESSION_DIR --out out.mp4 [--start F --stop F] [--scale 0.5]
  render_overlay.py VIDEO REPORT POSE --ball BALL --out out.mp4         (explicit files)
  render_overlay.py SESSION_DIR --audit N --out-dir dir                  (blind shot audit sheets)

Encodes H.264 through ffmpeg (hardware encoder on macOS) so the file plays in QuickTime / on phones.
"""
import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import court  # noqa: E402

COL = {"A": (255, 190, 60), "B": (60, 170, 255), "unknown": (180, 180, 180)}
BOUNCE = (80, 255, 80)


def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def load(args):
    if args.session:
        d = Path(args.session)
        video, report, pose, ball, cal = d / "video.mp4", d / "report.json", d / "pose.json", d / "ball.json", d / "calibration.json"
    else:
        video, report, pose, ball, cal = Path(args.video), Path(args.report), Path(args.pose), Path(args.ball), \
            (Path(args.calibration) if args.calibration else None)
    rep = json.loads(report.read_text())
    pz = json.loads(pose.read_text())
    bl = json.loads(ball.read_text())["frames"]
    calib = json.loads(cal.read_text()) if cal and cal.exists() else None
    return video, rep, pz, bl, calib


def mini_court(w=150, scale=10.0):
    """Top view, far baseline at the top. Returns image and a court->pixel function."""
    m = 12
    W, H = int(court.WIDTH * scale) + 2 * m, int(court.LENGTH * scale) + 2 * m
    img = np.full((H, W, 3), (60, 110, 70), np.uint8)
    to = lambda x, y: (int(m + x * scale), int(m + y * scale))
    for a, b in court.LINES.values():
        cv2.line(img, to(*a), to(*b), (235, 235, 235), 1)
    cv2.line(img, to(-0.5, court.NET_Y), to(court.WIDTH + 0.5, court.NET_Y), (40, 40, 40), 2)
    return img, to


def draw(img, f, rep, pz, ball, H_img, state, s):
    per = rep["perFrameIdentity"]
    g = next((x for x in (f, f - 1, f + 1, f - 2, f + 2) if 0 <= x < len(per) and per[x]), None)
    if g is not None and pz["frames"][g]:
        for pid, a in per[g].items():
            p = pz["frames"][g][a["det"]]
            x1, y1, x2, y2 = (int(v * s) for v in p[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), COL[pid], 2)
            cv2.putText(img, rep["players"][pid]["name"], (x1, max(18, y1 - 6)), 0, 0.6, COL[pid], 2)
    for k in range(max(0, f - 10), f + 1):
        b = ball[k] if k < len(ball) else None
        if b:
            cv2.circle(img, (int(b[0] * s), int(b[1] * s)), 3 if k < f else 6, (0, 255, 255), -1 if k < f else 2)
    for kind, fr, txt, col, pos in state["recent"]:
        age = f - fr
        if kind == "bounce" and pos is not None and age < 25:
            cv2.circle(img, (int(pos[0] * s), int(pos[1] * s)), 14, BOUNCE, 2)
    # banner with the latest shot
    lines = [r for r in state["recent"] if f - r[1] < 45][-2:]
    for i, (kind, fr, txt, col, _) in enumerate(lines):
        cv2.rectangle(img, (10, 10 + 34 * i), (10 + 11 * len(txt) + 16, 40 + 34 * i), (0, 0, 0), -1)
        cv2.putText(img, txt, (18, 32 + 34 * i), 0, 0.65, col, 2)
    # point status + clock
    t = f / state["fps"]
    pt = next((p for p in rep.get("points", []) if p["start"] - 15 <= f <= p["end"] + 30), None)
    clock = f"{int(t // 60):02d}:{t % 60:04.1f}"
    status = f"POINT {state['point_no'].get(pt['start'], '')}" if pt else "between points"
    W = img.shape[1]
    cv2.rectangle(img, (W - 260, 10), (W - 10, 40), (0, 0, 0), -1)
    cv2.putText(img, f"{clock}  {status}", (W - 252, 32), 0, 0.6, (255, 255, 255) if pt else (150, 150, 150), 1)
    # mini court with landings so far
    mc = state["mini"].copy()
    for (x, y, pid, fr) in state["landings"]:
        if fr <= f:
            cv2.circle(mc, state["to"](x, y), 3 if f - fr > 60 else 5, COL.get(pid, COL["unknown"]), -1)
    h0, w0 = mc.shape[:2]
    y0 = img.shape[0] - h0 - 10
    img[y0:y0 + h0, W - w0 - 10:W - 10] = mc
    return img


def render(video, rep, pz, ball, cal, out, start, stop, scale):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(5)
    w, h = int(cap.get(3)), int(cap.get(4))
    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    H_img = court.image_from_court(cal) if cal else None
    mini, to = mini_court()
    shots = {s["hitFrame"]: s for s in rep["shots"]}
    bounces = set(rep["events"]["bounces"])
    landings = [(s["landing"]["courtM"][0], s["landing"]["courtM"][1], s["playerId"], s["landing"]["bounceFrame"])
                for s in rep["shots"] if s["landing"].get("courtM") and s["landing"].get("region")]
    state = {"recent": [], "fps": fps, "mini": mini, "to": to, "landings": landings,
             "point_no": {p["start"]: i + 1 for i, p in enumerate(rep.get("points", []))}}
    # static court-line layer (lens-aware)
    lines_layer = None
    if H_img is not None:
        lines_layer = np.zeros((oh, ow, 3), np.uint8)
        lines_layer = court.draw(lines_layer, _scaled(H_img, scale), (200, 200, 200), 1)
    enc = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{ow}x{oh}", "-r", f"{fps}", "-i", "-", "-c:v",
                            "h264_videotoolbox" if sys.platform == "darwin" else "libx264", "-b:v", "6M",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
    cap.set(1, start)
    for f in range(start, stop):
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
        if lines_layer is not None:
            img = cv2.addWeighted(img, 1.0, lines_layer, 0.35, 0)
        if f in shots:
            sh = shots[f]
            nm = rep["players"].get(sh["playerId"], {}).get("name", "unknown player")
            land = sh["landing"].get("region") or sh["landing"].get("status", "")
            sp = f", {sh['speed']['avgKmh']:.0f} km/h" if "speed" in sh else ""
            state["recent"].append(("hit", f, f"HIT  {nm}: {sh['stroke']} -> {land}{sp}", COL.get(sh["playerId"], COL["unknown"]), None))
        if f in bounces:
            state["recent"].append(("bounce", f, "BOUNCE", BOUNCE, ball[f] if f < len(ball) else None))
        draw(img, f, rep, pz, ball, H_img, state, scale)
        enc.stdin.write(img.tobytes())
        if (f - start) % 3000 == 0:
            print(f"render {f - start}/{stop - start}", flush=True)
    enc.stdin.close()
    enc.wait()


def _scaled(M, s):
    """Wrap a court->image mapping so it returns coordinates in the scaled output frame."""
    class S:
        def apply(self, pts):
            return M.apply(pts) * s
    m = court.Mapping.__new__(court.Mapping)
    m.direction, m.H, m.lens = "image", None, None
    m.apply = S().apply
    return m


def audit(video, rep, pz, ball, n, out_dir, seed=3, per_sheet=4):
    """Blind audit sheets: for each sampled predicted hit, one row of 5 full frames (f-6..f+6 step 3)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    shots = list(rep["shots"])
    random.Random(seed).shuffle(shots)
    shots = sorted(shots[:n], key=lambda s: s["hitFrame"])
    cap = cv2.VideoCapture(str(video))
    rows, key = [], []
    for i, s in enumerate(shots):
        f = s["hitFrame"]
        tiles = []
        for k in range(f - 6, f + 7, 3):
            cap.set(1, k); ok, img = cap.read()
            if not ok:
                img = np.zeros((1080, 1920, 3), np.uint8)
            t = cv2.resize(img, (384, 216))
            cv2.putText(t, f"#{i} {k}{' <' if k == f else ''}", (4, 16), 0, 0.5, (0, 255, 255), 1)
            tiles.append(t)
        rows.append(np.hstack(tiles))
        key.append({"audit": i, "hitFrame": f, "pred": {"playerId": s["playerId"], "side": s["side"], "stroke": s["stroke"],
                                                         "landing": s["landing"].get("status"), "region": s["landing"].get("region")}})
    for sh in range(0, len(rows), per_sheet):
        cv2.imwrite(str(out_dir / f"audit_{sh // per_sheet:02d}.jpg"), np.vstack(rows[sh:sh + per_sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    (out_dir / "audit_key.json").write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="SESSION_DIR, or VIDEO REPORT POSE")
    ap.add_argument("--ball"); ap.add_argument("--calibration")
    ap.add_argument("--out"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--audit", type=int); ap.add_argument("--out-dir"); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    a.session = a.paths[0] if len(a.paths) == 1 else None
    if not a.session:
        a.video, a.report, a.pose = a.paths
    video, rep, pz, ball, cal = load(a)
    if a.audit:
        audit(video, rep, pz, ball, a.audit, a.out_dir, seed=a.seed)
    else:
        render(video, rep, pz, ball, cal, a.out, a.start, a.stop or len(ball), a.scale)
