#!/usr/bin/env python3
"""Render a session report onto the video: A/B boxes, ball trail, hit/bounce markers, stroke labels,
and a mini-court with landing points. Also renders audit strips for individual shots.

  render_overlay.py video report.json pose.json --out out.mp4 [--start F --stop F]
  render_overlay.py video report.json pose.json --audit N --out-dir dir   (random shot strips)
"""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "scripts"))
from pipeline import court  # noqa: E402

COL = {"A": (255, 190, 60), "B": (60, 170, 255), "unknown": (180, 180, 180)}


def load(video, report, pose):
    rep = json.loads(Path(report).read_text())
    pz = json.loads(Path(pose).read_text())
    ball = json.loads(Path(rep["ballPath"]).read_text())["frames"] if "ballPath" in rep else rep.get("ballTrack")
    return rep, pz, ball


def draw_frame(img, f, rep, pz, ball, shots_by_hit, bounce_set, recent):
    per = rep["perFrameIdentity"]
    ppl = pz["frames"][f] or []
    # nearest frame with identity (pose every 3rd frame)
    g = next((x for x in (f, f - 1, f + 1, f - 2, f + 2) if 0 <= x < len(per) and per[x]), None)
    if g is not None:
        for pid, a in per[g].items():
            p = (pz["frames"][g] or [])[a["det"]] if pz["frames"][g] else None
            if p is None:
                continue
            x1, y1, x2, y2 = map(int, p[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), COL[pid], 3)
            name = rep["players"][pid]["name"]
            cv2.putText(img, f"{name} ({a['side']})", (x1, max(20, y1 - 8)), 0, 0.8, COL[pid], 2)
    for k in range(max(0, f - 12), f + 1):
        b = ball[k] if ball and k < len(ball) else None
        if b:
            cv2.circle(img, (int(b[0]), int(b[1])), 4 if k < f else 8, (0, 255, 255), -1 if k < f else 2)
    for (kind, fr, txt, col) in recent:
        if f - fr < 20:
            cv2.putText(img, txt, (30, 120 + 40 * (kind == "bounce")), 0, 1.1, col, 3)
    return img


def render(video, rep, pz, ball, out, start, stop):
    cap = cv2.VideoCapture(str(video)); fps = cap.get(5)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.set(1, start)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w // 2, h // 2))
    shots = {s["hitFrame"]: s for s in rep["shots"]}
    bounces = set(rep["events"]["bounces"])
    recent = []
    for f in range(start, stop):
        ok, img = cap.read()
        if not ok:
            break
        if f in shots:
            s = shots[f]
            nm = rep["players"].get(s["playerId"], {}).get("name", "?")
            sp = f" {s['speed']['avgKmh']:.0f} km/h" if "speed" in s else ""
            recent.append(("hit", f, f"HIT {nm}: {s['stroke']}{sp}", COL.get(s["playerId"], COL["unknown"])))
        if f in bounces:
            recent.append(("bounce", f, "BOUNCE", (80, 255, 80)))
            b = ball[f]
            if b:
                cv2.circle(img, (int(b[0]), int(b[1])), 18, (80, 255, 80), 3)
        draw_frame(img, f, rep, pz, ball, shots, bounces, recent)
        cv2.putText(img, f"{f}", (w - 160, 40), 0, 1, (255, 255, 255), 2)
        vw.write(cv2.resize(img, (w // 2, h // 2)))
    vw.release()


def audit(video, rep, pz, ball, n, out_dir, seed=3, per_sheet=4):
    """Blind audit sheets: for each sampled predicted hit, one row of 5 full frames (f-6..f+6 step 3)
    with a far-court zoom band on top. Only frame numbers are drawn; the key with predictions is
    written separately so the reviewer judges contact / hitter / stroke independently."""
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
            full = cv2.resize(img[80:1000], (384, 184)); zoom = cv2.resize(img[120:420, 300:1700], (384, 82))
            t = np.vstack([zoom, full])
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
    ap.add_argument("video"); ap.add_argument("report"); ap.add_argument("pose")
    ap.add_argument("--out"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int)
    ap.add_argument("--audit", type=int); ap.add_argument("--out-dir"); ap.add_argument("--seed", type=int, default=3)
    a = ap.parse_args()
    rep, pz, ball = load(a.video, a.report, a.pose)
    if a.audit:
        audit(a.video, rep, pz, ball, a.audit, a.out_dir, seed=a.seed)
    else:
        render(a.video, rep, pz, ball, a.out, a.start, a.stop or len(ball))
