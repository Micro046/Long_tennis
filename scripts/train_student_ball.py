#!/usr/bin/env python3
"""Distil a small TrackNetV3 ball tracker (phone speed) from the RacketVision teacher.

Training targets: Gaussian heatmaps (sigma 2.5 @ 512x288, as upstream) centred on
  - TrackNet dataset ground-truth ball positions (broadcast, 10 games; game10 held out),
  - teacher positions on the amateur fixed-camera match (first 70% of the video; the rest held out).
Loss: upstream weighted (focal-style) BCE. Architecture: upstream TrackNetV3 with a narrower width.

Frames are cached once at 512x288 uint8 in work/student_cache (memmap) for fast random access.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/RacketVision/source/BallTrack"))
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "scripts"))
W, H, SEQ = 512, 288, 4
CACHE = ROOT / "work/student_cache"


def build_model(d_model):
    from model.tracknet_v3 import TrackNet
    return TrackNet(in_dim=3 * (SEQ + 1), out_dim=SEQ, last_only=True, d_model=d_model)


# ------------------------------------------------------------------ caching
def cache_source(name, frame_iter, n, targets, median_fn):
    """frame_iter yields BGR frames; targets: (n,2) model-space coords or NaN."""
    CACHE.mkdir(parents=True, exist_ok=True)
    fpath = CACHE / f"{name}_frames.u8"
    meta = CACHE / f"{name}.json"
    if meta.exists():
        return
    mm = np.memmap(fpath, dtype=np.uint8, mode="w+", shape=(n, H, W, 3))
    for i, f in enumerate(frame_iter):
        if i >= n:
            break
        mm[i] = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
    mm.flush()
    np.save(CACHE / f"{name}_targets.npy", targets.astype(np.float32))
    np.save(CACHE / f"{name}_median.npy", median_fn(mm))
    meta.write_text(json.dumps({"n": n}))


def prepare(teacher_youtube):
    # TrackNet: one source per clip (median per clip), GT positions
    for lab in sorted((ROOT / "data/public/tracknet/Dataset").glob("game*/Clip*/Label.csv")):
        game, clip = lab.parent.parent.name, lab.parent.name
        df = pd.read_csv(lab)
        xy = df[["x-coordinate", "y-coordinate"]].to_numpy(float)
        bad = (df["visibility"].to_numpy() == 0) | ~np.isfinite(xy).all(1) | (xy[:, 0] <= 0)
        xy[bad] = np.nan
        xy = xy * [W / 1280, H / 720]
        files = [lab.parent / f for f in df["file name"]]
        cache_source(f"tn_{game}_{clip}", (cv2.imread(str(p)) for p in files), len(files), xy,
                     lambda mm: np.median(mm[np.linspace(0, len(mm) - 1, min(48, len(mm))).astype(int)], 0).astype(np.uint8))
    # amateur match: teacher positions, background per 2-minute segment (store per-frame segment id)
    if not Path(teacher_youtube).exists():
        print("teacher track for the amateur match not ready; skipped"); return
    ball = json.loads(Path(teacher_youtube).read_text())
    vid = ROOT / ball["video"]["path"] if not Path(ball["video"]["path"]).is_absolute() else Path(ball["video"]["path"])
    sx, sy = W / ball["video"]["widthPx"], H / ball["video"]["heightPx"]
    fr = ball["frames"]
    xy = np.array([[f[0] * sx, f[1] * sy] if f else [np.nan, np.nan] for f in fr], float)
    cap = cv2.VideoCapture(str(vid))

    def it():
        while True:
            ok, f = cap.read()
            if not ok:
                return
            yield f

    def seg_medians(mm):
        seg = int(120 * ball["video"]["fps"])
        out = []
        for s in range(0, len(mm), seg):
            idx = np.linspace(s, min(len(mm), s + seg) - 1, 48).astype(int)
            out.append(np.median(mm[idx], 0).astype(np.uint8))
        return np.stack(out)

    cache_source("yt_4apx", it(), len(fr), xy, seg_medians)


# ------------------------------------------------------------------ dataset
class Sources:
    def __init__(self, names):
        self.src = []
        for n in names:
            m = json.loads((CACHE / f"{n}.json").read_text())
            fr = np.memmap(CACHE / f"{n}_frames.u8", dtype=np.uint8, mode="r", shape=(m["n"], H, W, 3))
            self.src.append((n, fr, np.load(CACHE / f"{n}_targets.npy"), np.load(CACHE / f"{n}_median.npy")))

    def sample(self, rng, k, frac_range=(0.0, 1.0), positive=0.8):
        s = self.src[rng.integers(len(self.src))]
        name, fr, tg, med = s
        lo, hi = int(frac_range[0] * len(fr)), int(frac_range[1] * len(fr))
        for _ in range(20):
            i = int(rng.integers(max(SEQ - 1, lo), hi))
            if rng.random() > positive or np.isfinite(tg[i]).all():
                break
        seq = fr[i - SEQ + 1:i + 1].astype(np.float32) / 255.0
        m = med if med.ndim == 3 else med[min(len(med) - 1, i // int(120 * 29.97))]
        x = np.concatenate([m[None].astype(np.float32) / 255.0, seq], 0)  # (5,H,W,3)
        t = tg[i].copy()
        if rng.random() < 0.5:  # horizontal flip
            x = x[:, :, ::-1]
            if np.isfinite(t).all():
                t[0] = W - 1 - t[0]
        x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
        return np.moveaxis(np.clip(x, 0, 1), -1, 1).reshape(15, H, W), t


def heatmap(t, sigma=2.5):
    if not np.isfinite(t).all():
        return np.zeros((1, H, W), np.float32)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    return np.exp(-0.5 * ((xx - t[0]) ** 2 + (yy - t[1]) ** 2) / sigma ** 2)[None].astype(np.float32)


def wbce(p, y):
    return (-(torch.square(1 - p) * y * torch.log(torch.clamp(p, 1e-7, 1))
              + torch.square(p) * (1 - y) * torch.log(torch.clamp(1 - p, 1e-7, 1)))).mean()


def train(d_model, steps, batch, out, device):
    tn = [p.stem for p in CACHE.glob("tn_game*.json") if not p.stem.startswith("tn_game10_")]
    train_tn, train_yt = Sources(tn), Sources(["yt_4apx"])
    net = build_model(d_model).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(steps):
        xs, ys = [], []
        for b in range(batch):
            if b % 2 == 0:
                x, t = train_yt.sample(rng, 1, (0.0, 0.7))
            else:
                x, t = train_tn.sample(rng, 1)
            xs.append(x); ys.append(heatmap(t))
        x = torch.from_numpy(np.stack(xs)).to(device); y = torch.from_numpy(np.stack(ys)).to(device)
        p = net(frames=x)[0]
        loss = wbce(p, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if device == "mps" and step % 50 == 0:
            torch.mps.synchronize(); torch.mps.empty_cache()
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.5f} {time.time()-t0:.0f}s", flush=True)
        if (step + 1) % 2000 == 0 or step + 1 == steps:
            torch.save(net.state_dict(), out)
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--teacher-youtube", default=str(ROOT / "outputs/v2/ball/youtube/4apx6gd5Uxs_ball.json"))
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "work/models/student_ball_d16.pt"))
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()
    if a.prepare:
        prepare(a.teacher_youtube)
    else:
        train(a.d_model, a.steps, a.batch, a.out, a.device)
