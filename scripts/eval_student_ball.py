#!/usr/bin/env python3
"""Evaluate the distilled student ball tracker against ground truth and against the teacher.

Held-out data only:
  - TrackNet game10 (never used in student training) -- ground-truth positions
  - RacketVision test clips (never used)             -- sparse ground-truth positions
  - amateur match frames after 70%                   -- agreement with the teacher track
Exports the student to ONNX for the app with --export.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/RacketVision/source/BallTrack"))
from pipeline import ball as B  # noqa: E402
from train_student_ball import build_model  # noqa: E402


def load_student(path, d_model, device):
    net = build_model(d_model)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net.to(device).eval()


def track(net, src, device, start=0, stop=None):
    """Same streaming procedure as pipeline/ball.run (median per 120 s, last-frame heatmap)."""
    stop = src.count if stop is None else stop
    out = [None] * src.count
    seg_len = int(120 * src.fps)
    sx, sy = src.width / B.W, src.height / B.H
    for seg_start in range(start - start % seg_len, stop, seg_len):
        seg_stop = min(src.count, seg_start + seg_len)
        med = np.moveaxis(B.background_median(src, seg_start, seg_stop), -1, 0)
        win, batch, idx = [], [], []

        def flush():
            if not batch:
                return
            with torch.no_grad():
                x = torch.from_numpy(np.stack(batch)).to(device)
                p = net(frames=x)[0].float().cpu().numpy()
            if device == "mps":
                torch.mps.synchronize(); torch.mps.empty_cache()
            for i, hm in zip(idx, p):
                pt = B.heatmap_to_point(hm[-1])
                out[i] = None if pt is None else [pt[0] * sx, pt[1] * sy]
            batch.clear(); idx.clear()

        for i, f in src.stream(max(seg_start, start - B.SEQ + 1), min(seg_stop, stop)):
            win.append(np.moveaxis(cv2.resize(f, (B.W, B.H)).astype(np.float32) / 255.0, -1, 0))
            win = win[-B.SEQ:]
            if i < start:
                continue
            seq = [win[0]] * (B.SEQ - len(win)) + win
            batch.append(np.concatenate([med, *seq], 0)); idx.append(i)
            if len(batch) == 8:
                flush()
        flush()
    return out


def score(pred, gt):
    d, fn, fp, tn = [], 0, 0, 0
    for p, g in zip(pred, gt):
        if g is None:
            fp += p is not None; tn += p is None
        elif p is None:
            fn += 1
        else:
            d.append(np.hypot(p[0] - g[0], p[1] - g[1]))
    d = np.array(d)
    return {"visibleGT": len(d) + fn, "recall": round(len(d) / max(1, len(d) + fn), 3),
            "falsePresence": fp, "medianPx": round(float(np.median(d)), 2) if len(d) else None,
            "within10px": round(float((d <= 10).mean()), 3) if len(d) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default=str(ROOT / "work/models/student_ball_d16.pt"))
    ap.add_argument("--d-model", type=int, default=16)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--export", default=str(ROOT / "android/app/src/main/assets/models/ball_student_d16.onnx"))
    ap.add_argument("--skip-amateur", action="store_true")
    a = ap.parse_args()
    net = load_student(a.student, a.d_model, a.device)
    res = {"student": Path(a.student).name}
    # TrackNet game10 (held out) vs GT, and teacher on the same frames
    st, te = [], []
    for lab in sorted((ROOT / "data/public/tracknet/Dataset/game10").glob("Clip*/Label.csv")):
        df = pd.read_csv(lab)
        gt = [None if (v == 0 or not np.isfinite(x) or x <= 0) else [x, y]
              for v, x, y in zip(df["visibility"], df["x-coordinate"], df["y-coordinate"])]
        src = B.FrameSource(lab.parent)
        p = track(net, src, a.device)
        t = json.loads((ROOT / f"outputs/v2/ball/tracknet/game10_{lab.parent.name}_ball.json").read_text())["frames"]
        st.append((p, gt)); te.append(([None if f is None else f[:2] for f in t[:len(gt)]], gt))
    res["tracknet_game10"] = {"student": score(sum([p for p, _ in st], []), sum([g for _, g in st], [])),
                              "teacher": score(sum([p for p, _ in te], []), sum([g for _, g in te], []))}
    # RacketVision sparse GT
    sp, sg, tp = [], [], []
    for c in ("match100_000", "match101_000", "match102_000"):
        src = B.FrameSource(ROOT / f"data/public/racketvision/{c}.mp4")
        p = track(net, src, a.device)
        t = json.loads((ROOT / f"work/v2test/{c}_ball.json").read_text())["frames"]
        gt = {int(r["Frame"]): r for r in csv.DictReader(open(ROOT / f"data/public/racketvision/annotations/{c}_ball.csv", encoding="utf-8-sig"))}
        for f, r in gt.items():
            g = [float(r["X"]), float(r["Y"])] if int(r["Visibility"]) > 0 else None
            sp.append(p[f]); sg.append(g); tp.append(None if t[f] is None else t[f][:2])
    res["racketvision_test"] = {"student": score(sp, sg), "teacher": score(tp, sg)}
    # amateur held-out: agreement with teacher (teacher as reference)
    if not a.skip_amateur:
        src = B.FrameSource(ROOT / "data/public/youtube_amateur/4apx6gd5Uxs.mp4")
        start = int(0.7 * src.count)
        p = track(net, src, a.device, start=start)
        t = json.loads((ROOT / "outputs/v2/ball/youtube/4apx6gd5Uxs_ball.json").read_text())["frames"]
        res["amateur_heldout_vs_teacher"] = score(p[start:], [None if f is None else f[:2] for f in t[start:]])
    # export + CPU latency
    class Wrap(torch.nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m._forward(x)[:, -1:]
    cpu = load_student(a.student, a.d_model, "cpu")
    x = torch.rand(1, 15, B.H, B.W)
    torch.onnx.export(Wrap(cpu).eval(), x, a.export, input_names=["frames"], output_names=["heatmap"], opset_version=17, dynamo=False)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 4
    s = ort.InferenceSession(a.export, so)
    s.run(None, {"frames": x.numpy()}); t0 = time.time()
    for _ in range(10):
        s.run(None, {"frames": x.numpy()})
    res["onnxCpu4ThreadsMsPerFrame"] = round((time.time() - t0) * 100, 1)
    res["onnxBytes"] = Path(a.export).stat().st_size
    print(json.dumps(res, indent=1))
    (ROOT / "outputs/v2/student_ball_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
