#!/usr/bin/env python3
"""Streaming RacketVision TrackNetV3 ball tracker for long videos or frame folders.

Unlike the milestone-0 adapter it never writes every frame to disk. The background
median is estimated from evenly spaced samples per segment (fixed-camera friendly).

Output: compact JSON {video:{...}, frames:[[x, y, conf] | null, ...]} in original pixels.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BALLTRACK_ROOT = ROOT / "third_party/RacketVision/source/BallTrack"
CKPT = ROOT / "work/models/racketvision/balltrack_best.pth"
W, H, SEQ = 512, 288, 4


def load_model(device, student=None, d_model=16, half=None):
    """half: run in float16 (default on GPUs: ~1.2x faster, detections identical within 1/900 frames)."""
    net = _load_model(device, student, d_model)
    if half is None:
        half = torch.device(device).type in ("mps", "cuda")
    return net.half() if half else net


def _load_model(device, student=None, d_model=16):
    sys.path.insert(0, str(BALLTRACK_ROOT))
    if student:  # distilled phone model (scripts/train_student_ball.py)
        from model.tracknet_v3 import TrackNet
        net = TrackNet(in_dim=3 * (SEQ + 1), out_dim=SEQ, last_only=True, d_model=d_model)
        net.load_state_dict(torch.load(student, map_location="cpu"))
        return net.to(device).eval()
    from mmengine.config import Config
    from mmengine.registry import MODELS
    import model  # noqa: F401  registers TrackNetV3
    cfg = Config.fromfile(str(BALLTRACK_ROOT / "configs/tracknetv3_base.py"), lazy_import=False)
    net = MODELS.build(cfg.model)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = state.get("state_dict", state)
    net.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()})
    return net.to(device).eval()


class FrameSource:
    """Uniform access to a video file or a folder of numbered jpg frames."""

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.is_dir():
            self.files = sorted(self.path.glob("*.jpg"))
            first = cv2.imread(str(self.files[0]))
            self.height, self.width = first.shape[:2]
            self.count, self.fps = len(self.files), 30.0
        else:
            cap = cv2.VideoCapture(str(self.path))
            self.width, self.height = int(cap.get(3)), int(cap.get(4))
            self.count, self.fps = int(cap.get(7)), float(cap.get(5))
            cap.release()
            self.files = None

    def read_at(self, indices):
        if self.files is not None:
            return [cv2.imread(str(self.files[i])) for i in indices]
        cap, out = cv2.VideoCapture(str(self.path)), []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if ok:
                out.append(f)
        cap.release()
        return out

    def stream(self, start=0, stop=None):
        stop = self.count if stop is None else stop
        if self.files is not None:
            for i in range(start, stop):
                yield i, cv2.imread(str(self.files[i]))
            return
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        while i < stop:
            ok, f = cap.read()
            if not ok:
                break
            yield i, f
            i += 1
        cap.release()


def background_median(src: FrameSource, start, stop, samples=48):
    idx = np.linspace(start, max(start, stop - 1), min(samples, stop - start)).astype(int)
    frames = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in src.read_at(idx)]
    return np.median(np.stack(frames), axis=0).astype(np.float32) / 255.0


def heatmap_to_point(hm, thr=0.5):
    """Largest connected blob above threshold; mirrors RacketVision's post-process."""
    mask = (hm > thr).astype(np.uint8)
    if not mask.any():
        return None
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[best, :4]
    conf = float(hm[y:y + h, x:x + w].mean())
    return float(x + w / 2), float(y + h / 2), conf


def run(path, out, device="mps", segment_s=120.0, batch=8, thr=0.5, limit=None, net=None, start=0, stop=None, progress=None,
        fast=True):
    if fast and Path(path).is_file():
        return run_fast(path, out, device, segment_s, batch, thr, limit, net, start, stop, progress)
    src = FrameSource(path)
    n = src.count if limit is None else min(limit, src.count)
    stop = n if stop is None else min(stop, n)
    net = net if net is not None else load_model(torch.device(device))
    sx, sy = src.width / W, src.height / H
    seg_len = max(SEQ + 1, int(segment_s * src.fps))
    results: list = [None] * n
    started = time.perf_counter()
    for seg_start in range(start - start % seg_len, stop, seg_len):
        seg_stop = min(n, seg_start + seg_len)
        if seg_stop <= start:
            continue
        median = np.moveaxis(background_median(src, seg_start, seg_stop), -1, 0)
        window: list[np.ndarray] = []
        pending_idx, pending = [], []

        def flush():
            if not pending:
                return
            arr = np.stack(pending)
            if len(arr) < batch:  # constant batch shape avoids MPS allocator growth
                arr = np.concatenate([arr, np.repeat(arr[-1:], batch - len(arr), 0)], 0)
            with torch.no_grad():
                x = torch.from_numpy(arr).to(device).to(next(net.parameters()).dtype)
                preds = net.forward(frames=x)[0].float().cpu().numpy()[:len(pending)]
                del x
            if device == "mps":  # the MPS caching allocator otherwise holds ~10 GB of activations
                torch.mps.synchronize(); torch.mps.empty_cache()
            for i, p in zip(pending_idx, preds):
                pt = heatmap_to_point(p[0], thr) if p.shape[0] == 1 else heatmap_to_point(p[-1], thr)
                results[i] = None if pt is None else [round(pt[0] * sx, 1), round(pt[1] * sy, 1), round(pt[2], 4)]
            pending_idx.clear(); pending.clear()

        for i, frame in src.stream(max(seg_start, start - SEQ + 1), min(seg_stop, stop)):
            img = np.moveaxis(cv2.resize(frame, (W, H)).astype(np.float32) / 255.0, -1, 0)
            window.append(img)
            if len(window) > SEQ:
                window.pop(0)
            seq = [window[0]] * (SEQ - len(window)) + window  # pad start like upstream (repeat first)
            if i < start:
                continue  # warm-up frames for the temporal window only
            pending.append(np.concatenate([median, *seq], 0))
            pending_idx.append(i)
            if len(pending) == batch:
                flush()
        flush()
        if device == "mps":
            torch.mps.empty_cache()
        done = seg_stop
        el = time.perf_counter() - started
        print(f"{Path(path).name}: {done}/{n} frames, {done/el:.1f} fps", flush=True)
        if progress:
            progress(done, n)
    doc = {"schemaVersion": "2.0", "type": "ball_track",
           "video": {"path": str(Path(path).resolve().relative_to(ROOT)) if str(Path(path).resolve()).startswith(str(ROOT)) else str(path),
                     "widthPx": src.width, "heightPx": src.height, "fps": src.fps, "frameCount": n},
           "producer": {"component": "RacketVision/TrackNetV3", "input": [W, H], "seq": SEQ,
                        "threshold": thr, "backgroundSegmentSeconds": segment_s},
           "runtimeSeconds": round(time.perf_counter() - started, 2),
           "frames": results}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def run_fast(path, out, device="mps", segment_s=120.0, batch=8, thr=0.5, limit=None, net=None, start=0, stop=None,
             progress=None):
    """Same model, inputs and outputs as the reference loop, about twice as fast: frames are decoded in a
    background thread, uploaded to the GPU once as uint8, and the 15-channel inputs (background + 4 frames)
    are assembled on the GPU instead of being rebuilt and copied for every frame."""
    import queue
    import threading
    src = FrameSource(path)
    n = src.count if limit is None else min(limit, src.count)
    stop = n if stop is None else min(stop, n)
    dev = torch.device(device)
    net = net if net is not None else load_model(dev)
    sx, sy = src.width / W, src.height / H
    seg_len = max(SEQ + 1, int(segment_s * src.fps))
    results: list = [None] * n
    started = time.perf_counter()

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    for seg_start in range(start - start % seg_len, stop, seg_len):
        seg_stop = min(n, seg_start + seg_len)
        if seg_stop <= start:
            continue
        median = torch.from_numpy(np.ascontiguousarray(np.moveaxis(background_median(src, seg_start, seg_stop), -1, 0))).to(dev)
        q: queue.Queue = queue.Queue(maxsize=64)
        a = max(seg_start, start - SEQ + 1)
        b = min(seg_stop, stop)

        def reader():
            for i, frame in src.stream(a, b):
                q.put((i, cv2.resize(frame, (W, H))))
            q.put(None)
        th = threading.Thread(target=reader, daemon=True)
        th.start()
        window: list = []
        pend_idx, pend_x = [], []

        def flush():
            if not pend_x:
                return
            x = torch.stack(pend_x)
            dt = next(net.parameters()).dtype
            if x.dtype != dt:
                x = x.to(dt)
            k = len(pend_x)
            if k < batch:   # constant batch shape avoids MPS allocator growth
                x = torch.cat([x, x[-1:].expand(batch - k, -1, -1, -1)], 0)
            with torch.no_grad():
                hm = net.forward(frames=x)[0][:k]
                hm = hm[:, -1] if hm.shape[1] > 1 else hm[:, 0]
                hm = hm.float().cpu().numpy()
            if device == "mps":
                torch.mps.empty_cache()
            for i, h in zip(pend_idx, hm):
                pt = heatmap_to_point(h, thr)
                results[i] = None if pt is None else [round(pt[0] * sx, 1), round(pt[1] * sy, 1), round(pt[2], 4)]
            pend_idx.clear(); pend_x.clear()

        while True:
            item = q.get()
            if item is None:
                break
            i, small = item
            t = torch.from_numpy(small).to(dev).permute(2, 0, 1).float().div_(255.0)
            window.append(t)
            if len(window) > SEQ:
                window.pop(0)
            if i < start:
                continue
            seq = [window[0]] * (SEQ - len(window)) + window
            pend_x.append(torch.cat([median, *seq], 0))
            pend_idx.append(i)
            if len(pend_x) == batch:
                flush()
        flush()
        th.join()
        sync()
        el = time.perf_counter() - started
        print(f"{Path(path).name}: {seg_stop}/{n} frames, {(seg_stop - max(start, seg_start)) / max(1e-6, el):.1f} fps (fast)", flush=True)
        if progress:
            progress(seg_stop, n)
    doc = {"schemaVersion": "2.0", "type": "ball_track",
           "video": {"path": str(path), "widthPx": src.width, "heightPx": src.height, "fps": src.fps, "frameCount": n},
           "producer": {"component": "RacketVision/TrackNetV3", "input": [W, H], "seq": SEQ, "threshold": thr,
                        "backgroundSegmentSeconds": segment_s, "loop": "fast"},
           "runtimeSeconds": round(time.perf_counter() - started, 2), "frames": results}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="video files or frame folders")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--name-depth", type=int, default=1, help="path parts joined for output name")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--student", help="student checkpoint (.pt); default = RacketVision teacher")
    a = ap.parse_args()
    net = load_model(torch.device(a.device), a.student)
    for p in a.inputs:
        p = Path(p)
        name = "_".join(p.with_suffix("").parts[-a.name_depth:])
        out = a.out_dir / f"{name}_ball{a.suffix}.json"
        if out.exists():
            print("skip", out); continue
        run(p, out, a.device, limit=a.limit, net=net, start=a.start, stop=a.stop)


if __name__ == "__main__":
    main()
