#!/usr/bin/env python3
"""Person detection + COCO-17 pose for fixed-camera tennis video.

Two detector passes per frame, exactly as the phone will do it:
  1. full frame (near player is large),
  2. tiles covering the far half of the court (far player is small), placed from the
     court calibration so the far athlete is ~2x larger at detector input.
Detections are merged with NMS; only people whose feet map inside the court plus a
run-off margin are kept (removes umpires, spectators, people on adjacent courts).

Output JSON: {video, calibration, frames: [[ [x1,y1,x2,y2,score, cx_m, cy_m, kp(17x3 flat)...] ...], ...]}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "work/cache"))
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rtmlib import YOLOX, RTMPose  # noqa: E402

from pipeline import court  # noqa: E402

CK = ROOT / "work/cache/rtmlib/hub/checkpoints"
DET = CK / "yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx"
POSE = CK / "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx"
MARGIN_X, MARGIN_BACK = 4.0, 7.0  # metres of run-off kept around the court


class Detector:
    def __init__(self):
        self.det = YOLOX(str(DET), model_input_size=(416, 416), backend="onnxruntime", device="cpu")
        self.pose = RTMPose(str(POSE), model_input_size=(192, 256), backend="onnxruntime", device="cpu")
        threads = int(os.environ.get("POSE_THREADS", "4"))
        if threads > 0:  # bounded, non-spinning threads so the GPU ball job is not starved
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            for tool, path in ((self.det, DET), (self.pose, POSE)):
                tool.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])

    def detect(self, img, min_score=0.3):
        x, ratio = self.det.preprocess(img)
        out = self.det.inference(x)[0]
        boxes, scores = out[0, :, :4] / ratio, out[0, :, 4]
        keep = scores > min_score
        return np.concatenate([boxes[keep], scores[keep, None]], 1)


def far_tiles(H_img_from_court, w, h, n_tiles=2):
    pts = court.project(H_img_from_court, [(-MARGIN_X, -MARGIN_BACK), (court.WIDTH + MARGIN_X, -MARGIN_BACK),
                                           (-MARGIN_X, court.NET_Y), (court.WIDTH + MARGIN_X, court.NET_Y),
                                           (0, 0), (1, 0)])
    px_per_m = np.linalg.norm(pts[5] - pts[4])
    x1, x2 = max(0, pts[:4, 0].min()), min(w, pts[:4, 0].max())
    y2 = min(h, pts[:4, 1].max())
    y1 = max(0, pts[:4, 1].min() - 2.6 * px_per_m)  # room for a standing player
    tiles, tw = [], (x2 - x1) / n_tiles
    for i in range(n_tiles):
        a = x1 + i * tw - (0.15 * tw if i else 0)
        b = x1 + (i + 1) * tw + (0.15 * tw if i < n_tiles - 1 else 0)
        tiles.append([int(a), int(y1), int(b), int(y2)])
    return tiles


def nms(dets, thr=0.5):
    if len(dets) == 0:
        return dets
    idx = cv2.dnn.NMSBoxes([[float(d[0]), float(d[1]), float(d[2] - d[0]), float(d[3] - d[1])] for d in dets],
                           [float(d[4]) for d in dets], 0.0, thr)
    return dets[np.array(idx).ravel()] if len(idx) else dets[:0]


def suppress_contained(dets, frac=0.6):
    """Drop a box when most of it lies inside a larger kept box (limb/part detections)."""
    order = np.argsort(-(dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))
    kept = []
    for i in order:
        a = dets[i]
        area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        inside = False
        for j in kept:
            b = dets[j]
            iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if iw * ih / area > frac:
                inside = True; break
        if not inside:
            kept.append(i)
    return dets[sorted(kept)]


def process_frame(D, frame, H_img, H_court, tiles):
    dets = [D.detect(frame)]
    for x1, y1, x2, y2 in tiles:
        d = D.detect(frame[y1:y2, x1:x2])
        # a box touching the tile's bottom/side edge is a truncated person (usually the near player)
        if len(d):
            d = d[(d[:, 3] < (y2 - y1) - 4) & ((d[:, 0] > 2) | (x1 == 0)) & ((d[:, 2] < (x2 - x1) - 2) | (x2 >= frame.shape[1] - 1))]
        if len(d):
            d[:, [0, 2]] += x1; d[:, [1, 3]] += y1
            dets.append(d)
    dets = suppress_contained(nms(np.concatenate(dets, 0))) if dets else np.zeros((0, 5))
    people = []
    feet = np.stack([(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]], 1) if len(dets) else np.zeros((0, 2))
    cm = court.project(H_court, feet) if len(dets) else np.zeros((0, 2))
    keep = [i for i in range(len(dets)) if -MARGIN_X <= cm[i, 0] <= court.WIDTH + MARGIN_X
            and -MARGIN_BACK <= cm[i, 1] <= court.LENGTH + MARGIN_BACK]
    if not keep:
        return people
    boxes = dets[keep, :4]
    kps, sc = D.pose(frame, bboxes=boxes.tolist())
    for j, i in enumerate(keep):
        people.append([*np.round(dets[i, :4], 1).tolist(), round(float(dets[i, 4]), 3),
                       round(float(cm[i, 0]), 2), round(float(cm[i, 1]), 2),
                       *np.round(np.concatenate([kps[j], sc[j][:, None]], 1).ravel(), 1).tolist()])
    return people


def run(video, calib_path, out, stride=1, limit=None, progress=None):
    cal = json.loads(Path(calib_path).read_text())
    H_img, H_court = court.image_from_court(cal), court.court_from_image(cal)
    cap = cv2.VideoCapture(str(video))
    w, h, n, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(7)), cap.get(5)
    n = n if limit is None else min(n, limit)
    tiles = far_tiles(H_img, w, h)
    D = Detector()
    frames, t0, i = [None] * n, time.perf_counter(), 0
    prev_small, cuts = None, []
    while i < n:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
        if prev_small is not None and np.abs(small - prev_small).mean() > 25:
            cuts.append(i)
        prev_small = small
        if i % stride == 0:
            frames[i] = process_frame(D, frame, H_img, H_court, tiles)
        else:
            frames[i] = None
        i += 1
        if i % 1000 == 0:
            print(f"{Path(video).name}: {i}/{n} {i/(time.perf_counter()-t0):.1f} fps", flush=True)
        if progress and i % 150 == 0:
            progress(i, n)
    doc = {"schemaVersion": "2.0", "type": "pose_track",
           "video": {"path": str(video), "widthPx": w, "heightPx": h, "fps": fps, "frameCount": n},
           "calibration": str(calib_path), "farTiles": tiles, "stride": stride, "sceneCuts": cuts,
           "personLayout": "x1,y1,x2,y2,score,courtX_m,courtY_m,17x(x,y,score)",
           "producer": {"detector": DET.name, "pose": POSE.name, "passes": "full+far_tiles"},
           "runtimeSeconds": round(time.perf_counter() - t0, 1), "frames": frames}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video"); ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    run(a.video, a.calibration, a.out, a.stride, a.limit)
