#!/usr/bin/env python3
"""Pose on THETIS RGB clips (single player facing the camera) for stroke-rule validation."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "work/cache"))
os.environ.setdefault("POSE_THREADS", "3")
sys.path.insert(0, str(ROOT / "scripts"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pipeline.pose import Detector  # noqa: E402

CLASSES = ["forehand_flat", "forehand_openstands", "backhand", "backhand2hands"]
out = ROOT / "outputs/v2/thetis_pose.json"
D = Detector()
res = json.loads(out.read_text()) if out.exists() else {}
for cls in CLASSES:
    for v in sorted((ROOT / "third_party/THETIS/VIDEO_RGB" / cls).glob("*.avi")):
        key = f"{cls}/{v.name}"
        if key in res:
            continue
        cap = cv2.VideoCapture(str(v)); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            d = D.detect(f)
            if len(d) == 0:
                frames.append(None); continue
            b = d[np.argmax((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]))]  # largest person
            k, s = D.pose(f, bboxes=[b[:4].tolist()])
            frames.append([*np.round(b[:4], 1).tolist(), *np.round(np.concatenate([k[0], s[0][:, None]], 1).ravel(), 1).tolist()])
        res[key] = {"fps": cap.get(5), "frames": frames}
        if len(res) % 50 == 0:
            out.write_text(json.dumps(res)); print(len(res), flush=True)
out.write_text(json.dumps(res)); print("done", len(res))
