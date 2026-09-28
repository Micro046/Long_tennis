#!/usr/bin/env python3
"""Pose (every 3rd frame, same detector passes as the server) on calibrated TrackNet clips.
Writes outputs/v2/pose/tracknet/<game>_<clip>_pose.json in the pipeline/pose.py format."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "work/cache"))
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from pipeline import pose  # noqa: E402

STRIDE = 3
out_dir = ROOT / "outputs/v2/pose/tracknet"; out_dir.mkdir(parents=True, exist_ok=True)
D = pose.Detector()
for cal_path in sorted((ROOT / "data/public/tracknet/calibration").glob("*.json")):
    cal = json.loads(cal_path.read_text())
    out = out_dir / f"{cal['clip']}_pose.json"
    if not cal.get("ok") or out.exists():
        continue
    game, clip = cal["clip"].split("_")
    files = sorted((ROOT / "data/public/tracknet/Dataset" / game / clip).glob("*.jpg"))
    first = cv2.imread(str(files[0])); h, w = first.shape[:2]
    Hi, Hc = np.array(cal["imageFromCourt"]), np.array(cal["courtFromImage"])
    tiles = pose.far_tiles(Hi, w, h)
    frames = [pose.process_frame(D, cv2.imread(str(f)), Hi, Hc, tiles) if i % STRIDE == 0 else None
              for i, f in enumerate(files)]
    out.write_text(json.dumps({"schemaVersion": "2.0", "type": "pose_track",
                               "video": {"path": str(files[0].parent), "widthPx": w, "heightPx": h, "fps": 30.0,
                                         "frameCount": len(files)},
                               "calibration": str(cal_path), "stride": STRIDE, "sceneCuts": [], "frames": frames},
                              separators=(",", ":")))
    print(cal["clip"], len(files), flush=True)
