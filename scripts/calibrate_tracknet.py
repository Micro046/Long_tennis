#!/usr/bin/env python3
"""Court calibration for every TrackNet broadcast clip: TennisCourtDetector (14 keypoints, trained on
broadcast) gives the initial homography, then pipeline/court.calibrate snaps it to the painted lines.
Writes data/public/tracknet/calibration/<game>_<clip>.json in the project format."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
TCD = ROOT / "third_party/yastrebksv-TennisCourtDetector"
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(TCD))
from pipeline import court  # noqa: E402
from tracknet import BallTrackerNet  # noqa: E402
from postprocess import postprocess  # noqa: E402
from homography import get_trans_matrix  # noqa: E402

# reference-court pixels -> metres (doubles 286..1379 x 561..2935)
S = np.array([[1093 / 10.97, 0, 286], [0, 2374 / 23.77, 561], [0, 0, 1]], float)


def main():
    net = BallTrackerNet(out_channels=15)
    net.load_state_dict(torch.load(ROOT / "work/models/court/model_tennis_court_det.pt", map_location="cpu"))
    net.eval()
    out_dir = ROOT / "data/public/tracknet/calibration"; out_dir.mkdir(parents=True, exist_ok=True)
    ok_n = 0
    for clip in sorted((ROOT / "data/public/tracknet/Dataset").glob("game*/Clip*")):
        files = sorted(clip.glob("*.jpg"))
        img = np.median(np.stack([cv2.imread(str(f)) for f in files[::max(1, len(files) // 9)][:9]]), 0).astype(np.uint8)
        h, w = img.shape[:2]
        x = torch.tensor(np.rollaxis(cv2.resize(img, (640, 360)).astype(np.float32) / 255., 2, 0))[None]
        with torch.no_grad():
            pred = F.sigmoid(net(x)[0]).numpy()
        pts = []
        for k in range(14):
            px, py = postprocess((pred[k] * 255).astype(np.uint8), low_thresh=170, max_radius=25)
            pts.append((px * w / 1280, py * h / 720) if px is not None else (None, None))
        M = get_trans_matrix(pts)
        rec = {"clip": f"{clip.parent.name}_{clip.name}", "ok": False}
        if M is not None:
            H0 = M @ S  # metres -> image
            init = {k: tuple(court.project(H0, [p])[0]) for k, p in court.MODEL_KP.items()
                    if k in ("far_left_singles", "far_right_singles", "near_left_singles", "near_right_singles")}
            try:
                r = court.calibrate(img, init)
                good = r["reprojectionRmsPx"] is not None and r["reprojectionRmsPx"] < 6 and r["lineSupport"] > 0.6
                rec = {**r, "clip": rec["clip"], "ok": bool(good), "source": "TennisCourtDetector + line snap"}
                ok_n += good
            except Exception as e:
                rec["error"] = str(e)
            if not rec["ok"]:  # fall back to the detector's own fit when it matches the paint reasonably
                sup = court.line_support(court.white_mask(img), H0)
                if sup > 0.45:
                    rec = {"clip": rec["clip"], "ok": True, "imageFromCourt": H0.tolist(),
                           "courtFromImage": np.linalg.inv(H0).tolist(), "lineSupport": float(sup),
                           "reprojectionRmsPx": None, "source": "TennisCourtDetector only"}
                    ok_n += 1
        (out_dir / f"{rec['clip']}.json").write_text(json.dumps(rec))
        print(rec["clip"], rec["ok"], rec.get("reprojectionRmsPx"), rec.get("lineSupport"), flush=True)
    print("calibrated", ok_n)


if __name__ == "__main__":
    main()
