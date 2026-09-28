#!/usr/bin/env python3
"""Fixture for the Kotlin end-to-end parity test: per-frame people (with the Python appearance
descriptors), ball track, calibration, and the Python session result for one clip."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "scripts"))
from pipeline import session as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("--ball", required=True); ap.add_argument("--pose", required=True)
ap.add_argument("--calibration", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--out", default=str(ROOT / "android/app/src/test/resources/session_fixture.json"))
a = ap.parse_args()

# The app stores ball positions as float32: run the reference on float32-rounded positions too.
b32 = json.loads(Path(a.ball).read_text())
b32["frames"] = [None if f is None else [float(np.float32(v)) for v in f] for f in b32["frames"]]
Path("/tmp/claude-ball32.json").write_text(json.dumps(b32))
a.ball = "/tmp/claude-ball32.json"
rep = S.run(a.video, a.ball, a.pose, a.calibration, a.model, "/tmp/claude-fixture-report.json", {"A": "A", "B": "B"})
pose = json.loads(Path(a.pose).read_text())
frames = [S.person_list(f) if i % 3 == 0 else None for i, f in enumerate(pose["frames"])]
S.compute_appearance(a.video, frames, stride=3)


def enc(d):
    return {"box": [float(x) for x in d["box"]], "score": float(d["score"]), "court": [float(x) for x in d["court"]],
            "kps": np.asarray(d["kps"]).ravel().round(3).tolist(),
            "desc": None if d.get("desc") is None else [None if np.isnan(x) else float(x) for x in d["desc"]],
            "overlap": bool(d.get("overlap", False))}


ball = json.loads(Path(a.ball).read_text())
cal = json.loads(Path(a.calibration).read_text())
doc = {"fps": ball["video"]["fps"], "height": ball["video"]["heightPx"], "ball": ball["frames"],
       "imageFromCourt": cal["imageFromCourt"],
       "people": [None if f is None else [enc(d) for d in f] for f in frames],
       "expected": {"hits": rep["events"]["hits"], "bounces": rep["events"]["bounces"],
                    "identity": rep["perFrameIdentity"],
                    "shots": [{"hitFrame": s["hitFrame"], "playerId": s["playerId"], "stroke": s["stroke"],
                               "landing": s["landing"]["status"], "region": s["landing"].get("region")} for s in rep["shots"]]}}
Path(a.out).write_text(json.dumps(doc, separators=(",", ":")))
print("hits", len(doc["expected"]["hits"]), "shots", len(doc["expected"]["shots"]))
