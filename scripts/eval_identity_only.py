#!/usr/bin/env python3
"""Run only the identity stage on a pose file (pose stride as on the phone) and score it against
the independent identity labels. Writes a minimal report usable by eval_amateur.identity_eval."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline import identity as idm, session as S  # noqa: E402
import eval_amateur as EA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("pose"); ap.add_argument("labels")
ap.add_argument("--stride", type=int, default=3); ap.add_argument("--out")
a = ap.parse_args()
pose = json.loads(Path(a.pose).read_text())
fps = pose["video"]["fps"]
frames = [S.person_list(f) if i % a.stride == 0 else None for i, f in enumerate(pose["frames"])]
S.compute_appearance(a.video, frames, stride=a.stride)
ident, tracks, seed = idm.assign_identities(frames, pose.get("sceneCuts", []), fps)
rep = {"identity": {"seedFrame": seed["seedFrame"]},
       "perFrameIdentity": [{k: {"det": v["det"], "side": v["side"]} for k, v in f.items()} for f in ident]}
res = EA.identity_eval(rep, json.loads(Path(a.labels).read_text()))
print(json.dumps({k: v for k, v in res.items() if k != "items"}, indent=1))
print("errors:", [i for i in res["items"] if i["outcome"] != "correct"])
cov = sum(1 for f in ident[::a.stride] if len(f) == 2) / max(1, len(ident[::a.stride]))
print("tracklets", len(tracks), "labelled", sum(t["label"] in ("A", "B") for t in tracks),
      "pose-frames with both A and B:", round(cov, 3), "sceneCuts", len(pose.get("sceneCuts", [])))
if a.out:
    Path(a.out).write_text(json.dumps({"result": res, "tracklets": len(tracks)}, indent=1))
