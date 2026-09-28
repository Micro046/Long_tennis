#!/usr/bin/env python3
"""Run pipeline/ball.py in fresh processes per chunk (works around a PyTorch MPS memory
leak on long videos) and merge. Chunks are aligned to the 120 s background segments, so
the merged output is identical to a single uninterrupted run."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/bin/python")


def run_video(video, out_dir, seg_s=120.0, segs_per_chunk=1, extra=()):
    video = Path(video)
    cap = cv2.VideoCapture(str(video)); n, fps = int(cap.get(7)), cap.get(5); cap.release()
    seg = max(5, int(seg_s * fps))
    chunks = [(a, min(n, a + seg * segs_per_chunk)) for a in range(0, n, seg * segs_per_chunk)]
    parts = []
    for a, b in chunks:
        part = Path(out_dir) / f"{video.stem}_ball_c{a}.json"
        if not part.exists():
            subprocess.run([PY, str(ROOT / "scripts/pipeline/ball.py"), str(video), "--out-dir", str(out_dir),
                            "--start", str(a), "--stop", str(b), "--suffix", f"_c{a}", *extra], check=True)
        parts.append((a, b, part))
    merged = None
    for a, b, part in parts:
        d = json.loads(part.read_text())
        if merged is None:
            merged = d; merged["runtimeSeconds"] = 0
        merged["frames"][a:b] = d["frames"][a:b]
        merged["runtimeSeconds"] += d["runtimeSeconds"]
    (Path(out_dir) / f"{video.stem}_ball.json").write_text(json.dumps(merged, separators=(",", ":")))
    for _, _, part in parts:
        part.unlink()


def run_folders(folders, out_dir, per_process=10, extra=()):
    todo = [f for f in folders if not (Path(out_dir) / ("_".join(Path(f).parts[-2:]) + "_ball.json")).exists()]
    for i in range(0, len(todo), per_process):
        subprocess.run([PY, str(ROOT / "scripts/pipeline/ball.py"), *todo[i:i + per_process], "--out-dir", str(out_dir),
                        "--name-depth", "2", *extra], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--folders", action="store_true")
    ap.add_argument("--student")
    a = ap.parse_args()
    extra = ("--student", a.student) if a.student else ()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    if a.folders:
        run_folders(a.inputs, a.out_dir, extra=extra)
    else:
        for v in a.inputs:
            run_video(v, a.out_dir, extra=extra)
