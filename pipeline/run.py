#!/usr/bin/env python3
"""End-to-end analysis of one session directory (the unit of work the server queues).

Session directory layout:
    meta.json          names, seedASide, handedness, quality
    video.mp4          uploaded recording
    calibration.json   court homography from the setup markers
  outputs written here:
    ball.json  pose.json  report.json  progress.json  run.log

Ball tracking (GPU / MPS) and pose (CPU) run concurrently in separate processes; the ball tracker
runs one 120-s segment per process (bounded GPU memory). Then identity/events/shots -> report.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


class Progress:
    """progress.json = {stage, fraction, etaSeconds, stages: {ball: f, pose: f, analysis: f}}"""
    WEIGHTS = {"ball": 0.45, "pose": 0.45, "analysis": 0.10}

    def __init__(self, d: Path):
        self.path = d / "progress.json"
        self.t0 = time.time()
        self.stages = {k: 0.0 for k in self.WEIGHTS}
        self.stage = "starting"

    def update(self, stage, fraction):
        self.stages[stage] = max(0.0, min(1.0, fraction))
        self.stage = stage
        total = sum(self.WEIGHTS[k] * v for k, v in self.stages.items())
        el = time.time() - self.t0
        eta = el * (1 - total) / total if total > 0.02 else None
        write_json(self.path, {"stage": stage, "fraction": round(total, 4), "etaSeconds": None if eta is None else round(eta),
                               "stages": {k: round(v, 4) for k, v in self.stages.items()}, "elapsedSeconds": round(el)})


def _ball_worker(video, out, start, stop, device, student, progress_file):
    """Child-process entry: one segment of the ball tracker."""
    sys.path.insert(0, str(ROOT))
    import torch
    from pipeline import ball
    net = ball.load_model(torch.device(device), student)

    def prog(done, n):
        write_json(progress_file, {"done": done, "n": n})
    ball.run(video, out, device, net=net, start=start, stop=stop, progress=prog)


def run_ball(d: Path, prog: Progress, device, student=None):
    import cv2
    video = d / "video.mp4"
    cap = cv2.VideoCapture(str(video)); n, fps = int(cap.get(7)), cap.get(5); cap.release()
    seg = int(120 * fps)
    merged = None
    for a in range(0, n, seg):
        b = min(n, a + seg)
        part = d / f"ball_c{a}.json"
        if not part.exists():
            code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); from pipeline.run import _ball_worker; "
                    f"_ball_worker({str(video)!r}, {str(part)!r}, {a}, {b}, {device!r}, {student!r}, {str(d / 'ball_progress.json')!r})")
            with open(d / "run.log", "a") as log:
                subprocess.run([PY, "-c", code], check=True, stdout=log, stderr=subprocess.STDOUT)
        doc = json.loads(part.read_text())
        if merged is None:
            merged = doc; merged["runtimeSeconds"] = 0
        merged["frames"][a:b] = doc["frames"][a:b]
        merged["runtimeSeconds"] += doc["runtimeSeconds"]
        prog.update("ball", b / n)
    write_json(d / "ball.json", merged)
    for p in d.glob("ball_c*.json"):
        p.unlink()


def run_pose_subprocess(d: Path, stride: int, threads: int):
    code = (f"import sys, json; sys.path.insert(0, {str(ROOT)!r}); from pipeline import pose\n"
            f"def prog(i, n):\n"
            f"    open({str(d / 'pose_progress.json')!r}, 'w').write(json.dumps({{'done': i, 'n': n}}))\n"
            f"pose.run({str(d / 'video.mp4')!r}, {str(d / 'calibration.json')!r}, {str(d / 'pose.json')!r}, stride={stride}, progress=prog)")
    env = dict(os.environ, POSE_THREADS=str(threads), XDG_CACHE_HOME=str(ROOT / "work/cache"))
    log = open(d / "pose.log", "a")
    return subprocess.Popen([PY, "-c", code], stdout=log, stderr=subprocess.STDOUT, env=env)


def analyze(d: Path, device=None, student=None, pose_threads=4):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    prog = Progress(d)
    prog.update("ball", 0)
    device = device or ("mps" if sys.platform == "darwin" else "cuda")
    stride = 1 if meta.get("quality") == "best" else 3
    pose_proc = None
    if not (d / "pose.json").exists():
        pose_proc = run_pose_subprocess(d, stride, pose_threads)
    try:
        if not (d / "ball.json").exists():
            import threading

            def watch_pose():
                while pose_proc and pose_proc.poll() is None:
                    try:
                        p = json.loads((d / "pose_progress.json").read_text())
                        prog.update("pose", p["done"] / max(1, p["n"]))
                    except Exception:
                        pass
                    time.sleep(3)
            threading.Thread(target=watch_pose, daemon=True).start()
            run_ball(d, prog, device, student)
        if pose_proc:
            while pose_proc.poll() is None:
                try:
                    p = json.loads((d / "pose_progress.json").read_text())
                    prog.update("pose", p["done"] / max(1, p["n"]))
                except Exception:
                    pass
                time.sleep(3)
            if pose_proc.returncode != 0:
                raise RuntimeError(f"pose failed (see {d / 'pose.log'})")
        prog.update("pose", 1.0)
        prog.update("analysis", 0.1)
        sys.path.insert(0, str(ROOT))
        from pipeline import session
        names = {"A": meta.get("nameA", "Player A"), "B": meta.get("nameB", "Player B")}
        session.run(str(d / "video.mp4"), str(d / "ball.json"), str(d / "pose.json"), str(d / "calibration.json"),
                    str(ROOT / "models/event_model.pkl"), str(d / "report.json"), names,
                    seed_a_side=meta.get("seedASide", "near"), pose_stride=stride,
                    handed={"A": meta.get("handedA", "right"), "B": meta.get("handedB", "right")})
        prog.update("analysis", 1.0)
    finally:
        if pose_proc and pose_proc.poll() is None:
            pose_proc.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    ap.add_argument("--device")
    ap.add_argument("--student")
    ap.add_argument("--pose-threads", type=int, default=4)
    a = ap.parse_args()
    analyze(a.session_dir, a.device, a.student, a.pose_threads)
