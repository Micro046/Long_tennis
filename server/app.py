"""Long Tennis analysis server (API v1, see docs/API.md).

Run:  .venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8770
Config: server/config.json {"token": "...", "storage": "server/storage", "device": "mps|cuda|cpu", "poseThreads": 4}
One background worker thread processes queued sessions one at a time (each analysis runs in a
child process via pipeline/run.py so a crash never takes the API down).
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import court  # noqa: E402

VERSION = "1.0.0"
CONFIG = {"token": "dev-token", "storage": str(ROOT / "server/storage"), "device": None, "poseThreads": 4}
_cfg = ROOT / "server/config.json"
if _cfg.exists():
    CONFIG.update(json.loads(_cfg.read_text()))
CONFIG["token"] = os.environ.get("LT_TOKEN", CONFIG["token"])
STORE = Path(os.environ.get("LT_STORAGE", CONFIG["storage"]))
(STORE / "sessions").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Long Tennis", version=VERSION)
jobs: "queue.Queue[str]" = queue.Queue()
lock = threading.Lock()


# ---------------------------------------------------------------------------- helpers
def auth(authorization: str | None = Header(default=None)):
    if authorization != f"Bearer {CONFIG['token']}":
        raise HTTPException(401, "bad or missing token")


def sdir(sid: str) -> Path:
    d = STORE / "sessions" / sid
    if not sid.isalnum() or not d.exists():
        raise HTTPException(404, "no such session")
    return d


def load(sid: str) -> dict:
    d = sdir(sid)
    meta = json.loads((d / "meta.json").read_text())
    p = d / "progress.json"
    if meta["status"] in ("queued", "processing") and p.exists():
        try:
            meta["progress"] = json.loads(p.read_text())
        except Exception:
            pass
    return meta


def save(meta: dict):
    d = STORE / "sessions" / meta["id"]
    with lock:
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        tmp.replace(d / "meta.json")


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    w, h, n, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(7)), cap.get(5)
    cap.release()
    if n <= 0 or fps <= 0:
        raise HTTPException(422, "uploaded file is not a readable video")
    return {"widthPx": w, "heightPx": h, "fps": fps, "frameCount": n, "durationS": round(n / fps, 2)}


def background(d: Path) -> np.ndarray:
    """Median of 31 frames spread over the video: players/ball vanish, court lines stay (cached)."""
    bg = d / "background.jpg"
    if bg.exists():
        return cv2.imread(str(bg))
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    n = int(cap.get(7))
    frames = []
    for i in np.linspace(0, max(0, n - 1), 31).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    img = np.median(np.stack(frames), axis=0).astype(np.uint8)
    cv2.imwrite(str(bg), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return img


EDITABLE = ("nameA", "nameB", "seedASide", "handedA", "handedB", "quality")


# ---------------------------------------------------------------------------- endpoints
@app.get("/api/v1/health")
def health():
    return {"ok": True, "version": VERSION, "device": CONFIG.get("device") or ("mps" if sys.platform == "darwin" else "cuda"),
            "queue": jobs.qsize()}


@app.post("/api/v1/sessions", dependencies=[Depends(auth)])
def create(body: dict):
    sid = uuid.uuid4().hex[:8]
    d = STORE / "sessions" / sid
    d.mkdir(parents=True)
    meta = {"id": sid, "createdAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nameA": "Player A", "nameB": "Player B", "seedASide": "near", "handedA": "right", "handedB": "right",
            "quality": "standard", "status": "created", "progress": None, "video": None, "error": None}
    meta.update({k: v for k, v in body.items() if k in EDITABLE})
    save(meta)
    return meta


@app.get("/api/v1/sessions", dependencies=[Depends(auth)])
def list_sessions():
    out = []
    for d in (STORE / "sessions").iterdir():
        if (d / "meta.json").exists():
            out.append(load(d.name))
    return sorted(out, key=lambda m: m["createdAt"], reverse=True)


@app.get("/api/v1/sessions/{sid}", dependencies=[Depends(auth)])
def get_session(sid: str):
    return load(sid)


@app.patch("/api/v1/sessions/{sid}", dependencies=[Depends(auth)])
def patch_session(sid: str, body: dict):
    meta = load(sid)
    meta.update({k: v for k, v in body.items() if k in EDITABLE})
    save(meta)
    return meta


@app.delete("/api/v1/sessions/{sid}", dependencies=[Depends(auth)])
def delete_session(sid: str):
    shutil.rmtree(sdir(sid))
    return {"deleted": True}


@app.get("/api/v1/sessions/{sid}/upload", dependencies=[Depends(auth)])
def upload_status(sid: str):
    d = sdir(sid)
    part = d / "video.part"
    meta = load(sid)
    return {"received": part.stat().st_size if part.exists() else (0 if meta["video"] is None else None),
            "complete": meta["video"] is not None}


@app.put("/api/v1/sessions/{sid}/upload", dependencies=[Depends(auth)])
async def upload_chunk(sid: str, offset: int, request: Request):
    d = sdir(sid)
    part = d / "video.part"
    have = part.stat().st_size if part.exists() else 0
    if offset != have:
        return JSONResponse({"received": have}, status_code=409)
    with open(part, "ab") as f:
        async for chunk in request.stream():
            f.write(chunk)
    meta = load(sid)
    if meta["status"] == "created":
        meta["status"] = "uploading"; save(meta)
    return {"received": part.stat().st_size}


@app.post("/api/v1/sessions/{sid}/upload/complete", dependencies=[Depends(auth)])
def upload_complete(sid: str, body: dict):
    d = sdir(sid)
    part = d / "video.part"
    if not part.exists() or part.stat().st_size != int(body.get("size", -1)):
        raise HTTPException(409, f"size mismatch: have {part.stat().st_size if part.exists() else 0}")
    part.replace(d / "video.mp4")
    meta = load(sid)
    meta["video"] = video_info(d / "video.mp4")
    meta["video"]["filename"] = body.get("filename")
    meta["status"] = "uploaded"
    save(meta)
    return meta


@app.get("/api/v1/sessions/{sid}/frame", dependencies=[Depends(auth)])
def frame(sid: str, t: float | None = None, maxWidth: int = 1920):
    """t omitted -> the median background image used for calibration (no players)."""
    d = sdir(sid)
    if t is None:
        img = background(d)
    else:
        cap = cv2.VideoCapture(str(d / "video.mp4"))
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(404, "no frame at that time")
    if img.shape[1] > maxWidth:
        img = cv2.resize(img, (maxWidth, int(img.shape[0] * maxWidth / img.shape[1])))
    return Response(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes(), media_type="image/jpeg")


@app.post("/api/v1/sessions/{sid}/calibration", dependencies=[Depends(auth)])
def calibrate(sid: str, body: dict):
    d = sdir(sid)
    meta = load(sid)
    if not meta.get("video"):
        raise HTTPException(409, "upload the video first")
    markers = body.get("markers", {})
    unknown = [k for k in markers if k not in court.MODEL_KP]
    if len(markers) < 4 or unknown:
        raise HTTPException(422, f"need >= 4 known court keypoints; unknown: {unknown}")
    img = background(d)
    h, w = img.shape[:2]
    try:
        r = court.calibrate(img, {k: (v[0] * w, v[1] * h) for k, v in markers.items()})
    except Exception as e:  # singular fit etc.
        raise HTTPException(422, f"calibration failed: {e}")
    ok = len(r["keypointsUsed"]) >= 4 and (r["reprojectionRmsPx"] or 99) < 12 and r.get("lineSupport", 0) > 0.6
    lines = []
    for a, b in court.LINES.values():
        p, q = court.project(r["imageFromCourt"], [a, b])
        lines.append([round(p[0] / w, 5), round(p[1] / h, 5), round(q[0] / w, 5), round(q[1] / h, 5)])
    (d / "calibration.json").write_text(json.dumps({**r, "markers": markers, "source": "median background"}))
    if meta["status"] in ("uploaded", "calibrated", "failed", "done"):
        meta["status"] = "calibrated"; save(meta)
    return {"ok": ok, "rmsPx": r["reprojectionRmsPx"], "lineSupport": r.get("lineSupport"),
            "keypointsUsed": r["keypointsUsed"], "lines": lines}


@app.post("/api/v1/sessions/{sid}/analyze", dependencies=[Depends(auth)])
def analyze(sid: str):
    d = sdir(sid)
    meta = load(sid)
    if not (d / "calibration.json").exists():
        raise HTTPException(409, "calibrate the court first")
    if meta["status"] in ("queued", "processing"):
        return meta
    for f in ("report.json", "progress.json"):
        (d / f).unlink(missing_ok=True)
    meta.update(status="queued", error=None, progress=None)
    save(meta)
    jobs.put(sid)
    return meta


def compact_report(r: dict) -> dict:
    """What the app needs: players, shots, events (drop per-frame arrays)."""
    return {k: v for k, v in r.items() if k not in ("perFrameIdentity", "ballPath")}


@app.get("/api/v1/sessions/{sid}/report", dependencies=[Depends(auth)])
def report(sid: str):
    p = sdir(sid) / "report.json"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    return compact_report(json.loads(p.read_text()))


@app.get("/api/v1/sessions/{sid}/video", dependencies=[Depends(auth)])
def video(sid: str):
    return FileResponse(sdir(sid) / "video.mp4", media_type="video/mp4")


# ---------------------------------------------------------------------------- worker
def worker():
    while True:
        sid = jobs.get()
        try:
            meta = load(sid)
        except HTTPException:
            continue
        meta["status"] = "processing"; save(meta)
        d = STORE / "sessions" / sid
        cmd = [sys.executable, str(ROOT / "pipeline/run.py"), str(d), "--pose-threads", str(CONFIG.get("poseThreads", 4))]
        if CONFIG.get("device"):
            cmd += ["--device", CONFIG["device"]]
        if CONFIG.get("ballModel"):
            cmd += ["--student", CONFIG["ballModel"]]
        with open(d / "run.log", "a") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
        meta = load(sid)
        if code == 0 and (d / "report.json").exists():
            meta.update(status="done", progress={"stage": "done", "fraction": 1.0, "etaSeconds": 0})
        else:
            tail = (d / "run.log").read_text()[-600:] if (d / "run.log").exists() else ""
            meta.update(status="failed", error=f"analysis failed (exit {code}): {tail.strip().splitlines()[-1] if tail.strip() else ''}")
        save(meta)


@app.on_event("startup")
def start_worker():
    # re-queue work interrupted by a restart
    for d in (STORE / "sessions").iterdir():
        try:
            m = json.loads((d / "meta.json").read_text())
        except Exception:
            continue
        if m.get("status") in ("queued", "processing"):
            m["status"] = "queued"; save(m); jobs.put(m["id"])
    threading.Thread(target=worker, daemon=True).start()
