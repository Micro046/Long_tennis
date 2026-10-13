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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

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
    (d / "camera_motion.json").write_text(json.dumps(camera_motion(frames)))
    return img


def camera_motion(frames, limit_px=8.0):
    """How far the camera moved during the recording: image shift of each sampled frame against the first
    (phase correlation on edge images of the lower 2/3, where the court is; players are small against it).
    The court is calibrated once, so a moved camera shifts every court position after the move."""
    def edges(f):
        g = cv2.cvtColor(cv2.resize(f, (640, 360)), cv2.COLOR_BGR2GRAY)[120:]
        e = cv2.Canny(g, 60, 160).astype(np.float32)
        return cv2.GaussianBlur(e, (5, 5), 0) * np.outer(np.hanning(e.shape[0]), np.hanning(e.shape[1])).astype(np.float32)
    if len(frames) < 2:
        return {"maxShiftPx": 0.0, "moved": False}
    ref = edges(frames[0]); sc = frames[0].shape[1] / 640.0
    shifts = []
    for f in frames[1:]:
        (dx, dy), _ = cv2.phaseCorrelate(ref, edges(f))
        shifts.append(float(np.hypot(dx, dy) * sc))
    # a single odd sample is a player in front of the lens or an edit cut; a moved camera stays moved:
    # require the shift in at least 3 samples (and 10% of them), and report the typical moved shift
    over = [v for v in shifts if v > limit_px]
    moved = len(over) >= max(3, int(0.1 * len(shifts)))
    mx = float(np.median(over)) if moved else float(np.median(shifts))
    first = next((i + 1 for i, v in enumerate(shifts) if v > limit_px), None) if moved else None
    return {"maxShiftPx": round(mx, 1), "moved": bool(moved),
            "firstMovedAtFraction": None if first is None else round(first / (len(frames) - 1), 2),
            "shiftsPx": [round(v, 1) for v in shifts]}


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
    # court lines for the app overlay; split into short segments so lens-bent lines are drawn as curves
    lines = []
    M = court.image_from_court(r)
    for a, b in court.LINES.values():
        pts = M.apply([np.asarray(a, float) + t * (np.asarray(b, float) - np.asarray(a, float)) for t in np.linspace(0, 1, 9)])
        for p, q in zip(pts, pts[1:]):
            lines.append([round(p[0] / w, 5), round(p[1] / h, 5), round(q[0] / w, 5), round(q[1] / h, 5)])
    (d / "calibration.json").write_text(json.dumps({**r, "markers": markers, "source": "median background"}))
    if meta["status"] in ("uploaded", "calibrated", "failed", "done"):
        meta["status"] = "calibrated"; save(meta)
    cm = json.loads((d / "camera_motion.json").read_text()) if (d / "camera_motion.json").exists() else None
    if cm is not None:
        cal_doc = json.loads((d / "calibration.json").read_text()); cal_doc["cameraMotion"] = cm
        (d / "calibration.json").write_text(json.dumps(cal_doc))
    return {"ok": ok, "rmsPx": r["reprojectionRmsPx"], "lineSupport": r.get("lineSupport"),
            "cameraMoved": bool(cm and cm["moved"]), "cameraMaxShiftPx": cm and cm["maxShiftPx"],
            "warning": ("The camera moved during the recording (up to %.0f px): court positions and in/out calls "
                        "after the move will be less accurate. Keep the phone fixed." % cm["maxShiftPx"]) if cm and cm["moved"] else None,
            "keypointsUsed": r["keypointsUsed"], "lines": lines, "lensDistortion": r.get("distortion"),
            "markerFallback": r.get("markerFallback")}


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


STROKES = ("serve", "forehand", "backhand", "volley", "overhead", "unknown")


def apply_corrections(rep: dict, corr: dict) -> dict:
    """Shots reviewed in the app: delete / change player or stroke / add missed shots. The summary is
    recomputed so every number the player sees reflects the corrections."""
    from pipeline import session as S
    by_frame = {c["hitFrame"]: c for c in corr.get("shots", [])}
    shots = []
    for s in rep["shots"]:
        c = by_frame.get(s["hitFrame"])
        if c and c.get("action") == "delete":
            continue
        s = dict(s)
        if c:
            if c.get("playerId") in ("A", "B"):
                s["playerId"] = c["playerId"]
            if c.get("stroke") in STROKES:
                s["stroke"] = c["stroke"]
            s["reviewed"] = True
        shots.append(s)
    fps = rep.get("fps", 30.0)
    for a in corr.get("added", []):
        shots.append({"hitFrame": int(round(a["timeS"] * fps)), "timeS": a["timeS"], "playerId": a.get("playerId", "unknown"),
                      "side": None, "stroke": a.get("stroke", "unknown"), "landing": {"status": "not_observed"},
                      "reviewed": True, "addedByUser": True})
    shots.sort(key=lambda s: s["hitFrame"])
    names = {k: v["name"] for k, v in rep["players"].items()}
    return {**rep, "shots": shots, "players": S.summarize(shots, names),
            "review": {"corrections": len(by_frame), "added": len(corr.get("added", []))}}


def load_report(d: Path) -> dict:
    p = d / "report.json"
    if not p.exists():
        raise HTTPException(404, "report not ready")
    rep = json.loads(p.read_text())
    c = d / "corrections.json"
    return apply_corrections(rep, json.loads(c.read_text())) if c.exists() else rep


@app.get("/api/v1/sessions/{sid}/report", dependencies=[Depends(auth)])
def report(sid: str):
    return compact_report(load_report(sdir(sid)))


@app.get("/api/v1/sessions/{sid}/corrections", dependencies=[Depends(auth)])
def get_corrections(sid: str):
    c = sdir(sid) / "corrections.json"
    return json.loads(c.read_text()) if c.exists() else {"shots": [], "added": []}


@app.put("/api/v1/sessions/{sid}/corrections", dependencies=[Depends(auth)])
def put_corrections(sid: str, body: dict):
    """body: {shots: [{hitFrame, action: confirm|delete|edit, playerId?, stroke?}], added: [{timeS, playerId, stroke}]}
    Replaces the stored corrections. Also written to storage/labels/ as review-mode training labels."""
    d = sdir(sid)
    if not (d / "report.json").exists():
        raise HTTPException(409, "no report yet")
    corr = {"shots": [c for c in body.get("shots", []) if isinstance(c.get("hitFrame"), int)],
            "added": [a for a in body.get("added", []) if isinstance(a.get("timeS"), (int, float))],
            "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "corrections.json").write_text(json.dumps(corr, indent=1))
    lab = STORE / "labels"; lab.mkdir(exist_ok=True)
    (lab / f"{sid}_review.json").write_text(json.dumps({"session": sid, "mode": "user_review", **corr}, indent=1))
    return compact_report(load_report(d))


@app.get("/api/v1/sessions/{sid}/video", dependencies=[Depends(auth)])
def video(sid: str):
    return FileResponse(sdir(sid) / "video.mp4", media_type="video/mp4")


# ---------------------------------------------------------------------------- downloads (LAN demo)
DIST = ROOT / "dist"


@app.get("/dl", response_class=HTMLResponse)
def downloads():
    """Phone-friendly page with the app and test videos (files in dist/)."""
    def item(p):
        mb = p.stat().st_size / 1e6
        return f'<li><a href="/dl/{p.relative_to(DIST).as_posix()}">{p.name}</a> ({mb:.0f} MB)</li>'
    apks = sorted(DIST.glob("*.apk"))
    vids = sorted((DIST / "test_videos").glob("*.mp4"))
    return f"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Long Tennis downloads</title><body style="font-family:sans-serif;max-width:640px;margin:auto;padding:16px">
<h2>Long Tennis prototype</h2>
<h3>1. App</h3><ul>{''.join(item(p) for p in apks)}</ul>
<p>Open the downloaded file and allow installing apps from this browser.</p>
<h3>2. Test videos</h3><ul>{''.join(item(p) for p in vids)}</ul>
<p>Download one, then in the app: New session &rarr; Choose video.</p>
<h3>3. Review reports</h3><ul>{''.join(item(p) for p in sorted(DIST.glob("*.pdf")))}</ul></body>"""


@app.get("/dl/{path:path}")
def download_file(path: str):
    p = (DIST / path).resolve()
    if not str(p).startswith(str(DIST.resolve())) or not p.is_file():
        raise HTTPException(404, "not found")
    mt = "application/vnd.android.package-archive" if p.suffix == ".apk" else None
    return FileResponse(p, media_type=mt, filename=p.name)


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
