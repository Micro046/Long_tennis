#!/usr/bin/env python3
"""Command-line client for the Long Tennis server (same calls the Flutter app makes).

  python -m server.client analyze VIDEO --markers "far_left_singles:0.367,0.294;..." [--names A=Red,B=Grey]
  python -m server.client status ID | report ID
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

URL = os.environ.get("LT_URL", "http://127.0.0.1:8770/api/v1")
H = {"Authorization": f"Bearer {os.environ.get('LT_TOKEN', 'dev-token')}"}
CHUNK = 8 * 1024 * 1024


def upload(c, sid, path):
    size = Path(path).stat().st_size
    got = c.get(f"{URL}/sessions/{sid}/upload", headers=H).json()["received"] or 0
    with open(path, "rb") as f:
        f.seek(got)
        while got < size:
            data = f.read(CHUNK)
            r = c.put(f"{URL}/sessions/{sid}/upload", params={"offset": got}, content=data,
                      headers={**H, "Content-Type": "application/octet-stream"}, timeout=300)
            got = r.json()["received"]
            print(f"\rupload {got / size:6.1%}", end="", flush=True)
            if r.status_code == 409:
                f.seek(got)
    print()
    r = c.post(f"{URL}/sessions/{sid}/upload/complete", json={"size": size, "filename": Path(path).name}, headers=H, timeout=300)
    r.raise_for_status()
    return r.json()


def analyze(a):
    with httpx.Client(timeout=120) as c:
        body = {"quality": a.quality}
        for kv in (a.names or "").split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                body[f"name{k}"] = v
        s = c.post(f"{URL}/sessions", json=body, headers=H).json()
        sid = s["id"]
        print("session", sid)
        meta = upload(c, sid, a.video)
        print("video", meta["video"])
        markers = {kv.split(":")[0]: [float(x) for x in kv.split(":")[1].split(",")] for kv in a.markers.split(";") if kv}
        cal = c.post(f"{URL}/sessions/{sid}/calibration", json={"markers": markers}, headers=H).json()
        print("calibration", {k: cal.get(k) for k in ("ok", "rmsPx", "lineSupport")})
        if not cal.get("ok") and not a.force:
            sys.exit("calibration not ok; adjust markers or pass --force")
        c.post(f"{URL}/sessions/{sid}/analyze", headers=H).raise_for_status()
        wait(c, sid)


def wait(c, sid):
    while True:
        s = c.get(f"{URL}/sessions/{sid}", headers=H).json()
        p = s.get("progress") or {}
        print(f"\r{s['status']:10s} {p.get('stage', ''):9s} {p.get('fraction', 0):6.1%} eta {p.get('etaSeconds')}s   ", end="", flush=True)
        if s["status"] in ("done", "failed"):
            print()
            if s["status"] == "failed":
                sys.exit(s["error"])
            r = c.get(f"{URL}/sessions/{sid}/report", headers=H).json()
            for k, p in r["players"].items():
                print(k, p["name"], "shots", p["shots"], p["strokes"])
            return
        time.sleep(5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze"); p.add_argument("video"); p.add_argument("--markers", required=True)
    p.add_argument("--names"); p.add_argument("--quality", default="standard"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("status"); p.add_argument("id")
    a = ap.parse_args()
    if a.cmd == "analyze":
        analyze(a)
    else:
        with httpx.Client() as c:
            wait(c, a.id)
