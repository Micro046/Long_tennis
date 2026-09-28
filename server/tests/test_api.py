"""API tests: session lifecycle, resumable upload, calibration (no analysis run)."""
import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "data/test_clips/rally30s.mp4"
TMP = tempfile.mkdtemp(prefix="lt-test-")
os.environ["LT_STORAGE"] = TMP
os.environ["LT_TOKEN"] = "t"

from fastapi.testclient import TestClient  # noqa: E402
from server.app import app  # noqa: E402

H = {"Authorization": "Bearer t"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c
    shutil.rmtree(TMP, ignore_errors=True)


def test_auth_required(client):
    assert client.get("/api/v1/sessions").status_code == 401
    assert client.get("/api/v1/health").json()["ok"]


def test_full_setup_flow(client):
    s = client.post("/api/v1/sessions", json={"nameA": "Red", "nameB": "Grey", "bogus": 1}, headers=H).json()
    sid = s["id"]
    assert s["nameA"] == "Red" and "bogus" not in s and s["status"] == "created"

    data = CLIP.read_bytes()
    half = len(data) // 2
    r = client.put(f"/api/v1/sessions/{sid}/upload?offset=0", content=data[:half], headers=H)
    assert r.json()["received"] == half
    # wrong offset -> 409 tells the client where to resume
    r = client.put(f"/api/v1/sessions/{sid}/upload?offset=0", content=data[half:], headers=H)
    assert r.status_code == 409 and r.json()["received"] == half
    assert client.get(f"/api/v1/sessions/{sid}/upload", headers=H).json()["received"] == half
    client.put(f"/api/v1/sessions/{sid}/upload?offset={half}", content=data[half:], headers=H)
    r = client.post(f"/api/v1/sessions/{sid}/upload/complete", json={"size": len(data), "filename": "rally.mp4"}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["video"]["frameCount"] == 900 and r.json()["status"] == "uploaded"

    img = client.get(f"/api/v1/sessions/{sid}/frame?t=1.0", headers=H)
    assert img.headers["content-type"] == "image/jpeg" and len(img.content) > 10000
    bg = client.get(f"/api/v1/sessions/{sid}/frame", headers=H)
    assert bg.status_code == 200 and len(bg.content) > 10000

    markers = {"far_left_singles": [705 / 1920, 318 / 1080], "far_right_singles": [1195 / 1920, 318 / 1080],
               "near_service_left": [370 / 1920, 524 / 1080], "near_service_right": [1530 / 1920, 524 / 1080]}
    r = client.post(f"/api/v1/sessions/{sid}/calibration", json={"markers": markers}, headers=H).json()
    assert r["ok"] and r["rmsPx"] < 3 and len(r["lines"]) == 9
    assert client.get(f"/api/v1/sessions/{sid}", headers=H).json()["status"] == "calibrated"
    assert client.get(f"/api/v1/sessions/{sid}/report", headers=H).status_code == 404

    assert client.delete(f"/api/v1/sessions/{sid}", headers=H).json()["deleted"]
    assert client.get(f"/api/v1/sessions/{sid}", headers=H).status_code == 404


def test_calibration_rejects_bad_markers(client):
    sid = client.post("/api/v1/sessions", json={}, headers=H).json()["id"]
    r = client.post(f"/api/v1/sessions/{sid}/calibration", json={"markers": {"x": [0, 0]}}, headers=H)
    assert r.status_code == 409  # no video yet


def test_corrections_recompute_summary(client, tmp_path=None):
    from server.app import apply_corrections
    rep = {"fps": 30.0, "players": {"A": {"name": "Red"}, "B": {"name": "Grey"}},
           "shots": [{"hitFrame": 10, "timeS": 0.33, "playerId": "A", "stroke": "forehand", "landing": {"status": "in", "region": "deep_left"}},
                     {"hitFrame": 50, "timeS": 1.67, "playerId": "B", "stroke": "backhand", "landing": {"status": "not_observed"}},
                     {"hitFrame": 90, "timeS": 3.0, "playerId": "A", "stroke": "serve", "landing": {"status": "not_observed"}}]}
    out = apply_corrections(rep, {"shots": [{"hitFrame": 50, "action": "edit", "stroke": "forehand"},
                                            {"hitFrame": 90, "action": "delete"}],
                                  "added": [{"timeS": 4.0, "playerId": "B", "stroke": "volley"}]})
    assert [s["hitFrame"] for s in out["shots"]] == [10, 50, 120]
    assert out["players"]["A"]["shots"] == 1 and out["players"]["B"]["strokes"]["forehand"] == 1
    assert out["players"]["B"]["strokes"]["volley"] == 1
