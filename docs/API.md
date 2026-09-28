# Long Tennis server API v1

Base URL: `http://<server>:8770/api/v1`. Every request carries `Authorization: Bearer <token>`
(token from `server/config.json`, default `dev-token` on the test Mac). JSON unless noted.

## Session lifecycle

```text
create -> upload video (resumable) -> frame -> calibrate (user markers) -> analyze -> poll -> report
```

| Method | Path | Body / query | Response |
|---|---|---|---|
| GET | `/health` | – | `{ok, version, device, queue}` (no auth) |
| POST | `/sessions` | `{nameA, nameB, seedASide: near\|far, handedA, handedB, quality: standard\|best}` | `Session` |
| GET | `/sessions` | – | `[Session]` newest first |
| GET | `/sessions/{id}` | – | `Session` (includes `status`, `progress`) |
| PATCH | `/sessions/{id}` | any of the create fields | `Session` |
| DELETE | `/sessions/{id}` | – | `{deleted: true}` (video + results removed) |
| GET | `/sessions/{id}/upload` | – | `{received: bytes, complete}` — where to resume |
| PUT | `/sessions/{id}/upload?offset=N` | raw bytes (`application/octet-stream`), a chunk starting at byte N | `{received}`; 409 with `{received}` if N ≠ received |
| POST | `/sessions/{id}/upload/complete` | `{size, filename}` | `Session` (`status: uploaded`, video metadata) |
| GET | `/sessions/{id}/frame?t=1.0` | – | JPEG of the video at t seconds (for court setup) |
| POST | `/sessions/{id}/calibration` | `{markers: {name: [x, y]}}`, x,y normalised 0..1 | `{ok, rmsPx, lineSupport, keypointsUsed, lines: [[x1,y1,x2,y2]...] normalised}` |
| POST | `/sessions/{id}/analyze` | – | `Session` (`status: queued`) |
| GET | `/sessions/{id}/report` | – | report JSON (see `docs/REPORT_SCHEMA.md`) |
| GET | `/sessions/{id}/video` | HTTP Range supported | the uploaded video (for playback) |

Marker names: `far_left_singles, far_right_singles, near_service_left, near_service_right` (default set)
or any 4+ keypoints of the court model (`near_left_singles`, `near_right_singles`, …).

## Session object

```json
{"id": "a1b2c3d4", "createdAt": "2026-09-28T12:00:00Z", "nameA": "Red", "nameB": "Grey",
 "seedASide": "near", "handedA": "right", "handedB": "right", "quality": "standard",
 "status": "created|uploading|uploaded|calibrated|queued|processing|done|failed",
 "progress": {"stage": "ball|pose|analysis", "fraction": 0.42, "etaSeconds": 610},
 "video": {"widthPx": 1920, "heightPx": 1080, "fps": 29.97, "frameCount": 38808, "durationS": 1294.9},
 "error": null}
```
