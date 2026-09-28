# Engineering workflow

Decided 2026-09-28: **analysis runs on a server** (Mac = test server, the owner's RTX laptop = GPU
server later, cloud optional); the **client is a Flutter app** (Android first, iOS later from the same
code). Nothing earlier is discarded: the Python pipeline is the server engine, and the report JSON is
the contract between server and app.

## Repository layout (target)

```text
pipeline/        analysis engine (python package; moved from scripts/pipeline)
server/          FastAPI service: sessions, resumable upload, calibration, job queue, results
app/             Flutter client (record/import, court setup, upload, progress, report)
eval/            benchmark definitions + regression runner (labels live in data/, results in outputs/)
docs/            contracts (REPORT_SCHEMA.md, API.md), results, runbooks, this file
scripts/         one-off tools (dataset fetchers, label sheet renderers, audits, exports)
android/         legacy native Kotlin client (kept as reference / offline fallback; frozen)
data/ work/ outputs/ third_party/   large, NOT in git (provenance in THIRD_PARTY.md, data/public/SOURCES_V2.md)
```

## Loop for every change

1. **Issue → benchmark first.** Every problem gets a measurable check before it is fixed
   (labels in `data/…/labels_*.json`, runner in `eval/`).
2. **Change the engine** (`pipeline/`), never the evaluation data to fit a result.
3. **Run the regression suite** `eval/run.py` — identity, hits/bounces, strokes on all labelled
   videos + TrackNet leave-one-game-out. Numbers are written to `outputs/eval/<date>.json` and the
   table in `docs/RESULTS.md` is regenerated. A change that makes any metric worse needs a stated reason.
4. **Unit tests** `python -m pytest tests/` (engine), `server/tests`, `flutter test` (app).
5. **Local git commit** with the metric deltas in the message. (Nothing is pushed anywhere.)

## Held-out rule

Labels used to *choose* rules/thresholds (development) are separate from labels used to *report*
accuracy (held-out). Each labelled video is tagged `dev` or `heldout` in `eval/benchmarks.json`.
Reported numbers come from held-out items only.

## Where we were stuck, and the plan

| Stuck point | Plan |
|---|---|
| Phone too slow | Server processing (this change). Server can afford bigger/better models. |
| Unknown recall of hits/bounces on amateur video | Exhaustive hit+bounce labels on windows of 3 videos, labelled from full-resolution crops |
| Landing zones / speed not validated | Hand-labelled bounce contact points → metres error + zone confusion; speed: physics sanity checks now, radar later |
| Far player weak (serves missed) | Server-grade detector + pose (larger models, higher-resolution far tiles) |
| ~17% false hits (dead-ball contacts) | Rally segmentation: a shot must belong to an exchange that crosses the net |
| Only one video tested | 3 additional behind-baseline videos with identity labels (2 in progress) |
