# Handoff — read this first (state on 2026-10-13)

Written for the next working session (new laptop with an NVIDIA RTX 3050). It summarises ~2 weeks of work
with the project owner so a new Claude session can continue without the old chat history.

## What the product is
A fixed phone behind one baseline records a singles match; afterwards every player gets: shots, forehand /
backhand / serve / volley / overhead, where each shot landed (6 regions or out), average shot speed, points
won, serve stats. Player identity (A/B) is kept across changeovers.

## Decisions by the owner (do not re-open)
- **Architecture:** Flutter app (`app/`) records/uploads; a **server** (`server/`, FastAPI, port 8770) runs the
  Python engine (`pipeline/`). Phones are too slow for the models. Mac was the test server; **the RTX laptop
  is the next server**, cloud later. Nothing is discarded; always work in a structured way.
- **Licensing** of models/datasets is not an issue for now (provenance kept in `THIRD_PARTY.md`).
- **Generalise:** every fix must be a tennis/physics rule or learned model that holds for any video from a
  fixed camera behind the baseline — never thresholds tuned to one clip. Test on all labelled videos.
- **The phone is set up once and never moved during a session** (camera-movement handling is not a goal;
  the server only warns).
- Work autonomously; the owner verifies results by eye (review PDFs, annotated videos).
- iOS / Apple Developer account: later.

## How to work (the loop)
1. Change the engine in `pipeline/`.
2. `python scripts/eval_clips.py -v` (3 exhaustively labelled clips, ~1 min) and
   `python scripts/eval_windows.py -v` (4apx dev + MZZ held-out windows, ~5 min).
3. Full suite before committing: `python eval/run.py` (writes `docs/RESULTS.md`).
4. Keep a change only if nothing gets worse; commit with before→after numbers in the message.
5. Debug tools: `scripts/rally_debug.py SESSION 300-450` (decoder decisions), `scripts/label_timeline.py`
   (ball-track plots), `scripts/label_grid.py` (frame crops), `scripts/review_pdf.py` (frame-by-frame review PDF).

## Engine (rally-v3) in one paragraph
Court calibration from ≥4 user markers + line snapping + lens-distortion search (`court.py`) → ball tracker
TrackNetV3 every frame, fp16, 17 fps on the Mac (`ball.py`) ‖ YOLOX-tiny + RTMPose-s every 3rd frame with
far-court tiles (`pose.py`) → identity by court-metre tracklets + shirt/shorts colours (`identity.py`) →
static-ball suppression and hidden-bounce gap bridging (`rally.py`) → per-frame hit/bounce probabilities from
a gradient-boosting model on 70 trajectory features (`events.py`, `models/event_model.pkl`) → **rally decoder**:
Viterbi over candidates under tennis rules (alternating sides, bounce on court, net / out / double bounce,
point gaps, flight-time limits, tosses/dribbles/catches = dead ball) (`rally.py`) → shots, stroke (ball side
at contact, follow-through fallback, handedness auto-detected from serves), landing region, speed with
plausibility caps (`session.py`) → point winners and serve stats (`outcome.py`) → `report.json`.
Full explanation with figures: `docs/workflow_guide/guide.html` (PDF in `dist/`, regenerate when needed).

## Current accuracy (docs/RESULTS.md)
| Benchmark | Hits F1 | Bounces F1 | Hitter | Stroke |
|---|---|---|---|---|
| test1 (60 s, dev) | 0.943 | 0.958 | 25/25 | 16/17 |
| test2 (90 s, dev) | 0.875 | 0.979 | 20/21 | 5/5 |
| test3 (30 s, dev) | 1.000 | 1.000 | 10/10 | 3/3 |
| 4apx windows (dev) | 0.889 | 0.889 | 12/12 | 8/12 |
| MZZ windows (**held-out**) | 0.552 | 0.857 | 8/8 | 6/8 |
Identity 95–98 %, 0 swaps on three long matches.

## Data that is NOT in git (copy from the Mac if needed)
- Model weights + RacketVision code: `dist/LongTennis_server_assets.zip` (see `docs/SETUP_GPU_LAPTOP.md`).
- Test sessions used by `eval_clips.py`: `server/storage/sessions/{3ce88198,d1075620,641cfe35}` (test1/2/3)
  and `fcddd441` (test4 outdoor 60 fps), `eb6f4aad` (60-min match, old engine run).
- Videos: `dist/test_videos/` (incl. `match_60min_outdoor.mp4`, 1.2 GB), public datasets under `data/`
  (`data/public/youtube_amateur/*.mp4`, TrackNet dataset) and their ball/pose outputs under `outputs/v2/`.
- Results of the 60-min run: `results/match_60min/` (annotated video, analysis.html, shots.csv).

## Known weaknesses / next steps (owner's priority list, 2026-10-13)
1. **Ball tracker** finds the ball in only ~35 % of frames on the outdoor 60-min match; misses far-court
   serves above the frame (MZZ held-out hits 0.55). Next: fine-tune TrackNetV3 on phone footage (labelled
   clips + pseudo-labels), skip tracking between points.
2. **Speed on the RTX laptop**: measure the 60-min timing there (Mac: 3 h 31 min first run; ≈1 h 50 min
   expected after the 2× speed-up). Consider `onnxruntime-gpu` for pose.
3. Point endings are detected for only ~30 % of points (ball lost before the last bounce) → limits points
   won / serve stats.
4. Strokes for the small far player; a learned stroke classifier on pose sequences (THETIS + our labels).
5. More labelled test videos (ideally recorded with the app) to keep generalisation honest.

## Owner contact preferences
Plain language, short status updates while long jobs run, honest numbers (say when something is worse),
links/paths to files they can open on the phone or computer.
