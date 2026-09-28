# Reproduce the first ball-tracking spike

This runbook evaluates an existing open-source component on held-out public videos. It is a desktop experiment, not the Flutter application.

## Inputs already prepared

- RacketVision is pinned in `third_party/RacketVision` at commit `c44af2a08524d3cb54d818f19686f4cdea4d2793`.
- Three videos and sparse ball annotations are in `data/public/racketvision`.
- The released checkpoint is at `work/models/racketvision/balltrack_best.pth`.
- Checkpoint SHA-256: `00d707b9db7a49561c411e4765956e79bcd7c7e20c7a0a535073440b3e972342`.

The upstream repository is MIT licensed and the dataset card declares MIT. The model repository has no model card or explicit license, so the checkpoint is suitable for evaluation but is **not yet cleared for redistribution in a commercial app**.

## Commands

```bash
cd /Users/inspiredsquare/Downloads/long_tennis
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-balltrack.txt

python scripts/inspect_public_dataset.py
python scripts/run_racketvision_ball.py --device auto --batch-size 16
python scripts/export_normalized_ball_events.py

deactivate
python3.12 -m venv .venv-pose
source .venv-pose/bin/activate
python -m pip install -r requirements-pose.txt
python scripts/run_rtmpose_players.py

# Build a multi-frame court consensus, then regenerate pose selection using it.
python scripts/run_court_calibration.py --samples 24
python scripts/run_rtmpose_players.py
python scripts/refresh_court_player_outputs.py
```

Use `--device cpu` where Apple Metal is unavailable. It is much slower on this machine.

## Outputs

- `outputs/dataset_inspection/dataset_report.json`: input integrity and video metadata.
- `outputs/dataset_inspection/*_contact_sheet.jpg`: ground-truth sanity checks.
- `outputs/racketvision_ball/report.json`: per-clip runtime, presence, and localization metrics.
- `outputs/racketvision_ball/*/predictions.json`: raw per-frame predictions.
- `outputs/racketvision_ball/*/overlay.mp4`: prediction/ground-truth visualizations.
- `outputs/racketvision_ball/contact_sheets`: quick visual samples from the overlays.
- `outputs/normalized/*_ball_observations.json`: project-owned, timestamped interchange data described in `SCHEMA.md`.
- `outputs/rtmpose_players/report.json`: pose throughput and player-selection proxy metrics.
- `outputs/rtmpose_players/*/pose_observations.json`: timestamped COCO-17 player poses.
- `outputs/rtmpose_players/*/overlay.mp4`: selected-player boxes and skeletons, with all other detected people shown in gray.
- `outputs/court_calibration/report.json`: multi-frame detection, stability, and line-support proxy metrics.
- `outputs/court_calibration/*/court_calibration.json`: court corners and both homography matrices.
- `outputs/court_calibration/*/player_court_positions.json`: pose-derived player ground points in court metres.
- `outputs/court_calibration/*/overlay.mp4`: court model and projected player coordinates over the source video.

Do not compare the reported desktop throughput directly with Flutter/mobile performance. A mobile candidate will need conversion to ONNX or a native mobile runtime and a fresh benchmark on real Android and iOS hardware.

## Milestone 2: benchmark ingestion and static mobile audit

Install the separate benchmark dependencies in `.venv` and run the audit:

```bash
.venv/bin/python -m pip install -r requirements-benchmark.txt
.venv/bin/python scripts/audit_benchmark.py
.venv/bin/python scripts/check_mobile_viability.py
.venv/bin/python -m unittest discover -s tests -p test_benchmark.py -v
.venv/bin/python -m unittest discover -s tests -p test_milestone1.py -v
```

The Kaggle back-view dataset is diagnostic-only; its underlying footage rights and ball-coordinate meaning are not verified. `outputs/milestone2/backview_audit.json` holds the per-clip facts and `outputs/milestone2/timestamps/` holds exact decoded-frame timestamps. `outputs/milestone2/mobile_viability.json` is a static format and toolchain audit, not Android throughput.

For a new consented phone recording, follow `BENCHMARK_PROTOCOL.md`, then generate exact timestamps and validate independent labels:

```bash
.venv/bin/python scripts/probe_phone_video.py --video /path/session.mp4 --session-id session_001 --output /path/session_timestamps.json
.venv/bin/python scripts/validate_phone_labels.py --video /path/session.mp4 --labels /path/session_labels.json --output /path/session_validation.json
```

## Pipeline v2 (end-to-end) and Android app

```bash
# ball (chunked to avoid the PyTorch MPS memory growth), pose, then the session report
.venv/bin/python scripts/run_ball_chunked.py VIDEO.mp4 --out-dir outputs/v2/ball/youtube
POSE_THREADS=4 .venv-pose/bin/python scripts/pipeline/pose.py VIDEO.mp4 --calibration CAL.json --out POSE.json
.venv/bin/python scripts/pipeline/session.py VIDEO.mp4 --ball BALL.json --pose POSE.json --calibration CAL.json \
    --model outputs/v2/events/event_model_pred.pkl --out REPORT.json --names "A=Name,B=Name"

# evaluation
.venv/bin/python scripts/eval_tracknet_events.py --source pred --save-model
.venv/bin/python scripts/eval_identity_only.py VIDEO.mp4 POSE.json data/public/youtube_amateur/labels_identity_4apx6gd5Uxs.json
.venv/bin/python scripts/render_overlay.py VIDEO.mp4 REPORT.json POSE.json --audit 32 --seed 101 --out-dir work/audit2

# model export for the app + parity fixtures
.venv/bin/python scripts/export_event_model.py --fixture-ball outputs/v2/ball/tracknet/game1_Clip1_ball.json
.venv/bin/python scripts/export_session_fixture.py data/test_clips/rally30s.mp4 --ball ... --pose ... --calibration ... --model ...

# Android
cd android && export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
./gradlew :app:testDebugUnitTest        # Kotlin == Python parity tests
./gradlew :app:assembleRelease :app:bundleRelease
```

Calibration for a new video: open the setup screen in the app, or call `pipeline.court.calibrate(img, {4 rough keypoints})`.
