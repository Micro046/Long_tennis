# Runbook: server + Flutter app

## 1. Start the analysis server (Mac test server)

```bash
cd ~/Downloads/long_tennis
.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8770
```

- Health check: `curl http://127.0.0.1:8770/api/v1/health`
- Token: `dev-token` by default. Set a private one in `server/config.json` (`{"token": "..."}`) before
  exposing the server beyond your home network.
- Storage: `server/storage/sessions/<id>/` (video, calibration, ball/pose tracks, report, logs).
- Port 8765 is already used by another program on this Mac, so the server uses 8770.
- Find the Mac's LAN address for the phone: `ipconfig getifaddr en0`.

## 2. Install the app on an Android phone

```bash
cd ~/Downloads/long_tennis/app
export PATH="$HOME/development/flutter/bin:$PATH" JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"
flutter build apk --release        # app/build/app/outputs/flutter-apk/app-release.apk
```

In the app: Settings → server address `http://<mac-ip>:8770`, token → *Save and test*.
Phone and Mac must be on the same Wi-Fi.

## 3. Using it

New session → names, handedness → Record (or Choose video) → Upload (resumes after interruptions)
→ Court setup: drag the 4 circles onto the line crossings on the players-removed image → Check court
→ Start analysis → report (summary, landing map, shot list with video playback).

## 4. Command-line client (same API)

```bash
.venv/bin/python -m server.client analyze VIDEO.mp4 --names A=Red,B=Grey \
  --markers "far_left_singles:0.367,0.294;far_right_singles:0.622,0.294;near_service_left:0.193,0.485;near_service_right:0.797,0.485"
```

## 5. Moving the server to the RTX laptop

Same code; install Python 3.12, `pip install -r requirements-server.txt` plus PyTorch with CUDA and
`onnxruntime-gpu`; copy `models/`, `work/models/racketvision/`, `work/cache/rtmlib/`. Set
`"device": "cuda"` in `server/config.json`. The engine picks the device from config.

## 6. Quality checks

```bash
.venv/bin/python eval/run.py            # all benchmarks -> docs/RESULTS.md
.venv/bin/python -m pytest server/tests -q
cd app && flutter analyze
```
