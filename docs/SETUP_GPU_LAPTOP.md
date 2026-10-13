# Running the server on an NVIDIA laptop (e.g. RTX 3050)

The Mac was the test server. An NVIDIA GPU runs the ball tracker (the bottleneck) with CUDA; the code is the
same — `pipeline/run.py` picks `cuda` automatically on non-macOS machines.

## 1. Code
```bash
git clone https://github.com/Micro046/Long_tennis.git long_tennis
cd long_tennis
```

## 2. Model weights and ball-tracker source (not in git)
Copy `LongTennis_server_assets.zip` (≈ 82 MB) from the Mac — on the same Wi-Fi it is at
`http://<mac-ip>:8770/dl/LongTennis_server_assets.zip` — and unzip it **inside** the repo folder. It contains:

| Path | What |
|---|---|
| `work/models/racketvision/balltrack_best.pth` | TrackNetV3 ball tracker weights (RacketVision) |
| `work/cache/rtmlib/hub/checkpoints/yolox_tiny_…onnx` | person detector |
| `work/cache/rtmlib/hub/checkpoints/rtmpose-s_…onnx` | body pose |
| `third_party/RacketVision/source/BallTrack/` | TrackNetV3 model code (MIT) |

## 3. Python environment (Python 3.12)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-server.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
The last line must print `True` and the GPU name. Optional, faster pose: `pip install onnxruntime-gpu`
(then the pose stage can use CUDA as well; without it pose runs on the CPU in parallel with the ball tracker).

## 4. Check it works
```bash
python -m pytest -q tests
python scripts/eval_clips.py          # needs the three test sessions copied from the Mac (server/storage/sessions)
```

## 5. Start the server
```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8770
```
In the phone app → Settings, set the server address to `http://<laptop-ip>:8770` (token `dev-token`).
Allow port 8770 through the firewall (Windows asks on first start).

## 6. Analyse a long video from the command line
```bash
python -m server.client analyze path/to/match.mp4 --markers "far_left_singles:0.378,0.498;far_right_singles:0.600,0.498;near_service_left:0.223,0.654;near_service_right:0.787,0.654" --names "A=Player1,B=Player2"
python scripts/session_summary.py server/storage/sessions/<id> results/<name>
python scripts/render_overlay.py server/storage/sessions/<id> --out results/<name>/annotated.mp4
```
Markers are normalised image positions (0–1) of at least 4 court points; the app's court-setup screen produces
them. `render_overlay.py` uses macOS' hardware encoder on a Mac and `libx264` elsewhere.

## Reference timings (60-min 1080p30 video)
| Machine | Ball tracking | Pose (parallel) | Total |
|---|---|---|---|
| Mac (M-series, MPS), first run | 3 h 29 min | 1 h 19 min | 3 h 31 min |
| Mac, after the 2× tracker speed-up (commit 853ea4d) | ≈ 1 h 45 min (17 fps) | 1 h 19 min | ≈ 1 h 50 min (est.) |
| RTX 3050 laptop | to be measured | | |
