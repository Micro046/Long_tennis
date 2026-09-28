# Pipeline v2 — end-to-end analysis and Android app (2026-09-28)

This milestone turns the research spikes into one working chain, from a fixed-phone singles
video to per-player statistics, implemented twice with identical logic:

* **Python reference** — `scripts/pipeline/` (used for evaluation on public data)
* **Android app** — `android/` (Kotlin + ONNX Runtime, on-device, post-session)

Kotlin/Python parity is enforced by JVM unit tests (`android/app/src/test/.../ParityTest.kt`):
event features (max diff < 1e-4), event-model probabilities (< 1e-5), court calibration
(keypoints within 1.65 px of Python).

## Stages

| Stage | Python | Kotlin | What it does |
|---|---|---|---|
| Court setup | `pipeline/court.py` | `CourtCalibrator.kt` | User drags 4 markers roughly onto line crossings; each projected court line is snapped to white paint (robust line fit), all 14 intersections re-fit the homography (1.6 px RMS on the amateur match). Works when baseline corners are outside the frame. Broadcast-trained court detectors (TennisCourtDetector, Good-Tennis) failed on the low amateur view. |
| Ball | `pipeline/ball.py` | `BallTracker` in `Models.kt` | RacketVision TrackNetV3 (512x288, 4 frames + median background per 2-min segment). **Fixed a one-frame lag in the milestone-0 adapter**: median error on the RacketVision test clips drops from 14.8–21.0 px to 1.5–2.2 px (100% within 25 px). |
| Players + pose | `pipeline/pose.py` | `PoseEstimator` in `Models.kt` | YOLOX-tiny on the full frame **plus two far-court tiles** placed from the calibration (the far player is ~90 px tall and was missed at 416 px input), NMS + truncated/contained-box suppression, court-area filter, RTMPose-s. Runs every 3rd frame on the phone. |
| Identity | `pipeline/identity.py` | `Identity.kt` | Whole-session: tracklets in court metres → shirt + shorts HSV descriptors from pose-guided regions → A/B assignment seeded at setup (A = near player unless the user says otherwise), with the constraint that simultaneous tracklets differ. Side (near/far) is derived per frame; identity is independent of side. |
| Events | `pipeline/events.py` | `Events.kt` (`GbmModel`) | 70 scale-normalised trajectory features → 3-class gradient-boosted trees (none/hit/bounce), peak picking. Trained on the TrackNet dataset, exported to JSON (`export_event_model.py`). |
| Dead-ball filters | `pipeline/session.py` | `Analyzer.kt` | Remove ball-handling events (ball stays in one player's reach and on their side of the net ±1 s), the earlier of two same-player hits within 1.5 s (serve toss, feeds), and hits not answered within 2.5 s by the opponent or a bounce in the opponent's half. |
| Shots | `pipeline/session.py` | `Analyzer.kt` | Hitter = detected person whose racket hand is nearest the ball in reach units (an unidentified nearest person gives "unknown" rather than a guess); first bounce before the next hit; landing region (6 zones of the opponent's singles half, from the hitter's view; out/near-line explicit); stroke class: serve = wrist above head before contact at the baseline, overhead elsewhere, volley = no bounce and < 7 m from the net, forehand/backhand by the follow-through side of the racket-hand wrist relative to the hips (handedness from setup); average flight speed hitter→first bounce (withheld above 230 km/h). |
| Report | `session.py` | `Analyzer.report` | Per-player shots, stroke counts, region counts/percentages with explicit denominator, speeds, shot list with timestamps. |

## Results (2026-09-28)

All numbers below are measured; "not validated" means no independent ground truth exists yet.

### Ball tracking (RacketVision test clips, sparse GT)
| | Recall | Median error | Within 25 px |
|---|---:|---:|---:|
| milestone-0 adapter | 89.8% | 19.2 px | 67.4% |
| v2 (one-frame lag fixed) | 92–100% | 1.5–2.2 px | 100% |

### Hit / bounce detection — TrackNet dataset, leave-one-game-out (10 broadcast games, 516 hits, 523 bounces, exhaustive per-frame labels, ±3-frame tolerance, threshold 0.5)
| Ball input | Hit P / R / F1 | Bounce P / R / F1 | Mean timing error |
|---|---|---|---|
| Ground-truth ball | 0.843 / 0.862 / 0.852 | 0.880 / 0.954 / 0.916 | 0.45 frames |
| **RacketVision tracker (realistic)** | **0.852 / 0.835 / 0.843** | **0.864 / 0.950 / 0.905** | 0.5 frames |
`outputs/v2/events/tracknet_logo_{gt,pred}.json`

### Amateur fixed-camera match (21.6 min, 38,808 frames, real end changes) — `data/public/youtube_amateur/4apx6gd5Uxs`
- **Player identity** (56 independently labelled random frames, 96 player-observations, ≥6 end changes): **94 correct, 0 wrong, 2 missed (97.9%)**; no identity swap. `outputs/v2/identity_eval_4apx.json`
- **Racket-contact precision** (blind audits of randomly sampled predicted hits, judged without seeing predictions): 62% before the dead-ball filters → **83%** after (24 true / 5 false, 3 undecidable). `outputs/v2/audit_4apx_v{1,2}.json`
- **Hitter attribution** on audited true contacts: 18 correct, 2 wrong, 4 left "unknown".
- **Stroke class** on audited, correctly attributed shots: **17/20** (forehand 7/7, backhand 3/5, serve 7/8). Follow-through-side rule replaced the contact-side rule (≈50%).
- **Recall on the amateur match** is not yet measured (the far player's groundstroke contacts cannot be timed reliably by eye at this resolution; exhaustive labels need a higher-resolution source).
- **Landing regions and speed: not validated.** Speed is the average ground-projected speed from the hitter to the first bounce, withheld above 230 km/h (implausible pairing). Validation needs radar and independent landing labels.
- Final per-player report: `outputs/v2/session/4apx6gd5Uxs_report_final.json`.

### Android app
- Builds (AGP 9.1, Compose, ONNX Runtime 1.22); installs and runs the full flow on an Android 35 arm64 emulator: automatic analysis → report screen (see `work/emu_report.png`).
- **Kotlin ≡ Python**: JVM tests reproduce identity (540/540 frames), hits, bounces, and every shot's player/stroke/landing on the 30 s fixture; event features < 1e-4, model probabilities < 1e-5, court calibration < 1.65 px.
- On-device vs desktop on the same clip: ball positions median 1.8 px apart; 17/17 bounces and 11/14 hits agree (differences from float32 decode/resampling at threshold boundaries).
- **Speed is the open problem**: with the full-size ball model the emulator analysed 0.46 frames/s. A distilled 0.7 M-parameter student (13x faster on CPU) is being trained (`scripts/train_student_ball.py`, evaluation `scripts/eval_student_ball.py`).

### Known limitations / next work
1. Far-player serves are under-detected (pose of a ~100 px player); grey's serve count (20) is too low relative to red's (72).
2. Contacts during dead-ball time (feeding balls between points) are only partly filtered (~17% false contacts remain).
3. Recording-side: the unedited phone video will contain long breaks and walking between points, which this edited YouTube match lacks.
4. All numbers come from one venue, one camera placement, two players. Additional sessions (different courts, lighting, left-handers, far-player-heavy views) are required before release claims.
