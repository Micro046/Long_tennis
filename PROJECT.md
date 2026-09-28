# Long Tennis Analysis — Mobile Product and Integration Plan

## Product goals — updated 2026-09-27

Build a commercial Android tennis-analysis app: the user places a phone, starts a session, plays without ongoing interaction, and receives player-specific statistics afterwards. Deliver an installable APK for testing and an Android App Bundle (AAB) for Google Play publication. Monetization is a product goal; the pricing model is still undecided.

User-requested capabilities:

1. Persistent player identities: player A stays A when players change court ends, cross, or temporarily disappear. Identity must be separate from the current near/far side.
2. Racket-contact and court-bounce detection, with attribution to the player who hit the shot.
3. Named court regions and end-of-session shot landing counts and percentages per player.
4. Per-player forehand and backhand counts, with room for serve, volley, and other stroke categories.
5. Ball-speed estimates where the recording supports a validated estimate.
6. Unattended recording and analysis during normal play after a brief setup.
7. A publishable, monetizable Android application using commercially cleared components.

These are product requirements, not claims about the current prototype.

## Current engineering objective

Prove an end-to-end, reuse-first tennis-video analysis pipeline before building Flutter:

```text
public/owned video
  -> existing ball/player/pose/court components
  -> normalized timestamped events
  -> overlays + evaluation report
  -> later: Flutter timeline viewer
```

## Rules

- Reuse legally compatible components; do not rebuild established CV algorithms without evidence.
- Keep every third-party repository, checkpoint, and dataset pinned with its own license decision.
- Code license, checkpoint license, and footage/dataset license are separate decisions.
- Use public/licensed or owner-recorded video only.
- Compare components on identical clips and metrics.
- Do not start the Flutter product until the analysis JSON and overlays are useful.

## First milestone

1. Validate the public RacketVision clips and ground-truth annotations.
2. Run the reusable desktop pipelines on the same clips.
3. Normalize output into the project event schema.
4. Compare accuracy, runtime, memory, model size, and failure modes.
5. Keep the best legal component for each task.

The existing desktop outputs demonstrate ball, pose, and court integration on short broadcast clips. See `outputs/STATUS.md` for the 2026-09-24 baseline and `outputs/milestone1/report.json` for the current identity/event proposals. A portable review workspace and an external oracle-box identity experiment have been added. `GENERALIZATION.md` defines the held-out phone evaluation gate. This milestone is incomplete until phone footage and independent annotations support the product claims.

## Current capability boundaries

| Requirement | Existing foundation | Work still required |
|---|---|---|
| Player A/B identity | Temporal A/B association with separate current side; external oracle-box baseline and synthetic crossing/changeover tests | Independent end-to-end validation on full fixed-phone sessions with real end changes; improve wrong attribution and abstention behavior |
| Hit and bounce detection | Ball image coordinates, timestamps, poses, court calibration, candidate heuristics and review UI | Independently annotate exhaustive phone-video windows; measure contact and bounce precision, recall, timing, and hitter attribution |
| Court-region statistics | Six player-relative singles regions, provisional first-bounce linking, calibration and boundary uncertainty gates | Verify bounce ground points and court mapping against independent labels across views; measure per-player count and region errors |
| Forehand/backhand counts | Seventeen body joints per selected player | Classify motion around a contact using body orientation, handedness, and available racket/ball evidence; validate on phone footage |
| Ball speed | Timestamped 2D ball observations | Camera calibration and estimated 3D trajectory or another validated method; compare with an independent speed reference |
| Unattended mobile use | Offline Python research scripts | Mobile capture, setup validation, mobile inference, long-session reliability, storage/battery/thermal management, and recovery |
| Commercial Android release | Third-party provenance register | Clear or replace evaluation-only weights, build Android app, test, sign, prepare store disclosures and billing |

## Proposed user experience and reporting rules

- Brief guided setup: fixed phone with the whole playing area visible, automatic court proposal, framing/lighting checks, and initial player assignment. Manual correction is a fallback. Exact supported placement and capture settings must be established through phone tests.
- During play: preserve session player IDs independently of court side and avoid requiring intervention at normal changeovers. Keep unresolved identities or events uncertain; offer correction after the session when necessary.
- After play: player-specific stroke totals, landing maps, counts and percentages by named region, and links to the corresponding video moments. Speed is explicitly estimated and unavailable when evidence is insufficient.
- A starting region design is left/centre/right crossed with short/deep on each court half; precise boundaries and names are configurable. Persist camera-fixed court coordinates and derive a player-relative attacking view using the player's side at the time of the shot, so changeovers do not reverse tactical statistics.
- Count one shot's first landing event once, not every tracked video frame or subsequent bounce. A shot intercepted by the opponent's volley has no landing before that return. Keep out, no-bounce, and unknown outcomes explicit; near-line uncertainty must not be reported as a certain line call.
- State the denominator for percentages, such as confidently assigned in-court first landings, and show how many detected shots were excluded or unresolved. Do not claim complete shot coverage without evaluating missed contacts.
- Handedness is independent of court side. Ambiguous strokes remain unknown instead of being forced into forehand/backhand totals.
- Do not calculate physical ball speed by applying the ground-plane homography to an airborne trajectory. Distinguish estimated speed at contact from average speed over a flight segment.

## Proposed delivery sequence

1. **Phone benchmark and release viability:** obtain licensed representative phone sessions, including changeovers and temporary occlusions; annotate identities, contacts, first bounces, landing regions, and stroke types. Identify commercially usable model candidates and run an early Android inference/recording feasibility check.
2. **Identity and event pipeline:** add persistent player IDs; detect racket contacts and bounces; link shots to hitters and first landing outcomes. Review errors on labelled clips and full sessions.
3. **Useful session report:** add named-region counts/maps and forehand/backhand classification. Validate the complete path from video to each player's statistics, including missed or uncertain events.
4. **Speed research:** estimate 3D trajectories with camera calibration and validate speed against an independent reference such as radar. Retain this goal while keeping unvalidated speed out of release claims.
5. **Android product:** build the Flutter capture/review flow and native inference integration once analysis outputs are useful. Test full intended session duration on representative phones for recording continuity, processing time, battery, heat, and storage recovery.
6. **Commercial release:** resolve model/data redistribution rights, choose and test monetization, prepare privacy/data disclosures, signed APK/AAB builds, and complete the applicable Play testing and publication requirements.

Recommended initial validation scope is singles with a fixed phone and post-session reporting, followed by doubles. This is a proposed prioritization, not a removal of doubles from the long-term scope. Fully local processing versus optional cloud processing remains undecided and depends on device benchmarks, privacy expectations, processing delay, and cost.

Before launch, set measurable acceptance thresholds and verify identity switches across changeovers, contact/bounce detection accuracy, landing-position and zone errors, per-stroke classification/count errors, and speed error where supported. Evaluate unseen phone sessions; the existing sparse broadcast ball labels cannot validate these product claims.

## Milestone 2 status — 2026-09-28

Benchmark ingestion now preserves decoded presentation timestamps and audits source hashes, annotation coverage, frame timing, and dataset split leakage. `BENCHMARK_PROTOCOL.md` and the phone manifest/label templates define an independent fixed-phone evaluation path. The Kaggle back-view set was downloaded and audited as an external diagnostic source, but its short clips, metadata discrepancy, corner-clustered ball points, and unverified underlying footage rights prevent it from serving as the phone release test. The two existing pose models pass an ONNX structural check; the ball checkpoint remains PyTorch only. None of the three model weights is currently cleared for a commercial app. No Android device, SDK toolchain, or full phone session is available, so no Android performance or changeover metric can be claimed. See `outputs/milestone2/` and the current user-facing milestone report.

## Pipeline v2 status — 2026-09-28

Licensing was explicitly deferred by the owner ("not an issue for now"), so public datasets and
weights were used to complete the chain. `PIPELINE_V2.md` has the design and measured results;
`data/public/SOURCES_V2.md` records provenance. Summary:

| Requirement | Status |
|---|---|
| Persistent A/B identity | Working on a 21-min amateur match with end changes: 97.9% of labelled player-observations correct, 0 swaps |
| Hit / bounce detection | TrackNet leave-one-game-out F1 0.84 / 0.91; amateur-match contact precision 83% (blind audit) |
| Hitter attribution | 18/20 correct on audited contacts, abstains when unsure |
| Landing regions + % per player | Implemented (6 zones, explicit denominators); not yet validated against landing labels |
| Forehand / backhand / serve / volley | 17/20 on audited shots |
| Ball speed | Average hitter→bounce flight speed with plausibility cap; not validated (needs radar) |
| Android app | Kotlin + ONNX Runtime app (capture, court setup, background analysis, report) builds and runs on an arm64 emulator; Kotlin output identical to the Python reference in unit tests |
| Remaining | Faster ball model for phone speed (student model), exhaustive labels on phone footage, multi-venue evaluation, real-device benchmarks, commercial rights |

## Technical and distribution references

References checked 2026-09-27; candidate research does not imply component selection or commercial clearance.

- Appearance-assisted tracking: https://arxiv.org/abs/1703.07402
- Monocular 3D ball trajectory research: https://arxiv.org/abs/2506.05763
- Mobile model deployment and device benchmarking: https://onnxruntime.ai/docs/tutorials/mobile/
- Google Play publishing format: https://developer.android.com/guide/app-bundle
- Google Play digital purchases and subscriptions: https://support.google.com/googleplay/android-developer/answer/10281818

## Directory layout

```text
data/public/       licensed public evaluation clips and annotations
scripts/           project-owned inspection/evaluation adapters
third_party/       pinned upstream repositories (not product code by default)
work/              intermediate files and model downloads
outputs/           reports, overlays, and normalized events
```
