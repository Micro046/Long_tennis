# Generalization and release evaluation plan

The first supported product is unattended, fixed-phone singles recording with a short setup and post-session report. The analytics must work on **unseen people, venues, cameras, and sessions** within a documented supported capture range. Current broadcast and spectator experiments are research baselines, not evidence of phone readiness.

## Capture and annotation corpus

Collect commercially usable, consented fixed-phone recordings, using different phones, mounting heights, distances, portrait/landscape orientation where supported, frame rates, resolution, video compression, lighting, court surface/color, clothing (including similar kits), player skill, handedness, occlusion, and camera shake. Include full sessions with players changing ends, temporary exits, warmups, spectators, and interrupted recordings. Record actual presentation timestamps to handle variable frame rate. Keep footage rights, annotator instructions, and source version with the data.

Annotate player identity independently of near/far side, side-change times, each visible racket contact, hitter, first ground contact, in/out or unknown, landing point and named court region, stroke class and handedness where visible, and ambiguity. Use exhaustive time windows for event recall. Double-label a sample and resolve disagreements. A radar or other independent reference is required for speed validation; 2D image displacement alone is not physical speed.

## Split and measure

- Partition by **physical recording session, venue, camera, and players**; adjacent frames from one video must never be split across train and test. Preserve an unseen held-out phone set untouched while developing models and thresholds. Record near-duplicate checks and dataset version.
- Freeze preprocessing, models, thresholds, region rules, and uncertainty gates before the held-out run. Report each camera/view and condition slice plus worst-slice results, not only the pooled mean. Report sample sizes and uncertainty intervals.
- For identity, report correctly attributed visible-player frames, wrong attributions, abstentions, switches per player-hour, and changeover recovery time. Distinguish an identity swap from a player simply disappearing.
- For hits and bounces, report one-to-one precision, recall, and temporal error on exhaustive windows. Report hitter attribution conditional on correct hit matching and end-to-end. Track missed events and false events separately.
- For landings, report court-coordinate position error in metres against independent contact points, region confusion, boundary abstentions, wrong-half outcomes, and per-player count error. Count only a shot's first bounce. Percentages must show the denominator and excluded shots.
- For forehand/backhand, report confusion and per-player count error on events with an observed contact, plus abstentions and right/left handed slices. Keep serve and volley distinct rather than forcing either class.
- For ball speed, first define contact or flight-segment speed, then measure error against a reference across the supported camera geometry. Withhold speed whenever calibration or trajectory evidence is inadequate.
- For the app, run full-duration sessions on representative Android phones and measure completed recordings, analysis latency, battery use, thermal throttling, storage, crash recovery, and privacy/data handling. Inference runtime from a desktop adapter is not a phone benchmark.

Set numerical launch thresholds **before** the held-out evaluation in a versioned protocol. Averages cannot mask a failing view or device slice. When a threshold fails, improve on training/development data and evaluate on a new held-out set; do not retune against the release set.

## Current evidence and decision

The three UVY spectator clips evaluate only the association step with supplied person boxes and labelled initialization. They do not test player detection, a phone placement, or a verified end change. The motion-only baseline outperformed the simple shirt-histogram variant on one clip and tied or nearly tied on the others. Therefore appearance matching is an unselected research candidate, and the app must not promise automatic identity continuity yet. The synthetic changeover test confirms the data model can preserve an ID when side changes; it is not a field validation.

Any public dataset is an external diagnostic set, not a substitute for the supported phone corpus. Do not train on or tune to the held-out sessions. Distribute only components and data with commercial rights checked separately for source code, weights, and footage.
