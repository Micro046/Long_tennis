# Analysis interchange schema v0.1 / milestone 0.2 research extensions

The research adapters must export project-owned JSON. The future Flutter application should consume this format rather than any upstream repository's native output.

## Ball observations

One document represents one video and contains:

```json
{
  "schemaVersion": "0.1",
  "sessionId": "match100_000",
  "video": {
    "fileName": "match100_000.mp4",
    "widthPx": 1920,
    "heightPx": 1080,
    "fps": 25.0,
    "frameCount": 250
  },
  "producer": {
    "component": "RacketVision/BallTrack",
    "sourceCommit": "c44af2a08524d3cb54d818f19686f4cdea4d2793",
    "modelSha256": "00d707b9db7a49561c411e4765956e79bcd7c7e20c7a0a535073440b3e972342",
    "threshold": 0.5
  },
  "observations": []
}
```

Each observation contains:

- `id`: stable within the session (`ball:<frame index>`).
- `type`: `ball_observation`.
- `frameIndex`: zero-based decoded frame number.
- `timestampUs`: integer microseconds from the start of the video, derived from frame index and FPS.
- `visible`: whether the component emitted a ball at its configured threshold.
- `confidence`: component confidence in the range 0–1.
- `image`: `null` when not visible; otherwise pixel and normalized coordinates in the **original video**, not the model input.
- `provenance`: identifies the producing adapter and preserves the raw model-input coordinates for debugging.

Normalized image coordinates use a top-left origin: `xNorm = xPx / widthPx`, `yNorm = yPx / heightPx`. They are camera-image coordinates, not court coordinates.

Pose estimators can extrapolate a joint a few pixels outside the decoded image, so pose `xNorm`/`yNorm` values are not forcibly clamped to 0–1. Consumers should use joint confidence and clip only for display.

Future pose, court, hit, bounce, stroke, rally, and score records will use the same `frameIndex`, `timestampUs`, confidence, and provenance conventions. Derived events must also list their input observation IDs.

## Player pose observations

Pose documents follow the same session/video/producer envelope. Each frame contains zero or more `player_pose_observation` records with:

- a view-relative `role` (`near`/`far` for singles; numbered near/far candidates for doubles);
- the person bounding box in pixels and normalized coordinates;
- 17 named COCO body joints, each with pixels, normalized coordinates, and a raw model score stored in the legacy `confidence` field;
- the model adapter and raw detection index in `provenance`.

The numbered doubles candidate roles are **not persistent player identities**. Their order can change when a player is missed or players cross. Stable player IDs must not be claimed until a temporal tracker has been evaluated.

The RTMPose adapter has emitted some joint scores above 1.0. Treat pose joint `confidence` as an **uncalibrated raw score**, not a probability. Do not display it as a percentage without model-specific validation. The milestone 0.2 pose export preserves every person detection in `candidates`, so a tracker can see both athletes even when they share the same court half. `role` remains a view label. Derived `playerId` is a separate temporal identity with an unconfirmed or reviewed seed, and `side` can change without changing `playerId`.

Phone video can use variable frame rate. Its source presentation timestamps must be decoded and preserved; `frameIndex / nominal fps` is only valid for the constant-frame-rate research clips. `candidate_review` files correct proposed events and do not establish detection recall. Accuracy evaluation requires independent exhaustive labels with explicit coverage windows.

## Court calibration

A `court_calibration.json` document stores:

- four consensus image corners ordered far-left, far-right, near-right, near-left;
- the same corners in normalized image coordinates;
- image-to-court and court-to-image 3×3 homography matrices;
- detector confidence, multi-frame corner stability, reference-line support, and every sampled-frame diagnostic;
- the source component commit and the fact that no model weights are used.

Court coordinates use metres. The origin is the far-left doubles baseline as seen by the camera; `x` runs left-to-right across the 10.97 m doubles width and `y` runs far-to-near along the 23.77 m court length.

`player_court_observation` records reference their source pose ID and contain the ankle-derived image ground point plus projected court metres. A homography assumes the point lies on the court plane; it must not be used as the true 3D position of an airborne ball.
