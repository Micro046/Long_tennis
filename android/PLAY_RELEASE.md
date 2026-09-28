# Google Play release kit (draft)

## Build outputs
- `app/build/outputs/apk/release/app-release.apk` — installable test APK (sideload / internal testing)
- `app/build/outputs/bundle/release/app-release.aab` — upload to Play Console
- Build: `./gradlew :app:assembleRelease :app:bundleRelease` (JDK: Android Studio JBR)

## Signing
`release.keystore` in this folder is a **development upload key** (alias `longtennis`, passwords via
`LT_STORE_PASSWORD` / `LT_KEY_PASSWORD`, default `longtennis`). Before the first Play upload create
a private upload key, keep it out of the repository, and enrol in Play App Signing.

## Store listing (draft)
**Title:** Long Tennis — Match Stats
**Short description:** Film your singles match with one phone and get shot stats for each player.
**Full description:**
Place your phone behind the baseline, mark the court once, and play. After the session Long Tennis
analyses the video on your phone and reports, for each player:
- shots, serves, forehands, backhands and volleys
- where first bounces landed in six zones of the court, with percentages
- estimated average ball speed from racket to bounce
- a shot list that jumps to the moment in the video
Players keep their identity when you change ends. All processing happens on the device.

## Data safety answers
- Data collected: none transmitted. Video and results stay in app-private storage on the device.
- Camera: used only to record the session the user starts.
- No account, no analytics SDK, no ads SDK in this build.

## Privacy policy (draft text to host at a public URL)
Long Tennis records video only when you start a session. Recordings, court setup and results are
stored in the app's private storage on your device and are never uploaded by the app. Deleting a
session deletes its video and results. Uninstalling the app removes all data. The app does not
collect personal information, contacts, location, or advertising identifiers. People appearing in a
recording should be informed and consent to being filmed.

## Open items before production
- Replace evaluation-only model weights or obtain rights (see `THIRD_PARTY.md`, `data/public/SOURCES_V2.md`).
- Monetisation decision (e.g. free tier + subscription via Play Billing).
- Real-device benchmarks (analysis time per hour of play, battery, heat) on low/mid/high phones.
- Target API level and 16 KB page-size compliance check for native libraries (ONNX Runtime).
