# Mobile model candidate decisions — 2026-09-28

These are research decisions, not commercial clearance or phone performance claims. Keep code, exact model artifact, training-data provenance, and footage decisions separate.

| Candidate | Evidence | Current decision |
|---|---|---|
| Existing YOLOX-tiny HumanArt detector | Structurally valid ONNX, 19.3 MiB; HumanArt training-data restriction recorded in `THIRD_PARTY.md` | Evaluation only; replace or obtain clearance before product distribution |
| Existing RTMPose-s | Structurally valid ONNX, 20.9 MiB; training-data chain pending | Evaluation only; retain as comparison baseline |
| Existing RacketVision ball checkpoint | PyTorch checkpoint, 44.4 MiB; weight redistribution terms not stated | Evaluation only; mobile conversion and artifact terms pending |
| MediaPipe Pose Landmarker Lite/Full | Official Android implementation; current API exposes `num_poses`; linked BlazePose GHUM model card lists Apache 2.0 | Candidate for an isolated pose comparison, not selected for full-court use |
| Official YOLOX COCO Tiny/Nano | YOLOX source is Apache 2.0; official documentation describes small variants for mobile deployment | Candidate replacement for HumanArt detector; verify exact weight artifact terms and phone accuracy before selection |

## MediaPipe scope caveat

The official [Pose Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) describes multiple-pose configuration and versioned task bundles. Its linked [BlazePose GHUM model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf) is dated 2021, lists Apache 2.0, and describes single-person mobile/fitness use. It warns about multiple people, people more than roughly four metres away, and scale/orientation sensitivity. The current API's multiple-pose option does not establish far-player quality on a tennis court. Do not transfer published fitness latency or accuracy into our product claims, and do not use estimated body depth as physical ball-speed evidence.

The next comparison should measure near/far player acquisition and labelled joint accuracy across the phone capture range, with exact task bundle hashes and SDK version pinned. Cropped-player pose is a possible architecture only after a separately validated person detector produces reliable crops. Both players' scale, occlusion, clothing and lighting slices must be measured.

## YOLOX source caveat

The official [YOLOX repository license](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE) is Apache 2.0, and its [custom-data guide](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md) offers COCO pretrained weights and small mobile variants. This supports evaluating a COCO-based alternative, but a source-code license alone does not close the exact checkpoint's redistribution and training-data provenance review. No replacement checkpoint has been installed or selected in this milestone.
