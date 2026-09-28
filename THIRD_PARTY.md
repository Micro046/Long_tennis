# Third-party provenance register

Reviewed 2026-09-24. A listed repository license does not automatically clear external weights or footage.

| Component | Pinned commit | Repository license | Current use decision |
|---|---|---|---|
| Good-Tennis | `62759624f8aaf3fd1abd5a6a03695150e90277e2` | Apache-2.0 | Primary desktop integration reference. Audit every downloaded YOLO/RTMPose weight and dependency separately. |
| WASB-SBDT | `923462cacdeb3353b84ddebdedb3f4b7a8553b0f` | MIT | Ball-tracker candidate. Confirm exact checkpoint redistribution terms before a product build. |
| RacketVision | `c44af2a08524d3cb54d818f19686f4cdea4d2793` | MIT | Ball/racket research candidate and public test data source. Model hub currently has no model card; do not assume checkpoint terms. |
| philippdubach/tennis-vision | `a501953194f5207534273c7b2b59c0b642df7a05` | MIT | Event schema, tests, and isolated hit/rally logic reference. Do not import its unlicensed upstream TennisProject dependency. |
| RTMLib | PyPI `0.0.16` | Apache-2.0 | Lightweight ONNX evaluation runtime for YOLOX + RTMPose. Candidate for the desktop spike; mobile integration would use the underlying ONNX models/runtime rather than Python. |
| MMPose RTMPose/YOLOX weights | Exact SHA-256 values in pose reports | OpenMMLab distribution; training-data terms are separate | **Evaluation only.** The selected YOLOX detector is advertised as HumanArt+COCO trained, while HumanArt access is restricted to non-commercial use. Do not ship this detector in a commercial product without written clearance or replacement. RTMPose's complete training-data chain also requires review. |

## Public test data

`data/public/racketvision` contains three clips plus ball annotations from the RacketVision Hugging Face dataset, whose dataset card declares MIT. Preserve dataset citation and provenance in derived reports.

`data/public/tenis_backview` contains ten short clips and CSV annotations from [the Kaggle dataset](https://www.kaggle.com/datasets/gastonarielfrancois/tenis-backview/data), version 1, whose card declares CC0. The actual files are 1920×1080 despite the card's 1080×1920 claim. Several frames show broadcast-style graphics and source watermarks, and the card does not document ownership of the embedded footage. More than half of each clip's reported ball positions cluster near the top-right image corner. **Diagnostic-only; original footage rights and annotation meaning are unverified. Do not redistribute, train a commercial model on, or use as a held-out phone benchmark until verified.** See `outputs/milestone2/backview_audit.json` and `data/public/tenis_backview/PROVENANCE.json`.
