"""Project-owned v2 analysis pipeline: video -> ball/pose/court -> events -> player report.

Every stage here is written so it can be ported one-to-one to the Android app
(ONNX models + small deterministic post-processing).
"""
