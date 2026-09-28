#!/usr/bin/env python3
"""Static Android packaging audit; deliberately not a device performance test."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import onnx

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_shapes(values):
    return [{"name": v.name, "shape": [d.dim_value or d.dim_param or None
             for d in v.type.tensor_type.shape.dim]} for v in values]


def inspect_onnx(path, name, commercial_status):
    model = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(model)
    return {"component": name, "fileName": path.name, "sizeBytes": path.stat().st_size,
            "sha256": file_hash(path), "format": "ONNX", "irVersion": model.ir_version,
            "opsets": [{"domain": x.domain, "version": x.version} for x in model.opset_import],
            "nodeCount": len(model.graph.node), "operatorCounts": dict(Counter(n.op_type for n in model.graph.node)),
            "inputs": tensor_shapes(model.graph.input), "outputs": tensor_shapes(model.graph.output),
            "onnxStructuralCheck": "passed", "commercialProductStatus": commercial_status,
            "androidRuntimeTest": "not_run"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT/"outputs/milestone2/mobile_viability.json")
    args = p.parse_args()
    model_dir = ROOT/"work/cache/rtmlib/hub/checkpoints"
    detector = next(model_dir.glob("yolox_tiny*.onnx"))
    pose = next(model_dir.glob("rtmpose-s*.onnx"))
    models = [inspect_onnx(detector, "person_detector", "blocked: HumanArt training-data terms"),
              inspect_onnx(pose, "body_pose", "evaluation_only: full training-data chain pending")]
    ball = ROOT/"work/models/racketvision/balltrack_best.pth"
    models.append({"component":"ball_tracker", "fileName":ball.name, "sizeBytes":ball.stat().st_size,
                   "sha256":file_hash(ball), "format":"PyTorch checkpoint",
                   "commercialProductStatus":"evaluation_only: weight license not stated",
                   "androidRuntimeTest":"not_run", "conversionStatus":"not_converted_to_mobile_format"})
    toolchain = {key: shutil.which(key) for key in ("adb", "flutter", "gradle", "java")}
    report = {"status":"static_audit_only", "models": models,
              "combinedModelBytes":sum(model["sizeBytes"] for model in models),
              "toolchain":toolchain,
              "androidDeviceBenchmarked":False,
              "mobileInferencesPerSecond":None,
              "batteryAndThermalMeasured":False,
              "productBuildReady":False,
              "blockers":["No licensed fixed-phone benchmark with end changes and independent labels",
                          "Existing detector training-data terms block commercial use",
                          "Ball checkpoint redistribution terms are unspecified",
                          "Ball checkpoint has not been converted and validated in an Android runtime",
                          "No Android hardware latency, battery, storage, or thermal result"],
              "reference":"https://onnxruntime.ai/docs/tutorials/mobile/"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(args.output)


if __name__ == "__main__":
    main()
