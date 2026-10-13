#!/usr/bin/env python3
"""Quick scoring of the exhaustive-window event benchmarks (4apx dev, MZZ held-out) with the current engine."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "scripts"), str(ROOT)]
import run  # noqa: E402

for b in [x for x in json.loads((ROOT / "eval/benchmarks.json").read_text()) if x["kind"] == "events"]:
    r = run.events(b)
    h, bo = r["hits"], r["bounces"]
    print(f"{b['name'][:5]} {b['split']:7s} hits P{h['precision']:.2f} R{h['recall']:.2f} F1 {h['f1']:.3f} | bounces P{bo['precision']:.2f} "
          f"R{bo['recall']:.2f} F1 {bo['f1']:.3f} | hitter {r['hitter']} | stroke {r['stroke']['correct']}/{r['stroke']['total']}", flush=True)
    if "-v" in sys.argv:
        for d in r["details"]:
            print("   ", d["window"], d["kind"], "missed", d["missed"], "false", d["false"])
