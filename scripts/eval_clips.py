#!/usr/bin/env python3
"""Quick scoring of the labelled test clips (clip_events benchmarks) with the current engine.

  eval_clips.py [-v]      # -v: list missed / false events per clip
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT / "scripts"), str(ROOT)]
import run  # noqa: E402

for b in [x for x in json.loads((ROOT / "eval/benchmarks.json").read_text()) if x["kind"] == "clip_events"]:
    if not (ROOT / b["session"]).exists() or not (ROOT / b["labels"]).exists():
        print(b["name"][:6], "skipped (inputs missing)"); continue
    r = run.clip_events(b)
    h, bo = r["hits"], r["bounces"]
    print(f"{b['name'][:6]} hits P{h['precision']:.2f} R{h['recall']:.2f} F1 {h['f1']:.3f} | bounces P{bo['precision']:.2f} "
          f"R{bo['recall']:.2f} F1 {bo['f1']:.3f} | hitter {r['hitter']} | stroke {r['stroke']['correct']}/{r['stroke']['total']} "
          f"| dead {r['deadBallEvents']}", flush=True)
    if "-v" in sys.argv:
        for d in r["details"]:
            print("   ", d["kind"], "missed", d["missed"], "false", d["false"])
