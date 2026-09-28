#!/usr/bin/env python3
"""Regression suite: run every benchmark in eval/benchmarks.json against the current engine and
write outputs/eval/<timestamp>.json + regenerate docs/RESULTS.md.

  python eval/run.py            # all
  python eval/run.py identity   # one kind
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "scripts"))
PY = sys.executable


def identity(b):
    from pipeline import identity as idm, session as S
    import eval_amateur as EA
    pose = json.loads((ROOT / b["pose"]).read_text())
    frames = [S.person_list(f) if i % 3 == 0 else None for i, f in enumerate(pose["frames"])]
    S.compute_appearance(str(ROOT / b["video"]), frames, stride=3)
    ident, tracks, seed = idm.assign_identities(frames, pose.get("sceneCuts", []), pose["video"]["fps"])
    rep = {"identity": {"seedFrame": seed["seedFrame"]},
           "perFrameIdentity": [{k: {"det": v["det"], "side": v["side"]} for k, v in f.items()} for f in ident]}
    r = EA.identity_eval(rep, json.loads((ROOT / b["labels"]).read_text()))
    return {k: r[k] for k in ("correct", "wrong", "missed", "falsePresent", "accuracyOfVisible", "wrongRate")}


def tracknet_events(b):
    out = subprocess.run([PY, str(ROOT / "scripts/eval_tracknet_events.py"), "--source", b.get("source", "pred")],
                         capture_output=True, text=True, cwd=ROOT).stdout
    d = json.loads(out[out.index("{"):])
    return d["byThreshold"]["0.5"]


def events(b):
    """Full session analysis, then exhaustive-window scoring of hits/bounces/hitter/stroke."""
    from pipeline import session as S
    import eval_amateur as EA
    out = ROOT / "outputs/eval/tmp" / (Path(b["video"]).stem + "_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rep = S.run(str(ROOT / b["video"]), str(ROOT / b["ball"]), str(ROOT / b["pose"]), str(ROOT / b["calibration"]),
                os.environ.get("LT_EVENT_MODEL", str(ROOT / "models/event_model.pkl")), str(out), {"A": "A", "B": "B"})
    ident = EA.identity_eval(rep, json.loads((ROOT / b["identityLabels"]).read_text()))
    lab = json.loads((ROOT / b["labels"]).read_text())
    lab["pidToColour"] = {"A": ident["colourOfA"], "B": "grey" if ident["colourOfA"] == "red" else "red"}
    r = EA.events_eval(rep, lab)
    return {k: r[k] for k in ("hits", "bounces", "hitter", "stroke")} | {"details": r["details"]}


RUNNERS = {"events": events, "identity": identity, "tracknet_events": tracknet_events}


def main(kinds):
    bench = json.loads((ROOT / "eval/benchmarks.json").read_text())
    results = []
    for b in bench:
        if kinds and b["kind"] not in kinds:
            continue
        if not all((ROOT / b[k]).exists() for k in ("video", "pose", "labels", "ball", "calibration") if k in b):
            results.append({**b, "skipped": "inputs missing"}); continue
        t0 = time.time()
        results.append({**b, "result": RUNNERS[b["kind"]](b), "seconds": round(time.time() - t0)})
        print(b["name"], {k: v for k, v in results[-1]["result"].items() if k != "details"}, flush=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = ROOT / "outputs/eval" / f"{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"commit": commit, "results": results}, indent=1))
    lines = ["# Regression results", "", f"Generated {stamp} at commit `{commit}` by `eval/run.py`.", "",
             "| Benchmark | Split | Kind | Result |", "|---|---|---|---|"]
    for r in results:
        res = r.get("result")
        if r["kind"] == "identity" and res:
            txt = f"{res['correct']} correct / {res['wrong']} wrong / {res['missed']} missed -> **{res['accuracyOfVisible']:.1%}**"
        elif r["kind"] == "events" and res:
            h, bo, ht, st = res["hits"], res["bounces"], res["hitter"], res["stroke"]
            txt = (f"hits P {h['precision']} R {h['recall']} (**F1 {h['f1']}**); bounces P {bo['precision']} R {bo['recall']} "
                   f"(**F1 {bo['f1']}**); hitter {ht['correct']}/{ht['correct'] + ht['wrong'] + ht['unknown']}; "
                   f"stroke {st['correct']}/{st['total']}")
        elif r["kind"] == "tracknet_events" and res:
            txt = f"hit F1 **{res['hit']['f1']}** (P {res['hit']['precision']}, R {res['hit']['recall']}); bounce F1 **{res['bounce']['f1']}** (P {res['bounce']['precision']}, R {res['bounce']['recall']})"
        else:
            txt = r.get("skipped", "")
        lines.append(f"| {r['name']} | {r['split']} | {r['kind']} | {txt} |")
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
