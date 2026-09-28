#!/usr/bin/env python3
"""Regression suite: run every benchmark in eval/benchmarks.json against the current engine and
write outputs/eval/<timestamp>.json + regenerate docs/RESULTS.md.

  python eval/run.py            # all
  python eval/run.py identity   # one kind
"""
import json
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


RUNNERS = {"identity": identity, "tracknet_events": tracknet_events}


def main(kinds):
    bench = json.loads((ROOT / "eval/benchmarks.json").read_text())
    results = []
    for b in bench:
        if kinds and b["kind"] not in kinds:
            continue
        if not all((ROOT / b[k]).exists() for k in ("video", "pose", "labels") if k in b):
            results.append({**b, "skipped": "inputs missing"}); continue
        t0 = time.time()
        results.append({**b, "result": RUNNERS[b["kind"]](b), "seconds": round(time.time() - t0)})
        print(b["name"], results[-1]["result"], flush=True)
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
        elif r["kind"] == "tracknet_events" and res:
            txt = f"hit F1 **{res['hit']['f1']}** (P {res['hit']['precision']}, R {res['hit']['recall']}); bounce F1 **{res['bounce']['f1']}** (P {res['bounce']['precision']}, R {res['bounce']['recall']})"
        else:
            txt = r.get("skipped", "")
        lines.append(f"| {r['name']} | {r['split']} | {r['kind']} | {txt} |")
    (ROOT / "docs/RESULTS.md").write_text("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
