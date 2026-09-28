#!/usr/bin/env python3
"""Evaluate a session report on the amateur fixed-camera match against independent labels.

Identity:  labels_identity_*.json gives, at sampled frames, the court half of the red- and
           grey-shirted players (or null = not in frame). We map A/B -> red/grey using the seed
           (A = near player at the seed frame) and score each labelled (player, frame).
Events:    labels_events_*.json gives exhaustive windows with hits (player, stroke) and bounces.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def identity_eval(report, labels):
    per = report["perFrameIdentity"]
    seed = report["identity"]["seedFrame"]
    # which shirt is A? the player labelled near at the labelled frame closest to the seed frame
    lab = sorted(labels["labels"], key=lambda l: abs(l["frameIndex"] - seed))[0]
    a_side_at_seed = per[seed]["A"]["side"]
    colour_of_a = "red" if lab["red"] == a_side_at_seed else "grey"
    colour = {"A": colour_of_a, "B": "grey" if colour_of_a == "red" else "red"}
    res = {"correct": 0, "wrong": 0, "missed": 0, "falsePresent": 0, "items": []}
    for l in labels["labels"]:
        f = l["frameIndex"]
        # use the nearest frame with a pose result (pose runs on every 3rd frame)
        cands = [g for g in range(f - 2, f + 3) if 0 <= g < len(per) and per[g]]
        g = min(cands, key=lambda x: abs(x - f)) if cands else f
        for pid in ("A", "B"):
            truth = l[colour[pid]]
            pred = per[g].get(pid, {}).get("side") if g < len(per) else None
            if truth is None and pred is None:
                continue
            if truth is None:
                res["falsePresent"] += 1; outcome = "false_present"
            elif pred is None:
                res["missed"] += 1; outcome = "missed"
            elif pred == truth:
                res["correct"] += 1; outcome = "correct"
            else:
                res["wrong"] += 1; outcome = "wrong"
            res["items"].append({"frame": f, "player": pid, "truth": truth, "pred": pred, "outcome": outcome})
    n = res["correct"] + res["wrong"] + res["missed"]
    res["accuracyOfVisible"] = round(res["correct"] / max(1, n), 3)
    res["wrongRate"] = round(res["wrong"] / max(1, n), 3)
    res["colourOfA"] = colour_of_a
    return res


def match(pred, truth, tol):
    pairs = sorted((abs(p - t), p, t) for p in pred for t in truth if abs(p - t) <= tol)
    up, ut, m = set(), set(), {}
    for d, p, t in pairs:
        if p in up or t in ut:
            continue
        up.add(p); ut.add(t); m[t] = p
    return m


def events_eval(report, labels, tol=4):
    shots = {s["hitFrame"]: s for s in report["shots"]}
    colour_to_pid = None
    out = {"hits": {"tp": 0, "fp": 0, "fn": 0}, "bounces": {"tp": 0, "fp": 0, "fn": 0},
           "hitter": {"correct": 0, "wrong": 0, "unknown": 0}, "stroke": {"confusion": {}}, "windows": []}
    for w in labels["windows"]:
        a, b = w["start"], w["end"]
        ph = [h for h in report["events"]["hits"] if a <= h < b]
        pb = [x for x in report["events"]["bounces"] if a <= x < b]
        th = [h["frameIndex"] for h in w["hits"]]
        tb = [x["frameIndex"] for x in w["bounces"]]
        mh, mb = match(ph, th, tol), match(pb, tb, tol)
        out["hits"]["tp"] += len(mh); out["hits"]["fp"] += len(ph) - len(mh); out["hits"]["fn"] += len(th) - len(mh)
        out["bounces"]["tp"] += len(mb); out["bounces"]["fp"] += len(pb) - len(mb); out["bounces"]["fn"] += len(tb) - len(mb)
        for h in w["hits"]:
            p = mh.get(h["frameIndex"])
            if p is None:
                continue
            s = shots.get(p)
            pred_colour = w.get("colourOfA_", None)
            if s is None:
                continue
            pc = labels["pidToColour"].get(s["playerId"]) if "pidToColour" in labels else None
            key = "unknown" if s["playerId"] == "unknown" else ("correct" if pc == h["player"] else "wrong")
            out["hitter"][key] += 1
            cm = out["stroke"]["confusion"].setdefault(h["stroke"], {})
            cm[s["stroke"]] = cm.get(s["stroke"], 0) + 1
        out["windows"].append({"start": a, "end": b, "predHits": ph, "trueHits": th, "predBounces": pb, "trueBounces": tb})
    for k in ("hits", "bounces"):
        d = out[k]; p = d["tp"] / max(1, d["tp"] + d["fp"]); r = d["tp"] / max(1, d["tp"] + d["fn"])
        d.update(precision=round(p, 3), recall=round(r, 3), f1=round(2 * p * r / max(1e-9, p + r), 3))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--identity-labels", default=str(ROOT / "data/public/youtube_amateur/labels_identity_4apx6gd5Uxs.json"))
    ap.add_argument("--event-labels")
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = json.loads(Path(a.report).read_text())
    res = {"identity": identity_eval(rep, json.loads(Path(a.identity_labels).read_text()))}
    if a.event_labels:
        el = json.loads(Path(a.event_labels).read_text())
        el["pidToColour"] = {"A": res["identity"]["colourOfA"], "B": "grey" if res["identity"]["colourOfA"] == "red" else "red"}
        res["events"] = events_eval(rep, el)
    summary = {k: {kk: vv for kk, vv in v.items() if kk not in ("items", "windows")} for k, v in res.items()}
    print(json.dumps(summary, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
