"""Evaluation against independent event labels with explicit coverage windows."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def evaluate_events(predictions, labels, fps, tolerance_seconds=.12):
    """One-to-one matching; count recall only inside exhaustively labelled ranges.

    Reviewing suggested candidates is not an exhaustive label set and cannot
    establish recall. Exclude those files from accuracy measurement entirely.
    """
    if labels.get("annotationMode") != "independent_exhaustive":
        return {"status":"not_evaluable", "reason":"Independent exhaustive annotations required",
                "precision":None, "recall":None, "f1":None}
    windows = labels.get("coverageWindows", [])
    if not windows:
        raise ValueError("Exhaustive labels require coverageWindows")
    windows = sorted(windows, key=lambda w:w["startFrame"])
    for i,w in enumerate(windows):
        if w["startFrame"] < 0 or w["endFrame"] < w["startFrame"]:
            raise ValueError("Invalid coverage window")
        if i and w["startFrame"] <= windows[i-1]["endFrame"]:
            raise ValueError("Coverage windows overlap")
    inside = lambda f:any(w["startFrame"] <= f <= w["endFrame"] for w in windows)
    report = {}
    for kind in ("hit", "bounce"):
        gt = [e for e in labels["events"] if e["type"] == kind and inside(e["frameIndex"])]
        pred = [e for e in predictions if e["type"] == kind and inside(e["frameIndex"])
                and e.get("status") != "rejected"]
        # Unmatched cost dominates temporal error, maximizing cardinality first.
        cost = np.full((len(gt),len(pred)),1e6)
        for i,g in enumerate(gt):
            for j,p in enumerate(pred):
                if "timestampUs" in g or "timestampUs" in p:
                    if "timestampUs" not in g or "timestampUs" not in p:
                        raise ValueError("Both event sources need presentation timestamps")
                    gap = abs(g["timestampUs"]-p["timestampUs"])/1_000_000
                else:
                    gap = abs(g["frameIndex"]-p["frameIndex"])/fps
                if gap <= tolerance_seconds:
                    cost[i,j] = gap
        padded = np.concatenate([cost,np.full((len(gt),len(gt)),1000)],axis=1)
        rows,cols = linear_sum_assignment(padded)
        matches = [(int(i),int(j)) for i,j in zip(rows,cols) if j < len(pred) and cost[i,j] < 1000]
        tp = len(matches); fp=len(pred)-tp; fn=len(gt)-tp
        precision=tp/(tp+fp) if tp+fp else None
        recall=tp/(tp+fn) if tp+fn else None
        attribution = [(g,p) for i,j in matches for g,p in [(gt[i],pred[j])]
                       if kind=="hit" and g.get("playerId") in ("A","B")]
        report[kind] = {"truePositives":tp,"falsePositives":fp,"falseNegatives":fn,
            "precision":precision,"recall":recall,
            "f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "matchedTimingErrorsMs":[round(cost[i,j]*1000,3) for i,j in matches],
            "correctHitterAmongMatched":sum(g.get("playerId")==p.get("playerId") for g,p in attribution),
            "hitterDenominator":len(attribution)}
    return {"status":"evaluated", "coverageWindows":windows,
            "toleranceSeconds":tolerance_seconds,"metrics":report}
