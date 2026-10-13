"""Point outcomes and serve statistics from a session report (rally-v3 engine).

For every decoded point:
  serve      the first shot when it is a serve: in (lands in the receiver's service area), fault (lands
             elsewhere or in the net), or unknown (no landing seen); first or second serve (a second serve
             follows a fault by the same server within SECOND_SERVE_GAP_S).
  winner     from how the point ended -- out / net: the last hitter loses; double bounce on one side: the
             player on that side loses; double fault: the receiver wins; a first-serve fault is not a point.
             Points whose ending was not seen keep winner None.
Per player: points won, points won on own serve, first-serve %, double faults -- each with its denominator.
"""
from __future__ import annotations

from pipeline import court

SECOND_SERVE_GAP_S = 25.0
SERVICE_MARGIN_M = 0.15


def serve_result(shot):
    l = shot.get("landing", {})
    if l.get("status") == "net":
        return "fault"
    if not l.get("courtM") or shot.get("side") not in ("near", "far"):
        return "unknown"
    x, y = l["courtM"]
    m = SERVICE_MARGIN_M
    if not (court.SL - m <= x <= court.WIDTH - court.SL + m):
        return "fault"
    if shot["side"] == "near":      # receiver's service area: far half between service line and net
        ok = court.NET_Y - court.SERVICE - m <= y <= court.NET_Y + m
    else:
        ok = court.NET_Y - m <= y <= court.NET_Y + court.SERVICE + m
    return "in" if ok else "fault"


def annotate(report):
    fps = report["fps"]
    shots = sorted(report["shots"], key=lambda s: s["hitFrame"])
    other = {"A": "B", "B": "A"}
    pts = report.get("points", [])
    prev = None                                   # previous serve point: (server, result, start frame)
    for p in pts:
        ps = [s for s in shots if p["start"] - 2 <= s["hitFrame"] <= p["end"] + 2]
        p["shots"] = len(ps)
        p["serve"] = None
        if ps and ps[0]["stroke"] == "serve" and ps[0]["playerId"] in ("A", "B"):
            sv = ps[0]
            res = serve_result(sv)
            if res == "fault" and len(ps) >= 3:   # returned and played on: the players treated it as in
                res = "in"
            second = bool(prev and prev[0] == sv["playerId"] and prev[1] == "fault"
                          and (p["start"] - prev[2]) / fps <= SECOND_SERVE_GAP_S and not prev[3])
            p["serve"] = {"server": sv["playerId"], "result": res, "number": 2 if second else 1}
            prev = (sv["playerId"], res, p["start"], second)
        else:
            prev = None
        # winner
        p["winner"], p["winnerReason"] = None, None
        last = next((s for s in reversed(ps) if s["playerId"] in ("A", "B")), None)
        sv = p["serve"]
        if sv and sv["result"] == "fault" and len(ps) == 1:
            if sv["number"] == 2:
                p["winner"], p["winnerReason"] = other[sv["server"]], "double fault"
            else:
                p["winnerReason"] = "first-serve fault (no point)"
            continue
        why = p.get("endReason")
        if last is None or why is None:
            continue
        if why in ("out", "net"):
            p["winner"], p["winnerReason"] = other[last["playerId"]], f"{why} by {last['playerId']}"
        elif why == "double":
            p["winner"], p["winnerReason"] = last["playerId"], "not returned (double bounce)"
    # per-player statistics
    for pid in ("A", "B"):
        mine = [p for p in pts if p.get("serve") and p["serve"]["server"] == pid]
        first = [p for p in mine if p["serve"]["number"] == 1 and p["serve"]["result"] != "unknown"]
        decided = [p for p in pts if p.get("winner")]
        on_serve = [p for p in decided if p.get("serve") and p["serve"]["server"] == pid]
        report["players"][pid]["serveStats"] = {
            "serves": len(mine),
            "firstServesJudged": len(first),
            "firstServeInPercent": round(100 * sum(p["serve"]["result"] == "in" for p in first) / len(first), 1) if first else None,
            "doubleFaults": sum(1 for p in pts if p.get("winnerReason") == "double fault" and p["serve"]["server"] == pid),
        }
        report["players"][pid]["pointsWon"] = {
            "won": sum(1 for p in decided if p["winner"] == pid),
            "decidedPoints": len(decided),
            "wonOnOwnServe": sum(1 for p in on_serve if p["winner"] == pid),
            "decidedOnOwnServe": len(on_serve),
            "definition": "only points whose ending (out, net, double bounce, double fault) was detected",
        }
    report["pointsSummary"] = {"points": len(pts), "decided": sum(1 for p in pts if p.get("winner")),
                               "firstServeFaults": sum(1 for p in pts if p.get("winnerReason") == "first-serve fault (no point)")}
    return report
