#!/usr/bin/env python3
"""Readable analysis report for one analysed session: an HTML page (with charts) + a CSV of every shot.

  session_summary.py SESSION_DIR OUT_DIR [--timing timing.json]

timing.json (optional): {"stage": seconds, ...} measured by the caller, shown in the report.
"""
import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import court  # noqa: E402

PCOL = {"A": "#3c8dde", "B": "#e0703c", "unknown": "#999999"}


def b64(path):
    import base64
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def landing_plot(rep, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 7.2))
    for ax, pid in zip(axes, ("A", "B")):
        for a, b in court.LINES.values():
            ax.plot([a[0], b[0]], [a[1], b[1]], color="white", lw=1.2)
        ax.plot([-0.9, court.WIDTH + 0.9], [court.NET_Y] * 2, color="#222", lw=2.5)
        ax.set_facecolor("#3f7a52")
        for s in rep["shots"]:
            l = s["landing"]
            if s["playerId"] != pid or not l.get("courtM"):
                continue
            x, y = l["courtM"]
            c = PCOL[pid] if l.get("region") else ("#ff3030" if l.get("status") == "out" else "#dddddd")
            ax.scatter([x], [y], s=26, color=c, edgecolor="black", lw=0.4, zorder=3)
        ax.set_xlim(-3, court.WIDTH + 3); ax.set_ylim(court.LENGTH + 4, -4)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{rep['players'][pid]['name']}: where shots landed\n(red = out)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def timeline_plot(rep, fps, out):
    n = rep["video"]["frameCount"]
    fig, ax = plt.subplots(figsize=(11, 2.2))
    for p in rep.get("points", []):
        ax.axvspan(p["start"] / fps / 60, p["end"] / fps / 60, color="#9cc9a8", alpha=0.8, lw=0)
    for s in rep["shots"]:
        ax.plot([s["hitFrame"] / fps / 60] * 2, [0.55 if s["playerId"] == "A" else 0.05, 0.95 if s["playerId"] == "A" else 0.45],
                color=PCOL.get(s["playerId"], "#999"), lw=0.8)
    ax.set_xlim(0, n / fps / 60); ax.set_ylim(0, 1); ax.set_yticks([0.25, 0.75])
    ax.set_yticklabels([rep["players"]["B"]["name"], rep["players"]["A"]["name"]])
    ax.set_xlabel("minutes (green = detected points, lines = shots)")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("out"); ap.add_argument("--timing")
    a = ap.parse_args()
    d, out = Path(a.session), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((d / "report.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    cal = json.loads((d / "calibration.json").read_text())
    fps = rep["fps"]
    dur = rep["video"]["frameCount"] / fps
    timing = json.loads(Path(a.timing).read_text()) if a.timing else {}
    landing_plot(rep, out / "landing.png")
    timeline_plot(rep, fps, out / "timeline.png")
    # CSV of shots
    with open(out / "shots.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time", "frame", "player", "side", "stroke", "landing", "landing_x_m", "landing_y_m", "speed_kmh"])
        for s in rep["shots"]:
            l = s["landing"]
            w.writerow([mmss(s["timeS"]), s["hitFrame"], rep["players"].get(s["playerId"], {}).get("name", "unknown"),
                        s.get("side"), s["stroke"], l.get("region") or l.get("status"),
                        *(l.get("courtM") or ["", ""]), (s.get("speed") or {}).get("avgKmh", "")])
    P = rep["players"]
    ends = Counter(p.get("endReason") or "not detected" for p in rep.get("points", []))
    unknown_shots = sum(1 for s in rep["shots"] if s["playerId"] not in ("A", "B"))
    rows = []
    for pid in ("A", "B"):
        p = P[pid]
        st = p["strokes"]; ld = p["landing"]; sp = p["speed"]
        reg = " · ".join(f"{k.replace('_', ' ')} {v['count']} ({v['percent'] or 0:.0f}%)" for k, v in ld["regions"].items())
        rows.append(f"""<tr><td><b style="color:{PCOL[pid]}">{html.escape(p['name'])}</b></td><td>{p['shots']}</td>
<td>{st['serve']}</td><td>{st['forehand']}</td><td>{st['backhand']}</td><td>{st['volley']}</td><td>{st['overhead']}</td><td>{st['unknown']}</td>
<td>{ld['inCourtFirstBounces']}</td><td>{ld['out']}</td><td>{ld['noObservedBounce']}</td>
<td>{sp['medianAvgKmh'] or '–'}</td><td>{sp['maxAvgKmh'] or '–'}</td></tr><tr><td></td><td colspan=12 class=small>{reg}</td></tr>""")
    def pct(v):
        return "–" if v is None else f"{v:.0f}%"
    orows = "".join(
        f"<tr><td><b style='color:{PCOL[pid]}'>{html.escape(P[pid]['name'])}</b></td>"
        f"<td>{P[pid].get('pointsWon', {}).get('won', '–')} / {P[pid].get('pointsWon', {}).get('decidedPoints', '–')}</td>"
        f"<td>{P[pid].get('pointsWon', {}).get('wonOnOwnServe', '–')} / {P[pid].get('pointsWon', {}).get('decidedOnOwnServe', '–')}</td>"
        f"<td>{P[pid].get('serveStats', {}).get('serves', '–')}</td><td>{pct(P[pid].get('serveStats', {}).get('firstServeInPercent'))}</td>"
        f"<td>{P[pid].get('serveStats', {}).get('doubleFaults', '–')}</td></tr>" for pid in ("A", "B")) if "pointsWon" in P["A"] else ""
    trows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v / 60:.1f} min</td><td>{v / dur:.2f}× video length</td></tr>"
                    for k, v in timing.items() if isinstance(v, (int, float)))
    shot_rows = "".join(
        f"<tr><td>{mmss(s['timeS'])}</td><td style='color:{PCOL.get(s['playerId'], '#999')}'>{html.escape(P.get(s['playerId'], {}).get('name', 'unknown'))}</td>"
        f"<td>{s['stroke']}</td><td>{s['landing'].get('region') or s['landing'].get('status')}</td>"
        f"<td>{(s.get('speed') or {}).get('avgKmh', '')}</td></tr>" for s in rep["shots"])
    page = f"""<!doctype html><html><head><meta charset=utf-8><title>Match analysis</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1d2226}}
h1{{color:#1f6f5c}} table{{border-collapse:collapse;width:100%;margin:8px 0 18px;font-size:14px}} td,th{{border:1px solid #ddd;padding:4px 7px;text-align:left}}
th{{background:#f3f6f5}} .small{{font-size:12px;color:#555}} img{{max-width:100%;border:1px solid #ddd}} .box{{background:#f3f6f5;padding:8px 12px;border-left:4px solid #1f6f5c;margin:10px 0}}
details summary{{cursor:pointer;color:#1f6f5c}}</style></head><body>
<h1>Match analysis</h1>
<p><b>{html.escape(meta['video'].get('filename') or '')}</b> · {mmss(dur)} min · {rep['video']['widthPx']}×{rep['video']['heightPx']} @ {fps:.0f} fps ·
engine {rep.get('engine', '')} · lens correction {('k1 = ' + str(cal['distortion']['k1'])) if cal.get('distortion') else 'none'}</p>
<div class=box><b>Detected:</b> {len(rep.get('points', []))} points, {len(rep['shots'])} shots ({unknown_shots} without a player), {len(rep['events']['bounces'])} bounces.
Point endings: {', '.join(f'{k} {v}' for k, v in ends.items())}.</div>
{('<h2>Processing time</h2><table><tr><th>Stage</th><th>Time</th><th>Relative</th></tr>' + trows + '</table>') if trows else ''}
<h2>Per player</h2>
<table><tr><th>Player</th><th>Shots</th><th>Serve</th><th>FH</th><th>BH</th><th>Volley</th><th>Overhead</th><th>Unknown</th>
<th>Landed in</th><th>Out</th><th>No bounce seen</th><th>Median km/h</th><th>Max km/h</th></tr>{''.join(rows)}</table>
<p class=small>Speed = average speed from the hitter to the first bounce (lower than a radar gun). Landing regions are in the opponent's singles half, seen from the hitter.</p>
{('<h2>Points and serve</h2><table><tr><th>Player</th><th>Points won / decided</th><th>Won on own serve</th><th>Serves</th><th>1st serve in</th><th>Double faults</th></tr>' + orows + '</table><p class=small>Only points whose ending (out, net, double bounce, double fault) was seen are decided; the rest are left open rather than guessed.</p>') if orows else ''}
<h2>Where the shots landed</h2><img src="{b64(out / 'landing.png')}">
<h2>Timeline</h2><img src="{b64(out / 'timeline.png')}">
<h2>All shots</h2><details><summary>{len(rep['shots'])} shots (also in shots.csv)</summary>
<table><tr><th>Time</th><th>Player</th><th>Stroke</th><th>Landing</th><th>km/h</th></tr>{shot_rows}</table></details>
</body></html>"""
    (out / "analysis.html").write_text(page)
    (out / "report.json").write_text(json.dumps({k: v for k, v in rep.items() if k != "perFrameIdentity"}, indent=1))
    print("wrote", out / "analysis.html")


if __name__ == "__main__":
    main()
