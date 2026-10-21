#!/usr/bin/env python3
"""Start page for the results folder: one card per analysed video (annotated video + analysis page + key
numbers), for showing the work to others.

  results_index.py [RESULTS_DIR]        -> RESULTS_DIR/index.html
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DESCRIPTIONS = {
    "test1_swingvision_60s": "Indoor, phone behind the baseline (SwingVision-style view), 60 s.",
    "test2_indoor_90s": "Indoor club match, camera behind the baseline, edited between points, 90 s.",
    "test3_rally_30s": "Short rally uploaded from the phone app, 30 s.",
    "test4_outdoor_60fps_90s": "Outdoor, wide-angle low camera, 60 fps, 90 s (lens correction).",
    "match_60min": "Full hour of an unedited outdoor match (camera was moved during recording).",
}
ORDER = ["test1_swingvision_60s", "test2_indoor_90s", "test3_rally_30s", "test4_outdoor_60fps_90s", "match_60min"]


def card(d):
    rep = json.loads((d / "report.json").read_text())
    vids = sorted(d.glob("*_ANNOTATED.mp4"))
    P = rep["players"]
    dur = rep["video"]["frameCount"] / rep["fps"]
    rows = "".join(
        f"<tr><td>{html.escape(P[p]['name'])}</td><td>{P[p]['shots']}</td><td>{P[p]['strokes']['forehand']}</td>"
        f"<td>{P[p]['strokes']['backhand']}</td><td>{P[p]['strokes']['serve']}</td>"
        f"<td>{P[p]['speed']['medianAvgKmh'] or '–'}</td></tr>" for p in ("A", "B"))
    timing = ""
    if (d / "timing.json").exists():
        t = json.loads((d / "timing.json").read_text())
        tot = t.get("Total, start to report")
        if tot:
            timing = f"<p class=small>Processing on the Mac: {tot / 60:.0f} min for {dur / 60:.0f} min of video.</p>"
    v = vids[0].name if vids else None
    return f"""<div class=card><h2>{html.escape(d.name.replace('_', ' '))}</h2>
<p>{html.escape(DESCRIPTIONS.get(d.name, ''))}</p>
<p><b>{len(rep.get('points', []))}</b> points · <b>{len(rep['shots'])}</b> shots · <b>{len(rep['events']['bounces'])}</b> bounces · {dur / 60:.1f} min</p>
<table><tr><th>Player</th><th>Shots</th><th>FH</th><th>BH</th><th>Serve</th><th>Median km/h</th></tr>{rows}</table>
{timing}
<p>{f'<a href="{d.name}/{v}">▶ Annotated video</a> · ' if v else ''}<a href="{d.name}/analysis.html">Analysis page</a> · <a href="{d.name}/shots.csv">Shots (CSV)</a></p></div>"""


def main():
    res = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    dirs = [res / n for n in ORDER if (res / n / "report.json").exists()]
    dirs += [d for d in sorted(res.iterdir()) if d.is_dir() and d not in dirs and (d / "report.json").exists()]
    page = f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Long Tennis results</title><style>
body{{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1d2226;background:#fff}}
h1{{color:#1f6f5c}} .card{{border:1px solid #dde3e1;border-radius:8px;padding:12px 16px;margin:14px 0}} .card h2{{margin:0 0 4px;font-size:18px;text-transform:capitalize}}
table{{border-collapse:collapse;font-size:14px;margin:6px 0}} td,th{{border:1px solid #ddd;padding:3px 8px;text-align:left}} th{{background:#f3f6f5}}
.small{{font-size:12px;color:#555}} a{{color:#1f6f5c;font-weight:600}} .box{{background:#f3f6f5;border-left:4px solid #1f6f5c;padding:8px 12px}}</style></head><body>
<h1>Long Tennis — analysed videos</h1>
<div class=box>A phone behind the baseline records a singles match; the server finds the ball, both players, every hit and bounce,
and reports per player: shots, forehand/backhand/serve, where each shot landed, speed, points and serve stats.
In each annotated video: player boxes and names, the ball trail, every hit (player, stroke, landing, speed) and bounce, the point
number, and a mini-court that fills with landing spots.</div>
<p class=small>Measured accuracy on hand-labelled clips (test1–3): 81–100 % of hits and 92–100 % of bounces found, right player for 55 of 56 hits, stroke type right for 24 of 25.
On new outdoor footage the ball tracker loses the ball more often, so some shots and point endings are missing.</p>
{''.join(card(d) for d in dirs)}
</body></html>"""
    (res / "index.html").write_text(page)
    print("wrote", res / "index.html", len(dirs), "videos")


if __name__ == "__main__":
    main()
