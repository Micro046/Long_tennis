#!/usr/bin/env python3
"""Ball-track timeline for exhaustive event labelling.

  label_timeline.py SESSION_DIR OUT_PREFIX [--panel 300]

Per panel of `panel` frames: ball image-y (top) and image-x (bottom) per frame as dots, the players'
vertical extent (near = orange band, far = blue band, from pose + identity), the wrists of the player
nearest the ball (x marks), system hits (red lines) and bounces (green lines), scene cuts (black).
Hits show as turn-arounds of the ball close to a player's band; bounces as V-shapes away from players.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main(d, prefix, panel=300):
    d = Path(d)
    ball = json.loads((d / "ball.json").read_text())["frames"]
    pose = json.loads((d / "pose.json").read_text())
    rep = json.loads((d / "report.json").read_text())
    ident = rep["perFrameIdentity"]
    n = len(ball)
    bx = np.array([b[0] if b else np.nan for b in ball]); by = np.array([b[1] if b else np.nan for b in ball])
    bands = {"near": [], "far": []}
    for f, ppl in enumerate(pose["frames"][:n]):
        if not ppl or f >= len(ident):
            continue
        for pid, a in ident[f].items():
            p = ppl[a["det"]]
            bands[a["side"]].append((f, p[1], p[3], p[0], p[2]))
    hits, bounces = rep["events"]["hits"], rep["events"]["bounces"]
    cuts = pose.get("sceneCuts", [])
    pages = []
    per_fig = 3
    starts = list(range(0, n, panel))
    for k in range(0, len(starts), per_fig):
        fig, axs = plt.subplots(2 * per_fig, 1, figsize=(18, 4.2 * per_fig), gridspec_kw={"height_ratios": [2, 1] * per_fig})
        for j, a in enumerate(starts[k:k + per_fig]):
            b = min(n, a + panel)
            ay, ax = axs[2 * j], axs[2 * j + 1]
            for side, col in (("near", "#f0a060"), ("far", "#6090f0")):
                pts = [p for p in bands[side] if a <= p[0] < b]
                for f, y1, y2, x1, x2 in pts:
                    ay.plot([f, f], [y1, y2], color=col, alpha=0.35, lw=3)
                    ax.plot([f, f], [x1, x2], color=col, alpha=0.35, lw=3)
            t = np.arange(a, b)
            ay.scatter(t, by[a:b], s=7, color="k", zorder=3)
            ax.scatter(t, bx[a:b], s=7, color="k", zorder=3)
            for e in hits:
                if a <= e < b:
                    for axx in (ay, ax):
                        axx.axvline(e, color="r", alpha=.5)
                    ay.text(e, 10, f"H{e}", color="r", fontsize=7, rotation=90, va="top")
            for e in bounces:
                if a <= e < b:
                    for axx in (ay, ax):
                        axx.axvline(e, color="g", alpha=.5)
                    ay.text(e, 1070, f"b{e}", color="g", fontsize=7, rotation=90, va="bottom")
            for c in cuts:
                if a <= c < b:
                    for axx in (ay, ax):
                        axx.axvline(c, color="k", lw=2)
            ay.set_xlim(a, a + panel); ax.set_xlim(a, a + panel)
            ay.set_ylim(1080, 0); ax.set_ylim(0, 1920)
            ay.set_ylabel("ball y"); ax.set_ylabel("ball x")
            ay.set_xticks(range(a, a + panel + 1, 10), minor=True); ay.grid(True, which="both", alpha=.25)
            ax.set_xticks(range(a, a + panel + 1, 10), minor=True); ax.grid(True, which="both", alpha=.25)
        fig.tight_layout()
        out = f"{prefix}_{k // per_fig:02d}.png"
        fig.savefig(out, dpi=72); plt.close(fig)
        pages.append(out)
    print("\n".join(pages))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[sys.argv.index("--panel") + 1]) if "--panel" in sys.argv else 300)
