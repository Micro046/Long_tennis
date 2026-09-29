#!/usr/bin/env python3
"""Frame-by-frame review of one analysed session.

  review_pdf.py SESSION_DIR sheets OUT_DIR      # event strips + 3-s ball-path plots (for reviewing)
  review_pdf.py SESSION_DIR pdf VERDICTS.json OUT.pdf

Each event (hit / bounce) gets a strip of raw frames around it (-6..+6 frames, crop centred on the
tracked ball, predicted ball marked with a thin circle) and a small full-frame context with the ball
path. The PDF adds the system's output and the reviewer's verdict for every event.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OFFS = (-6, -3, 0, 3, 6)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def load(d):
    d = Path(d)
    rep = json.loads((d / "report.json").read_text())
    ball = json.loads((d / "ball.json").read_text())["frames"]
    meta = json.loads((d / "meta.json").read_text())
    return d, rep, ball, meta


def events(rep, meta):
    names = {"A": meta["nameA"], "B": meta["nameB"]}
    shots = {s["hitFrame"]: s for s in rep["shots"]}
    ev = []
    for h in rep["events"]["hits"]:
        s = shots.get(h)
        if s is None:
            continue
        l = s["landing"]
        ev.append({"frame": h, "kind": "hit", "player": names.get(s["playerId"], "unknown"), "side": s.get("side"),
                   "stroke": s["stroke"], "landing": l.get("region") or l.get("status"),
                   "speed": (s.get("speed") or {}).get("avgKmh")})
    for b in rep["events"]["bounces"]:
        land = [s for s in rep["shots"] if s["landing"].get("bounceFrame") == b]
        ev.append({"frame": b, "kind": "bounce",
                   "landing": (land[0]["landing"].get("region") or land[0]["landing"].get("status")) if land else "not a first bounce",
                   "courtM": land[0]["landing"].get("courtM") if land else None})
    return sorted(ev, key=lambda e: (e["frame"], e["kind"]))


def read(cap, f):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    ok, img = cap.read()
    return img if ok else None


def crop(img, c, w=400, h=300):
    H, W = img.shape[:2]
    x1 = int(np.clip(c[0] - w / 2, 0, W - w)); y1 = int(np.clip(c[1] - h / 2, 0, H - h))
    return img[y1:y1 + h, x1:x1 + w].copy(), (x1, y1)


def strip(cap, ball, f, fps):
    """Crops around the ball for OFFS frames + a context thumbnail with the ball path."""
    centre = None
    for k in [f] + [f + o for o in (1, -1, 2, -2, 3, -3, 4, -4, 6, -6)]:
        if 0 <= k < len(ball) and ball[k]:
            centre = ball[k][:2]; break
    tiles = []
    for o in OFFS:
        g = f + o
        img = read(cap, g)
        if img is None:
            tiles.append(np.zeros((300, 400, 3), np.uint8)); continue
        c = centre or (img.shape[1] / 2, img.shape[0] / 2)
        t, (x1, y1) = crop(img, c)
        if 0 <= g < len(ball) and ball[g]:
            bx, by = int(ball[g][0] - x1), int(ball[g][1] - y1)
            cv2.circle(t, (bx, by), 14, (255, 0, 255), 1)
        cv2.rectangle(t, (0, 0), (150, 28), (0, 0, 0), -1)
        cv2.putText(t, f"{'EVENT' if o == 0 else f'{o:+d}'}  f{g}", (4, 20), 0, 0.55, (0, 255, 255) if o == 0 else (255, 255, 255), 1)
        if o == 0:
            cv2.rectangle(t, (0, 0), (399, 299), (0, 255, 255), 3)
        tiles.append(t)
    return np.hstack(tiles + [cv2.resize(context(cap, ball, f), (533, 300))])


def context(cap, ball, f):
    """Full frame at f with the ball path of +-1 s (blue before, yellow at, red after)."""
    ctx = read(cap, f)
    for k in range(f - 30, f + 31):
        if 0 <= k < len(ball) and ball[k]:
            col = (255, 200, 0) if k < f else (0, 0, 255) if k > f else (0, 255, 255)
            cv2.circle(ctx, (int(ball[k][0]), int(ball[k][1])), 7 if k == f else 4, col, -1)
    return ctx


def paths(cap, ball, a, b, out):
    img = read(cap, (a + b) // 2)
    img = cv2.addWeighted(img, 0.6, np.zeros_like(img), 0.4, 0)
    pts = [(f, ball[f]) for f in range(a, b) if f < len(ball) and ball[f]]
    for (f1, p1), (f2, p2) in zip(pts, pts[1:]):
        if f2 - f1 <= 3:
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 200, 255), 1)
    for f, p in pts:
        c = (0, 255, 0) if f - a < (b - a) / 3 else (0, 255, 255) if f - a < 2 * (b - a) / 3 else (255, 100, 255)
        cv2.circle(img, (int(p[0]), int(p[1])), 4, c, -1)
        if f % 3 == 0:
            cv2.putText(img, str(f), (int(p[0]) + 5, int(p[1]) - 5), 0, .45, (255, 255, 255), 1)
    cv2.putText(img, f"frames {a}-{b - 1}", (10, 40), 0, 1, (255, 255, 255), 2)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def sheets(d, out):
    d, rep, ball, meta = load(d)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    fps = meta["video"]["fps"]
    ev = events(rep, meta)
    (out / "events.json").write_text(json.dumps(ev, indent=1))
    rows = []
    for i, e in enumerate(ev):
        s = strip(cap, ball, e["frame"], fps)
        lab = f"#{i + 1} {e['kind'].upper()} f{e['frame']} t={e['frame'] / fps:.2f}s  " + (
            f"{e['player']} {e['stroke']} -> {e['landing']}" if e["kind"] == "hit" else f"{e['landing']}")
        bar = np.zeros((34, s.shape[1], 3), np.uint8)
        cv2.putText(bar, lab, (6, 24), 0, 0.7, (0, 255, 255), 2)
        s = np.vstack([bar, s]); cv2.imwrite(str(out / f"event_{i:02d}.jpg"), s, [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append(s)
    for k in range(0, len(rows), 3):
        g = rows[k:k + 3]
        cv2.imwrite(str(out / f"sheet_{k // 3:02d}.jpg"), cv2.resize(np.vstack(g), None, fx=0.75, fy=0.75), [cv2.IMWRITE_JPEG_QUALITY, 85])
    n = len(ball); step = int(round(3 * fps))
    for a in range(0, n, step):
        paths(cap, ball, a, min(n, a + step), out / f"path_{a:05d}.jpg")
    print(len(ev), "events;", len(rows), "strips")


# ----------------------------------------------------------------------------- PDF
def fnt(size, bold=False):
    try:
        return ImageFont.truetype(FONT_B if bold else FONT, size)
    except Exception:
        return ImageFont.load_default()


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) > width and cur:
            lines.append(cur); cur = w
        else:
            cur = t
    return lines + ([cur] if cur else [])


def text_page(title, paragraphs, W=2480, H=1754):
    page = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(page)
    dr.text((100, 80), title, fill="black", font=fnt(64, True))
    y = 200
    for p in paragraphs:
        if isinstance(p, (tuple, list)):  # table rows (JSON turns tuples into lists)
            for row in p:
                x = 100
                for cell, w in row:
                    dr.text((x, y), str(cell), fill="black", font=fnt(30)); x += w
                y += 44
            y += 20; continue
        for line in wrap(dr, p, fnt(34), W - 200):
            dr.text((100, y), line, fill="black", font=fnt(34)); y += 46
        y += 18
    return page


COL = {"correct": (30, 140, 60), "wrong": (200, 40, 40), "partly": (220, 130, 0), "unsure": (120, 120, 120),
       "missed": (140, 60, 180)}


def fit(im, w, h):
    s = min(w / im.width, h / im.height)
    return im.resize((int(im.width * s), int(im.height * s)))


def event_page(img_path, head, sysline, verdict, reason, ctx=None, W=2480, H=1754):
    """Strip across the top; below it the full frame (left) and the verdict (right). Without a
    full frame (missed-event grids) the image takes the page and the verdict goes underneath."""
    page = Image.new("RGB", (W, H), "white"); dr = ImageDraw.Draw(page)
    dr.text((80, 50), head, fill="black", font=fnt(54, True))
    dr.text((80, 125), sysline, fill=(40, 40, 40), font=fnt(36))
    im = Image.open(img_path).convert("RGB")
    if ctx is not None:
        im = fit(im, W - 160, 600); page.paste(im, (80, 190))
        y0 = 210 + im.height
        full = fit(Image.fromarray(cv2.cvtColor(ctx, cv2.COLOR_BGR2RGB)), 1500, H - y0 - 110)
        page.paste(full, (80, y0))
        x, y, tw = 80 + full.width + 50, y0, W - (80 + full.width + 50) - 80
    else:
        im = fit(im, W - 160, H - 640); page.paste(im, (80, 190))
        x, y, tw = 80, 210 + im.height, W - 160
    c = COL.get(verdict.split()[0], (0, 0, 0))
    dr.rectangle([x, y, x + 420, y + 70], fill=c)
    dr.text((x + 20, y + 12), verdict.upper(), fill="white", font=fnt(40, True))
    y += 100
    for line in wrap(dr, reason, fnt(36), tw):
        dr.text((x, y), line, fill="black", font=fnt(36)); y += 50
    note = ("Top: frames -6, -3, EVENT, +3, +6 zoomed on the ball (magenta circle = where the tracker put the ball). "
            "Below: the full frame at the event with the ball path of +-1 s (blue = before, yellow = event, red = after)."
            if ctx is not None else "Frames around the event (frame number top-left), magenta circle = where the tracker put the ball.")
    dr.text((80, H - 70), note, fill=(90, 90, 90), font=fnt(26))
    return page


def pdf(d, verdicts_path, out):
    d, rep, ball, meta = load(d)
    V = json.loads(Path(verdicts_path).read_text())
    sheets_dir = Path(V["sheetsDir"])
    ev = json.loads((sheets_dir / "events.json").read_text())
    fps = meta["video"]["fps"]
    cap = cv2.VideoCapture(str(d / "video.mp4"))
    pages = [text_page(V["title"], V["intro"])]
    pages.append(text_page("Summary", V["summary"]))
    for i, e in enumerate(ev):
        v = V["events"].get(str(i), {"verdict": "unsure", "reason": "not reviewed"})
        head = f"Event {i + 1}/{len(ev)}  -  {e['kind'].upper()} at {e['frame'] / fps:.2f} s (frame {e['frame']})"
        sysl = ("System: " + (f"{e['player']} hit, stroke = {e['stroke']}, first bounce = {e['landing']}"
                              + (f", speed {e['speed']} km/h" if e.get('speed') else "")
                              if e["kind"] == "hit" else f"bounce, landing = {e['landing']}"))
        pages.append(event_page(sheets_dir / f"event_{i:02d}.jpg", head, sysl, v["verdict"], v["reason"],
                                ctx=context(cap, ball, e["frame"])))
    for m in V.get("missed", []):
        pages.append(event_page(sheets_dir / m["image"], f"MISSED {m['kind'].upper()} at {m['frame'] / fps:.2f} s (frame {m['frame']})",
                                "System: nothing detected here", "missed", m["reason"]))
    pages.append(text_page("Root causes and what to fix", V["causes"]))
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=200)
    print("wrote", out, len(pages), "pages")


if __name__ == "__main__":
    if sys.argv[2] == "sheets":
        sheets(sys.argv[1], sys.argv[3])
    else:
        pdf(sys.argv[1], sys.argv[3], sys.argv[4])
