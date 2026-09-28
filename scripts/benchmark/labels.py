"""Validate independently annotated phone sessions against decoded video frames."""
from __future__ import annotations


def validate_labels(labels, timing, width, height):
    if labels.get("schemaVersion") != "0.1" or labels.get("annotationMode") not in ("draft", "independent_exhaustive"):
        raise ValueError("Expected phone label schema 0.1 and a valid annotation mode")
    if labels["annotationMode"] == "independent_exhaustive" and labels.get("independentOfModelProposals") is not True:
        raise ValueError("Exhaustive labels must be produced independently of model proposals")
    if labels.get("sessionId") != timing.get("sessionId") or labels.get("videoSha256") != timing.get("videoSha256"):
        raise ValueError("Labels and decoded video sidecar do not match")
    pts = timing["relativePresentationTimestampsUs"]
    n = len(pts)
    windows = sorted(labels.get("coverageWindows", []), key=lambda w: w["startFrame"])
    if labels["annotationMode"] == "independent_exhaustive" and not windows:
        raise ValueError("Exhaustive annotation requires coverage windows")
    for i, window in enumerate(windows):
        start, end = window["startFrame"], window["endFrame"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < n:
            raise ValueError("Invalid coverage window")
        if i and start <= windows[i-1]["endFrame"]:
            raise ValueError("Coverage windows overlap")
    inside = lambda frame: any(w["startFrame"] <= frame <= w["endFrame"] for w in windows)
    event_ids = set()
    hits = {}
    bounce_links = []
    for event in labels.get("events", []):
        eid, frame = event.get("id"), event.get("frameIndex")
        if not isinstance(eid, str) or not eid or eid in event_ids:
            raise ValueError("Event IDs must be unique")
        event_ids.add(eid)
        if event.get("type") not in ("hit", "bounce") or type(frame) is not int or not 0 <= frame < n:
            raise ValueError(f"Invalid event frame or type: {eid}")
        if not inside(frame):
            raise ValueError(f"Event outside exhaustive coverage: {eid}")
        if type(event.get("timestampUs")) is not int or abs(event["timestampUs"]-pts[frame]) > 1000:
            raise ValueError(f"Event timestamp differs from decoded presentation time: {eid}")
        if event["type"] == "hit":
            if event.get("playerId") not in ("A", "B", "unknown"):
                raise ValueError(f"Hit needs A/B/unknown attribution: {eid}")
            if event.get("strokeClass") not in ("forehand", "backhand", "serve", "volley", "other", "unknown"):
                raise ValueError(f"Hit needs a stroke category: {eid}")
            hits[eid] = frame
        else:
            point = event.get("imagePx")
            if point is not None and (not isinstance(point, list) or len(point) != 2 or
                                      not all(isinstance(v, (int, float)) for v in point) or
                                      not 0 <= point[0] < width or not 0 <= point[1] < height):
                raise ValueError(f"Invalid bounce point: {eid}")
            if event.get("landingRegion") not in (None, "short_left", "short_centre", "short_right",
                                                   "deep_left", "deep_centre", "deep_right", "outside", "unknown"):
                raise ValueError(f"Invalid landing region: {eid}")
            link = event.get("firstLandingForHitId")
            if link is not None:
                bounce_links.append((eid, frame, link))
            if event.get("landingRegion") not in (None, "unknown") and link is None:
                raise ValueError(f"Landing region needs a linked hit: {eid}")
    linked_hits = set()
    for eid, frame, hit_id in bounce_links:
        if hit_id not in hits or hits[hit_id] >= frame or hit_id in linked_hits:
            raise ValueError(f"Invalid or duplicate first-landing link: {eid}")
        linked_hits.add(hit_id)
    seen_player_frames = set()
    for observation in labels.get("playerObservations", []):
        frame, player = observation.get("frameIndex"), observation.get("playerId")
        if type(frame) is not int or not 0 <= frame < n or player not in ("A", "B"):
            raise ValueError("Invalid player observation")
        if (frame, player) in seen_player_frames:
            raise ValueError("Duplicate player observation")
        seen_player_frames.add((frame, player))
        if observation.get("side") not in ("near", "far", "unknown"):
            raise ValueError("Invalid player side")
        if observation.get("visibility") not in ("visible", "occluded", "out_of_frame"):
            raise ValueError("Invalid player visibility")
        box = observation.get("bboxPx")
        if observation["visibility"] == "visible" and box is None:
            raise ValueError("Visible player needs an image box")
        if box is not None and (not isinstance(box, list) or len(box) != 4 or
                                not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
            raise ValueError("Invalid player box")
    for change in labels.get("changeovers", []):
        frame = change.get("frameIndex")
        if (type(frame) is not int or not 0 <= frame < n or change.get("playerId") not in ("A", "B") or
            change.get("sideBefore") not in ("near", "far") or
            change.get("sideAfter") not in ("near", "far") or
            change["sideBefore"] == change["sideAfter"]):
            raise ValueError("Invalid changeover annotation")
    return {"eventEvaluationReady": labels["annotationMode"] == "independent_exhaustive" and bool(windows),
            "identityEvaluationFrames": len({frame for frame, _ in seen_player_frames}),
            "annotatedPlayerSideChanges": len(labels.get("changeovers", [])),
            "eventCount": len(event_ids), "coverageWindowCount": len(windows),
            "note": "Coverage windows establish event recall only where every contact and bounce was independently checked."}
