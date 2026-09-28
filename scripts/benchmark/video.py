"""Video inspection using decoded presentation timestamps, including variable FPS."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from statistics import median

import av


def summarize_timestamps(timestamps_us):
    if not timestamps_us:
        raise ValueError("Video has no decoded frames")
    if any(b <= a for a, b in zip(timestamps_us, timestamps_us[1:])):
        raise ValueError("Decoded video timestamps must be strictly increasing")
    gaps = [b-a for a, b in zip(timestamps_us, timestamps_us[1:])]
    if not gaps:
        return {"medianFrameIntervalUs": None, "minFrameIntervalUs": None,
                "maxFrameIntervalUs": None, "irregularIntervalCount": 0,
                "variableFrameRateObserved": None}
    typical = median(gaps)
    tolerance = max(2000, .05*typical)
    irregular = sum(abs(gap-typical) > tolerance for gap in gaps)
    return {"medianFrameIntervalUs": typical, "minFrameIntervalUs": min(gaps),
            "maxFrameIntervalUs": max(gaps), "irregularIntervalCount": irregular,
            "variableFrameRateObserved": irregular > 0}


def probe_video(path):
    path = Path(path)
    with av.open(str(path)) as container:
        videos = container.streams.video
        if len(videos) != 1:
            raise ValueError(f"Expected one video track in {path}; found {len(videos)}")
        stream = videos[0]
        timestamps = []
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Missing presentation timestamp in {path}")
            source_us = round(Fraction(frame.pts) * frame.time_base * 1_000_000)
            timestamps.append(source_us)
        timing = summarize_timestamps(timestamps)
        first = timestamps[0]
        return {"widthPx": stream.width, "heightPx": stream.height,
                "frameCount": len(timestamps), "nominalFps": float(stream.average_rate) if stream.average_rate else None,
                "timeBase": str(stream.time_base), "sourceFirstTimestampUs": first,
                "sourceLastTimestampUs": timestamps[-1],
                "relativePresentationTimestampsUs": [t-first for t in timestamps],
                "decodedDurationSeconds": round((timestamps[-1]-first)/1_000_000, 6),
                **timing}
