#!/usr/bin/env python3
"""Create a decoded-frame timestamp sidecar for a new phone recording."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark.video import probe_video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--session-id", required=True)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    digest = hashlib.sha256()
    with a.video.open("rb") as file:
        for chunk in iter(lambda: file.read(1024*1024), b""):
            digest.update(chunk)
    report = {"sessionId":a.session_id, "videoSha256":digest.hexdigest(), **probe_video(a.video)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(a.output)


if __name__ == "__main__":main()
