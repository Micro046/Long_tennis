#!/usr/bin/env python3
"""Validate a phone annotation file against its decoded presentation timestamps."""
import argparse
import hashlib
import json
from pathlib import Path

from benchmark.labels import validate_labels
from benchmark.video import probe_video


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    labels = json.loads(a.labels.read_text())
    video = probe_video(a.video)
    timing = {"sessionId":labels["sessionId"], "videoSha256":sha(a.video),
              "relativePresentationTimestampsUs":video["relativePresentationTimestampsUs"]}
    result = validate_labels(labels, timing, video["widthPx"], video["heightPx"])
    result["sessionId"] = labels["sessionId"]
    result["videoSha256"] = timing["videoSha256"]
    result["labelsSha256"] = sha(a.labels)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+"\n")
    print(a.output)


if __name__ == "__main__":main()
