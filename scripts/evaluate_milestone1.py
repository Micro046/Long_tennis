#!/usr/bin/env python3
"""Evaluate event candidates without treating reviewed suggestions as truth."""
import argparse
from pathlib import Path
from milestone1.core import read_json, write_json, file_hash
from milestone1.evaluation import evaluate_events

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events",required=True,type=Path)
    p.add_argument("--labels",required=True,type=Path)
    p.add_argument("--fps",required=True,type=float)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    events,labels=read_json(a.events),read_json(a.labels)
    if events["sessionId"] != labels["sessionId"]:
        raise ValueError("Session mismatch")
    if a.fps <= 0:
        raise ValueError("FPS must be positive")
    result=evaluate_events(events["events"],labels,a.fps)
    result["provenance"]={"eventsSha256":file_hash(a.events),"labelsSha256":file_hash(a.labels)}
    write_json(a.output,result)
    print(result["status"])

if __name__=="__main__":main()
