"""Validate benchmark group splits and stop disguised public clips becoming phone truth."""
from __future__ import annotations

from collections import defaultdict

SPLITS = {"development", "heldout", "diagnostic_only"}
GROUP_KEYS = ("physicalSessionId", "venueId", "cameraDeviceId", "playerGroupId")


def validate_manifest(manifest):
    if manifest.get("schemaVersion") != "0.1" or not isinstance(manifest.get("sessions"), list):
        raise ValueError("Expected benchmark manifest schema 0.1 with sessions")
    seen_ids = set()
    seen_paths = set()
    player_splits = defaultdict(set)
    groups = {key: defaultdict(set) for key in GROUP_KEYS}
    for session in manifest["sessions"]:
        sid, split = session.get("sessionId"), session.get("split")
        if not isinstance(sid, str) or not sid or sid in seen_ids:
            raise ValueError("Session IDs must be unique nonempty strings")
        seen_ids.add(sid)
        if split not in SPLITS:
            raise ValueError(f"Invalid split for {sid}")
        if not isinstance(session.get("videoPath"), str) or not session["videoPath"]:
            raise ValueError(f"Missing video path for {sid}")
        if session["videoPath"] in seen_paths:
            raise ValueError(f"Duplicate video path in benchmark: {session['videoPath']}")
        seen_paths.add(session["videoPath"])
        if split != "diagnostic_only":
            if session.get("captureMode") != "fixed_phone":
                raise ValueError(f"Evaluation split requires confirmed fixed-phone capture: {sid}")
            if not session.get("rights", {}).get("commercialEvaluationCleared"):
                raise ValueError(f"Evaluation split requires documented video rights: {sid}")
            players = session.get("playerIds")
            if (not isinstance(players, list) or len(players) != 2 or
                not all(isinstance(player, str) and player for player in players) or
                len(set(players)) != 2):
                raise ValueError(f"Singles evaluation needs two distinct pseudonymous player IDs: {sid}")
            for player in players:
                player_splits[player].add(split)
            for key in GROUP_KEYS:
                value = session.get("groups", {}).get(key)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Evaluation split needs {key}: {sid}")
                groups[key][value].add(split)
    for player, splits in player_splits.items():
        if len(splits) > 1:
            raise ValueError(f"Data leakage: individual player {player} crosses development and heldout")
    for key, values in groups.items():
        for value, splits in values.items():
            if len(splits) > 1:
                raise ValueError(f"Data leakage: {key}={value} crosses development and heldout")
    return {"sessionCount": len(seen_ids),
            "splitCounts": {s: sum(row["split"] == s for row in manifest["sessions"]) for s in sorted(SPLITS)}}
