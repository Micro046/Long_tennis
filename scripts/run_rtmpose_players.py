#!/usr/bin/env python3
"""Run reusable RTMLib/RTMPose player pose inference on public test clips."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "work/cache"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from rtmlib import Body  # noqa: E402


KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

SKELETON = (
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (0, 1),
    (0, 2), (1, 3), (2, 4),
)

ROLE_COLORS = {
    "near": (90, 230, 90),
    "far": (0, 170, 255),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_manifest() -> list[dict]:
    checkpoint_dir = ROOT / "work/cache/rtmlib/hub/checkpoints"
    models = []
    for path in sorted(checkpoint_dir.glob("*.onnx")):
        if "yolox_tiny" in path.name or "rtmpose-s_simcc-body7" in path.name:
            models.append(
                {
                    "fileName": path.name,
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return models


def make_people(
    boxes: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> list[dict]:
    people = []
    for index, (box, person_keypoints, person_scores) in enumerate(
        zip(boxes, keypoints, scores)
    ):
        x1, y1, x2, y2 = [float(value) for value in box]
        mean_score = float(np.mean(person_scores))
        confident_joints = int(np.sum(person_scores >= 0.3))
        people.append(
            {
                "rawIndex": index,
                "bbox": [x1, y1, x2, y2],
                "bottomCenterNorm": [(x1 + x2) / (2 * width), y2 / height],
                "bboxAreaNorm": max(0.0, x2 - x1) * max(0.0, y2 - y1) / (width * height),
                "meanJointConfidence": mean_score,
                "confidentJointCount": confident_joints,
                "keypoints": np.asarray(person_keypoints),
                "scores": np.asarray(person_scores),
            }
        )
    return people


def person_ground_point(person: dict) -> tuple[float, float]:
    ankle_indices = [index for index in (15, 16) if person["scores"][index] >= 0.3]
    if ankle_indices:
        return (
            float(np.mean([person["keypoints"][index][0] for index in ankle_indices])),
            float(np.mean([person["keypoints"][index][1] for index in ankle_indices])),
        )
    x1, _y1, x2, y2 = person["bbox"]
    return (x1 + x2) / 2.0, y2


def image_to_court(point: tuple[float, float], matrix: np.ndarray) -> tuple[float, float]:
    source = np.asarray([[point]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(source, matrix)[0][0]
    return float(mapped[0]), float(mapped[1])


def select_players(
    people: list[dict],
    players_per_side: int,
    court_matrix: np.ndarray | None = None,
) -> list[tuple[str, dict]]:
    candidates = [
        person
        for person in people
        if 0.14 <= person["bottomCenterNorm"][0] <= 0.86
        and person["meanJointConfidence"] >= 0.3
        and person["confidentJointCount"] >= 6
    ]
    if court_matrix is None:
        near = [person for person in candidates if person["bottomCenterNorm"][1] > 0.35]
        far = [person for person in candidates if person["bottomCenterNorm"][1] <= 0.35]
        ranking = lambda person: (
            person["bottomCenterNorm"][1],
            person["bboxAreaNorm"],
            person["meanJointConfidence"],
        )
        near = sorted(near, key=ranking, reverse=True)[:players_per_side]
        far = sorted(far, key=ranking, reverse=True)[:players_per_side]
    else:
        for person in candidates:
            person["courtPoint"] = image_to_court(person_ground_point(person), court_matrix)
        candidates = [
            person
            for person in candidates
            if -1.5 <= person["courtPoint"][0] <= 10.97 + 1.5
            and -7.0 <= person["courtPoint"][1] <= 23.77 + 7.0
        ]
        near = [person for person in candidates if person["courtPoint"][1] > 23.77 / 2]
        far = [person for person in candidates if person["courtPoint"][1] <= 23.77 / 2]
        # Prefer players nearer the net over staff stationed behind a baseline.
        near = sorted(
            near,
            key=lambda person: (
                person["courtPoint"][1],
                -person["meanJointConfidence"],
            ),
        )[:players_per_side]
        far = sorted(
            far,
            key=lambda person: (
                person["courtPoint"][1],
                person["meanJointConfidence"],
            ),
            reverse=True,
        )[:players_per_side]
    near.sort(key=lambda person: person["bottomCenterNorm"][0])
    far.sort(key=lambda person: person["bottomCenterNorm"][0])

    selected = []
    for side, side_people in (("near", near), ("far", far)):
        if players_per_side == 1:
            if side_people:
                selected.append((side, side_people[0]))
        else:
            for index, person in enumerate(side_people):
                # Candidate order is view-relative for this frame and is not a
                # persistent player identity. Proper IDs require tracking.
                selected.append((f"{side}_candidate_{index + 1}", person))
    return selected


def serialize_player(
    role: str,
    person: dict,
    frame_index: int,
    fps: float,
    width: int,
    height: int,
) -> dict:
    x1, y1, x2, y2 = person["bbox"]
    joints = []
    for name, point, score in zip(KEYPOINT_NAMES, person["keypoints"], person["scores"]):
        x, y = float(point[0]), float(point[1])
        joints.append(
            {
                "name": name,
                "xPx": round(x, 3),
                "yPx": round(y, 3),
                "xNorm": round(x / width, 8),
                "yNorm": round(y / height, 8),
                "confidence": round(float(score), 6),
            }
        )
    return {
        "id": f"pose:{frame_index}:{role}",
        "type": "player_pose_observation",
        "role": role,
        "frameIndex": frame_index,
        "timestampUs": round(frame_index * 1_000_000 / fps),
        "confidence": round(person["meanJointConfidence"], 6),
        "bbox": {
            "x1Px": round(x1, 3),
            "y1Px": round(y1, 3),
            "x2Px": round(x2, 3),
            "y2Px": round(y2, 3),
            "x1Norm": round(x1 / width, 8),
            "y1Norm": round(y1 / height, 8),
            "x2Norm": round(x2 / width, 8),
            "y2Norm": round(y2 / height, 8),
        },
        "confidentJointCount": person["confidentJointCount"],
        "joints": joints,
        "provenance": {
            "adapter": "rtmlib_rtmpose_players_v1",
            "rawDetectionIndex": person["rawIndex"],
            "selection": (
                "court_aware_heuristic_v2"
                if "courtPoint" in person
                else "view_relative_heuristic_v1"
            ),
        },
    }


def draw_player(frame: np.ndarray, role: str, person: dict, threshold: float = 0.3) -> None:
    side = role.split("_")[0]
    color = ROLE_COLORS[side]
    x1, y1, x2, y2 = [round(value) for value in person["bbox"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame,
        f"{role} {person['meanJointConfidence']:.2f}",
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    for start, end in SKELETON:
        if person["scores"][start] >= threshold and person["scores"][end] >= threshold:
            point_a = tuple(np.rint(person["keypoints"][start]).astype(int))
            point_b = tuple(np.rint(person["keypoints"][end]).astype(int))
            cv2.line(frame, point_a, point_b, color, 3, cv2.LINE_AA)
    for point, score in zip(person["keypoints"], person["scores"]):
        if score >= threshold:
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 4, color, -1, cv2.LINE_AA)


def process_clip(
    video_path: Path,
    output_dir: Path,
    body: Body,
    session_config: dict,
    court_matrix: np.ndarray | None,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    players_per_side = int(session_config["playersPerSide"])
    expected_players = players_per_side * 2

    clip_output = output_dir / video_path.stem
    clip_output.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(clip_output / "overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    observations = []
    raw_person_counts = []
    selected_counts = []
    confident_joint_rates = []
    expected_roles = (
        ["near", "far"]
        if players_per_side == 1
        else [
            "near_candidate_1",
            "near_candidate_2",
            "far_candidate_1",
            "far_candidate_2",
        ]
    )
    role_hits = {role: 0 for role in expected_roles}
    frame_index = 0
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        boxes = body.det_model(frame)
        if len(boxes):
            keypoints, scores = body.pose_model(frame, bboxes=boxes)
        else:
            keypoints = np.empty((0, 17, 2), dtype=np.float32)
            scores = np.empty((0, 17), dtype=np.float32)
        people = make_people(boxes, keypoints, scores, width, height)
        selected = select_players(people, players_per_side, court_matrix)
        players = [
            serialize_player(role, person, frame_index, fps, width, height)
            for role, person in selected
        ]
        observations.append(
            {
                "frameIndex": frame_index,
                "timestampUs": round(frame_index * 1_000_000 / fps),
                "rawPersonDetectionCount": len(people),
                "players": players,
                # Preserve detections before side-based selection. Both athletes
                # can occupy the same half while changing ends.
                "candidates": [
                    serialize_player(
                        f"candidate_{person['rawIndex']}", person,
                        frame_index, fps, width, height,
                    )
                    for person in people
                ],
            }
        )
        raw_person_counts.append(len(people))
        selected_counts.append(len(players))
        for player in players:
            role_hits[player["role"]] += 1
        confident_joint_rates.extend(
            player["confidentJointCount"] / len(KEYPOINT_NAMES) for player in players
        )

        for person in people:
            x1, y1, x2, y2 = [round(value) for value in person["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (130, 130, 130), 1)
        for role, person in selected:
            draw_player(frame, role, person)
        cv2.putText(
            frame,
            f"selected players {len(players)}/{expected_players} | raw people {len(people)}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_index += 1
        if frame_index % 50 == 0 or frame_index == frame_count:
            print(f"{video_path.stem}: {frame_index}/{frame_count} frames", flush=True)

    elapsed = time.perf_counter() - started
    capture.release()
    writer.release()
    document = {
        "schemaVersion": "0.1",
        "sessionId": video_path.stem,
        "video": {
            "fileName": video_path.name,
            "widthPx": width,
            "heightPx": height,
            "fps": fps,
            "frameCount": frame_index,
        },
        "sessionConfig": session_config,
        "producer": {
            "component": "RTMLib Body / YOLOX-tiny + RTMPose-s",
            "rtmlibVersion": "0.0.16",
            "backend": "onnxruntime-cpu",
            "mode": "lightweight",
            "keypointFormat": "COCO-17",
            "models": model_manifest(),
            "selectionMode": (
                "court_aware_heuristic_v2"
                if court_matrix is not None
                else "view_relative_heuristic_v1"
            ),
        },
        "observations": observations,
    }
    (clip_output / "pose_observations.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    complete_frames = sum(count == expected_players for count in selected_counts)
    return {
        "clip": video_path.name,
        "format": session_config["format"],
        "expectedPlayers": expected_players,
        "frameCount": frame_index,
        "runtimeSeconds": elapsed,
        "analysisFps": frame_index / elapsed,
        "framesWithExpectedPlayerCount": complete_frames,
        "expectedPlayerCountRate": complete_frames / frame_index if frame_index else 0.0,
        "meanRawPersonDetections": float(np.mean(raw_person_counts)),
        "meanSelectedPlayers": float(np.mean(selected_counts)),
        "roleCoverageRate": {
            role: hits / frame_index if frame_index else 0.0
            for role, hits in role_hits.items()
        },
        "meanSelectedJointCoverageAt0.3": (
            float(np.mean(confident_joint_rates)) if confident_joint_rates else 0.0
        ),
        "warning": "Selection-rate metrics are proxy checks, not accuracy against pose ground truth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/public/racketvision")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/rtmpose_players")
    parser.add_argument(
        "--court-dir", type=Path, default=ROOT / "outputs/court_calibration"
    )
    parser.add_argument("--clip", action="append", help="Video stem to process; repeat as needed")
    args = parser.parse_args()

    configs = json.loads((args.data_dir / "session_config.json").read_text(encoding="utf-8"))
    body = Body(mode="lightweight", backend="onnxruntime", device="cpu")
    reports = []
    for video_path in sorted(args.data_dir.glob("*.mp4")):
        if args.clip and video_path.stem not in set(args.clip):
            continue
        calibration_path = args.court_dir / video_path.stem / "court_calibration.json"
        court_matrix = None
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            court_matrix = np.asarray(
                calibration["calibration"]["imageToCourtHomography"], dtype=np.float32
            )
        reports.append(
            process_clip(
                video_path,
                args.output_dir,
                body,
                configs[video_path.stem],
                court_matrix,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "component": "RTMLib Body / YOLOX-tiny + RTMPose-s",
        "models": model_manifest(),
        "clips": reports,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
