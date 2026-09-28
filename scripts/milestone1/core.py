"""Small, inspectable adapters around OpenCV Kalman filters and SciPy assignment.

Scores are heuristic ranking values, never calibrated probabilities. No detector
or re-identification network is trained here. Shirt histograms are deliberately
a low-cost baseline whose limitations must be measured on phone footage.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

WIDTH, LENGTH, SINGLES = 10.97, 23.77, 8.23
NET = LENGTH / 2
SINGLE_LEFT = (WIDTH - SINGLES) / 2


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def project(point, matrix):
    p = np.asarray(matrix, dtype=float) @ np.array([*point, 1.0])
    if not np.isfinite(p).all() or abs(p[2]) < 1e-9:
        return None
    return (p[:2] / p[2]).tolist()


def ground_point(person):
    ankles = [j for j in person["joints"]
              if j["name"] in ("left_ankle", "right_ankle") and j["confidence"] >= .3]
    if ankles:
        return np.mean([[j["xPx"], j["yPx"]] for j in ankles], axis=0).tolist()
    b = person["bbox"]
    return [(b["x1Px"] + b["x2Px"]) / 2, b["y2Px"]]


def shirt_descriptor(image, bbox):
    """Central upper-body crop; H/S and value histograms include white/black kits."""
    x1, y1, x2, y2 = [bbox[k] for k in ("x1Px", "y1Px", "x2Px", "y2Px")]
    h, w = image.shape[:2]
    left, right = int(max(0, x1 + .25*(x2-x1))), int(min(w, x2 - .25*(x2-x1)))
    top, bottom = int(max(0, y1 + .2*(y2-y1))), int(min(h, y1 + .6*(y2-y1)))
    if right-left < 3 or bottom-top < 3:
        return None
    hsv = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2HSV)
    hs = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).ravel()
    value = cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()
    result = np.concatenate([.7 * hs / hs.sum(), .3 * value / value.sum()])
    return result.astype(np.float32)


def appearance_distance(a, b):
    if a is None or b is None:
        return None
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def calibration_usable(calibration, frame):
    """Conservative bracketing-sample check, NOT a calibration accuracy score."""
    samples = sorted(calibration.get("samples", []), key=lambda s: s["frameIndex"])
    before = [s for s in samples if s["frameIndex"] <= frame]
    after = [s for s in samples if s["frameIndex"] >= frame]
    if not before or not after:
        return False
    return all(s.get("status") == "accepted"
               and s.get("cornerRmsErrorPxFromConsensus", float("inf")) <= 12
               and s.get("consensusReferenceScore", 0) >= .90
               for s in (before[-1], after[0]))


@dataclass
class IdentityState:
    player_id: str
    filter: object
    descriptor: object
    last_seen_us: int


class IdentityTracker:
    """Global A/B association without near/far restrictions.

    Initial prototypes are frozen to avoid silent identity drift. A long gap
    requires distinctive appearance. Ambiguous candidates remain unassigned.
    This is a baseline, not proof of long-term identity retention.
    """
    def __init__(self, appearance=True):
        self.states = []
        self.last_update_us = None
        self.use_appearance = appearance

    def seed(self, player_id, point, descriptor, timestamp_us):
        if any(s.player_id == player_id for s in self.states):
            raise ValueError("Duplicate player seed")
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.statePost = np.array([[point[0]], [point[1]], [0], [0]], np.float32)
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * .3
        self.states.append(IdentityState(player_id, kf, descriptor, timestamp_us))
        self.last_update_us = timestamp_us

    def update(self, candidates, timestamp_us):
        if self.last_update_us is not None and timestamp_us <= self.last_update_us:
            raise ValueError("Tracker timestamps must strictly increase after seeding")
        dt = (timestamp_us - self.last_update_us) / 1e6 if self.last_update_us is not None else .04
        costs = np.full((len(self.states), len(candidates)), 1e6)
        details = {}
        for i, state in enumerate(self.states):
            state.filter.transitionMatrix = np.array(
                [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            state.filter.processNoiseCov = np.eye(4, dtype=np.float32) * max(.01, dt*.5)
            pred = state.filter.predict()[:2, 0]
            gap = (timestamp_us - state.last_seen_us) / 1e6
            for j, candidate in enumerate(candidates):
                motion = float(np.linalg.norm(np.array(candidate["point"]) - pred))
                app = appearance_distance(state.descriptor, candidate.get("descriptor"))
                if gap <= 1.0:
                    gate = 1.3 + 9 * gap
                    if motion > gate or (self.use_appearance and app is not None and app > .75):
                        continue
                    cost = .55 * min(motion/gate, 1) + .45 * (app if app is not None else .7)
                    if not self.use_appearance:
                        cost = motion/gate
                else:
                    # Do not extrapolate motion through an entire changeover.
                    if not self.use_appearance or app is None or app > .38:
                        continue
                    cost = app
                costs[i, j] = cost
                details[i, j] = {"appearanceDistance": app, "motionResidualTrackingUnits": motion,
                                 "secondsSinceSeen": gap}
        self.last_update_us = timestamp_us
        if not candidates or not self.states:
            return []
        # A private dummy column for each identity makes 'unknown' an option.
        assignment = np.concatenate([costs, np.full((len(self.states), len(self.states)), .8)], axis=1)
        rows, cols = linear_sum_assignment(assignment)
        matches = []
        for i, j in zip(rows, cols):
            if j >= len(candidates) or costs[i, j] >= .8:
                continue
            row_other = np.delete(costs[i], j)
            col_other = np.delete(costs[:, j], i)
            alternatives = np.concatenate([row_other, col_other, [.8]])
            margin = float(alternatives.min() - costs[i, j])
            if margin < .08:
                continue
            state = self.states[i]
            state.filter.correct(np.array(candidates[j]["point"], np.float32).reshape(2, 1))
            state.last_seen_us = timestamp_us
            matches.append({"candidateIndex": int(j), "playerId": state.player_id,
                            "associationScore": float(1-costs[i, j]),
                            "associationMargin": margin,
                            "method": "appearance_motion" if self.use_appearance else "motion_only",
                            **details[i, j]})
        return matches


def region_for_landing(point, hitter_side, uncertainty_m=.35):
    """Six opponent-half singles zones, rotated with the hitter's direction.

    The default uncertainty is a configurable review buffer, not a statistically
    estimated error bound. No precise line calls or airborne projection.
    """
    if point is None or hitter_side not in ("near", "far"):
        return {"status": "unknown", "region": None}
    x, y = map(float, point)
    if not np.isfinite([x, y]).all() or uncertainty_m < 0:
        return {"status": "unknown", "region": None}
    # Rotate the entire court 180 degrees when the hitter is at the far end.
    if hitter_side == "far":
        x, y = WIDTH-x, LENGTH-y
    left, right = SINGLE_LEFT, WIDTH-SINGLE_LEFT
    if x < left-uncertainty_m or x > right+uncertainty_m or y < -uncertainty_m or y > NET+uncertainty_m:
        return {"status": "outside_target_half", "region": None}
    boundaries = [abs(x-left), abs(x-right), abs(y), abs(y-NET),
                  abs(x-(left+SINGLES/3)), abs(x-(left+2*SINGLES/3)), abs(y-(NET-6.40))]
    if min(boundaries) <= uncertainty_m:
        return {"status": "boundary_uncertain", "region": None}
    if not (left <= x <= right and 0 <= y <= NET):
        return {"status": "outside_target_half", "region": None}
    lateral = ("left", "centre", "right")[min(2, int((x-left)/(SINGLES/3)))]
    depth = "deep" if y < NET-6.40 else "short"
    return {"status": "in_target_half", "region": f"{depth}_{lateral}"}


def landing_statistics(shots):
    result = {}
    for player_id in ("A", "B"):
        selected = [s for s in shots if s.get("playerId") == player_id]
        region_counts = Counter(s["landing"]["region"] for s in selected
                                if s.get("landing", {}).get("region"))
        denominator = sum(region_counts.values())
        result[player_id] = {
            "shots": len(selected), "assignedFirstLandings": denominator,
            "excludedOrUnresolved": len(selected)-denominator,
            "denominator": "first landings assigned to one opponent-half singles region",
            "regions": {f"{d}_{l}": {"count": region_counts[f"{d}_{l}"],
                "percent": round(100*region_counts[f"{d}_{l}"]/denominator, 2) if denominator else None}
                for d in ("short", "deep") for l in ("left", "centre", "right")}}
    return {"players": result,
            "shotsWithUnknownPlayer": sum(s.get("playerId") not in ("A", "B") for s in shots)}


def audit_inputs(pose, ball, calibration):
    errors, warnings = [], []
    if len({d["sessionId"] for d in (pose, ball, calibration)}) != 1:
        errors.append("Session IDs disagree")
    for key in ("widthPx", "heightPx", "fps", "frameCount"):
        if len({d["video"][key] for d in (pose, ball, calibration)}) != 1:
            errors.append(f"Video metadata disagree: {key}")
    n, fps = ball["video"]["frameCount"], ball["video"]["fps"]
    if fps <= 0 or n <= 0:
        errors.append("Invalid video duration")
    joint_out_of_range = 0
    for label, doc in (("pose", pose), ("ball", ball)):
        obs = doc["observations"]
        if [o["frameIndex"] for o in obs] != list(range(n)):
            errors.append(f"{label}: missing, duplicate, or out-of-order frames")
        if fps > 0 and any(abs(o["timestampUs"]-round(o["frameIndex"]*1e6/fps)) > 1 for o in obs):
            errors.append(f"{label}: timestamp mismatch")
        if label == "ball":
            if any(o["visible"] != (o["image"] is not None) for o in obs):
                errors.append("Ball visibility/coordinate mismatch")
        else:
            joint_out_of_range = sum(not 0 <= j["confidence"] <= 1 for f in obs
                for p in f["players"] for j in p["joints"])
    if joint_out_of_range:
        warnings.append(f"{joint_out_of_range} joint scores outside [0,1]; treat as raw model scores, not probabilities")
    unstable = sum(not calibration_usable(calibration, i) for i in range(n))
    if unstable:
        warnings.append(f"{unstable} frames fail bracketing calibration checks; landing statistics withheld there")
    return {"errors": errors, "warnings": warnings, "frameCount": n,
            "unstableCalibrationFrames": unstable, "outOfRangeJointScores": joint_out_of_range,
            "timestampBasis": "frame index / nominal fps; phone variable frame rate requires decoded PTS"}
