import csv
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from benchmark.manifest import validate_manifest
from benchmark.video import probe_video, summarize_timestamps
from benchmark.labels import validate_labels
from audit_benchmark import audit_csv


def phone_session(sid, split, venue):
    return {"sessionId": sid, "videoPath": f"{sid}.mp4", "split": split,
            "playerIds": [f"{sid}_A", f"{sid}_B"],
            "captureMode": "fixed_phone", "rights": {"commercialEvaluationCleared": True},
            "groups": {"physicalSessionId": sid, "venueId": venue,
                       "cameraDeviceId": sid, "playerGroupId": sid}}


class BenchmarkManifest(unittest.TestCase):
    def test_cross_venue_split_leakage_rejected(self):
        m={"schemaVersion":"0.1","sessions":[phone_session("a","development","court1"),
                                              phone_session("b","heldout","court1")]}
        with self.assertRaisesRegex(ValueError,"Data leakage"):
            validate_manifest(m)

    def test_unconfirmed_public_video_cannot_be_heldout_phone_test(self):
        s=phone_session("x","heldout","court1")
        s["captureMode"]="unknown_back_view"
        with self.assertRaisesRegex(ValueError,"fixed-phone"):
            validate_manifest({"schemaVersion":"0.1","sessions":[s]})

    def test_grouped_distinct_sessions_pass(self):
        m={"schemaVersion":"0.1","sessions":[phone_session("a","development","court1"),
                                              phone_session("b","heldout","court2")]}
        self.assertEqual(validate_manifest(m)["splitCounts"]["heldout"],1)

    def test_renamed_session_cannot_reuse_same_video_path(self):
        a=phone_session("a","development","court1")
        b=phone_session("b","heldout","court2");b["videoPath"]=a["videoPath"]
        with self.assertRaisesRegex(ValueError,"Duplicate video path"):
            validate_manifest({"schemaVersion":"0.1","sessions":[a,b]})

    def test_shared_person_in_different_player_pairs_is_leakage(self):
        a=phone_session("a","development","court1")
        b=phone_session("b","heldout","court2");b["playerIds"][0]=a["playerIds"][0]
        with self.assertRaisesRegex(ValueError,"individual player"):
            validate_manifest({"schemaVersion":"0.1","sessions":[a,b]})


class TimestampAudit(unittest.TestCase):
    def test_vfr_video_preserves_decoded_pts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"sample.mkv"
            with av.open(str(path),"w") as out:
                stream=out.add_stream("libx264",rate=30)
                stream.width=64;stream.height=48;stream.pix_fmt="yuv420p"
                stream.time_base=Fraction(1,1000)
                stream.codec_context.time_base=Fraction(1,1000)
                stream.options={"bf":"0"}
                for pts in (0,33,66,150,200):
                    frame=av.VideoFrame.from_ndarray(np.zeros((48,64,3),dtype=np.uint8),format="rgb24")
                    frame.pts=pts;frame.time_base=Fraction(1,1000)
                    for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            result=probe_video(path)
            self.assertEqual(result["relativePresentationTimestampsUs"],[0,33000,66000,150000,200000])
            self.assertTrue(result["variableFrameRateObserved"])
            self.assertEqual(result["frameCount"],5)

    def test_duplicate_pts_rejected(self):
        with self.assertRaisesRegex(ValueError,"strictly increasing"):
            summarize_timestamps([0,33333,33333])


class AnnotationAudit(unittest.TestCase):
    def test_corner_cluster_flagged_without_declaring_missing_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"ball.csv"
            with path.open("w",newline="") as file:
                w=csv.writer(file);w.writerow(("frame","ball_x","ball_y"))
                w.writerows((("frame_000",1910,5),("frame_001",1911,5),("frame_002",900,400)))
            result=audit_csv(path,3,1920,1080,"ball")
            self.assertEqual(result["topRightCornerCoordinateCount"],2)
            self.assertEqual(result["qualityFlag"],"top_right_cluster_needs_label_review")


class PhoneLabels(unittest.TestCase):
    def timing(self):
        return {"sessionId":"phone1","videoSha256":"abc",
                "relativePresentationTimestampsUs":[0,33000,66000,150000,200000]}

    def labels(self):
        return {"schemaVersion":"0.1","annotationMode":"independent_exhaustive",
                "sessionId":"phone1","videoSha256":"abc","independentOfModelProposals":True,
                "coverageWindows":[{"startFrame":0,"endFrame":4}],
                "events":[{"id":"hit1","type":"hit","frameIndex":3,"timestampUs":150000,
                           "playerId":"A","strokeClass":"forehand"}],
                "playerObservations":[{"frameIndex":3,"playerId":"A","side":"near","visibility":"visible",
                                       "bboxPx":[5,5,25,40]}],
                "changeovers":[{"frameIndex":4,"playerId":"A","sideBefore":"near","sideAfter":"far"}]}

    def test_independent_labels_match_vfr_timestamps(self):
        result=validate_labels(self.labels(),self.timing(),100,80)
        self.assertTrue(result["eventEvaluationReady"])
        self.assertEqual(result["annotatedPlayerSideChanges"],1)

    def test_nominal_fps_timestamp_rejected_for_vfr_event(self):
        labels=self.labels();labels["events"][0]["timestampUs"]=100000
        with self.assertRaisesRegex(ValueError,"presentation time"):
            validate_labels(labels,self.timing(),100,80)

    def test_candidate_review_cannot_masquerade_as_independent_truth(self):
        labels=self.labels();labels["annotationMode"]="candidate_review"
        with self.assertRaises(ValueError):validate_labels(labels,self.timing(),100,80)


if __name__=="__main__":unittest.main()
