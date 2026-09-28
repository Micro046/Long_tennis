import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from milestone1.core import (IdentityTracker, WIDTH, LENGTH, region_for_landing,
    landing_statistics, calibration_usable, audit_inputs)
from milestone1.events import build_shots
from milestone1.evaluation import evaluate_events
from milestone1.review import validate_review, reviewed_results

RED=np.array([1,0,0],np.float32)
BLUE=np.array([0,1,0],np.float32)
GREEN=np.array([0,0,1],np.float32)
def c(x,y,d):return {"point":[x,y],"descriptor":d}

def calibrated():
    return {"calibration":{"imageToCourtHomography":np.eye(3).tolist()},
            "samples":[{"frameIndex":i,"status":"accepted","cornerRmsErrorPxFromConsensus":0,
                        "consensusReferenceScore":1} for i in (0,1000)]}

def e(kind,f,**kw):return {"id":f"{kind}:{f}","type":kind,"frameIndex":f,"status":"unreviewed_candidate",**kw}

class Tracking(unittest.TestCase):
    def tracker(self,distinct=True):
        t=IdentityTracker();t.seed("A",[3,20],RED,0);t.seed("B",[7,4],BLUE if distinct else RED,0)
        return t
    def test_detector_order_does_not_define_identity(self):
        m=self.tracker().update([c(7,4,BLUE),c(3,20,RED)],40000)
        self.assertEqual({x["playerId"]:x["candidateIndex"] for x in m},{"A":1,"B":0})
    def test_continuous_changeover_crossing_and_same_half(self):
        t=self.tracker()
        for i in range(1,251):
            a=c(3,20-16*i/250,RED);b=c(7,4+16*i/250,BLUE)
            candidates=[a,b] if i%2 else [b,a]
            m=t.update(candidates,i*40000)
            self.assertEqual(len(m),2)
            for x in m:
                self.assertIs(candidates[x["candidateIndex"]]["descriptor"],RED if x["playerId"]=="A" else BLUE)
    def test_missing_detections_emit_no_fake_observation(self):
        self.assertEqual(self.tracker().update([],40000),[])
    def test_distinct_appearance_recovers_after_long_gap_at_opposite_end(self):
        m=self.tracker().update([c(3,4,RED),c(7,20,BLUE)],30000000)
        self.assertEqual({x["playerId"]:x["candidateIndex"] for x in m},{"A":0,"B":1})
    def test_same_clothing_long_gap_is_unknown(self):
        self.assertEqual(self.tracker(False).update([c(3,4,RED),c(7,20,RED)],30000000),[])
    def test_unfamiliar_person_not_used_for_recovery(self):
        self.assertEqual(self.tracker().update([c(3,4,GREEN)],30000000),[])
    def test_non_monotonic_timestamp_rejected(self):
        with self.assertRaises(ValueError):self.tracker().update([],0)
    def test_duplicate_seed_rejected(self):
        t=self.tracker()
        with self.assertRaises(ValueError):t.seed("A",[3,4],RED,0)

class Regions(unittest.TestCase):
    def test_six_regions_and_changeover_rotation(self):
        for x,l in ((2.5,"left"),(5.485,"centre"),(8.4,"right")):
            for y,d in ((2,"deep"),(8,"short")):
                a=region_for_landing([x,y],"near")
                b=region_for_landing([WIDTH-x,LENGTH-y],"far")
                self.assertEqual(a["region"],f"{d}_{l}");self.assertEqual(a,b)
    def test_singles_alley_is_not_in_region(self):
        self.assertEqual(region_for_landing([.5,3],"near")["status"],"outside_target_half")
    def test_boundary_is_unknown(self):
        self.assertEqual(region_for_landing([1.37,3],"near")["status"],"boundary_uncertain")
    def test_wrong_half_not_counted(self):
        self.assertIsNone(region_for_landing([5,20],"near")["region"])
    def test_unknown_side_not_counted(self):
        self.assertEqual(region_for_landing([5,3],None)["status"],"unknown")

class Shots(unittest.TestCase):
    def test_only_first_bounce_counted(self):
        shots=build_shots([e("hit",10,playerId="A",hitterSide="near"),
             e("bounce",25,imagePx=[5,2]),e("bounce",45,imagePx=[2.5,8])],calibrated(),25)
        self.assertEqual(len(shots),1);self.assertEqual(shots[0]["bounceEventId"],"bounce:25")
        self.assertEqual(landing_statistics(shots)["players"]["A"]["assignedFirstLandings"],1)
    def test_next_hit_ends_previous_shot(self):
        shots=build_shots([e("hit",10,playerId="A",hitterSide="near"),
            e("hit",30,playerId="B",hitterSide="far"),e("bounce",50,imagePx=[5,20])],calibrated(),25)
        self.assertIsNone(shots[0]["bounceEventId"]);self.assertEqual(shots[1]["bounceEventId"],"bounce:50")
    def test_conflicting_hit_bounce_not_counted(self):
        s=build_shots([e("hit",10,playerId="A",hitterSide="near"),e("bounce",11,imagePx=[5,2])],calibrated(),25)
        self.assertIsNone(s[0]["landing"]["region"])
    def test_calibration_failure_withholds_landing(self):
        cal=calibrated();cal["samples"][1]["cornerRmsErrorPxFromConsensus"]=50
        s=build_shots([e("hit",10,playerId="A",hitterSide="near"),e("bounce",25,imagePx=[5,2])],cal,25)
        self.assertEqual(s[0]["landing"]["status"],"calibration_unstable")
    def test_missing_bounce_not_zero_percent(self):
        s=build_shots([e("hit",10,playerId="A",hitterSide="near")],calibrated(),25)
        a=landing_statistics(s)["players"]["A"]
        self.assertEqual(a["excludedOrUnresolved"],1)
        self.assertIsNone(a["regions"]["deep_left"]["percent"])
    def test_no_cross_rally_long_gap_link(self):
        s=build_shots([e("hit",10,playerId="A",hitterSide="near"),e("bounce",150,imagePx=[5,2])],calibrated(),25)
        self.assertIsNone(s[0]["bounceEventId"])

class Evaluation(unittest.TestCase):
    def test_reviewing_candidates_is_not_recall_ground_truth(self):
        r=evaluate_events([], {"annotationMode":"candidate_review","events":[]},25)
        self.assertIsNone(r["recall"]);self.assertEqual(r["status"],"not_evaluable")
    def test_one_to_one_matching_and_missing_events(self):
        labels={"annotationMode":"independent_exhaustive","coverageWindows":[{"startFrame":0,"endFrame":99}],
                "events":[e("bounce",20),e("bounce",50)]}
        r=evaluate_events([e("bounce",20),e("bounce",21)],labels,25)["metrics"]["bounce"]
        self.assertEqual((r["truePositives"],r["falsePositives"],r["falseNegatives"]),(1,1,1))
    def test_unlabelled_intervals_excluded(self):
        labels={"annotationMode":"independent_exhaustive","coverageWindows":[{"startFrame":0,"endFrame":10}],"events":[]}
        r=evaluate_events([e("bounce",50)],labels,25)["metrics"]["bounce"]
        self.assertEqual(r["falsePositives"],0);self.assertIsNone(r["recall"])
    def test_variable_frame_timing_uses_presentation_timestamps(self):
        labels={"annotationMode":"independent_exhaustive","coverageWindows":[{"startFrame":0,"endFrame":10}],
                "events":[e("bounce",4,timestampUs=300000)]}
        pred=[e("bounce",3,timestampUs=100000)]
        r=evaluate_events(pred,labels,25)["metrics"]["bounce"]
        self.assertEqual((r["truePositives"],r["falsePositives"],r["falseNegatives"]),(0,1,1))
    def test_overlapping_coverage_rejected(self):
        labels={"annotationMode":"independent_exhaustive","coverageWindows":[{"startFrame":0,"endFrame":10},{"startFrame":10,"endFrame":20}],"events":[]}
        with self.assertRaises(ValueError):evaluate_events([],labels,25)
    def test_duplicate_frames_fail_audit(self):
        meta={"widthPx":100,"heightPx":100,"fps":25,"frameCount":2}
        ball={"sessionId":"x","video":meta,"observations":[{"frameIndex":0,"timestampUs":0,"visible":False,"image":None}]*2}
        pose={"sessionId":"x","video":meta,"observations":[{"frameIndex":i,"timestampUs":i*40000,"players":[]} for i in range(2)]}
        cal={**calibrated(),"sessionId":"x","video":meta}
        self.assertTrue(audit_inputs(pose,ball,cal)["errors"])

class Review(unittest.TestCase):
    def test_source_mismatch_and_out_of_frame_contact_rejected(self):
        r={"sessionId":"s","videoSha256":"abc","annotationMode":"candidate_review",
           "events":[e("bounce",25,imagePx=[500,2])]}
        with self.assertRaises(ValueError):validate_review(r,"s","wrong",100,100,100)
        with self.assertRaises(ValueError):validate_review(r,"s","abc",100,100,100)

    def test_unreviewed_intervening_event_withholds_count(self):
        r={"events":[{**e("hit",10,playerId="A",hitterSide="near"),"status":"accepted"},
             e("bounce",20,imagePx=[5,2]),
             {**e("bounce",25,imagePx=[5,2]),"status":"accepted"}]}
        result=reviewed_results(r,calibrated(),25)
        self.assertEqual(result["status"],"partial_review_only")
        self.assertEqual(result["landingStatistics"]["players"]["A"]["assignedFirstLandings"],0)
        self.assertEqual(result["shots"][0]["landing"]["status"],"unreviewed_event_in_shot_window")

    def test_reviewed_shot_counts_only_accepted_first_bounce(self):
        r={"events":[{**e("hit",10,playerId="A",hitterSide="near"),"status":"accepted"},
             {**e("bounce",25,imagePx=[5,2]),"status":"accepted"}]}
        result=reviewed_results(r,calibrated(),25)
        self.assertEqual(result["landingStatistics"]["players"]["A"]["assignedFirstLandings"],1)
        self.assertFalse(result["accuracyEvaluated"])

if __name__=="__main__":unittest.main()
