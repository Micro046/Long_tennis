package com.longtennis.app

import com.longtennis.app.analysis.Court
import com.longtennis.app.analysis.CourtCalibrator
import com.longtennis.app.analysis.Events
import com.longtennis.app.analysis.GbmModel
import com.longtennis.app.analysis.Homography
import com.longtennis.app.analysis.Pt
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.abs

/** Kotlin port must reproduce the Python reference (scripts/pipeline) on the same inputs. */
class ParityTest {
    private fun res(name: String) = File(javaClass.classLoader!!.getResource(name)!!.toURI())

    @Test fun eventFeaturesAndModelMatchPython() {
        val fx = JSONObject(res("event_fixture.json").readText())
        val h = fx.getDouble("height"); val fps = fx.getDouble("fps")
        val frames = fx.getJSONArray("frames")
        val xy = Array(frames.length()) { i ->
            if (frames.isNull(i)) doubleArrayOf(Double.NaN, Double.NaN)
            else frames.getJSONArray(i).let { doubleArrayOf(it.getDouble(0) / h, it.getDouble(1) / h) }
        }
        val (clean, obs) = Events.cleanTrack(xy)
        val feats = Events.features(clean, obs, fps)
        val exp = fx.getJSONArray("features")
        var maxDiff = 0.0
        for (i in feats.indices) for (j in feats[i].indices) maxDiff = maxOf(maxDiff, abs(feats[i][j] - exp.getJSONArray(i).getDouble(j)))
        assertTrue("feature max diff $maxDiff", maxDiff < 1e-4)

        val gbm = GbmModel(File("src/main/assets/models/event_gbm.json").readText())
        val prob = fx.getJSONArray("prob")
        var pd = 0.0
        for (i in feats.indices) {
            val expRow = JSONObject.wrap(null).let { expectedFeatures(exp, i) }
            val p = gbm.predictProba(expRow)
            for (c in 0 until 3) pd = maxOf(pd, abs(p[c] - prob.getJSONArray(i).getDouble(c)))
        }
        assertTrue("probability max diff $pd", pd < 1e-5)
    }

    private fun expectedFeatures(exp: org.json.JSONArray, i: Int): FloatArray {
        val r = exp.getJSONArray(i); return FloatArray(r.length()) { r.getDouble(it).toFloat() }
    }

    @Test fun courtCalibrationMatchesPython() {
        val img = ImageIO.read(res("court_frame.png"))
        val w = img.width; val h = img.height
        val argb = img.getRGB(0, 0, w, h, null, 0, w)
        val init = mapOf("far_left_singles" to Pt(705.0, 318.0), "far_right_singles" to Pt(1195.0, 318.0),
            "near_service_left" to Pt(370.0, 524.0), "near_service_right" to Pt(1530.0, 524.0))
        val r = CourtCalibrator.calibrate(argb, w, h, init)!!
        val exp = JSONObject(res("court_expected.json").readText()).getJSONArray("imageFromCourt")
        val He = Homography.fromList((0 until 3).map { a -> (0 until 3).map { b -> exp.getJSONArray(a).getDouble(b) } })
        var worst = 0.0
        for ((_, p) in Court.MODEL_KP) {
            val a = r.imageFromCourt.apply(p)!!; val b = He.apply(p)!!
            if (b.x in 0.0..w.toDouble() && b.y in 0.0..h.toDouble()) worst = maxOf(worst, (a - b).norm())
        }
        println("court: lines=${r.linesSnapped.size} kps=${r.keypointsUsed.size} rms=${r.reprojectionRmsPx} worst diff vs python=$worst px")
        assertTrue("court keypoints differ from Python by $worst px", worst < 3.0)
        assertEquals(9, r.linesSnapped.size)
    }

    @Test fun courtCalibrationLowOffsetViewMatchesPython() {
        val img = ImageIO.read(res("court_frame2.png"))
        val w = img.width; val h = img.height
        val argb = img.getRGB(0, 0, w, h, null, 0, w)
        val init = mapOf("near_left_singles" to Pt(85.0, 683.0), "near_right_singles" to Pt(1800.0, 683.0),
            "near_service_left" to Pt(505.0, 400.0), "near_service_right" to Pt(1395.0, 400.0))
        val r = CourtCalibrator.calibrate(argb, w, h, init)!!
        val exp = JSONObject(res("court_expected2.json").readText()).getJSONArray("imageFromCourt")
        val He = Homography.fromList((0 until 3).map { a -> (0 until 3).map { b -> exp.getJSONArray(a).getDouble(b) } })
        var worst = 0.0
        for ((_, p) in Court.MODEL_KP) {
            val a = r.imageFromCourt.apply(p)!!; val b = He.apply(p)!!
            if (b.x in 0.0..w.toDouble() && b.y in 0.0..h.toDouble()) worst = maxOf(worst, (a - b).norm())
        }
        println("court2: kps=${r.keypointsUsed.size} rms=${r.reprojectionRmsPx} worst diff vs python=$worst px")
        assertTrue("low-view court keypoints differ from Python by $worst px", worst < 4.0)
    }

    @Test fun regionsMatchPython() {
        // near hitter hitting deep right (far half, far right from hitter's view)
        assertEquals("in" to "deep_right", Court.region(Pt(8.5, 2.0), "near"))
        assertEquals("in" to "short_left", Court.region(Pt(2.5, 9.0), "near"))
        // far hitter: target near half, left/right flipped
        assertEquals("in" to "deep_left", Court.region(Pt(8.5, 21.0), "far"))
        assertEquals("out", Court.region(Pt(0.5, 21.0), "far").first)
    }
}
