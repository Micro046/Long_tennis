package com.longtennis.app.analysis

import android.content.Context
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/**
 * On-device post-session analysis:
 *   pass over the video: ball tracker on every frame, detector+pose+appearance every POSE_STRIDE frames
 *   then: identity -> events -> shots -> report JSON.
 */
class Pipeline(private val context: Context) {
    companion object {
        const val POSE_STRIDE = 3
        const val SEGMENT_S = 120.0
        const val BG_SAMPLES = 48
    }

    data class Progress(val stage: String, val fraction: Float, val detail: String = "")

    class Cancelled : Exception()

    fun run(
        video: Uri, calibration: Calibration, names: Map<String, String>, seedASide: String = "near",
        handedA: String = "right", handedB: String = "right", quality: String = "fast",
        maxFrames: Int = Int.MAX_VALUE, onProgress: (Progress) -> Unit = {}, isCancelled: () -> Boolean = { false },
    ): JSONObject {
        val dec = VideoDecoder(context, video)
        val info = dec.info
        val total = min(maxFrames, max(1, info.estFrames))
        val ballModel = BallTracker.chooseModel(context, quality)
        val ball = BallTracker(context, ballModel)
        val pose = PoseEstimator(context, calibration.courtFromImage, calibration.imageFromCourt, info.width, info.height)
        val gbm = GbmModel(context.assets.open(BallTracker.eventModelFor(context, ballModel)).bufferedReader().readText())

        val balls = ArrayList<FloatArray?>(total); val pts = ArrayList<Long>(total)
        val people = ArrayList<List<Person>?>(total); val cuts = HashSet<Int>()
        var prevSmall: FloatArray? = null
        val segFrames = (SEGMENT_S * info.frameRate).toInt()
        val started = System.nanoTime()
        var tDecode = 0L; var tBall = 0L; var tPose = 0L; var tBg = 0L
        try {
            while (true) {
                if (isCancelled()) throw Cancelled()
                val t0 = System.nanoTime()
                val f = dec.next() ?: break
                tDecode += System.nanoTime() - t0
                val i = f.index
                if (i >= maxFrames) break
                if (i % segFrames == 0) {
                    onProgress(Progress("background", i.toFloat() / total, "segment ${i / segFrames + 1}"))
                    val tb = System.nanoTime()
                    ball.setBackground(background(video, f.ptsUs, f.ptsUs + (SEGMENT_S * 1e6).toLong(), info))
                    tBg += System.nanoTime() - tb
                }
                val t1 = System.nanoTime()
                balls.add(ball.push(f)); pts.add(f.ptsUs)
                tBall += System.nanoTime() - t1
                val small = graySmall(f)
                prevSmall?.let { p -> if (small.indices.sumOf { abs(small[it] - p[it]).toDouble() } / small.size > 25) cuts.add(i) }
                prevSmall = small
                if (i % POSE_STRIDE == 0) {
                    val t2 = System.nanoTime()
                    val ppl = pose.process(f)
                    for ((a, d) in ppl.withIndex()) {
                        d.overlap = ppl.withIndex().any { (b, e) -> b != a && iou(d.box, e.box) > 0.05f }
                        d.desc = Identity.appearance(f, d)
                    }
                    people.add(ppl)
                    tPose += System.nanoTime() - t2
                } else people.add(null)
                if (i % 15 == 0) {
                    val el = (System.nanoTime() - started) / 1e9
                    onProgress(Progress("tracking", (i + 1f) / total, "%.1f fps".format((i + 1) / el)))
                }
            }
        } finally { dec.close(); ball.close(); pose.close() }

        onProgress(Progress("identity", 1f))
        val ident = Identity.assign(people, cuts, info.frameRate, seedASide)
        onProgress(Progress("events", 1f))
        val an = Analyzer(balls.toTypedArray(), pts.toLongArray(), people, ident, calibration.courtFromImage, info.frameRate, info.height)
        an.detectEvents(gbm)
        an.removeBallHandling(calibration.imageFromCourt)
        an.dedupeHits()
        an.keepAnsweredHits()
        val handed = mapOf("A" to handedA, "B" to handedB)   // from setup; automatic votes proved unreliable
        val shots = an.buildShots(handed)
        val elapsed = (System.nanoTime() - started) / 1e9
        val meta = JSONObject().put("widthPx", info.width).put("heightPx", info.height).put("fps", info.frameRate)
            .put("frameCount", balls.size).put("durationS", (pts.lastOrNull() ?: 0L) / 1e6)
            .put("analysisSeconds", elapsed).put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("ballModel", ballModel)
            .put("timingMsPerFrame", JSONObject().put("decodeToBgr", tDecode / 1e6 / max(1, balls.size))
                .put("ball", tBall / 1e6 / max(1, balls.size)).put("poseAndAppearance", tPose / 1e6 / max(1, balls.size))
                .put("backgroundMedian", tBg / 1e6 / max(1, balls.size)))
        onProgress(Progress("done", 1f))
        return an.report(shots, names, handed, meta).put("ballTrack", org.json.JSONArray(balls.map { b ->
            if (b == null) JSONObject.NULL else org.json.JSONArray().put(b[0].toDouble()).put(b[1].toDouble())
        }))
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0])); val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6f)
    }

    private fun graySmall(f: BgrFrame): FloatArray {
        val w = 160; val h = 90; val out = FloatArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val sx = (x + 0.5) * f.width / w; val sy = (y + 0.5) * f.height / h
            val xi = min(f.width - 1, sx.toInt()); val yi = min(f.height - 1, sy.toInt())
            out[y * w + x] = 0.114f * f.b(xi, yi) + 0.587f * f.g(xi, yi) + 0.299f * f.r(xi, yi)
        }
        return out
    }

    /** Per-pixel median of BG_SAMPLES evenly spaced frames in [t0, t1), planar BGR/255 at 512x288. */
    private fun background(video: Uri, t0: Long, t1: Long, info: VideoInfo): FloatArray {
        val W = BallTracker.W; val H = BallTracker.H
        val mmr = MediaMetadataRetriever()
        mmr.setDataSource(context, video)
        val end = min(t1, if (info.durationUs > 0) info.durationUs else t1)
        val samples = ArrayList<IntArray>()
        for (k in 0 until BG_SAMPLES) {
            val t = t0 + (end - t0) * k / max(1, BG_SAMPLES - 1)
            val bmp: Bitmap? = if (Build.VERSION.SDK_INT >= 27) mmr.getScaledFrameAtTime(t, MediaMetadataRetriever.OPTION_CLOSEST_SYNC, W, H)
            else mmr.getFrameAtTime(t, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)?.let { Bitmap.createScaledBitmap(it, W, H, true) }
            bmp?.let { val s = if (it.width != W || it.height != H) Bitmap.createScaledBitmap(it, W, H, true) else it
                val px = IntArray(W * H); s.getPixels(px, 0, W, 0, 0, W, H); samples.add(px) }
        }
        mmr.release()
        val out = FloatArray(3 * W * H)
        if (samples.isEmpty()) return out
        val vals = IntArray(samples.size); val plane = W * H
        for (p in 0 until plane) for (c in 0 until 3) {
            val shift = when (c) { 0 -> 0; 1 -> 8; else -> 16 } // B, G, R from ARGB
            for (k in samples.indices) vals[k] = (samples[k][p] shr shift) and 255
            vals.sort()
            val m = if (vals.size % 2 == 1) vals[vals.size / 2].toFloat() else (vals[vals.size / 2 - 1] + vals[vals.size / 2]) / 2f
            out[c * plane + p] = m / 255f
        }
        return out
    }
}

/** Stored court calibration for a session. */
class Calibration(val imageFromCourt: Homography, val width: Int, val height: Int, val rmsPx: Double?) {
    val courtFromImage: Homography = imageFromCourt.inverse()
    fun toJson(): JSONObject = JSONObject().put("imageFromCourt", org.json.JSONArray(imageFromCourt.toList().map { org.json.JSONArray(it) }))
        .put("width", width).put("height", height).put("reprojectionRmsPx", rmsPx ?: JSONObject.NULL)

    companion object {
        fun fromJson(o: JSONObject): Calibration {
            val a = o.getJSONArray("imageFromCourt")
            val rows = (0 until 3).map { r -> (0 until 3).map { c -> a.getJSONArray(r).getDouble(c) } }
            return Calibration(Homography.fromList(rows), o.optInt("width"), o.optInt("height"),
                if (o.isNull("reprojectionRmsPx")) null else o.optDouble("reprojectionRmsPx"))
        }
    }
}
