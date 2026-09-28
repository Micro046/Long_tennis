package com.longtennis.app.analysis

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

object Ort {
    val env: OrtEnvironment by lazy { OrtEnvironment.getEnvironment() }

    fun session(context: Context, asset: String, threads: Int = 4): OrtSession {
        val f = File(context.cacheDir, asset.substringAfterLast('/'))
        if (!f.exists() || f.length() == 0L) context.assets.open(asset).use { i -> f.outputStream().use { i.copyTo(it) } }
        fun options(xnnpack: Boolean) = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (xnnpack) {
                // XNNPACK: optimised ARM kernels for the conv-heavy models; ORT threads = 1 per XNNPACK docs
                addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                setIntraOpNumThreads(1)
                addConfigEntry("session.intra_op.allow_spinning", "0")
            } else setIntraOpNumThreads(threads)
        }
        return runCatching { env.createSession(f.absolutePath, options(true)) }
            .getOrElse { env.createSession(f.absolutePath, options(false)) }
    }
}

/**
 * RacketVision TrackNetV3 ball tracker: input = [median background, f(t-3), f(t-2), f(t-1), f(t)]
 * as 15 x 288 x 512 BGR/255, output = heatmap for frame t. Port of scripts/pipeline/ball.py.
 */
class BallTracker(context: Context, model: String = chooseModel(context)) : AutoCloseable {
    companion object {
        const val W = 512; const val H = 288; const val SEQ = 4; const val PLANE = 3 * W * H
        /** "fast" -> distilled student (if packaged), "accurate" -> full RacketVision model. */
        fun chooseModel(context: Context, quality: String = "fast"): String =
            if (quality == "fast" && context.assets.list("models")?.contains("ball_student_d16.onnx") == true) "models/ball_student_d16.onnx"
            else "models/ball_tracknetv3.onnx"
        /** Hit/bounce model trained on tracks from the same ball model. */
        fun eventModelFor(context: Context, ballModel: String): String =
            if (ballModel.contains("student") && context.assets.list("models")?.contains("event_gbm_student.json") == true)
                "models/event_gbm_student.json" else "models/event_gbm.json"
    }
    private val session = Ort.session(context, model)
    private val window = ArrayDeque<FloatArray>()
    private var median: FloatArray? = null
    private val input = FloatArray((SEQ + 1) * PLANE)

    fun setBackground(bg: FloatArray) { median = bg; window.clear() }

    /** Pushes frame t and returns the ball position (original px) + confidence, or null. */
    fun push(frame: BgrFrame, thr: Float = 0.5f): FloatArray? {
        val planar = FloatArray(PLANE)
        Resample.areaToPlanar(frame, W, H, planar, 0)
        window.addLast(planar)
        if (window.size > SEQ) window.removeFirst()
        val bg = median ?: return null
        System.arraycopy(bg, 0, input, 0, PLANE)
        val pad = SEQ - window.size
        for (k in 0 until SEQ) {
            val src = if (k < pad) window.first() else window.elementAt(k - pad)
            System.arraycopy(src, 0, input, (k + 1) * PLANE, PLANE)
        }
        val t = OnnxTensor.createTensor(Ort.env, FloatBuffer.wrap(input), longArrayOf(1, 15, H.toLong(), W.toLong()))
        val hm = t.use { session.run(mapOf("frames" to it)).use { r -> (r[0] as OnnxTensor).floatBuffer.let { b -> FloatArray(b.remaining()).also { b.get(it) } } } }
        val pt = blob(hm, thr) ?: return null
        return floatArrayOf(pt[0] * frame.width / W, pt[1] * frame.height / H, pt[2])
    }

    /** Largest 8-connected component above thr: (cx, cy, mean value in its bounding box). */
    private fun blob(hm: FloatArray, thr: Float): FloatArray? {
        val label = IntArray(hm.size)
        var best: IntArray? = null; var bestArea = 0; var cur = 0
        val stack = IntArray(hm.size)
        for (i in hm.indices) {
            if (hm[i] <= thr || label[i] != 0) continue
            cur++; var sp = 0; stack[sp++] = i; label[i] = cur
            var area = 0; var x1 = W; var y1 = H; var x2 = -1; var y2 = -1
            while (sp > 0) {
                val p = stack[--sp]; area++
                val x = p % W; val y = p / W
                x1 = min(x1, x); x2 = max(x2, x); y1 = min(y1, y); y2 = max(y2, y)
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = x + dx; val ny = y + dy
                    if (nx in 0 until W && ny in 0 until H) {
                        val q = ny * W + nx
                        if (label[q] == 0 && hm[q] > thr) { label[q] = cur; stack[sp++] = q }
                    }
                }
            }
            // cv2 CC_STAT_AREA = pixel count
            if (area > bestArea) { bestArea = area; best = intArrayOf(x1, y1, x2, y2) }
        }
        val b = best ?: return null
        var s = 0.0; var n = 0
        for (y in b[1]..b[3]) for (x in b[0]..b[2]) { s += hm[y * W + x]; n++ }
        val w = b[2] - b[0] + 1; val h = b[3] - b[1] + 1
        return floatArrayOf(b[0] + w / 2f, b[1] + h / 2f, (s / n).toFloat())
    }

    override fun close() = session.close()
}

/** A detected person: box, score, court position of the feet, 17 keypoints (x, y, score). */
class Person(val box: FloatArray, val score: Float, val court: Pt, val kps: FloatArray) {
    var desc: FloatArray? = null
    var overlap = false
    fun kx(j: Int) = kps[j * 3]; fun ky(j: Int) = kps[j * 3 + 1]; fun ks(j: Int) = kps[j * 3 + 2]
    val height get() = max(20f, box[3] - box[1])
}

/** YOLOX-tiny person detector (full frame + far-court tiles) and RTMPose-s. Port of pose.py. */
class PoseEstimator(context: Context, private val courtFromImage: Homography, imageFromCourt: Homography,
                    width: Int, height: Int) : AutoCloseable {
    private val det = Ort.session(context, "models/yolox_tiny_person.onnx")
    private val pose = Ort.session(context, "models/rtmpose_s_body.onnx")
    val tiles: List<IntArray> = farTiles(imageFromCourt, width, height)

    companion object {
        const val DET = 416; const val PW = 192; const val PH = 256
        const val MARGIN_X = 4.0; const val MARGIN_BACK = 7.0
        private val MEAN = floatArrayOf(123.675f, 116.28f, 103.53f)
        private val STD = floatArrayOf(58.395f, 57.12f, 57.375f)

        fun farTiles(H: Homography, w: Int, h: Int, n: Int = 2): List<IntArray> {
            val pts = listOf(Pt(-MARGIN_X, -MARGIN_BACK), Pt(Court.WIDTH + MARGIN_X, -MARGIN_BACK),
                Pt(-MARGIN_X, Court.NET_Y), Pt(Court.WIDTH + MARGIN_X, Court.NET_Y), Pt(0.0, 0.0), Pt(1.0, 0.0)).map { H.apply(it)!! }
            val pxPerM = (pts[5] - pts[4]).norm()
            val x1 = max(0.0, pts.take(4).minOf { it.x }); val x2 = min(w.toDouble(), pts.take(4).maxOf { it.x })
            val y2 = min(h.toDouble(), pts.take(4).maxOf { it.y })
            val y1 = max(0.0, pts.take(4).minOf { it.y } - 2.6 * pxPerM)
            val tw = (x2 - x1) / n
            return (0 until n).map { i ->
                val a = x1 + i * tw - (if (i > 0) 0.15 * tw else 0.0)
                val b = x1 + (i + 1) * tw + (if (i < n - 1) 0.15 * tw else 0.0)
                intArrayOf(a.toInt(), y1.toInt(), b.toInt(), y2.toInt())
            }
        }
    }

    /** Returns boxes [x1,y1,x2,y2,score] in frame coordinates for a crop region. */
    private fun detect(f: BgrFrame, cx1: Int, cy1: Int, cx2: Int, cy2: Int, minScore: Float = 0.3f): List<FloatArray> {
        val cw = cx2 - cx1; val ch = cy2 - cy1
        val ratio = min(DET.toDouble() / ch, DET.toDouble() / cw)
        val rw = (cw * ratio).toInt(); val rh = (ch * ratio).toInt()
        val inp = FloatArray(3 * DET * DET) { 114f }
        val plane = DET * DET
        for (y in 0 until rh) for (x in 0 until rw) {
            // cv2 INTER_LINEAR sampling position (pixel centres)
            val sx = cx1 + (x + 0.5) / ratio - 0.5; val sy = cy1 + (y + 0.5) / ratio - 0.5
            val p = y * DET + x
            for (c in 0 until 3) inp[c * plane + p] = Resample.bilinear(f, sx, sy, c).roundToInt().toFloat()
        }
        val t = OnnxTensor.createTensor(Ort.env, FloatBuffer.wrap(inp), longArrayOf(1, 3, DET.toLong(), DET.toLong()))
        val out = ArrayList<FloatArray>()
        t.use {
            det.run(mapOf(det.inputNames.first() to it)).use { r ->
                @Suppress("UNCHECKED_CAST")
                val dets = (r[0].value as Array<Array<FloatArray>>)[0]
                for (d in dets) if (d[4] > minScore) out.add(floatArrayOf(
                    (d[0] / ratio).toFloat() + cx1, (d[1] / ratio).toFloat() + cy1,
                    (d[2] / ratio).toFloat() + cx1, (d[3] / ratio).toFloat() + cy1, d[4]))
            }
        }
        return out
    }

    fun process(f: BgrFrame): List<Person> {
        val all = ArrayList(detect(f, 0, 0, f.width, f.height))
        for (tl in tiles) {
            val tw = tl[2] - tl[0]; val th = tl[3] - tl[1]
            for (d in detect(f, tl[0], tl[1], tl[2], tl[3])) {
                val bottomOk = d[3] - tl[1] < th - 4
                val leftOk = d[0] - tl[0] > 2 || tl[0] == 0
                val rightOk = d[2] - tl[0] < tw - 2 || tl[2] >= f.width - 1
                if (bottomOk && leftOk && rightOk) all.add(d)
            }
        }
        val kept = suppressContained(nms(all, 0.5f))
        val people = ArrayList<Person>()
        for (d in kept) {
            val feet = courtFromImage.apply(Pt(((d[0] + d[2]) / 2).toDouble(), d[3].toDouble())) ?: continue
            if (feet.x < -MARGIN_X || feet.x > Court.WIDTH + MARGIN_X || feet.y < -MARGIN_BACK || feet.y > Court.LENGTH + MARGIN_BACK) continue
            people.add(Person(d.copyOf(4), d[4], feet, keypoints(f, d)))
        }
        return people
    }

    private fun keypoints(f: BgrFrame, b: FloatArray): FloatArray {
        val cx = (b[0] + b[2]) / 2.0; val cy = (b[1] + b[3]) / 2.0
        var sw = (b[2] - b[0]) * 1.25; var sh = (b[3] - b[1]) * 1.25
        val aspect = PW.toDouble() / PH
        if (sw > sh * aspect) sh = sw / aspect else sw = sh * aspect
        val inp = FloatArray(3 * PW * PH); val plane = PW * PH
        for (v in 0 until PH) for (u in 0 until PW) {
            val sx = cx + (u - PW / 2.0) * sw / PW; val sy = cy + (v - PH / 2.0) * sh / PH
            val p = v * PW + u
            for (c in 0 until 3) {
                val px = if (sx < 0 || sy < 0 || sx > f.width - 1 || sy > f.height - 1) 0f else Resample.bilinear(f, sx, sy, c)
                inp[c * plane + p] = (px - MEAN[c]) / STD[c]
            }
        }
        val out = FloatArray(17 * 3)
        val t = OnnxTensor.createTensor(Ort.env, FloatBuffer.wrap(inp), longArrayOf(1, 3, PH.toLong(), PW.toLong()))
        t.use {
            pose.run(mapOf(pose.inputNames.first() to it)).use { r ->
                @Suppress("UNCHECKED_CAST") val sx = (r[0].value as Array<Array<FloatArray>>)[0]
                @Suppress("UNCHECKED_CAST") val sy = (r[1].value as Array<Array<FloatArray>>)[0]
                for (k in 0 until 17) {
                    var ix = 0; var iy = 0
                    for (i in sx[k].indices) if (sx[k][i] > sx[k][ix]) ix = i
                    for (i in sy[k].indices) if (sy[k][i] > sy[k][iy]) iy = i
                    val score = (sx[k][ix] + sy[k][iy]) / 2
                    val lx = ix / 2.0; val ly = iy / 2.0
                    out[k * 3] = (lx / PW * sw + cx - sw / 2).toFloat()
                    out[k * 3 + 1] = (ly / PH * sh + cy - sh / 2).toFloat()
                    out[k * 3 + 2] = score
                }
            }
        }
        return out
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0])); val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
        val inter = iw * ih
        return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-6f)
    }

    private fun nms(d: List<FloatArray>, thr: Float): List<FloatArray> {
        val out = ArrayList<FloatArray>()
        for (x in d.sortedByDescending { it[4] }) if (out.none { iou(it, x) > thr }) out.add(x)
        return out
    }

    private fun suppressContained(d: List<FloatArray>, frac: Float = 0.6f): List<FloatArray> {
        val out = ArrayList<FloatArray>()
        for (a in d.sortedByDescending { (it[2] - it[0]) * (it[3] - it[1]) }) {
            val area = max(1e-6f, (a[2] - a[0]) * (a[3] - a[1]))
            val inside = out.any { b ->
                val iw = max(0f, min(a[2], b[2]) - max(a[0], b[0])); val ih = max(0f, min(a[3], b[3]) - max(a[1], b[1]))
                iw * ih / area > frac
            }
            if (!inside) out.add(a)
        }
        return out
    }

    override fun close() { det.close(); pose.close() }
}
