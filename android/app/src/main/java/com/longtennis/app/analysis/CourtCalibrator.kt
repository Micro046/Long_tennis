package com.longtennis.app.analysis

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Setup-time court calibration: rough user markers -> projected court lines ->
 * snap each line to white paint -> intersections -> refined homography.
 * Port of scripts/pipeline/court.py (calibrate).
 */
object CourtCalibrator {

    data class Result(
        val imageFromCourt: Homography,
        val courtFromImage: Homography,
        val linesSnapped: List<String>,
        val keypointsUsed: List<String>,
        val reprojectionRmsPx: Double?,
        val width: Int,
        val height: Int,
    )

    /** argb: packed ARGB pixels (Bitmap.getPixels order). */
    fun whiteMask(argb: IntArray, w: Int, h: Int): ByteArray {
        val v = IntArray(w * h); val s = IntArray(w * h)
        for (i in 0 until w * h) {
            val p = argb[i]
            val r = (p shr 16) and 255; val g = (p shr 8) and 255; val b = p and 255
            val mx = max(r, max(g, b)); val mn = min(r, min(g, b))
            v[i] = mx; s[i] = if (mx == 0) 0 else ((mx - mn) * 255 + mx / 2) / mx
        }
        // 31x31 box mean of V via integral image
        val integ = LongArray((w + 1) * (h + 1))
        for (y in 0 until h) {
            var row = 0L
            for (x in 0 until w) { row += v[y * w + x]; integ[(y + 1) * (w + 1) + x + 1] = integ[y * (w + 1) + x + 1] + row }
        }
        val r = 15
        val mask = ByteArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (s[i] >= 70 || v[i] <= 150) continue
            val x1 = max(0, x - r); val x2 = min(w, x + r + 1); val y1 = max(0, y - r); val y2 = min(h, y + r + 1)
            val sum = integ[y2 * (w + 1) + x2] - integ[y1 * (w + 1) + x2] - integ[y2 * (w + 1) + x1] + integ[y1 * (w + 1) + x1]
            val local = sum.toDouble() / ((x2 - x1) * (y2 - y1))
            if (v[i] > local + 25) mask[i] = 1
        }
        return mask
    }

    data class Line(val p: Pt, val d: Pt, val support: Int)

    fun snapLine(mask: ByteArray, w: Int, h: Int, p1: Pt, p2: Pt, band: Int, minPixels: Int = 40): Line? {
        val dv = p2 - p1; val len = dv.norm()
        if (len < 1) return null
        val u = dv * (1 / len); val n = Pt(-u.y, u.x)
        val xs = ArrayList<Double>(); val ys = ArrayList<Double>()
        val seen = HashSet<Int>()
        var t = 0.0
        while (t < len) {
            for (off in -band..band) {
                val q = p1 + u * t + n * off.toDouble()
                val xi = q.x.roundToInt(); val yi = q.y.roundToInt()
                if (xi in 0 until w && yi in 0 until h) {
                    val idx = yi * w + xi
                    if (mask[idx].toInt() != 0 && seen.add(idx)) { xs.add(xi.toDouble()); ys.add(yi.toDouble()) }
                }
            }
            t += 1.0
        }
        if (xs.size < minPixels) return null
        var px = xs.toDoubleArray(); var py = ys.toDoubleArray()
        var line = fitLine(px, py)
        repeat(3) {
            val keep = px.indices.filter { i -> abs((px[i] - line.p.x) * line.d.y - (py[i] - line.p.y) * line.d.x) < 2.5 }
            if (keep.size < minPixels) return Line(line.p, line.d, px.size)
            px = DoubleArray(keep.size) { px[keep[it]] }; py = DoubleArray(keep.size) { py[keep[it]] }
            line = fitLine(px, py)
        }
        return Line(line.p, line.d, px.size)
    }

    /** Total-least-squares line through points (principal axis). */
    private fun fitLine(x: DoubleArray, y: DoubleArray): Line {
        val mx = x.average(); val my = y.average()
        var sxx = 0.0; var syy = 0.0; var sxy = 0.0
        for (i in x.indices) { val dx = x[i] - mx; val dy = y[i] - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
        val theta = 0.5 * kotlin.math.atan2(2 * sxy, sxx - syy)
        return Line(Pt(mx, my), Pt(kotlin.math.cos(theta), kotlin.math.sin(theta)), x.size)
    }

    private fun intersect(a: Line, b: Line): Pt? {
        val cross = a.d.x * b.d.y - a.d.y * b.d.x
        if (abs(cross) < 1e-6) return null
        val q = b.p - a.p
        val t = (q.x * b.d.y - q.y * b.d.x) / cross
        return a.p + a.d * t
    }

    /** Fraction of in-image samples along projected court lines that lie on white paint (+-2 px). */
    fun lineSupport(mask: ByteArray, w: Int, h: Int, H: Homography, tol: Int = 2): Double {
        var hit = 0; var tot = 0
        for ((_, seg) in Court.LINES) {
            val p = H.apply(seg.first) ?: continue; val q = H.apply(seg.second) ?: continue
            val n = max(2, ((q - p).norm() / 2.0).toInt())
            for (k in 0 until n) {
                val t = k.toDouble() / (n - 1); val x = (p.x + t * (q.x - p.x)).toInt(); val y = (p.y + t * (q.y - p.y)).toInt()
                if (x !in 0 until w || y !in 0 until h) continue
                tot++
                var on = false
                loop@ for (dy in -tol..tol) for (dx in -tol..tol) {
                    val xx = x + dx; val yy = y + dy
                    if (xx in 0 until w && yy in 0 until h && mask[yy * w + xx].toInt() != 0) { on = true; break@loop }
                }
                if (on) hit++
            }
        }
        return if (tot == 0) 0.0 else hit.toDouble() / tot
    }

    fun calibrate(argb: IntArray, w: Int, h: Int, init: Map<String, Pt>, iterations: Int = 3, band: Int = 12): Result? {
        val names = init.keys.toList()
        var H = Homography.fit(names.map { Court.MODEL_KP.getValue(it) }, names.map { init.getValue(it) }) ?: return null
        val mask = whiteMask(argb, w, h)
        var fitted = mapOf<String, Line>()
        var used = listOf<String>(); var dst = listOf<Pt>()
        for (it in 0 until iterations) {
            val f = LinkedHashMap<String, Line>()
            for ((name, seg) in Court.LINES) {
                val a = H.apply(seg.first) ?: continue; val b = H.apply(seg.second) ?: continue
                snapLine(mask, w, h, a, b, if (it == 0) band else max(4, band / 2))?.let { f[name] = it }
            }
            val src = ArrayList<Pt>(); val d = ArrayList<Pt>(); val u = ArrayList<String>()
            for ((kp, pair) in Court.KEYPOINTS) {
                val la = f[pair.first] ?: continue; val lb = f[pair.second] ?: continue
                val p = intersect(la, lb) ?: continue
                if (p.x > -0.5 * w && p.x < 1.5 * w && p.y > -0.5 * h && p.y < 1.5 * h) {
                    src.add(Court.MODEL_KP.getValue(kp)); d.add(p); u.add(kp)
                }
            }
            fitted = f
            if (src.size < 4) break
            // candidates: all keypoints, near half, far half; keep the one whose lines sit on most white paint
            val subsets = listOf(src.indices.toList(),
                src.indices.filter { src[it].y > Court.NET_Y }, src.indices.filter { src[it].y < Court.NET_Y })
            var best: Triple<Double, Homography, List<Int>>? = null
            for (sub in subsets) {
                if (sub.size < 4) continue
                val hc = Homography.fitRobust(sub.map { src[it] }, sub.map { d[it] }) ?: continue
                val sc = lineSupport(mask, w, h, hc)
                if (best == null || sc > best.first) best = Triple(sc, hc, sub)
            }
            val b = best ?: break
            H = b.second
            val keep = src.indices.filter { ((H.apply(src[it]) ?: Pt(1e9, 1e9)) - d[it]).norm() < 4.0 }
            used = keep.map { u[it] }; dst = keep.map { d[it] }
        }
        val errs = used.indices.map { i -> (H.apply(Court.MODEL_KP.getValue(used[i]))!! - dst[i]).norm() }
        val rms = if (errs.isEmpty()) null else sqrt(errs.sumOf { it * it } / errs.size)
        return Result(H, H.inverse(), fitted.keys.sorted(), used, rms, w, h)
    }
}
