package com.longtennis.app.analysis

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Court model in metres. Origin = far-left doubles corner as seen from the camera,
 * x across (0..10.97), y far -> near (0..23.77). Mirrors scripts/pipeline/court.py.
 */
object Court {
    const val WIDTH = 10.97
    const val LENGTH = 23.77
    const val SINGLES = 8.23
    const val SL = (WIDTH - SINGLES) / 2
    const val NET_Y = LENGTH / 2
    const val SERVICE = 6.40
    const val FAR_SERVICE_Y = NET_Y - SERVICE
    const val NEAR_SERVICE_Y = NET_Y + SERVICE
    const val CX = WIDTH / 2

    val LINES: Map<String, Pair<Pt, Pt>> = linkedMapOf(
        "far_baseline" to (Pt(0.0, 0.0) to Pt(WIDTH, 0.0)),
        "near_baseline" to (Pt(0.0, LENGTH) to Pt(WIDTH, LENGTH)),
        "left_doubles" to (Pt(0.0, 0.0) to Pt(0.0, LENGTH)),
        "right_doubles" to (Pt(WIDTH, 0.0) to Pt(WIDTH, LENGTH)),
        "left_singles" to (Pt(SL, 0.0) to Pt(SL, LENGTH)),
        "right_singles" to (Pt(WIDTH - SL, 0.0) to Pt(WIDTH - SL, LENGTH)),
        "far_service" to (Pt(SL, FAR_SERVICE_Y) to Pt(WIDTH - SL, FAR_SERVICE_Y)),
        "near_service" to (Pt(SL, NEAR_SERVICE_Y) to Pt(WIDTH - SL, NEAR_SERVICE_Y)),
        "centre_service" to (Pt(CX, FAR_SERVICE_Y) to Pt(CX, NEAR_SERVICE_Y)),
    )

    val KEYPOINTS: Map<String, Pair<String, String>> = linkedMapOf(
        "far_left_doubles" to ("far_baseline" to "left_doubles"),
        "far_left_singles" to ("far_baseline" to "left_singles"),
        "far_right_singles" to ("far_baseline" to "right_singles"),
        "far_right_doubles" to ("far_baseline" to "right_doubles"),
        "near_left_doubles" to ("near_baseline" to "left_doubles"),
        "near_left_singles" to ("near_baseline" to "left_singles"),
        "near_right_singles" to ("near_baseline" to "right_singles"),
        "near_right_doubles" to ("near_baseline" to "right_doubles"),
        "far_service_left" to ("far_service" to "left_singles"),
        "far_service_right" to ("far_service" to "right_singles"),
        "far_service_centre" to ("far_service" to "centre_service"),
        "near_service_left" to ("near_service" to "left_singles"),
        "near_service_right" to ("near_service" to "right_singles"),
        "near_service_centre" to ("near_service" to "centre_service"),
    )

    val MODEL_KP: Map<String, Pt> = KEYPOINTS.mapValues { (_, v) ->
        val a = LINES.getValue(v.first); val b = LINES.getValue(v.second)
        if (a.first.x == a.second.x) Pt(a.first.x, b.first.y) else Pt(b.first.x, a.first.y)
    }

    /** Default setup markers: the four points the user drags onto the court. */
    val SETUP_POINTS = listOf("far_left_singles", "far_right_singles", "near_service_left", "near_service_right")

    /**
     * Six target regions of the opponent's singles half, from the hitter's point of view.
     * Returns status in {in, in_near_line, out, hitter_half, unknown} and region name.
     */
    fun region(pt: Pt?, hitterSide: String?, bufferM: Double = 0.15): Pair<String, String?> {
        if (pt == null || (hitterSide != "near" && hitterSide != "far")) return "unknown" to null
        val depth: Double; val lat: Double
        if (hitterSide == "near") { depth = NET_Y - pt.y; lat = pt.x - SL } else { depth = pt.y - NET_Y; lat = (WIDTH - SL) - pt.x }
        if (depth < -bufferM) return "hitter_half" to null
        if (lat < -bufferM || lat > SINGLES + bufferM || depth > NET_Y + bufferM) return "out" to null
        val lateral = arrayOf("left", "centre", "right")[min(2, max(0, (lat / (SINGLES / 3)).toInt()))]
        val d = if (depth < SERVICE) "short" else "deep"
        val nearLine = minOf(abs(lat), abs(lat - SINGLES), abs(depth - NET_Y)) < bufferM
        return (if (nearLine) "in_near_line" else "in") to "${d}_$lateral"
    }

    val REGIONS = listOf("deep_left", "deep_centre", "deep_right", "short_left", "short_centre", "short_right")
}

data class Pt(val x: Double, val y: Double) {
    operator fun minus(o: Pt) = Pt(x - o.x, y - o.y)
    operator fun plus(o: Pt) = Pt(x + o.x, y + o.y)
    operator fun times(s: Double) = Pt(x * s, y * s)
    fun norm() = hypot(x, y)
}

/** 3x3 homography, row-major. */
class Homography(val m: DoubleArray) {
    fun apply(p: Pt): Pt? {
        val w = m[6] * p.x + m[7] * p.y + m[8]
        if (abs(w) < 1e-12) return null
        return Pt((m[0] * p.x + m[1] * p.y + m[2]) / w, (m[3] * p.x + m[4] * p.y + m[5]) / w)
    }

    fun inverse(): Homography {
        val a = m
        val det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) + a[2] * (a[3] * a[7] - a[4] * a[6])
        val inv = doubleArrayOf(
            a[4] * a[8] - a[5] * a[7], a[2] * a[7] - a[1] * a[8], a[1] * a[5] - a[2] * a[4],
            a[5] * a[6] - a[3] * a[8], a[0] * a[8] - a[2] * a[6], a[2] * a[3] - a[0] * a[5],
            a[3] * a[7] - a[4] * a[6], a[1] * a[6] - a[0] * a[7], a[0] * a[4] - a[1] * a[3],
        )
        return Homography(DoubleArray(9) { inv[it] / det })
    }

    fun toList(): List<List<Double>> = listOf(m.slice(0..2), m.slice(3..5), m.slice(6..8))

    companion object {
        fun fromList(rows: List<List<Double>>) = Homography(rows.flatten().toDoubleArray())

        /** Least-squares DLT with h33 = 1 (normal equations), points normalised for stability. */
        fun fit(src: List<Pt>, dst: List<Pt>): Homography? {
            if (src.size < 4 || src.size != dst.size) return null
            val (ns, ts) = normalise(src); val (nd, td) = normalise(dst)
            val ata = Array(8) { DoubleArray(8) }; val atb = DoubleArray(8)
            for (i in ns.indices) {
                val (x, y) = ns[i]; val (u, v) = nd[i]
                val r1 = doubleArrayOf(x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y)
                val r2 = doubleArrayOf(0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y)
                for ((r, b) in listOf(r1 to u, r2 to v)) {
                    for (p in 0 until 8) { atb[p] += r[p] * b; for (q in 0 until 8) ata[p][q] += r[p] * r[q] }
                }
            }
            val h = solve(ata, atb) ?: return null
            val hn = Homography(doubleArrayOf(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0))
            // denormalise: H = Td^-1 * Hn * Ts
            val full = mul(mul(inv3(td), hn.m), ts)
            val s = full[8]
            return Homography(DoubleArray(9) { full[it] / s })
        }

        /** Robust fit: iteratively drop the worst point while its error exceeds thrPx. */
        fun fitRobust(src: List<Pt>, dst: List<Pt>, thrPx: Double = 4.0): Homography? {
            var s = src; var d = dst
            while (true) {
                val h = fit(s, d) ?: return null
                if (s.size <= 4) return h
                val errs = s.indices.map { (h.apply(s[it])?.minus(d[it])?.norm()) ?: 1e9 }
                val worst = errs.indices.maxBy { errs[it] }
                if (errs[worst] <= thrPx) return h
                s = s.filterIndexed { i, _ -> i != worst }; d = d.filterIndexed { i, _ -> i != worst }
            }
        }

        private fun normalise(p: List<Pt>): Pair<List<Pair<Double, Double>>, DoubleArray> {
            val cx = p.sumOf { it.x } / p.size; val cy = p.sumOf { it.y } / p.size
            val md = p.sumOf { hypot(it.x - cx, it.y - cy) } / p.size
            val s = if (md < 1e-9) 1.0 else sqrt(2.0) / md
            return p.map { (it.x - cx) * s to (it.y - cy) * s } to doubleArrayOf(s, 0.0, -s * cx, 0.0, s, -s * cy, 0.0, 0.0, 1.0)
        }

        fun mul(a: DoubleArray, b: DoubleArray) = DoubleArray(9) { i ->
            val r = i / 3; val c = i % 3
            a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c]
        }

        private fun inv3(a: DoubleArray) = Homography(a).inverse().m

        /** Gaussian elimination with partial pivoting. */
        fun solve(a0: Array<DoubleArray>, b0: DoubleArray): DoubleArray? {
            val n = b0.size
            val a = Array(n) { a0[it].copyOf() }; val b = b0.copyOf()
            for (c in 0 until n) {
                var p = c
                for (r in c + 1 until n) if (abs(a[r][c]) > abs(a[p][c])) p = r
                if (abs(a[p][c]) < 1e-12) return null
                val t = a[c]; a[c] = a[p]; a[p] = t
                val tb = b[c]; b[c] = b[p]; b[p] = tb
                for (r in c + 1 until n) {
                    val f = a[r][c] / a[c][c]
                    for (k in c until n) a[r][k] -= f * a[c][k]
                    b[r] -= f * b[c]
                }
            }
            val x = DoubleArray(n)
            for (r in n - 1 downTo 0) {
                var s = b[r]
                for (k in r + 1 until n) s -= a[r][k] * x[k]
                x[r] = s / a[r][r]
            }
            return x
        }
    }
}
