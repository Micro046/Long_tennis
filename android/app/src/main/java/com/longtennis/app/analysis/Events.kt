package com.longtennis.app.analysis

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round
import kotlin.math.sign

/** Port of scripts/pipeline/events.py (feature order must match exactly). */
object Events {
    val SCALES = intArrayOf(1, 2, 3, 5, 8)
    const val MAX_GAP = 6

    /** xy: [n][2] image px / image height, NaN = missing. Returns cleaned copy + observed mask. */
    fun cleanTrack(input: Array<DoubleArray>, maxJump: Double = 0.12): Pair<Array<DoubleArray>, BooleanArray> {
        val n = input.size
        val xy = Array(n) { input[it].copyOf() }
        fun fin(i: Int) = !xy[i][0].isNaN() && !xy[i][1].isNaN()
        fun d(a: Int, b: Int) = hypot(xy[a][0] - xy[b][0], xy[a][1] - xy[b][1])
        for (i in 1 until n - 1) {
            if (fin(i - 1) && fin(i) && fin(i + 1) && d(i, i - 1) > maxJump && d(i, i + 1) > maxJump && d(i - 1, i + 1) < maxJump)
                xy[i] = doubleArrayOf(Double.NaN, Double.NaN)
        }
        val f = BooleanArray(n) { fin(it) }
        for (i in 0 until n) {
            if (!f[i]) continue
            var c = 0
            for (j in max(0, i - 2) until min(n, i + 3)) if (f[j]) c++
            if (c == 1) xy[i] = doubleArrayOf(Double.NaN, Double.NaN)
        }
        val observed = BooleanArray(n) { fin(it) }
        val idx = (0 until n).filter { observed[it] }
        for (k in 0 until idx.size - 1) {
            val a = idx[k]; val b = idx[k + 1]
            if (b - a in 2..MAX_GAP + 1) for (j in a + 1 until b) {
                val t = (j - a).toDouble() / (b - a)
                xy[j] = doubleArrayOf(xy[a][0] * (1 - t) + xy[b][0] * t, xy[a][1] * (1 - t) + xy[b][1] * t)
            }
        }
        return xy to observed
    }

    private fun safe(v: Double) = if (v.isNaN() || v.isInfinite()) 0.0 else v
    private fun sgn(v: Double) = if (v.isNaN()) Double.NaN else sign(v)

    /** Returns [n][nFeatures] float features. */
    fun features(xy: Array<DoubleArray>, observed: BooleanArray, fps: Double): Array<FloatArray> {
        val n = xy.size
        val r = fps / 30.0
        val cols = ArrayList<DoubleArray>()
        fun at(i: Int, c: Int) = if (i in 0 until n) xy[i][c] else Double.NaN
        for (s in SCALES) {
            val k = max(1, round(s * r).toInt())
            val bx = DoubleArray(n); val by = DoubleArray(n); val ax = DoubleArray(n); val ay = DoubleArray(n)
            for (i in 0 until n) {
                bx[i] = (xy[i][0] - at(i - k, 0)) / s; by[i] = (xy[i][1] - at(i - k, 1)) / s
                ax[i] = (at(i + k, 0) - xy[i][0]) / s; ay[i] = (at(i + k, 1) - xy[i][1]) / s
            }
            val nb = DoubleArray(n) { hypot(bx[it], by[it]) }; val na = DoubleArray(n) { hypot(ax[it], ay[it]) }
            val cos = DoubleArray(n) { (bx[it] * ax[it] + by[it] * ay[it]) / (nb[it] * na[it] + 1e-9) }
            val lsr = DoubleArray(n) { ln((na[it] + 1e-4) / (nb[it] + 1e-4)) }
            val accx = DoubleArray(n) { ax[it] - bx[it] }; val accy = DoubleArray(n) { ay[it] - by[it] }
            val an = DoubleArray(n) { hypot(accx[it], accy[it]) }
            val flip = DoubleArray(n) { sgn(by[it]) * sgn(ay[it]) }
            val dvy = DoubleArray(n) { abs(ay[it]) - abs(by[it]) }
            cols.addAll(listOf(bx, by, ax, ay, nb, na, cos, lsr, accx, accy, an, flip, dvy))
        }
        val k3 = max(1, round(3 * r).toInt())
        val kink = DoubleArray(n); val kinky = DoubleArray(n)
        for (i in 0 until n) {
            val mx = (at(i - k3, 0) + at(i + k3, 0)) / 2; val my = (at(i - k3, 1) + at(i + k3, 1)) / 2
            kink[i] = hypot(xy[i][0] - mx, xy[i][1] - my); kinky[i] = xy[i][1] - my
        }
        cols.add(kink); cols.add(kinky)
        for (w in intArrayOf(3, 8)) {
            val kk = max(1, round(w * r).toInt())
            cols.add(DoubleArray(n) { i ->
                var c = 0
                for (j in i - kk..i + kk) if (j in 0 until n && observed[j]) c++
                c.toDouble() / (2 * kk + 1)
            })
        }
        cols.add(DoubleArray(n) { if (observed[it]) 1.0 else 0.0 })
        return Array(n) { i -> FloatArray(cols.size) { c -> safe(cols[c][i]).toFloat() } }
    }

    /** Local maxima above thr, at least radius frames apart (greedy by probability). */
    fun peaks(prob: DoubleArray, thr: Double, radius: Int): List<Int> {
        // deterministic across platforms: 6-decimal probabilities, ties -> earlier frame
        val order = prob.indices.sortedWith(compareBy({ -Math.round(prob[it] * 1e6) }, { it }))
        val taken = BooleanArray(prob.size); val out = ArrayList<Int>()
        for (i in order) {
            if (prob[i] < thr) break
            var blocked = false
            for (j in max(0, i - radius)..min(prob.size - 1, i + radius)) if (taken[j]) { blocked = true; break }
            if (blocked) continue
            taken[i] = true; out.add(i)
        }
        return out.sorted()
    }
}

/** HistGradientBoosting multiclass evaluator (JSON exported by scripts/export_event_model.py). */
class GbmModel(json: String) {
    private class Tree(val f: IntArray, val t: DoubleArray, val l: IntArray, val r: IntArray, val leaf: BooleanArray, val v: DoubleArray, val ml: BooleanArray)
    private val baseline: DoubleArray
    private val trees: List<List<Tree>>
    val nClasses: Int

    init {
        val o = JSONObject(json)
        baseline = o.getJSONArray("baseline").let { a -> DoubleArray(a.length()) { a.getDouble(it) } }
        nClasses = baseline.size
        fun ints(a: JSONArray) = IntArray(a.length()) { a.getInt(it) }
        fun dbl(a: JSONArray) = DoubleArray(a.length()) { a.getDouble(it) }
        val ta = o.getJSONArray("trees")
        trees = (0 until ta.length()).map { i ->
            val pc = ta.getJSONArray(i)
            (0 until pc.length()).map { c ->
                val t = pc.getJSONObject(c)
                Tree(ints(t.getJSONArray("f")), dbl(t.getJSONArray("t")), ints(t.getJSONArray("l")), ints(t.getJSONArray("r")),
                    ints(t.getJSONArray("leaf")).map { it != 0 }.toBooleanArray(), dbl(t.getJSONArray("v")),
                    ints(t.getJSONArray("ml")).map { it != 0 }.toBooleanArray())
            }
        }
    }

    fun predictProba(x: FloatArray): DoubleArray {
        val raw = baseline.copyOf()
        for (it in trees) for (c in it.indices) {
            val t = it[c]; var node = 0
            while (!t.leaf[node]) {
                val v = x[t.f[node]].toDouble()
                node = if (v.isNaN()) (if (t.ml[node]) t.l[node] else t.r[node]) else if (v <= t.t[node]) t.l[node] else t.r[node]
            }
            raw[c] += t.v[node]
        }
        val m = raw.max(); val e = raw.map { exp(it - m) }; val s = e.sum()
        return DoubleArray(raw.size) { e[it] / s }
    }
}
