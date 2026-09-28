package com.longtennis.app.analysis

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Port of scripts/pipeline/identity.py: whole-session A/B identity independent of court side. */
object Identity {
    const val HB = 12; const val SB = 4; const val VB = 4
    private const val PART = HB * SB + VB

    /** HSV (OpenCV scaling: H 0..180, S,V 0..255) histogram inside a convex polygon. */
    fun regionHist(f: BgrFrame, poly: List<Pt>, pad: Double = 0.0): FloatArray? {
        if (poly.size < 3) return null
        var pts = poly
        if (pad != 0.0) {
            val c = Pt(pts.sumOf { it.x } / pts.size, pts.sumOf { it.y } / pts.size)
            pts = pts.map { c + (it - c) * (1 + pad) }
        }
        val x1 = max(0, floor(pts.minOf { it.x }).toInt()); val y1 = max(0, floor(pts.minOf { it.y }).toInt())
        val x2 = min(f.width, ceil(pts.maxOf { it.x }).toInt()); val y2 = min(f.height, ceil(pts.maxOf { it.y }).toInt())
        if (x2 - x1 < 3 || y2 - y1 < 3) return null
        val hs = FloatArray(HB * SB); val vh = FloatArray(VB); var cnt = 0
        for (y in y1 until y2) for (x in x1 until x2) {
            if (!inside(pts, x.toDouble(), y.toDouble())) continue
            val b = f.b(x, y); val g = f.g(x, y); val r = f.r(x, y)
            val v = max(r, max(g, b)); val mn = min(r, min(g, b))
            val s = if (v == 0) 0 else (255 * (v - mn) + v / 2) / v
            var h = 0.0
            if (v != mn) {
                h = when (v) { r -> 60.0 * (g - b) / (v - mn); g -> 120.0 + 60.0 * (b - r) / (v - mn); else -> 240.0 + 60.0 * (r - g) / (v - mn) }
                if (h < 0) h += 360.0
            }
            val hh = min(179, (h / 2).toInt())
            hs[min(HB - 1, hh * HB / 180) * SB + min(SB - 1, s * SB / 256)] += 1f
            vh[min(VB - 1, v * VB / 256)] += 1f
            cnt++
        }
        if (cnt < 12) return null
        val out = FloatArray(PART)
        for (i in hs.indices) out[i] = 0.75f * hs[i] / cnt
        for (i in vh.indices) out[HB * SB + i] = 0.25f * vh[i] / cnt
        return out
    }

    private fun inside(p: List<Pt>, x: Double, y: Double): Boolean {
        var sgn = 0
        for (i in p.indices) {
            val a = p[i]; val b = p[(i + 1) % p.size]
            val c = (b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x)
            val s = if (c > 0) 1 else if (c < 0) -1 else 0
            if (s != 0) { if (sgn == 0) sgn = s else if (s != sgn) return false }
        }
        return true
    }

    fun appearance(f: BgrFrame, p: Person): FloatArray? {
        val ok = BooleanArray(17) { p.ks(it) > 0.3f }
        fun kp(j: Int) = Pt(p.kx(j).toDouble(), p.ky(j).toDouble())
        val upper = if (ok[5] && ok[6] && ok[11] && ok[12]) regionHist(f, listOf(kp(5), kp(6), kp(12), kp(11)), -0.1) else {
            val b = p.box; val hh = (b[3] - b[1]).toDouble(); val w = (b[2] - b[0]).toDouble()
            regionHist(f, listOf(Pt(b[0] + .3 * w, b[1] + .2 * hh), Pt(b[2] - .3 * w, b[1] + .2 * hh), Pt(b[2] - .3 * w, b[1] + .5 * hh), Pt(b[0] + .3 * w, b[1] + .5 * hh)))
        } ?: return null
        val lower = if (ok[11] && ok[12] && ok[13] && ok[14]) {
            val h0 = kp(11); val h1 = kp(12); val m0 = h0 + (kp(13) - h0) * 0.5; val m1 = h1 + (kp(14) - h1) * 0.5
            regionHist(f, listOf(h0, h1, m1, m0))
        } else null
        return upper + (lower ?: FloatArray(PART) { Float.NaN })
    }

    fun distance(a: FloatArray, b: FloatArray): Double {
        val ds = ArrayList<Double>()
        for (part in 0 until 2) {
            val o = part * PART
            if ((0 until PART).any { a[o + it].isNaN() || b[o + it].isNaN() }) continue
            var bc = 0.0; var sa = 0.0; var sb = 0.0
            for (i in 0 until PART) { bc += sqrt(max(0f, a[o + i]).toDouble() * max(0f, b[o + i])); sa += a[o + i]; sb += b[o + i] }
            bc /= sqrt(max(1e-9, sa * sb))
            ds.add(sqrt(max(0.0, 1 - bc)))
        }
        return if (ds.isEmpty()) 1.0 else ds.average()
    }

    class Tracklet(val id: Int, var pos: Pt, var vel: Pt, var last: Int) {
        val dets = LinkedHashMap<Int, Int>()
        var start = last; var end = last
        var desc: FloatArray? = null
        var label = "unknown"; var margin = 0.0; var seed: String? = null
        val overlaps = ArrayList<Int>()
        val n get() = dets.size
    }

    fun buildTracklets(frames: List<List<Person>?>, cuts: Set<Int>, gateM: Double = 1.2, maxGap: Int = 8): List<Tracklet> {
        var active = ArrayList<Tracklet>(); val done = ArrayList<Tracklet>(); var nextId = 0
        for (i in frames.indices) {
            val dets = frames[i] ?: continue
            if (i in cuts) { done.addAll(active); active = ArrayList() }
            val still = ArrayList<Tracklet>()
            for (t in active) if (i - t.last > maxGap) done.add(t) else still.add(t)
            active = still
            data class P(val d: Double, val ti: Int, val di: Int)
            val pairs = ArrayList<P>()
            for ((ti, t) in active.withIndex()) {
                val dt = i - t.last
                val pred = t.pos + t.vel * dt.toDouble()
                for ((di, d) in dets.withIndex()) {
                    val dist = (d.court - pred).norm()
                    if (dist < gateM + 0.25 * dt) pairs.add(P(dist, ti, di))
                }
            }
            pairs.sortBy { it.d }
            val usedT = HashSet<Int>(); val usedD = HashSet<Int>()
            for (p in pairs) {
                if (p.ti in usedT || p.di in usedD) continue
                val rivals = pairs.any { it.ti == p.ti && it.di != p.di && it.di !in usedD && it.d < p.d + 0.3 }
                if (rivals) continue
                usedT.add(p.ti); usedD.add(p.di)
                val t = active[p.ti]; val dt = (i - t.last).toDouble()
                val np = dets[p.di].court
                t.vel = t.vel * 0.6 + (np - t.pos) * (0.4 / dt)
                t.pos = np; t.last = i; t.dets[i] = p.di
            }
            for ((di, d) in dets.withIndex()) if (di !in usedD) {
                active.add(Tracklet(nextId++, d.court, Pt(0.0, 0.0), i).also { it.dets[i] = di })
            }
        }
        done.addAll(active)
        for (t in done) { t.start = t.dets.keys.min(); t.end = t.dets.keys.max() }
        return done
    }

    private fun isPlayer(t: Tracklet, frames: List<List<Person>?>, fps: Double): Boolean {
        if (t.n < max(5, (0.3 * fps).toInt())) return false
        val inside = t.dets.count { (f, di) -> val c = frames[f]!![di].court; c.x > -1.5 && c.x < 12.5 && c.y > -5.5 && c.y < 29.3 }
        return inside.toDouble() / t.n > 0.5
    }

    private fun descriptor(t: Tracklet, frames: List<List<Person>?>): FloatArray? {
        val ds = t.dets.mapNotNull { (f, di) -> val p = frames[f]!![di]; if (p.overlap) null else p.desc }
        if (ds.isEmpty()) return null
        val dim = ds[0].size
        return FloatArray(dim) { j ->
            val vals = ds.map { it[j] }.filter { !it.isNaN() }.sorted()
            if (vals.isEmpty()) Float.NaN else if (vals.size % 2 == 1) vals[vals.size / 2] else (vals[vals.size / 2 - 1] + vals[vals.size / 2]) / 2
        }
    }

    class Assigned(val det: Int, val tracklet: Int, val margin: Double, val side: String)

    fun assign(frames: List<List<Person>?>, cuts: Set<Int>, fps: Double, seedASide: String = "near",
               iterations: Int = 4, minMargin: Double = 0.04): List<Map<String, Assigned>> {
        val tracks = buildTracklets(frames, cuts).filter { isPlayer(it, frames, fps) }
        tracks.forEach { it.desc = descriptor(it, frames) }
        val ts = tracks.filter { it.desc != null }
        var protoA: FloatArray? = null; var protoB: FloatArray? = null
        val order = ts.sortedBy { it.start }
        loop@ for (i in frames.indices) {
            val live = order.filter { it.start <= i && i <= it.end && it.dets.containsKey(i) }
            if (live.size != 2) continue
            val sides = live.map { if (frames[i]!![it.dets[i]!!].court.y > Court.NET_Y) "near" else "far" }
            if (sides.sorted() != listOf("far", "near")) continue
            val a = live[sides.indexOf(seedASide)]; val b = live[1 - sides.indexOf(seedASide)]
            protoA = a.desc!!.copyOf(); protoB = b.desc!!.copyOf(); a.seed = "A"; b.seed = "B"
            break@loop
        }
        if (protoA == null || protoB == null) return frames.map { emptyMap() }
        for (t in ts) for (u in ts) if (u !== t && u.start <= t.end && t.start <= u.end && u.dets.keys.any { t.dets.containsKey(it) }) t.overlaps.add(u.id)
        val byId = ts.associateBy { it.id }
        repeat(iterations) {
            for (t in ts) {
                val da = distance(t.desc!!, protoA!!); val db = distance(t.desc!!, protoB!!)
                t.label = t.seed ?: if (da < db) "A" else "B"; t.margin = abs(da - db)
            }
            for (t in ts.sortedByDescending { it.margin }) for (oid in t.overlaps) {
                val o = byId[oid] ?: continue
                if (o.label == t.label && o.label != "unknown") {
                    var loser = if (o.margin < t.margin || (o.margin == t.margin && o.n < t.n)) o else t
                    if (loser.seed != null) loser = if (loser === o) t else o
                    loser.label = if (loser.label == "A") "B" else "A"; loser.margin = -loser.margin
                }
            }
            val conf = ts.filter { it.margin > 2 * minMargin }
            for (lab in listOf("A", "B")) {
                val sel = conf.filter { it.label == lab }
                if (sel.isEmpty()) continue
                val dim = sel[0].desc!!.size; val w = sel.sumOf { it.n }.toDouble()
                val proto = FloatArray(dim) { j -> (sel.sumOf { t -> (t.desc!![j].let { v -> if (v.isNaN()) 0f else v }).toDouble() * t.n } / w).toFloat() }
                if (lab == "A") protoA = proto else protoB = proto
            }
        }
        for (t in ts) if (abs(t.margin) < minMargin && t.seed == null) t.label = "unknown"
        val out = List(frames.size) { HashMap<String, Assigned>() }
        for (t in ts) if (t.label == "A" || t.label == "B") for ((f, di) in t.dets) {
            val d = frames[f]!![di]
            val prev = out[f][t.label]
            if (prev == null || t.margin > prev.margin)
                out[f][t.label] = Assigned(di, t.id, t.margin, if (d.court.y > Court.NET_Y) "near" else "far")
        }
        // singles completion: one identified player + exactly one unassigned person on the opposite half
        for (f in frames.indices) {
            val dets = frames[f] ?: continue
            if (out[f].size != 1) continue
            val (pid, a) = out[f].entries.first()
            val other = if (pid == "A") "B" else "A"
            val cands = dets.indices.filter { i -> val c = dets[i].court
                i != a.det && c.x > -1.5 && c.x < 12.5 && c.y > -6.0 && c.y < 29.8 && (c.y > Court.NET_Y) != (a.side == "near") }
            if (cands.size == 1) out[f][other] = Assigned(cands[0], -1, 0.0, if (dets[cands[0]].court.y > Court.NET_Y) "near" else "far")
        }
        return out
    }
}
