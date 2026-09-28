package com.longtennis.app.analysis

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/**
 * Post-session analysis on the per-frame ball track + sparse poses. Port of
 * scripts/pipeline/session.py (detect_events, build_shots, stroke_class, summarize).
 */
class Analyzer(
    private val ball: Array<FloatArray?>,          // per frame [x, y, conf] px or null
    private val ptsUs: LongArray,                  // per frame presentation time
    private val people: List<List<Person>?>,       // per frame (null = pose not run on this frame)
    private val ident: List<Map<String, Identity.Assigned>>,
    private val courtFromImage: Homography,
    private val fps: Double,
    private val height: Int,
) {
    companion object {
        const val NOSE = 0; const val LWR = 9; const val RWR = 10; const val LHIP = 11; const val RHIP = 12
    }

    val n = ball.size
    lateinit var ballPx: Array<DoubleArray>
    lateinit var hits: List<Int>
    lateinit var bounces: List<Int>

    private fun t(i: Int) = ptsUs[min(i, ptsUs.size - 1)] / 1e6

    fun detectEvents(model: GbmModel, thrHit: Double = 0.5, thrBounce: Double = 0.5) {
        val xy = Array(n) { i -> ball[i]?.let { doubleArrayOf(it[0].toDouble() / height, it[1].toDouble() / height) } ?: doubleArrayOf(Double.NaN, Double.NaN) }
        val (clean, obs) = Events.cleanTrack(xy)
        val feats = Events.features(clean, obs, fps)
        val prob = Array(n) { model.predictProba(feats[it]) }
        val r = round(6 * fps / 30).toInt()
        val ph = DoubleArray(n) { prob[it][1] }; val pb = DoubleArray(n) { prob[it][2] }
        val h0 = Events.peaks(ph, thrHit, r); val b0 = Events.peaks(pb, thrBounce, r)
        hits = h0.filter { h -> b0.none { b -> abs(h - b) <= 2 && pb[b] > ph[h] } }
        bounces = b0.filter { b -> hits.none { h -> abs(h - b) <= 2 && ph[h] >= pb[b] } }
        ballPx = Array(n) { doubleArrayOf(clean[it][0] * height, clean[it][1] * height) }
    }

    private fun finite(p: DoubleArray) = !p[0].isNaN() && !p[1].isNaN()

    fun nearestPose(frame: Int, pid: String, radius: Int = 4): Pair<Int, Identity.Assigned>? {
        for (off in (-radius..radius).sortedBy { abs(it) }) {
            val f = frame + off
            if (f in 0 until n) ident[f][pid]?.let { return f to it }
        }
        return null
    }

    /** Ball distance to the racket hand in units of reach (0.45 x body height). */
    private fun reach(b: DoubleArray, d: Person): Double {
        val h = d.height.toDouble()
        val wr = intArrayOf(LWR, RWR).filter { d.ks(it) > 0.3f }.map { hypot(b[0] - d.kx(it), b[1] - d.ky(it)) }
        return if (wr.isNotEmpty()) wr.min() / (0.45 * h)
        else hypot(b[0] - (d.box[0] + d.box[2]) / 2, b[1] - (d.box[1] + d.box[3]) / 2) / (0.6 * h)
    }

    /** Nearest racket hand over ALL detected people; unidentified nearest person -> unknown hitter. */
    private fun hitterFor(frame: Int): Pair<String?, Double>? {
        var best: Pair<Double, String?>? = null
        for (f in max(0, frame - 4) until min(n, frame + 5)) {
            val ppl = people[f] ?: continue
            val b = if (finite(ballPx[frame])) ballPx[frame] else ballPx[f]
            if (!finite(b)) continue
            val detToPid = ident[f].entries.associate { it.value.det to it.key }
            for ((di, d) in ppl.withIndex()) {
                val r = reach(b, d)
                if (best == null || r < best.first) best = r to detToPid[di]
            }
        }
        val b = best ?: return null
        return if (b.first > 1.5) null else b.second to round3(b.first)
    }

    /** Ball stays around one player's body and on that player's side of the net (bouncing before a serve etc.). */
    fun ballHandling(frame: Int, netImgY: Double): Boolean {
        val r = max(6, round(0.5 * fps).toInt())
        val idx = (max(0, frame - r)..min(n - 1, frame + r)).filter { finite(ballPx[it]) }
        if (idx.size < 6) return false
        for (pid in listOf("A", "B")) {
            val (pf, a) = nearestPose(frame, pid, 6) ?: continue
            val d = people[pf]!![a.det]
            val x1 = d.box[0].toDouble(); val y1 = d.box[1].toDouble(); val x2 = d.box[2].toDouble(); val y2 = d.box[3].toDouble(); val h = y2 - y1
            val near = idx.filter { hypot(ballPx[it][0] - (x1 + x2) / 2, ballPx[it][1] - (y1 + y2) / 2) < 3 * h }
            if (near.size < 6) continue
            val inside = near.count { ballPx[it][0] > x1 - 0.3 * h && ballPx[it][0] < x2 + 0.3 * h && ballPx[it][1] > y1 && ballPx[it][1] < y2 + 0.1 * h }
            if (inside.toDouble() / near.size >= 0.75 && near.size.toDouble() / idx.size >= 0.6) {
                val w = max(8, round(fps).toInt())
                val ys = (max(0, frame - w)..min(n - 1, frame + w)).filter { finite(ballPx[it]) }.map { ballPx[it][1] }
                if (ys.isEmpty()) continue
                if (a.side == "far" && ys.max() < netImgY + 0.02 * height) return true
                if (a.side == "near" && ys.min() > netImgY - 0.02 * height) return true
            }
        }
        return false
    }

    /** Removes ball-handling events (call after detectEvents, before buildShots). */
    fun removeBallHandling(imageFromCourt: Homography) {
        val net = imageFromCourt.apply(Pt(Court.CX, Court.NET_Y))!!.y
        val dh = hits.filter { ballHandling(it, net) }.toSet(); val db = bounces.filter { ballHandling(it, net) }.toSet()
        hits = hits.filter { it !in dh }; bounces = bounces.filter { it !in db }
    }

    /** Top of head from face/ear keypoints (visible from behind), else shoulders - 0.12 h; never the box top. */
    private fun headY(d: Person): Double? {
        val pts = (0 until 5).filter { d.ks(it) > 0.3f }.map { d.ky(it).toDouble() }
        if (pts.isNotEmpty()) return pts.min()
        val sh = intArrayOf(5, 6).filter { d.ks(it) > 0.3f }.map { d.ky(it).toDouble() }
        return if (sh.isEmpty()) null else sh.min() - 0.12 * d.height
    }

    /** Port of session.py stroke_class (serve/overhead/volley, FH/BH by follow-through side). */
    private fun strokeClass(frame: Int, pid: String, side: String?, prevGapS: Double?, bouncedBefore: Boolean, handed: String): Pair<String, JSONObject> {
        val (pf, a) = nearestPose(frame, pid) ?: return "unknown" to JSONObject()
        val d = people[pf]!![a.det]
        val y = d.court.y
        val atBaseline = if (side == "near") y > Court.LENGTH - 1.2 else y < 1.2
        var wristUp = false
        for (f in frame - 4 until frame + 2) {           // up to contact only
            val asg = if (f in 0 until n) ident[f][pid] else null
            val p = asg?.let { people[f]!![it.det] } ?: continue
            val head = headY(p) ?: continue
            if (intArrayOf(LWR, RWR).any { p.ks(it) > 0.3f && p.ky(it) < head }) wristUp = true
        }
        if (wristUp) return (if (atBaseline) "serve" else "overhead") to JSONObject().put("wristAboveHead", true).put("atBaseline", atBaseline)
        if (!bouncedBefore) {
            val dn = abs(y - Court.NET_Y)
            if (dn < 7.0) return "volley" to JSONObject().put("distanceToNetM", round2(dn))
        }
        val racket = if (handed == "right") RWR else LWR
        val xs = ArrayList<Double>()
        for (f in frame + 4 until frame + 13) {
            val asg = if (f in 0 until n) ident[f][pid] else null
            val p = asg?.let { people[f]!![it.det] } ?: continue
            if (p.ks(racket) <= 0.3f) continue
            val hx = if (p.ks(LHIP) > 0.3f && p.ks(RHIP) > 0.3f) (p.kx(LHIP) + p.kx(RHIP)) / 2.0 else (p.box[0] + p.box[2]) / 2.0
            xs.add((p.kx(racket) - hx) / p.height)
        }
        if (xs.isEmpty()) return "unknown" to JSONObject()
        val ft = xs.average()
        var sgn = if (side == "near") ft else -ft
        if (handed != "right") sgn = -sgn
        return (if (sgn < 0.1) "forehand" else "backhand") to JSONObject().put("followThroughBodyHeights", round3(ft))
    }

    /** Drop the earlier of two consecutive same-player hits within 1.5 s (toss, feed). */
    fun dedupeHits() {
        val hs = hits.sorted(); val who = hs.map { hitterFor(it)?.first }
        hits = hs.filterIndexed { i, h -> !(i + 1 < hs.size && who[i] != null && who[i + 1] == who[i] && t(hs[i + 1]) - t(h) < 1.5) }
    }

    /** A shot must be answered within 2.5 s by the other player's hit or a bounce in the opponent's half. */
    fun keepAnsweredHits() {
        val hs = hits.sorted(); val who = hs.associateWith { hitterFor(it)?.first }
        val bs = bounces.sorted()
        hits = hs.filter { h ->
            val pid = who[h]; val side = pid?.let { nearestPose(h, it)?.second?.side }
            val lim = h + 2.5 * fps
            val nextHit = hs.firstOrNull { it > h && it <= lim }
            val nextB = bs.firstOrNull { it > h + 2 && it <= lim }
            when {
                nextHit != null && (nextB == null || nextHit < nextB) -> !(who[nextHit] != null && pid != null && who[nextHit] == pid)
                nextB != null -> if (side == null || !finite(ballPx[nextB])) true else {
                    val yy = courtFromImage.apply(Pt(ballPx[nextB][0], ballPx[nextB][1]))?.y
                    yy == null || (if (side == "near") yy < Court.NET_Y else yy > Court.NET_Y)
                }
                else -> false
            }
        }
    }

    class Shot(val hitFrame: Int, val timeS: Double, val playerId: String, val side: String?, val attribution: Double?) {
        var stroke = "unknown"; var evidence = JSONObject()
        var landingStatus = "not_observed"; var region: String? = null; var bounceFrame: Int? = null; var courtM: Pt? = null
        var speedKmh: Double? = null; var distanceM: Double? = null; var flightS: Double? = null
    }

    fun buildShots(handed: Map<String, String> = mapOf("A" to "right", "B" to "right")): List<Shot> {
        val out = ArrayList<Shot>()
        val hs = hits.sorted(); val bs = bounces.sorted()
        for ((k, h) in hs.withIndex()) {
            val hp = hitterFor(h)
            val pid = hp?.first
            val side = pid?.let { nearestPose(h, it)?.second?.side }
            val nxt = if (k + 1 < hs.size) hs[k + 1] else h + (4 * fps).toInt()
            val prv = if (k > 0) hs[k - 1] else null
            val gap = prv?.let { t(h) - t(it) }
            val bouncedBefore = prv != null && bs.any { it in (prv + 1) until h }
            val firstB = bs.firstOrNull { it > h + 2 && it < min(nxt, h + (3.5 * fps).toInt()) }
            val s = Shot(h, round3(t(h)), pid ?: "unknown", side, hp?.second)
            if (pid != null) {
                val (c, ev) = strokeClass(h, pid, side, gap, bouncedBefore || gap == null || gap > 2.5, handed[pid] ?: "right")
                s.stroke = c; s.evidence = ev
            }
            if (firstB == null) {
                s.landingStatus = if (nxt - h < 3.5 * fps) "no_bounce_before_next_hit" else "not_observed"
            } else if (!finite(ballPx[firstB])) {
                s.landingStatus = "bounce_position_missing"; s.bounceFrame = firstB
            } else {
                val pt = courtFromImage.apply(Pt(ballPx[firstB][0], ballPx[firstB][1]))
                val (st, reg) = Court.region(pt, side)
                s.landingStatus = st; s.region = reg; s.bounceFrame = firstB; s.courtM = pt
                val pa = pid?.let { nearestPose(h, it) }
                if (pa != null && pt != null) {
                    val src = people[pa.first]!![pa.second.det].court
                    val dt = t(firstB) - t(h); val dist = (pt - src).norm()
                    if (dt > 0.15 && dist > 3 && dist < 32 && dist / dt * 3.6 <= 230) { s.speedKmh = round1(dist / dt * 3.6); s.distanceM = round2(dist); s.flightS = round3(dt) }
                }
            }
            out.add(s)
        }
        return out
    }

    fun handedness(shots: List<Shot>): Map<String, String> {
        val votes = mapOf("A" to HashMap<String, Int>(), "B" to HashMap())
        for (s in shots) {
            if (s.stroke != "serve" || s.playerId !in votes) continue
            val (pf, a) = nearestPose(s.hitFrame, s.playerId) ?: continue
            val d = people[pf]!![a.det]
            if (d.ks(LWR) > .3f && d.ks(RWR) > .3f) {
                val hi = if (d.ky(LWR) < d.ky(RWR)) LWR else RWR; val other = if (hi == LWR) RWR else LWR
                val imgRight = d.kx(hi) > d.kx(other)
                val playerRight = if (s.side == "near") imgRight else !imgRight
                val key = if (playerRight) "right" else "left"
                votes.getValue(s.playerId)[key] = (votes.getValue(s.playerId)[key] ?: 0) + 1
            }
        }
        return votes.mapValues { (_, v) -> if (v.values.sum() >= 3) v.maxBy { it.value }.key else "right" }
    }

    fun report(shots: List<Shot>, names: Map<String, String>, handed: Map<String, String>, meta: JSONObject): JSONObject {
        val players = JSONObject()
        for (pid in listOf("A", "B")) {
            val sel = shots.filter { it.playerId == pid }
            val strokes = JSONObject()
            for (k in listOf("serve", "forehand", "backhand", "volley", "overhead", "unknown")) strokes.put(k, sel.count { it.stroke == k })
            val landed = sel.filter { it.region != null }
            val outN = sel.count { it.landingStatus == "out" }
            val regions = JSONObject()
            for (r in Court.REGIONS) {
                val c = landed.count { it.region == r }
                regions.put(r, JSONObject().put("count", c).put("percent", if (landed.isEmpty()) JSONObject.NULL else round1(100.0 * c / landed.size)))
            }
            val speeds = sel.mapNotNull { it.speedKmh }.sorted()
            val serveSpeeds = sel.filter { it.stroke == "serve" }.mapNotNull { it.speedKmh }.sorted()
            fun med(l: List<Double>) = if (l.isEmpty()) JSONObject.NULL else if (l.size % 2 == 1) l[l.size / 2] else round1((l[l.size / 2 - 1] + l[l.size / 2]) / 2)
            players.put(pid, JSONObject()
                .put("name", names[pid] ?: pid).put("shots", sel.size).put("strokes", strokes)
                .put("landing", JSONObject().put("inCourtFirstBounces", landed.size).put("out", outN)
                    .put("noObservedBounce", sel.size - landed.size - outN)
                    .put("denominator", "shots with a detected first bounce inside the opponent's singles half")
                    .put("regions", regions))
                .put("speed", JSONObject().put("shotsWithEstimate", speeds.size).put("medianAvgKmh", med(speeds))
                    .put("maxAvgKmh", speeds.maxOrNull() ?: JSONObject.NULL).put("serveMedianAvgKmh", med(serveSpeeds))
                    .put("definition", "average ground-projected speed hitter->first bounce (not speed off the racket)")))
        }
        val shotArr = JSONArray()
        for (s in shots) shotArr.put(JSONObject().put("hitFrame", s.hitFrame).put("timeS", s.timeS).put("playerId", s.playerId)
            .put("side", s.side ?: JSONObject.NULL).put("stroke", s.stroke).put("strokeEvidence", s.evidence)
            .put("landing", JSONObject().put("status", s.landingStatus).put("region", s.region ?: JSONObject.NULL)
                .put("bounceFrame", s.bounceFrame ?: JSONObject.NULL)
                .put("courtM", s.courtM?.let { JSONArray().put(round2(it.x)).put(round2(it.y)) } ?: JSONObject.NULL))
            .also { o -> s.speedKmh?.let { o.put("speed", JSONObject().put("avgKmh", it).put("distanceM", s.distanceM).put("flightS", s.flightS)) } })
        return JSONObject().put("schemaVersion", "2.0").put("type", "session_report").put("video", meta).put("fps", fps)
            .put("players", players).put("handedness", JSONObject(handed))
            .put("identity", JSONObject().put("framesWithA", ident.count { "A" in it }).put("framesWithB", ident.count { "B" in it }))
            .put("events", JSONObject().put("hits", JSONArray(hits)).put("bounces", JSONArray(bounces)))
            .put("shots", shotArr)
    }

    private fun round1(v: Double) = round(v * 10) / 10
    private fun round2(v: Double) = round(v * 100) / 100
    private fun round3(v: Double) = round(v * 1000) / 1000
}
