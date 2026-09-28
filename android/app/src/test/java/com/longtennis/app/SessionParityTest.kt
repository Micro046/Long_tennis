package com.longtennis.app

import com.longtennis.app.analysis.Analyzer
import com.longtennis.app.analysis.GbmModel
import com.longtennis.app.analysis.Homography
import com.longtennis.app.analysis.Identity
import com.longtennis.app.analysis.Person
import com.longtennis.app.analysis.Pt
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** Identity + events + shot logic in Kotlin must reproduce the Python session on the same inputs. */
class SessionParityTest {
    @Test fun sessionMatchesPython() {
        val fx = JSONObject(File(javaClass.classLoader!!.getResource("session_fixture.json")!!.toURI()).readText())
        val fps = fx.getDouble("fps"); val h = fx.getInt("height")
        val ballArr = fx.getJSONArray("ball")
        val ball = Array(ballArr.length()) { i -> if (ballArr.isNull(i)) null else ballArr.getJSONArray(i).let { floatArrayOf(it.getDouble(0).toFloat(), it.getDouble(1).toFloat(), it.getDouble(2).toFloat()) } }
        val pa = fx.getJSONArray("people")
        val people: List<List<Person>?> = (0 until pa.length()).map { i ->
            if (pa.isNull(i)) null else pa.getJSONArray(i).let { arr -> (0 until arr.length()).map { j ->
                val o = arr.getJSONObject(j)
                fun fa(a: JSONArray) = FloatArray(a.length()) { if (a.isNull(it)) Float.NaN else a.getDouble(it).toFloat() }
                Person(fa(o.getJSONArray("box")), o.getDouble("score").toFloat(),
                    o.getJSONArray("court").let { Pt(it.getDouble(0), it.getDouble(1)) }, fa(o.getJSONArray("kps"))).also { p ->
                    p.desc = if (o.isNull("desc")) null else fa(o.getJSONArray("desc")); p.overlap = o.getBoolean("overlap")
                }
            } }
        }
        val H = fx.getJSONArray("imageFromCourt").let { a -> Homography.fromList((0 until 3).map { r -> (0 until 3).map { c -> a.getJSONArray(r).getDouble(c) } }) }
        val ident = Identity.assign(people, emptySet(), fps)
        val exp = fx.getJSONObject("expected")
        // identity per frame
        val ei = exp.getJSONArray("identity"); var agree = 0; var total = 0
        for (f in 0 until ei.length()) {
            val e = ei.getJSONObject(f)
            for (pid in listOf("A", "B")) {
                val want = if (e.has(pid)) e.getJSONObject(pid).getInt("det") else -1
                val got = ident[f][pid]?.det ?: -1
                if (want != -1 || got != -1) { total++; if (want == got) agree++ }
            }
        }
        println("identity agreement $agree / $total")
        assertTrue("identity agreement $agree/$total", agree >= 0.99 * total)

        val pts = LongArray(ball.size) { (it * 1e6 / fps).toLong() }
        val an = Analyzer(ball, pts, people, ident, H.inverse(), fps, h)
        an.detectEvents(GbmModel(File("src/main/assets/models/event_gbm.json").readText()))
        an.removeBallHandling(H)
        an.dedupeHits(); an.keepAnsweredHits()
        val shots = an.buildShots()
        fun ints(a: JSONArray) = (0 until a.length()).map { a.getInt(it) }
        println("kotlin hits ${an.hits} bounces ${an.bounces}")
        assertEquals(ints(exp.getJSONArray("bounces")), an.bounces.sorted())
        assertEquals(ints(exp.getJSONArray("hits")), an.hits.sorted())
        val es = exp.getJSONArray("shots")
        assertEquals(es.length(), shots.size)
        for (i in shots.indices) {
            val e = es.getJSONObject(i); val s = shots[i]
            assertEquals("shot $i player", e.getString("playerId"), s.playerId)
            assertEquals("shot $i stroke", e.getString("stroke"), s.stroke)
            assertEquals("shot $i landing", e.getString("landing"), s.landingStatus)
            assertEquals("shot $i region", if (e.isNull("region")) null else e.getString("region"), s.region)
        }
    }
}
