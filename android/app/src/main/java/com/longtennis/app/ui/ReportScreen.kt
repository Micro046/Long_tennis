package com.longtennis.app.ui

import android.net.Uri
import android.widget.VideoView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.ui.window.Dialog
import androidx.navigation.NavController
import com.longtennis.app.analysis.Court
import com.longtennis.app.data.SessionStore
import org.json.JSONObject

private val COLOR_A = Color(0xFF4FC3F7)
private val COLOR_B = Color(0xFFFFB74D)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReportScreen(nav: NavController, id: String) {
    val ctx = LocalContext.current
    val store = remember { SessionStore(ctx) }
    val session = remember { store.load(id)!! }
    val report = remember { store.loadReport(id) }
    var tab by remember { mutableIntStateOf(0) }
    var playAt by remember { mutableStateOf<Double?>(null) }
    Scaffold(topBar = { TopAppBar(title = { Text("${session.nameA} vs ${session.nameB}") },
        navigationIcon = { TextButton(onClick = { nav.popBackStack() }) { Text("Back") } }) }) { pad ->
        if (report == null) { Text("No report.", Modifier.padding(pad).padding(16.dp)); return@Scaffold }
        Column(Modifier.padding(pad)) {
            TabRow(tab) {
                listOf("Summary", "Landing", "Shots").forEachIndexed { i, t -> Tab(tab == i, { tab = i }, text = { Text(t) }) }
            }
            when (tab) {
                0 -> Summary(report)
                1 -> Landing(report)
                else -> Shots(report) { playAt = it }
            }
        }
    }
    playAt?.let { t ->
        Dialog(onDismissRequest = { playAt = null }) {
            Card { Column(Modifier.padding(8.dp)) {
                AndroidView({ c -> VideoView(c).apply {
                    setVideoURI(Uri.parse(session.videoUri))
                    setOnPreparedListener { it.seekTo(((t - 2.0).coerceAtLeast(0.0) * 1000).toLong(), android.media.MediaPlayer.SEEK_CLOSEST); start() }
                } }, Modifier.fillMaxWidth().aspectRatio(16f / 9f))
                TextButton(onClick = { playAt = null }) { Text("Close") }
            } }
        }
    }
}

private fun players(r: JSONObject) = listOf("A", "B").map { it to r.getJSONObject("players").getJSONObject(it) }

@Composable
private fun Summary(r: JSONObject) {
    LazyColumn(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            val v = r.optJSONObject("video")
            val dur = v?.optDouble("durationS", 0.0) ?: 0.0
            Text("Session %d:%02d · %d shots detected".format((dur / 60).toInt(), (dur % 60).toInt(), r.getJSONArray("shots").length()),
                style = MaterialTheme.typography.bodyMedium)
        }
        for ((pid, p) in players(r)) item {
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(p.getString("name"), style = MaterialTheme.typography.titleLarge, color = if (pid == "A") COLOR_A else COLOR_B)
                    Text("${p.getInt("shots")} shots · ${r.optJSONObject("handedness")?.optString(pid) ?: "right"}-handed (estimated)")
                    val s = p.getJSONObject("strokes")
                    StrokeBar(s)
                    val sp = p.getJSONObject("speed")
                    if (sp.getInt("shotsWithEstimate") > 0) Text("Avg flight speed: median ${sp.opt("medianAvgKmh")} km/h · max ${sp.opt("maxAvgKmh")} km/h" +
                        (if (!sp.isNull("serveMedianAvgKmh")) " · serves ${sp.opt("serveMedianAvgKmh")} km/h" else ""), style = MaterialTheme.typography.bodySmall)
                    val l = p.getJSONObject("landing")
                    Text("In-court first bounces: ${l.getInt("inCourtFirstBounces")} · out: ${l.getInt("out")} · not observed: ${l.getInt("noObservedBounce")}",
                        style = MaterialTheme.typography.bodySmall)
                }
            }
        }
        item {
            Text("Speeds are average ground-projected speeds from the hitter to the first bounce, not racket speed. " +
                "Percentages use shots whose first bounce was detected in the opponent's half as the denominator.",
                style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable
private fun StrokeBar(s: JSONObject) {
    val keys = listOf("serve" to "Serve", "forehand" to "Forehand", "backhand" to "Backhand", "volley" to "Volley", "overhead" to "Overhead", "unknown" to "Unclassified")
    Column {
        for ((k, label) in keys) {
            val v = s.optInt(k)
            if (v == 0 && k in setOf("overhead", "unknown")) continue
            Row { Text(label, Modifier.weight(1f)); Text("$v", fontWeight = FontWeight.Bold) }
        }
    }
}

@Composable
private fun Landing(r: JSONObject) {
    var pid by remember { mutableStateOf("A") }
    val p = r.getJSONObject("players").getJSONObject(pid)
    val shots = r.getJSONArray("shots")
    val pts = (0 until shots.length()).map { shots.getJSONObject(it) }.filter { it.getString("playerId") == pid }.mapNotNull { s ->
        val l = s.getJSONObject("landing"); if (l.isNull("courtM")) null else {
            val c = l.getJSONArray("courtM"); Triple(c.getDouble(0), c.getDouble(1), s.optString("side")) to l.optString("status")
        }
    }
    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            for ((k, pp) in players(r)) FilterChip(pid == k, { pid = k }, label = { Text(pp.getString("name")) })
        }
        Text("Where ${p.getString("name")}'s shots first bounced (viewed from the hitter's end)", style = MaterialTheme.typography.bodyMedium)
        val regions = p.getJSONObject("landing").getJSONObject("regions")
        val color = if (pid == "A") COLOR_A else COLOR_B
        Box(Modifier.fillMaxWidth().aspectRatio(0.72f)) {
            Canvas(Modifier.fillMaxWidth().aspectRatio(0.72f)) {
                // target half drawn with the net at the bottom, opponent's baseline at the top
                val margin = 3.0; val wM = Court.WIDTH + 2 * margin; val hM = Court.NET_Y + margin
                val s = minOf(size.width / wM, size.height / hM).toFloat()
                fun o(x: Double, depth: Double) = Offset(((x + margin) * s).toFloat(), ((margin + Court.NET_Y - depth) * s).toFloat())
                drawRect(Color(0xFF2E5E9E), o(0.0, Court.NET_Y), Size((Court.WIDTH * s).toFloat(), (Court.NET_Y * s).toFloat()))
                val white = Color.White
                drawLine(white, o(0.0, Court.NET_Y), o(Court.WIDTH, Court.NET_Y), 3f)
                drawLine(white, o(0.0, 0.0), o(Court.WIDTH, 0.0), 5f)
                for (x in listOf(0.0, Court.SL, Court.WIDTH - Court.SL, Court.WIDTH)) drawLine(white, o(x, 0.0), o(x, Court.NET_Y), 3f)
                drawLine(white, o(Court.SL, Court.SERVICE), o(Court.WIDTH - Court.SL, Court.SERVICE), 3f)
                drawLine(white, o(Court.CX, 0.0), o(Court.CX, Court.SERVICE), 3f)
                // region percentages
                val paint = android.graphics.Paint().apply { this.color = android.graphics.Color.WHITE; textSize = 34f; textAlign = android.graphics.Paint.Align.CENTER; isFakeBoldText = true }
                for ((i, name) in Court.REGIONS.withIndex()) {
                    val col = i % 3; val deep = i < 3
                    val cx = Court.SL + (col + 0.5) * Court.SINGLES / 3
                    val cd = if (deep) (Court.SERVICE + Court.NET_Y) / 2 else Court.SERVICE / 2
                    val reg = regions.getJSONObject(name)
                    val txt = if (reg.isNull("percent")) "–" else "${reg.getDouble("percent").toInt()}%"
                    val pos = o(cx, cd)
                    drawContext.canvas.nativeCanvas.drawText(txt, pos.x, pos.y, paint)
                }
                for ((pt, status) in pts) {
                    val (x, y, side) = pt
                    val (vx, depth) = if (side == "near") (x to Court.NET_Y - y) else ((Court.WIDTH - x) to y - Court.NET_Y)
                    if (depth < -1) continue
                    drawCircle(if (status == "out") Color(0xFFFF5252) else color, 9f, o(vx, depth))
                    drawCircle(Color.Black, 9f, o(vx, depth), style = Stroke(2f))
                }
            }
        }
        Text("Dots: first bounces (red = out). Numbers: share of in-court first bounces per zone.", style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun Shots(r: JSONObject, onPlay: (Double) -> Unit) {
    val shots = r.getJSONArray("shots")
    val names = players(r).associate { (k, p) -> k to p.getString("name") }
    LazyColumn(Modifier.padding(horizontal = 12.dp)) {
        items(shots.length()) { i ->
            val s = shots.getJSONObject(i)
            val t = s.getDouble("timeS")
            val l = s.getJSONObject("landing")
            val sp = s.optJSONObject("speed")
            Row(Modifier.fillMaxWidth().clickable { onPlay(t) }.padding(vertical = 10.dp)) {
                Text("%d:%02d".format((t / 60).toInt(), (t % 60).toInt()), Modifier.weight(0.6f))
                Text(names[s.getString("playerId")] ?: "?", Modifier.weight(1.2f),
                    color = when (s.getString("playerId")) { "A" -> COLOR_A; "B" -> COLOR_B; else -> Color.Gray })
                Text(s.getString("stroke"), Modifier.weight(1f))
                Text(if (l.isNull("region")) l.getString("status").replace('_', ' ') else l.getString("region").replace('_', ' '), Modifier.weight(1.3f),
                    style = MaterialTheme.typography.bodySmall)
                Text(sp?.let { "${it.getDouble("avgKmh").toInt()} km/h" } ?: "", Modifier.weight(0.8f), style = MaterialTheme.typography.bodySmall)
            }
            HorizontalDivider()
        }
    }
}
