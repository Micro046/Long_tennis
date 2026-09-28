package com.longtennis.app.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import com.longtennis.app.analysis.Court
import com.longtennis.app.analysis.Homography
import com.longtennis.app.analysis.Pt
import kotlin.math.hypot
import kotlin.math.min

/** Default marker positions (normalised) for a phone behind the near baseline at ~2 m height. */
val DEFAULT_MARKERS = mapOf(
    "far_left_singles" to Pt(0.37, 0.30), "far_right_singles" to Pt(0.62, 0.30),
    "near_service_left" to Pt(0.19, 0.49), "near_service_right" to Pt(0.80, 0.49),
)

val MARKER_LABELS = mapOf(
    "far_left_singles" to "Far baseline × left singles line",
    "far_right_singles" to "Far baseline × right singles line",
    "near_service_left" to "Near service line × left singles line",
    "near_service_right" to "Near service line × right singles line",
)

/**
 * Draggable court markers over an image of aspect `imageAspect` letterboxed into the box.
 * `court` (image-from-court homography in normalised image units) draws the snapped court.
 */
@Composable
fun CourtMarkerOverlay(
    imageAspect: Float, markers: Map<String, Pt>, onChange: (Map<String, Pt>) -> Unit,
    court: Homography?, modifier: Modifier = Modifier,
) {
    var dragging by remember { mutableStateOf<String?>(null) }
    val accent = MaterialTheme.colorScheme.primary
    Canvas(modifier.fillMaxSize().pointerInput(markers) {
        fun frame(): FloatArray {
            val w = size.width.toFloat(); val h = size.height.toFloat()
            val iw = min(w, h * imageAspect); val ih = iw / imageAspect
            return floatArrayOf((w - iw) / 2, (h - ih) / 2, iw, ih)
        }
        detectDragGestures(
            onDragStart = { o ->
                val f = frame()
                dragging = markers.minByOrNull { (_, p) -> hypot(f[0] + p.x * f[2] - o.x, f[1] + p.y * f[3] - o.y) }
                    ?.takeIf { (_, p) -> hypot(f[0] + p.x * f[2] - o.x, f[1] + p.y * f[3] - o.y) < 120 }?.key
            },
            onDragEnd = { dragging = null },
            onDrag = { change, amount ->
                val k = dragging ?: return@detectDragGestures
                change.consume()
                val f = frame(); val p = markers.getValue(k)
                onChange(markers + (k to Pt((p.x + amount.x / f[2]).coerceIn(0.0, 1.0), (p.y + amount.y / f[3]).coerceIn(0.0, 1.0))))
            },
        )
    }) {
        val w = size.width; val h = size.height
        val iw = min(w, h * imageAspect); val ih = iw / imageAspect
        val ox = (w - iw) / 2; val oy = (h - ih) / 2
        fun o(p: Pt) = Offset(ox + p.x.toFloat() * iw, oy + p.y.toFloat() * ih)
        court?.let { H ->
            for ((_, seg) in Court.LINES) {
                val a = H.apply(seg.first) ?: continue; val b = H.apply(seg.second) ?: continue
                drawLine(Color(0xFF39FF7A), o(a), o(b), strokeWidth = 3f)
            }
        }
        for ((k, p) in markers) {
            drawCircle(Color.Black.copy(alpha = 0.5f), 26f, o(p))
            drawCircle(if (k == dragging) Color.Yellow else accent, 22f, o(p), style = Stroke(5f))
            drawCircle(Color.White, 4f, o(p))
        }
    }
}

@Composable
fun HandedChips(label: String, handed: String, onChange: (String) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(label, modifier = Modifier.padding(top = 8.dp))
        FilterChip(handed == "right", { onChange("right") }, label = { Text("right-handed") })
        FilterChip(handed == "left", { onChange("left") }, label = { Text("left-handed") })
    }
}

@Composable
fun QualityChips(quality: String, onChange: (String) -> Unit) {
    Column {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Analysis", modifier = Modifier.padding(top = 8.dp))
            FilterChip(quality == "fast", { onChange("fast") }, label = { Text("Fast") })
            FilterChip(quality == "accurate", { onChange("accurate") }, label = { Text("Accurate") })
        }
        Text(if (quality == "fast") "About 12x quicker; slightly fewer shots detected."
             else "Best detection; much slower (keep the phone charging).", style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
fun PlayerFields(nameA: String, nameB: String, seedASide: String, onChange: (String, String, String) -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(nameA, { onChange(it, nameB, seedASide) }, label = { Text("Player A") }, singleLine = true, modifier = Modifier.weight(1f))
            OutlinedTextField(nameB, { onChange(nameA, it, seedASide) }, label = { Text("Player B") }, singleLine = true, modifier = Modifier.weight(1f))
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("At the start, $nameA is on the", modifier = Modifier.padding(top = 8.dp))
            FilterChip(seedASide == "near", { onChange(nameA, nameB, "near") }, label = { Text("near side") })
            FilterChip(seedASide == "far", { onChange(nameA, nameB, "far") }, label = { Text("far side") })
        }
    }
}

/** Convert a pixel-space homography into normalised image units for drawing. */
fun normalisedCourt(H: Homography, w: Int, h: Int): Homography =
    Homography(Homography.mul(doubleArrayOf(1.0 / w, 0.0, 0.0, 0.0, 1.0 / h, 0.0, 0.0, 0.0, 1.0), H.m))

@Composable
fun ImageWithMarkers(bitmap: Bitmap, markers: Map<String, Pt>, onChange: (Map<String, Pt>) -> Unit, court: Homography?, modifier: Modifier) {
    Box(modifier) {
        androidx.compose.foundation.Image(bitmap = bitmap.asImageBitmap(), contentDescription = "Court frame",
            modifier = Modifier.fillMaxSize(), contentScale = androidx.compose.ui.layout.ContentScale.Fit)
        CourtMarkerOverlay(bitmap.width.toFloat() / bitmap.height, markers, onChange, court, Modifier.fillMaxWidth())
    }
}
