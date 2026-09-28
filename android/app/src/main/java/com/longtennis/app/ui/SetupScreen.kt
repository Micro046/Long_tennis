package com.longtennis.app.ui

import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.longtennis.app.analysis.AnalysisService
import com.longtennis.app.analysis.Calibration
import com.longtennis.app.analysis.CourtCalibrator
import com.longtennis.app.data.SessionStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Setup for an imported video: place the court markers on a frame, name the players, analyse. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SetupScreen(nav: NavController, id: String) {
    val ctx = LocalContext.current
    val store = remember { SessionStore(ctx) }
    var session by remember { mutableStateOf(store.load(id)!!) }
    var frame by remember { mutableStateOf<Bitmap?>(null) }
    var markers by remember { mutableStateOf(session.markers.ifEmpty { DEFAULT_MARKERS }) }
    var result by remember { mutableStateOf<CourtCalibrator.Result?>(null) }
    var message by remember { mutableStateOf("Drag the four circles onto the court line crossings, then tap Check court.") }
    val scope = rememberCoroutineScope()

    LaunchedEffect(id) {
        frame = withContext(Dispatchers.IO) {
            runCatching {
                MediaMetadataRetriever().run {
                    setDataSource(ctx, Uri.parse(session.videoUri))
                    (getFrameAtTime(1_000_000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: frameAtTime).also { release() }
                }
            }.getOrNull()
        }
        if (frame == null) message = "Could not read this video."
    }

    Scaffold(topBar = { TopAppBar(title = { Text("Court setup") }) }) { pad ->
        Column(Modifier.padding(pad).padding(12.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            frame?.let { bmp ->
                ImageWithMarkers(bmp, markers, { markers = it; result = null },
                    result?.let { normalisedCourt(it.imageFromCourt, it.width, it.height) },
                    Modifier.fillMaxWidth().aspectRatio(bmp.width.toFloat() / bmp.height))
            }
            Text(message, style = MaterialTheme.typography.bodyMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = {
                    val bmp = frame ?: return@OutlinedButton
                    scope.launch {
                        val r = withContext(Dispatchers.Default) { AnalysisService.calibrateBitmap(bmp, markers) }
                        result = r
                        message = if (r == null || r.keypointsUsed.size < 4) "Court lines not found. Move the circles closer to the line crossings."
                        else "Court found: ${r.linesSnapped.size} lines, fit error %.1f px. Check the green overlay matches the court.".format(r.reprojectionRmsPx ?: 0.0)
                    }
                }) { Text("Check court") }
            }
            PlayerFields(session.nameA, session.nameB, session.seedASide) { a, b, s -> session = session.copy(nameA = a, nameB = b, seedASide = s) }
            HandedChips(session.nameA, session.handedA) { session = session.copy(handedA = it) }
            HandedChips(session.nameB, session.handedB) { session = session.copy(handedB = it) }
            QualityChips(session.quality) { session = session.copy(quality = it) }
            Button(enabled = result != null, modifier = Modifier.fillMaxWidth(), onClick = {
                val r = result!!
                store.saveCalibration(id, Calibration(r.imageFromCourt, r.width, r.height, r.reprojectionRmsPx))
                store.save(session.copy(markers = markers, status = "recorded"))
                AnalysisService.start(ctx, id)
                nav.navigate("analysis/$id") { popUpTo("home") }
            }) { Text("Analyse session") }
        }
    }
}
