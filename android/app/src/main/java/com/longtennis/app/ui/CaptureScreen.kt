package com.longtennis.app.ui

import android.Manifest
import android.app.Activity
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.view.WindowManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FileOutputOptions
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.Recorder
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.navigation.NavController
import com.longtennis.app.analysis.AnalysisService
import com.longtennis.app.analysis.CourtCalibrator
import com.longtennis.app.data.SessionStore
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/** Live setup + unattended recording. The court calibration is re-run on the recorded video. */
@Composable
fun CaptureScreen(nav: NavController, id: String) {
    val ctx = LocalContext.current
    val activity = ctx as Activity
    val lifecycle = LocalLifecycleOwner.current
    val store = remember { SessionStore(ctx) }
    var session by remember { mutableStateOf(store.load(id)!!) }
    var markers by remember { mutableStateOf(session.markers.ifEmpty { DEFAULT_MARKERS }) }
    var result by remember { mutableStateOf<CourtCalibrator.Result?>(null) }
    var message by remember { mutableStateOf("Frame the whole court (landscape), drag the circles onto the line crossings, tap Check.") }
    var granted by remember { mutableStateOf(ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    var recording by remember { mutableStateOf<Recording?>(null) }
    var startedAt by remember { mutableLongStateOf(0L) }
    var elapsed by remember { mutableLongStateOf(0L) }
    val previewView = remember { PreviewView(ctx).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    val videoCapture = remember {
        VideoCapture.withOutput(Recorder.Builder().setQualitySelector(
            QualitySelector.from(Quality.FHD, FallbackStrategy.lowerQualityOrHigherThan(Quality.HD))).build())
    }
    val scope = rememberCoroutineScope()

    DisposableEffect(Unit) {
        activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_SENSOR_LANDSCAPE
        activity.window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        onDispose {
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_UNSPECIFIED
            activity.window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        }
    }
    LaunchedEffect(granted) {
        if (!granted) { permission.launch(Manifest.permission.CAMERA); return@LaunchedEffect }
        val provider = ProcessCameraProvider.getInstance(ctx).get()
        val preview = Preview.Builder().build().also { it.surfaceProvider = previewView.surfaceProvider }
        provider.unbindAll()
        provider.bindToLifecycle(lifecycle, CameraSelector.DEFAULT_BACK_CAMERA, preview, videoCapture)
    }
    LaunchedEffect(recording) { while (recording != null) { elapsed = System.currentTimeMillis() - startedAt; delay(500) } }

    Row(Modifier.fillMaxSize().background(Color.Black)) {
        Box(Modifier.weight(1f).fillMaxHeight()) {
            AndroidView({ previewView }, Modifier.fillMaxSize())
            if (recording == null) CourtMarkerOverlay(16f / 9f, markers, { markers = it; result = null },
                result?.let { normalisedCourt(it.imageFromCourt, it.width, it.height) })
        }
        Column(Modifier.width(260.dp).fillMaxHeight().background(MaterialTheme.colorScheme.surface).padding(10.dp)
            .verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            if (recording == null) {
                Text(message, style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = {
                    val full = previewView.bitmap ?: return@OutlinedButton
                    // FIT_CENTER letterboxes a 16:9 image inside the view: crop to it
                    val iw = minOf(full.width.toFloat(), full.height * 16f / 9f).toInt(); val ih = (iw * 9f / 16f).toInt()
                    val bmp = android.graphics.Bitmap.createBitmap(full, (full.width - iw) / 2, (full.height - ih) / 2, iw, ih)
                    scope.launch {
                        val r = withContext(Dispatchers.Default) { AnalysisService.calibrateBitmap(bmp, markers) }
                        result = r
                        message = if (r == null || r.keypointsUsed.size < 4) "Court lines not found. Adjust the circles."
                        else "Court found (${r.linesSnapped.size} lines). Check the green overlay."
                    }
                }) { Text("Check court") }
                PlayerFields(session.nameA, session.nameB, session.seedASide) { a, b, s -> session = session.copy(nameA = a, nameB = b, seedASide = s) }
                HandedChips(session.nameA, session.handedA) { session = session.copy(handedA = it) }
                HandedChips(session.nameB, session.handedB) { session = session.copy(handedB = it) }
            QualityChips(session.quality) { session = session.copy(quality = it) }
                Button(enabled = granted && result != null, onClick = {
                    store.save(session.copy(markers = markers))
                    val out = FileOutputOptions.Builder(store.videoFile(id)).build()
                    startedAt = System.currentTimeMillis()
                    recording = videoCapture.output.prepareRecording(ctx, out).start(ContextCompat.getMainExecutor(ctx)) { ev ->
                        if (ev is VideoRecordEvent.Finalize) {
                            if (ev.hasError()) {
                                store.save(session.copy(markers = markers, status = "failed", error = "Recording error ${ev.error}"))
                            } else {
                                store.save(session.copy(markers = markers, status = "recorded"))
                                AnalysisService.start(ctx, id)
                            }
                            nav.navigate("analysis/$id") { popUpTo("home") }
                        }
                    }
                }) { Text("Start recording") }
            } else {
                val s = elapsed / 1000
                Text("Recording", color = Color(0xFFFF5252), style = MaterialTheme.typography.titleMedium)
                Text("%d:%02d:%02d".format(s / 3600, (s / 60) % 60, s % 60), style = MaterialTheme.typography.displaySmall)
                Text("Keep the phone still. Play normally; players may change ends.", style = MaterialTheme.typography.bodySmall)
                Button(colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD32F2F)), onClick = {
                    recording?.stop(); recording = null
                }) { Text("Stop & analyse") }
            }
        }
    }
}
