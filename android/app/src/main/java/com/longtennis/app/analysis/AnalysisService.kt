package com.longtennis.app.analysis

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import com.longtennis.app.data.SessionStore
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/** Global analysis progress, observed by the UI. */
object AnalysisState {
    data class State(val sessionId: String? = null, val stage: String = "idle", val fraction: Float = 0f,
                     val detail: String = "", val error: String? = null)
    val flow = MutableStateFlow(State())
    val state: StateFlow<State> = flow
    @Volatile var cancelRequested = false
}

/** Foreground service so a long session keeps analysing with the screen off. */
class AnalysisService : Service() {
    companion object {
        const val EXTRA_ID = "sessionId"
        const val EXTRA_MAX_FRAMES = "maxFrames"
        private const val CHANNEL = "analysis"

        fun start(context: Context, sessionId: String, maxFrames: Int = Int.MAX_VALUE) {
            val i = Intent(context, AnalysisService::class.java).putExtra(EXTRA_ID, sessionId).putExtra(EXTRA_MAX_FRAMES, maxFrames)
            if (Build.VERSION.SDK_INT >= 26) context.startForegroundService(i) else context.startService(i)
        }

        /** Calibrate from the video's frame at ~1 s using the normalised setup markers. */
        fun calibrateFromVideo(context: Context, uri: Uri, markers: Map<String, Pt>): CourtCalibrator.Result? {
            val mmr = MediaMetadataRetriever()
            mmr.setDataSource(context, uri)
            val bmp: Bitmap = mmr.getFrameAtTime(1_000_000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: mmr.frameAtTime ?: return null
            mmr.release()
            return calibrateBitmap(bmp, markers)
        }

        fun calibrateBitmap(bmp: Bitmap, markers: Map<String, Pt>): CourtCalibrator.Result? {
            val w = bmp.width; val h = bmp.height
            val px = IntArray(w * h); bmp.getPixels(px, 0, w, 0, 0, w, h)
            val init = markers.mapValues { (_, p) -> Pt(p.x * w, p.y * h) }
            if (init.size < 4) return null
            return CourtCalibrator.calibrate(px, w, h, init)
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val id = intent?.getStringExtra(EXTRA_ID) ?: return START_NOT_STICKY
        val maxFrames = intent.getIntExtra(EXTRA_MAX_FRAMES, Int.MAX_VALUE)
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= 26) nm.createNotificationChannel(NotificationChannel(CHANNEL, "Session analysis", NotificationManager.IMPORTANCE_LOW))
        val notif = notification("Analysing session…", 0)
        if (Build.VERSION.SDK_INT >= 29) startForeground(1, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC) else startForeground(1, notif)
        val wake = (getSystemService(POWER_SERVICE) as PowerManager).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "longtennis:analysis")
        wake.acquire(8 * 3600 * 1000L)
        AnalysisState.cancelRequested = false
        scope.launch {
            val store = SessionStore(this@AnalysisService)
            var s = store.load(id) ?: return@launch
            try {
                s = s.copy(status = "analyzing", error = null); store.save(s)
                AnalysisState.flow.value = AnalysisState.State(id, "calibration", 0f)
                val uri = Uri.parse(s.videoUri)
                var cal = store.loadCalibration(id)
                if (cal == null) {
                    val r = calibrateFromVideo(this@AnalysisService, uri, s.markers) ?: error("Court calibration failed: place the four markers on the court lines")
                    cal = Calibration(r.imageFromCourt, r.width, r.height, r.reprojectionRmsPx)
                    store.saveCalibration(id, cal)
                }
                var last = 0L
                val report = Pipeline(this@AnalysisService).run(uri, cal, mapOf("A" to s.nameA, "B" to s.nameB), s.seedASide, s.handedA, s.handedB, s.quality, maxFrames,
                    onProgress = { p ->
                        AnalysisState.flow.value = AnalysisState.State(id, p.stage, p.fraction, p.detail)
                        val now = System.currentTimeMillis()
                        if (now - last > 2000) { last = now; nm.notify(1, notification("${p.stage} ${(p.fraction * 100).toInt()}% ${p.detail}", (p.fraction * 100).toInt())) }
                    }, isCancelled = { AnalysisState.cancelRequested })
                store.saveReport(id, report)
                store.save(s.copy(status = "done"))
                AnalysisState.flow.value = AnalysisState.State(id, "done", 1f)
            } catch (e: Throwable) {
                Log.e("AnalysisService", "analysis failed", e)
                val msg = if (e is Pipeline.Cancelled) "Cancelled" else (e.message ?: e.javaClass.simpleName)
                store.save(s.copy(status = "failed", error = msg))
                AnalysisState.flow.value = AnalysisState.State(id, "failed", 0f, error = msg)
            } finally {
                if (wake.isHeld) wake.release()
                stopForeground(STOP_FOREGROUND_REMOVE); stopSelf()
            }
        }
        return START_NOT_STICKY
    }

    private fun notification(text: String, pct: Int): Notification {
        val b = if (Build.VERSION.SDK_INT >= 26) Notification.Builder(this, CHANNEL) else @Suppress("DEPRECATION") Notification.Builder(this)
        return b.setContentTitle("Long Tennis").setContentText(text).setSmallIcon(android.R.drawable.stat_notify_sync)
            .setProgress(100, pct, pct == 0).setOngoing(true).build()
    }
}
