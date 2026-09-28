package com.longtennis.app

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.longtennis.app.analysis.AnalysisService
import com.longtennis.app.analysis.Calibration
import com.longtennis.app.analysis.Pt
import com.longtennis.app.data.SessionStore
import com.longtennis.app.ui.AnalysisScreen
import com.longtennis.app.ui.CaptureScreen
import com.longtennis.app.ui.HomeScreen
import com.longtennis.app.ui.ReportScreen
import com.longtennis.app.ui.SetupScreen
import org.json.JSONObject
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val autoId = handleAutomation(intent)
        setContent { AppTheme { AppNav(if (autoId != null) "analysis/$autoId" else "home") } }
    }

    /**
     * Test automation (used for emulator/device verification):
     * adb shell am start -n com.longtennis.app/.MainActivity --es autoVideo <path in app external files>
     *   [--es autoCalibration <json file>] [--es autoMarkers "name:x,y;..."] [--ei maxFrames N]
     */
    private fun handleAutomation(i: Intent?): String? {
        val path = i?.getStringExtra("autoVideo") ?: return null
        val store = SessionStore(this)
        val f = File(path)
        val s0 = store.create("Automated: ${f.name}", Uri.fromFile(f).toString())
        var s = s0.copy(nameA = i.getStringExtra("nameA") ?: "Player A", nameB = i.getStringExtra("nameB") ?: "Player B",
            quality = i.getStringExtra("quality") ?: "fast")
        i.getStringExtra("autoMarkers")?.let { m ->
            s = s.copy(markers = m.split(";").filter { it.contains(":") }.associate { kv ->
                val (k, v) = kv.split(":"); val (x, y) = v.split(","); k to Pt(x.toDouble(), y.toDouble())
            })
        }
        store.save(s.copy(status = "recorded"))
        i.getStringExtra("autoCalibration")?.let { store.saveCalibration(s.id, Calibration.fromJson(JSONObject(File(it).readText()))) }
        AnalysisService.start(this, s.id, i.getIntExtra("maxFrames", Int.MAX_VALUE))
        return s.id
    }
}

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val scheme = if (dark) darkColorScheme(primary = Color(0xFFD7F036), secondary = Color(0xFF7FB2FF), onPrimary = Color(0xFF1B2600))
    else lightColorScheme(primary = Color(0xFF1E4D8C), secondary = Color(0xFF5F7A00))
    MaterialTheme(colorScheme = scheme, content = content)
}

@Composable
fun AppNav(start: String) {
    val nav = rememberNavController()
    NavHost(nav, startDestination = start) {
        composable("home") { HomeScreen(nav) }
        composable("capture/{id}") { CaptureScreen(nav, it.arguments!!.getString("id")!!) }
        composable("setup/{id}") { SetupScreen(nav, it.arguments!!.getString("id")!!) }
        composable("analysis/{id}") { AnalysisScreen(nav, it.arguments!!.getString("id")!!) }
        composable("report/{id}") { ReportScreen(nav, it.arguments!!.getString("id")!!) }
    }
}
