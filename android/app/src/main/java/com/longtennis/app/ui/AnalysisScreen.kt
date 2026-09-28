package com.longtennis.app.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.longtennis.app.analysis.AnalysisService
import com.longtennis.app.analysis.AnalysisState
import com.longtennis.app.data.SessionStore

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AnalysisScreen(nav: NavController, id: String) {
    val ctx = LocalContext.current
    val store = remember { SessionStore(ctx) }
    val st by AnalysisState.state.collectAsState()
    val session = remember(st.stage) { store.load(id) }
    LaunchedEffect(st.stage, st.sessionId) {
        if (st.sessionId == id && st.stage == "done") nav.navigate("report/$id") { popUpTo("home") }
    }
    Scaffold(topBar = { TopAppBar(title = { Text("Analysing") }) }) { pad ->
        Column(Modifier.padding(pad).padding(20.dp).fillMaxSize(), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            val mine = st.sessionId == id
            when {
                session?.status == "done" -> Button(onClick = { nav.navigate("report/$id") }) { Text("Open report") }
                mine && st.stage == "failed" || session?.status == "failed" -> {
                    Text("Analysis failed: ${st.error ?: session?.error}", color = MaterialTheme.colorScheme.error)
                    Button(onClick = { AnalysisService.start(ctx, id) }) { Text("Retry") }
                }
                mine -> {
                    Text(stageLabel(st.stage), style = MaterialTheme.typography.titleLarge)
                    LinearProgressIndicator(progress = { st.fraction }, modifier = Modifier.fillMaxWidth())
                    Text("${(st.fraction * 100).toInt()}%  ${st.detail}")
                    Text("You can leave this screen; analysis continues in the background (keep the phone charging for long sessions).",
                        style = MaterialTheme.typography.bodySmall)
                    OutlinedButton(onClick = { AnalysisState.cancelRequested = true }) { Text("Cancel") }
                }
                else -> {
                    Text("Session is ${statusLabel(session?.status ?: "")}.")
                    Button(onClick = { AnalysisService.start(ctx, id) }) { Text("Start analysis") }
                }
            }
            OutlinedButton(onClick = { nav.navigate("home") { popUpTo("home") { inclusive = true } } }) { Text("Home") }
        }
    }
}

private fun stageLabel(s: String) = when (s) {
    "calibration" -> "Checking court"; "background" -> "Preparing video"; "tracking" -> "Tracking ball and players"
    "identity" -> "Identifying players"; "events" -> "Detecting shots"; "done" -> "Done"; else -> "Starting"
}
