package com.longtennis.app.ui

import android.content.Intent
import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.Videocam
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.longtennis.app.data.Session
import com.longtennis.app.data.SessionStore

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(nav: NavController) {
    val ctx = LocalContext.current
    val store = remember { SessionStore(ctx) }
    var sessions by remember { mutableStateOf(store.list()) }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri: Uri? ->
        if (uri != null) {
            runCatching { ctx.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION) }
            val s = store.create("Imported video", uri.toString())
            nav.navigate("setup/${s.id}")
        }
    }
    Scaffold(
        topBar = { TopAppBar(title = { Text("Long Tennis") }) },
        floatingActionButton = {
            ExtendedFloatingActionButton(onClick = { val s = store.create("Match session"); nav.navigate("capture/${s.id}") },
                icon = { Icon(Icons.Default.Videocam, null) }, text = { Text("New session") })
        },
    ) { pad ->
        Column(Modifier.padding(pad).padding(horizontal = 16.dp).fillMaxSize()) {
            Text("Mount the phone behind a baseline with the whole court in view, then play. Stats per player appear after the session.",
                style = MaterialTheme.typography.bodyMedium)
            Spacer(Modifier.padding(6.dp))
            OutlinedButton(onClick = { picker.launch(arrayOf("video/*")) }) {
                Icon(Icons.Default.FileOpen, null); Spacer(Modifier.width(8.dp)); Text("Analyse an existing video")
            }
            Spacer(Modifier.padding(6.dp))
            if (sessions.isEmpty()) Text("No sessions yet.", style = MaterialTheme.typography.bodyLarge)
            LazyColumn(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                items(sessions, key = { it.id }) { s ->
                    SessionRow(s, onOpen = {
                        when (s.status) {
                            "done" -> nav.navigate("report/${s.id}")
                            "analyzing", "failed", "recorded" -> nav.navigate("analysis/${s.id}")
                            else -> nav.navigate(if (s.videoUri.startsWith("content")) "setup/${s.id}" else "capture/${s.id}")
                        }
                    }, onDelete = { store.delete(s.id); sessions = store.list() })
                }
            }
        }
    }
}

@Composable
private fun SessionRow(s: Session, onOpen: () -> Unit, onDelete: () -> Unit) {
    Card(Modifier.fillMaxWidth().clickable(onClick = onOpen)) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(Modifier.weight(1f)) {
                Text("${s.nameA} vs ${s.nameB}", style = MaterialTheme.typography.titleMedium)
                Text("${s.dateLabel} · ${statusLabel(s.status)}", style = MaterialTheme.typography.bodySmall)
                s.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
            }
            IconButton(onClick = onDelete) { Icon(Icons.Default.Delete, "Delete") }
        }
    }
}

fun statusLabel(s: String) = when (s) {
    "done" -> "Report ready"; "analyzing" -> "Analysing"; "failed" -> "Analysis failed"; "recorded" -> "Recorded"; else -> "Setup"
}
