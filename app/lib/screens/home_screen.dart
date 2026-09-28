import 'package:flutter/material.dart';

import '../api/client.dart';
import '../models/session.dart';
import 'court_setup_screen.dart';
import 'new_session_screen.dart';
import 'processing_screen.dart';
import 'report_screen.dart';
import 'settings_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late Future<List<Session>> _future = ApiClient.instance.sessions();

  void _refresh() => setState(() => _future = ApiClient.instance.sessions());

  Future<void> _open(Session s) async {
    final Widget page = switch (s.status) {
      'done' => ReportScreen(session: s),
      'queued' || 'processing' || 'failed' => ProcessingScreen(sessionId: s.id),
      'uploaded' || 'calibrated' => CourtSetupScreen(session: s),
      _ => NewSessionScreen(resume: s),
    };
    await Navigator.push(context, MaterialPageRoute(builder: (_) => page));
    _refresh();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Long Tennis'), actions: [
        IconButton(icon: const Icon(Icons.refresh), onPressed: _refresh),
        IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () async {
              await Navigator.push(context, MaterialPageRoute(builder: (_) => const SettingsScreen()));
              _refresh();
            }),
      ]),
      floatingActionButton: FloatingActionButton.extended(
        icon: const Icon(Icons.add),
        label: const Text('New session'),
        onPressed: () async {
          await Navigator.push(context, MaterialPageRoute(builder: (_) => const NewSessionScreen()));
          _refresh();
        },
      ),
      body: RefreshIndicator(
        onRefresh: () async => _refresh(),
        child: FutureBuilder<List<Session>>(
          future: _future,
          builder: (context, snap) {
            if (snap.connectionState != ConnectionState.done) return const Center(child: CircularProgressIndicator());
            if (snap.hasError) {
              return ListView(padding: const EdgeInsets.all(24), children: [
                const Icon(Icons.cloud_off, size: 48),
                const SizedBox(height: 12),
                Text(describeError(snap.error!), textAlign: TextAlign.center),
                const SizedBox(height: 12),
                Text('Server: ${ApiClient.instance.baseUrl}', textAlign: TextAlign.center, style: Theme.of(context).textTheme.bodySmall),
              ]);
            }
            final list = snap.data!;
            if (list.isEmpty) {
              return ListView(padding: const EdgeInsets.all(24), children: const [
                Text('No sessions yet.\n\nTap "New session" to record a match or upload a video. '
                    'Place the phone behind a baseline, raised if possible, with the whole court in view.'),
              ]);
            }
            return ListView.separated(
              padding: const EdgeInsets.only(bottom: 96),
              itemCount: list.length,
              separatorBuilder: (_, _) => const Divider(height: 1),
              itemBuilder: (_, i) {
                final s = list[i];
                return ListTile(
                  title: Text('${s.nameA} vs ${s.nameB}'),
                  subtitle: Text([
                    '${s.createdAt.toLocal()}'.substring(0, 16),
                    if (s.durationS != null) formatDuration(s.durationS!),
                    statusLabel(s.status) + (s.isBusy ? ' ${(s.fraction * 100).round()}%' : ''),
                  ].join(' · ')),
                  leading: Icon(switch (s.status) {
                    'done' => Icons.insights,
                    'failed' => Icons.error_outline,
                    'queued' || 'processing' => Icons.hourglass_top,
                    _ => Icons.sports_tennis,
                  }),
                  onTap: () => _open(s),
                  trailing: IconButton(
                    icon: const Icon(Icons.delete_outline),
                    onPressed: () async {
                      final ok = await showDialog<bool>(
                          context: context,
                          builder: (c) => AlertDialog(
                                title: const Text('Delete session?'),
                                content: const Text('The video and results are removed from the server.'),
                                actions: [
                                  TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Cancel')),
                                  FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Delete')),
                                ],
                              ));
                      if (ok == true) {
                        await ApiClient.instance.delete(s.id);
                        _refresh();
                      }
                    },
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }
}
