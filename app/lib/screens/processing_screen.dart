import 'dart:async';

import 'package:flutter/material.dart';

import '../api/client.dart';
import '../models/session.dart';
import 'report_screen.dart';

class ProcessingScreen extends StatefulWidget {
  const ProcessingScreen({super.key, required this.sessionId});
  final String sessionId;

  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  Session? _s;
  String? _error;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    _poll();
    _timer = Timer.periodic(const Duration(seconds: 4), (_) => _poll());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _poll() async {
    try {
      final s = await ApiClient.instance.get(widget.sessionId);
      if (!mounted) return;
      setState(() { _s = s; _error = null; });
      if (s.status == 'done') {
        _timer?.cancel();
        Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => ReportScreen(session: s)));
      }
    } catch (e) {
      if (mounted) setState(() => _error = describeError(e));
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = _s;
    return Scaffold(
      appBar: AppBar(title: const Text('Analysing')),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: s == null
            ? Center(child: _error != null ? Text(_error!) : const CircularProgressIndicator())
            : Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('${s.nameA} vs ${s.nameB}', style: Theme.of(context).textTheme.titleLarge),
                const SizedBox(height: 16),
                if (s.status == 'failed') ...[
                  Text(s.error ?? 'Analysis failed', style: TextStyle(color: Theme.of(context).colorScheme.error)),
                  const SizedBox(height: 12),
                  FilledButton(onPressed: () async { await ApiClient.instance.analyze(s.id); _poll(); }, child: const Text('Retry')),
                ] else ...[
                  Text(switch (s.stage) {
                    'ball' => 'Tracking the ball and players',
                    'pose' => 'Tracking players',
                    'analysis' => 'Building the report',
                    _ => statusLabel(s.status),
                  }),
                  const SizedBox(height: 12),
                  LinearProgressIndicator(value: s.status == 'queued' ? null : s.fraction),
                  const SizedBox(height: 8),
                  Text('${(s.fraction * 100).toStringAsFixed(0)}%'
                      '${s.etaSeconds != null ? ' · about ${formatDuration(s.etaSeconds!)} left' : ''}'),
                  const SizedBox(height: 24),
                  const Text('You can close the app; the server keeps working and the report appears in your session list.'),
                ],
                if (_error != null) Padding(padding: const EdgeInsets.only(top: 12), child: Text(_error!)),
              ]),
      ),
    );
  }
}
