import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../api/client.dart';
import '../models/session.dart';
import '../widgets/landing_map.dart';

const colorA = Color(0xFF4FC3F7);
const colorB = Color(0xFFFFB74D);

class ReportScreen extends StatefulWidget {
  const ReportScreen({super.key, required this.session});
  final Session session;

  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  late final Future<Map<String, dynamic>> _report = ApiClient.instance.report(widget.session.id);

  @override
  Widget build(BuildContext context) {
    final s = widget.session;
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: Text('${s.nameA} vs ${s.nameB}'),
          bottom: const TabBar(tabs: [Tab(text: 'Summary'), Tab(text: 'Landing'), Tab(text: 'Shots')]),
        ),
        body: FutureBuilder<Map<String, dynamic>>(
          future: _report,
          builder: (context, snap) {
            if (snap.hasError) return Center(child: Text(describeError(snap.error!)));
            if (!snap.hasData) return const Center(child: CircularProgressIndicator());
            final r = snap.data!;
            return TabBarView(children: [
              _Summary(r, s),
              _Landing(r),
              _Shots(r, s),
            ]);
          },
        ),
      ),
    );
  }
}

Map<String, dynamic> _player(Map<String, dynamic> r, String pid) => (r['players'] as Map<String, dynamic>)[pid] as Map<String, dynamic>;

class _Summary extends StatelessWidget {
  const _Summary(this.r, this.s);
  final Map<String, dynamic> r;
  final Session s;

  @override
  Widget build(BuildContext context) {
    final shots = (r['shots'] as List).length;
    final dur = ((r['video'] as Map?)?['frameCount'] ?? 0) / ((r['fps'] as num?) ?? 30);
    return ListView(padding: const EdgeInsets.all(12), children: [
      Text('Session ${formatDuration(dur)} · $shots shots detected', style: Theme.of(context).textTheme.bodyMedium),
      for (final pid in ['A', 'B']) _PlayerCard(pid: pid, p: _player(r, pid)),
      const SizedBox(height: 8),
      Text('Speed = average ground speed from the hitter to the first bounce (not racket speed). '
          'Zone percentages use shots whose first bounce was detected in the opponent\'s half.',
          style: Theme.of(context).textTheme.labelSmall),
    ]);
  }
}

class _PlayerCard extends StatelessWidget {
  const _PlayerCard({required this.pid, required this.p});
  final String pid;
  final Map<String, dynamic> p;

  @override
  Widget build(BuildContext context) {
    final strokes = p['strokes'] as Map<String, dynamic>;
    final speed = p['speed'] as Map<String, dynamic>;
    final landing = p['landing'] as Map<String, dynamic>;
    const labels = {'serve': 'Serves', 'forehand': 'Forehands', 'backhand': 'Backhands', 'volley': 'Volleys', 'overhead': 'Overheads', 'unknown': 'Unclassified'};
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(p['name'] as String, style: Theme.of(context).textTheme.titleLarge?.copyWith(color: pid == 'A' ? colorA : colorB)),
          Text('${p['shots']} shots'),
          const SizedBox(height: 8),
          for (final e in labels.entries)
            if ((strokes[e.key] ?? 0) > 0 || !['overhead', 'unknown'].contains(e.key))
              Row(children: [Expanded(child: Text(e.value)), Text('${strokes[e.key] ?? 0}', style: const TextStyle(fontWeight: FontWeight.bold))]),
          const Divider(),
          Text('First bounces in court: ${landing['inCourtFirstBounces']} · out: ${landing['out']} · not seen: ${landing['noObservedBounce']}'),
          if ((speed['shotsWithEstimate'] ?? 0) > 0)
            Text('Speed: median ${speed['medianAvgKmh']} km/h · top ${speed['maxAvgKmh']} km/h'
                '${speed['serveMedianAvgKmh'] != null ? ' · serves ${speed['serveMedianAvgKmh']} km/h' : ''}'),
        ]),
      ),
    );
  }
}

class _Landing extends StatefulWidget {
  const _Landing(this.r);
  final Map<String, dynamic> r;

  @override
  State<_Landing> createState() => _LandingState();
}

class _LandingState extends State<_Landing> {
  String _pid = 'A';

  @override
  Widget build(BuildContext context) {
    final p = _player(widget.r, _pid);
    final points = <LandingPoint>[];
    for (final s in (widget.r['shots'] as List).cast<Map<String, dynamic>>()) {
      final l = s['landing'] as Map<String, dynamic>;
      if (s['playerId'] != _pid || l['courtM'] == null) continue;
      final c = (l['courtM'] as List).cast<num>();
      points.add(LandingPoint(c[0].toDouble(), c[1].toDouble(), s['side'] as String? ?? 'near', l['status'] == 'out'));
    }
    final regions = ((p['landing'] as Map)['regions'] as Map).map((k, v) => MapEntry(k as String, (v as Map)['percent'] as num?));
    return ListView(padding: const EdgeInsets.all(12), children: [
      SegmentedButton<String>(
        segments: [
          ButtonSegment(value: 'A', label: Text(_player(widget.r, 'A')['name'] as String)),
          ButtonSegment(value: 'B', label: Text(_player(widget.r, 'B')['name'] as String)),
        ],
        selected: {_pid},
        onSelectionChanged: (v) => setState(() => _pid = v.first),
      ),
      const SizedBox(height: 12),
      Text('Where ${p['name']}\'s shots first bounced (seen from the hitter\'s end)'),
      const SizedBox(height: 8),
      LandingMap(points: points, regionPercent: regions, color: _pid == 'A' ? colorA : colorB),
      const SizedBox(height: 8),
      Text('Dots: first bounces (red = out). Numbers: share of in-court first bounces per zone.', style: Theme.of(context).textTheme.labelSmall),
    ]);
  }
}

class _Shots extends StatelessWidget {
  const _Shots(this.r, this.s);
  final Map<String, dynamic> r;
  final Session s;

  @override
  Widget build(BuildContext context) {
    final shots = (r['shots'] as List).cast<Map<String, dynamic>>();
    final names = {'A': s.nameA, 'B': s.nameB};
    return ListView.separated(
      itemCount: shots.length,
      separatorBuilder: (_, _) => const Divider(height: 1),
      itemBuilder: (_, i) {
        final sh = shots[i];
        final l = sh['landing'] as Map<String, dynamic>;
        final pid = sh['playerId'] as String;
        final t = (sh['timeS'] as num).toDouble();
        return ListTile(
          dense: true,
          leading: Text(formatDuration(t)),
          title: Text('${names[pid] ?? 'Unknown'} · ${sh['stroke']}', style: TextStyle(color: pid == 'A' ? colorA : pid == 'B' ? colorB : null)),
          subtitle: Text([
            ((l['region'] ?? l['status']) as String).replaceAll('_', ' '),
            if (sh['speed'] != null) '${((sh['speed'] as Map)['avgKmh'] as num).round()} km/h',
          ].join(' · ')),
          trailing: const Icon(Icons.play_circle_outline),
          onTap: () => showDialog(context: context, builder: (_) => _Clip(sessionId: s.id, t: t)),
        );
      },
    );
  }
}

/// Plays the uploaded video from the server, 2 s before the shot.
class _Clip extends StatefulWidget {
  const _Clip({required this.sessionId, required this.t});
  final String sessionId;
  final double t;

  @override
  State<_Clip> createState() => _ClipState();
}

class _ClipState extends State<_Clip> {
  late final VideoPlayerController _c = VideoPlayerController.networkUrl(
      Uri.parse(ApiClient.instance.videoUrl(widget.sessionId)), httpHeaders: ApiClient.instance.authHeaders);

  @override
  void initState() {
    super.initState();
    _c.initialize().then((_) async {
      await _c.seekTo(Duration(milliseconds: ((widget.t - 2).clamp(0, double.infinity) * 1000).round()));
      await _c.play();
      if (mounted) setState(() {});
    });
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Dialog(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          AspectRatio(
              aspectRatio: _c.value.isInitialized ? _c.value.aspectRatio : 16 / 9,
              child: _c.value.isInitialized ? VideoPlayer(_c) : const Center(child: CircularProgressIndicator())),
          TextButton(onPressed: () => Navigator.pop(context), child: const Text('Close')),
        ]),
      );
}
