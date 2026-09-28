import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import '../api/client.dart';
import '../models/session.dart';
import '../widgets/landing_map.dart';

const colorA = Color(0xFF4FC3F7);
const colorB = Color(0xFFFFB74D);
const strokes = ['serve', 'forehand', 'backhand', 'volley', 'overhead', 'unknown'];

class ReportScreen extends StatefulWidget {
  const ReportScreen({super.key, required this.session});
  final Session session;

  @override
  State<ReportScreen> createState() => _ReportScreenState();
}

class _ReportScreenState extends State<ReportScreen> {
  Map<String, dynamic>? _report;
  Map<String, dynamic> _corr = {'shots': [], 'added': []};
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final api = ApiClient.instance;
      final r = await api.report(widget.session.id);
      final c = await api.corrections(widget.session.id);
      setState(() { _report = r; _corr = {'shots': List.of(c['shots'] ?? []), 'added': List.of(c['added'] ?? [])}; });
    } catch (e) {
      setState(() => _error = describeError(e));
    }
  }

  /// Merge one correction into the stored set and save it; the server returns the recomputed report.
  Future<void> _correct({int? hitFrame, Map<String, dynamic>? added, String action = 'edit', String? playerId, String? stroke}) async {
    final shots = (_corr['shots'] as List).cast<Map>().where((c) => c['hitFrame'] != hitFrame).toList();
    if (hitFrame != null) {
      shots.add({'hitFrame': hitFrame, 'action': action, if (playerId != null) 'playerId': playerId, if (stroke != null) 'stroke': stroke});
    }
    final addedList = List.of(_corr['added'] as List)..addAll(added != null ? [added] : []);
    final body = {'shots': shots, 'added': addedList};
    try {
      final r = await ApiClient.instance.saveCorrections(widget.session.id, body);
      setState(() { _report = r; _corr = body; });
    } catch (e) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(describeError(e))));
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.session;
    final r = _report;
    return DefaultTabController(
      length: 3,
      child: Scaffold(
        appBar: AppBar(
          title: Text('${s.nameA} vs ${s.nameB}'),
          bottom: const TabBar(tabs: [Tab(text: 'Summary'), Tab(text: 'Landing'), Tab(text: 'Shots')]),
        ),
        body: r == null
            ? Center(child: _error != null ? Text(_error!) : const CircularProgressIndicator())
            : TabBarView(children: [
                _Summary(r, s),
                _Landing(r),
                _Shots(r, s, onCorrect: _correct),
              ]),
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
    final review = r['review'] as Map?;
    return ListView(padding: const EdgeInsets.all(12), children: [
      Text('Session ${formatDuration(dur)} · $shots shots'
          '${review != null ? ' · ${review['corrections']} reviewed, ${review['added']} added by you' : ''}',
          style: Theme.of(context).textTheme.bodyMedium),
      for (final pid in ['A', 'B']) _PlayerCard(pid: pid, p: _player(r, pid)),
      const SizedBox(height: 8),
      Text('Speed = average ground speed from the hitter to the first bounce (not racket speed). '
          'Zone percentages use shots whose first bounce was detected in the opponent\'s half. '
          'Fix wrong shots in the Shots tab; the numbers update.',
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
    final st = p['strokes'] as Map<String, dynamic>;
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
            if ((st[e.key] ?? 0) > 0 || !['overhead', 'unknown'].contains(e.key))
              Row(children: [Expanded(child: Text(e.value)), Text('${st[e.key] ?? 0}', style: const TextStyle(fontWeight: FontWeight.bold))]),
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

typedef CorrectFn = Future<void> Function({int? hitFrame, Map<String, dynamic>? added, String action, String? playerId, String? stroke});

class _Shots extends StatelessWidget {
  const _Shots(this.r, this.s, {required this.onCorrect});
  final Map<String, dynamic> r;
  final Session s;
  final CorrectFn onCorrect;

  @override
  Widget build(BuildContext context) {
    final shots = (r['shots'] as List).cast<Map<String, dynamic>>();
    final names = {'A': s.nameA, 'B': s.nameB};
    return Column(children: [
      Expanded(
        child: ListView.separated(
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
              title: Text('${names[pid] ?? 'Unknown'} · ${sh['stroke']}',
                  style: TextStyle(color: pid == 'A' ? colorA : pid == 'B' ? colorB : null)),
              subtitle: Text([
                ((l['region'] ?? l['status']) as String).replaceAll('_', ' '),
                if (sh['speed'] != null) '${((sh['speed'] as Map)['avgKmh'] as num).round()} km/h',
                if (sh['reviewed'] == true) 'reviewed',
              ].join(' · ')),
              trailing: const Icon(Icons.edit_outlined),
              onTap: () => showModalBottomSheet(
                  context: context,
                  isScrollControlled: true,
                  builder: (_) => _ShotEditor(session: s, t: t, shot: sh, names: names, onCorrect: onCorrect)),
            );
          },
        ),
      ),
      SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(8),
          child: OutlinedButton.icon(
            icon: const Icon(Icons.add),
            label: const Text('Add a missed shot'),
            onPressed: () => showModalBottomSheet(
                context: context,
                isScrollControlled: true,
                builder: (_) => _ShotEditor(session: s, t: 0, shot: null, names: names, onCorrect: onCorrect)),
          ),
        ),
      ),
    ]);
  }
}

/// Watch the shot, then confirm / change / delete it, or add a missed shot at the paused position.
class _ShotEditor extends StatefulWidget {
  const _ShotEditor({required this.session, required this.t, required this.shot, required this.names, required this.onCorrect});
  final Session session;
  final double t;
  final Map<String, dynamic>? shot;
  final Map<String, String> names;
  final CorrectFn onCorrect;

  @override
  State<_ShotEditor> createState() => _ShotEditorState();
}

class _ShotEditorState extends State<_ShotEditor> {
  late final VideoPlayerController _c = VideoPlayerController.networkUrl(
      Uri.parse(ApiClient.instance.videoUrl(widget.session.id)), httpHeaders: ApiClient.instance.authHeaders);
  late String _pid = (widget.shot?['playerId'] as String?) ?? 'A';
  late String _stroke = (widget.shot?['stroke'] as String?) ?? 'forehand';
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _c.initialize().then((_) async {
      if (widget.shot != null) {
        await _c.seekTo(Duration(milliseconds: ((widget.t - 2).clamp(0, double.infinity) * 1000).round()));
        await _c.play();
      }
      if (mounted) setState(() {});
    });
    _c.addListener(() { if (mounted) setState(() {}); });
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  Future<void> _do(Future<void> Function() f) async {
    setState(() => _busy = true);
    await f();
    if (mounted) Navigator.pop(context);
  }

  @override
  Widget build(BuildContext context) {
    final isNew = widget.shot == null;
    final pos = _c.value.position.inMilliseconds / 1000.0;
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.start, children: [
          AspectRatio(
              aspectRatio: _c.value.isInitialized ? _c.value.aspectRatio : 16 / 9,
              child: _c.value.isInitialized ? VideoPlayer(_c) : const Center(child: CircularProgressIndicator())),
          if (_c.value.isInitialized)
            Row(children: [
              IconButton(icon: Icon(_c.value.isPlaying ? Icons.pause : Icons.play_arrow),
                  onPressed: () => _c.value.isPlaying ? _c.pause() : _c.play()),
              IconButton(icon: const Icon(Icons.replay_5), onPressed: () => _c.seekTo(_c.value.position - const Duration(seconds: 2))),
              Text(formatDuration(pos)),
              Expanded(child: VideoProgressIndicator(_c, allowScrubbing: true)),
            ]),
          if (isNew) const Text('Pause the video at the moment of contact, then choose who hit it.'),
          const SizedBox(height: 6),
          Wrap(spacing: 8, children: [
            for (final p in ['A', 'B'])
              ChoiceChip(label: Text(widget.names[p]!), selected: _pid == p, onSelected: (_) => setState(() => _pid = p)),
          ]),
          Wrap(spacing: 6, children: [
            for (final s in strokes)
              ChoiceChip(label: Text(s), selected: _stroke == s, onSelected: (_) => setState(() => _stroke = s)),
          ]),
          const SizedBox(height: 8),
          Row(children: [
            if (!isNew)
              TextButton.icon(
                  icon: const Icon(Icons.delete_outline),
                  label: const Text('Not a shot'),
                  onPressed: _busy ? null : () => _do(() => widget.onCorrect(hitFrame: widget.shot!['hitFrame'] as int, action: 'delete'))),
            const Spacer(),
            FilledButton(
              onPressed: _busy
                  ? null
                  : () => _do(() => isNew
                      ? widget.onCorrect(added: {'timeS': pos, 'playerId': _pid, 'stroke': _stroke})
                      : widget.onCorrect(hitFrame: widget.shot!['hitFrame'] as int, action: 'edit', playerId: _pid, stroke: _stroke)),
              child: Text(isNew ? 'Add shot' : 'Save'),
            ),
          ]),
        ]),
      ),
    );
  }
}
