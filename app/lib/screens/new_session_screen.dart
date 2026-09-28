import 'dart:io';

import 'package:dio/dio.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../api/client.dart';
import '../models/session.dart';
import 'court_setup_screen.dart';
import 'record_screen.dart';

/// Players + source video, then a resumable upload. `resume` continues an interrupted upload.
class NewSessionScreen extends StatefulWidget {
  const NewSessionScreen({super.key, this.resume});
  final Session? resume;

  @override
  State<NewSessionScreen> createState() => _NewSessionScreenState();
}

class _NewSessionScreenState extends State<NewSessionScreen> {
  late final _nameA = TextEditingController(text: widget.resume?.nameA ?? 'Player A');
  late final _nameB = TextEditingController(text: widget.resume?.nameB ?? 'Player B');
  late String _seedASide = widget.resume?.seedASide ?? 'near';
  late String _handedA = widget.resume?.handedA ?? 'right';
  late String _handedB = widget.resume?.handedB ?? 'right';
  late String _quality = widget.resume?.quality ?? 'standard';
  File? _file;
  double? _progress;
  String? _error;
  CancelToken? _cancel;

  Map<String, dynamic> get _body => {
        'nameA': _nameA.text.trim(), 'nameB': _nameB.text.trim(), 'seedASide': _seedASide,
        'handedA': _handedA, 'handedB': _handedB, 'quality': _quality,
      };

  Future<void> _record() async {
    final f = await Navigator.push<File?>(context, MaterialPageRoute(builder: (_) => const RecordScreen()));
    if (f != null) setState(() => _file = f);
  }

  Future<void> _pick() async {
    final r = await FilePicker.pickFiles(type: FileType.video);
    final p = r.isEmpty ? null : r.first.path;
    if (p != null) setState(() => _file = File(p));
  }

  Future<void> _upload() async {
    if (_file == null) return;
    setState(() { _progress = 0; _error = null; _cancel = CancelToken(); });
    await WakelockPlus.enable();
    try {
      final api = ApiClient.instance;
      final s = widget.resume != null ? await api.patch(widget.resume!.id, _body) : await api.create(_body);
      final done = await api.upload(s.id, _file!, cancel: _cancel,
          onProgress: (sent, total) => setState(() => _progress = sent / total));
      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => CourtSetupScreen(session: done)));
    } catch (e) {
      setState(() { _error = describeError(e); _progress = null; });
    } finally {
      await WakelockPlus.disable();
    }
  }

  Widget _chips(String label, String value, List<(String, String)> options, void Function(String) onPick) => Wrap(
        crossAxisAlignment: WrapCrossAlignment.center,
        spacing: 8,
        children: [
          Text(label),
          for (final (v, l) in options) ChoiceChip(label: Text(l), selected: value == v, onSelected: (_) => setState(() => onPick(v))),
        ],
      );

  @override
  Widget build(BuildContext context) {
    final uploading = _progress != null;
    return Scaffold(
      appBar: AppBar(title: Text(widget.resume == null ? 'New session' : 'Continue upload')),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        Row(children: [
          Expanded(child: TextField(controller: _nameA, decoration: const InputDecoration(labelText: 'Player A', border: OutlineInputBorder()))),
          const SizedBox(width: 12),
          Expanded(child: TextField(controller: _nameB, decoration: const InputDecoration(labelText: 'Player B', border: OutlineInputBorder()))),
        ]),
        const SizedBox(height: 12),
        _chips('At the start, Player A is on the', _seedASide, [('near', 'near side'), ('far', 'far side')], (v) => _seedASide = v),
        _chips('Player A is', _handedA, [('right', 'right-handed'), ('left', 'left-handed')], (v) => _handedA = v),
        _chips('Player B is', _handedB, [('right', 'right-handed'), ('left', 'left-handed')], (v) => _handedB = v),
        _chips('Analysis', _quality, [('standard', 'Standard'), ('best', 'Best (slower)')], (v) => _quality = v),
        const Divider(height: 32),
        Text('Video', style: Theme.of(context).textTheme.titleMedium),
        const SizedBox(height: 8),
        const Text('Phone behind a baseline, as high as possible, whole court visible, landscape. Do not move it while recording.'),
        const SizedBox(height: 12),
        Row(children: [
          Expanded(child: FilledButton.tonalIcon(onPressed: uploading ? null : _record, icon: const Icon(Icons.videocam), label: const Text('Record'))),
          const SizedBox(width: 12),
          Expanded(child: OutlinedButton.icon(onPressed: uploading ? null : _pick, icon: const Icon(Icons.video_library), label: const Text('Choose video'))),
        ]),
        if (_file != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(_file!.path.split('/').last)),
        const SizedBox(height: 24),
        if (uploading) ...[
          LinearProgressIndicator(value: _progress),
          const SizedBox(height: 8),
          Text('Uploading ${(_progress! * 100).toStringAsFixed(1)}% — keep the app open; an interrupted upload resumes where it stopped.'),
          TextButton(onPressed: () => _cancel?.cancel(), child: const Text('Cancel')),
        ] else
          FilledButton.icon(onPressed: _file == null ? null : _upload, icon: const Icon(Icons.cloud_upload), label: const Text('Upload')),
        if (_error != null) Padding(padding: const EdgeInsets.only(top: 12), child: Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error))),
      ]),
    );
  }
}
