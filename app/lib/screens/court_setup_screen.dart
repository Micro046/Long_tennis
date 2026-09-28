import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import '../api/client.dart';
import '../models/session.dart';
import 'processing_screen.dart';

const _markerLabels = {
  'far_left_singles': 'Far baseline × left singles line',
  'far_right_singles': 'Far baseline × right singles line',
  'near_service_left': 'Near service line × left singles line',
  'near_service_right': 'Near service line × right singles line',
};

/// Default positions for a phone behind the near baseline (normalised image coordinates).
const _defaults = {
  'far_left_singles': Offset(0.37, 0.30),
  'far_right_singles': Offset(0.62, 0.30),
  'near_service_left': Offset(0.19, 0.49),
  'near_service_right': Offset(0.80, 0.49),
};

class CourtSetupScreen extends StatefulWidget {
  const CourtSetupScreen({super.key, required this.session});
  final Session session;

  @override
  State<CourtSetupScreen> createState() => _CourtSetupScreenState();
}

class _CourtSetupScreenState extends State<CourtSetupScreen> {
  ui.Image? _image;
  Uint8List? _bytes;
  final Map<String, Offset> _markers = Map.of(_defaults);
  String? _dragging;
  CalibrationResult? _result;
  String? _message;
  bool _busy = false;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    try {
      final b = await ApiClient.instance.background(widget.session.id);
      final img = await decodeImageFromList(b);
      setState(() { _bytes = b; _image = img; _message = 'Drag the four circles onto the line crossings, then tap Check court.'; });
    } catch (e) {
      setState(() => _message = describeError(e));
    }
  }

  Future<void> _check() async {
    setState(() => _busy = true);
    try {
      final r = await ApiClient.instance.calibrate(
          widget.session.id, _markers.map((k, v) => MapEntry(k, [v.dx, v.dy])));
      _result = r;
      _message = r.ok
          ? 'Court found (fit error ${r.rmsPx?.toStringAsFixed(1)} px). Check that the green lines sit on the court, then start the analysis.'
          : 'The court lines were not found reliably. Move the circles closer to the crossings and check again.';
    } catch (e) {
      _message = describeError(e);
    }
    setState(() => _busy = false);
  }

  Future<void> _analyze() async {
    setState(() => _busy = true);
    try {
      await ApiClient.instance.analyze(widget.session.id);
      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => ProcessingScreen(sessionId: widget.session.id)));
    } catch (e) {
      setState(() { _message = describeError(e); _busy = false; });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Court setup')),
      body: ListView(padding: const EdgeInsets.all(12), children: [
        if (_image != null)
          AspectRatio(
            aspectRatio: _image!.width / _image!.height,
            child: LayoutBuilder(builder: (context, box) {
              final size = Size(box.maxWidth, box.maxHeight);
              Offset toPx(Offset n) => Offset(n.dx * size.width, n.dy * size.height);
              return GestureDetector(
                onPanStart: (d) {
                  String? best; var bestD = 60.0;
                  _markers.forEach((k, v) {
                    final dd = (toPx(v) - d.localPosition).distance;
                    if (dd < bestD) { bestD = dd; best = k; }
                  });
                  setState(() => _dragging = best);
                },
                onPanUpdate: (d) {
                  final k = _dragging;
                  if (k == null) return;
                  final p = _markers[k]! + Offset(d.delta.dx / size.width, d.delta.dy / size.height);
                  setState(() {
                    _markers[k] = Offset(p.dx.clamp(0.0, 1.0), p.dy.clamp(0.0, 1.0));
                    _result = null;
                  });
                },
                onPanEnd: (_) => setState(() => _dragging = null),
                child: Stack(fit: StackFit.expand, children: [
                  Image.memory(_bytes!, fit: BoxFit.fill, gaplessPlayback: true),
                  CustomPaint(painter: _OverlayPainter(_markers, _result?.lines, _dragging)),
                ]),
              );
            }),
          )
        else
          const SizedBox(height: 200, child: Center(child: CircularProgressIndicator())),
        const SizedBox(height: 8),
        if (_dragging != null) Text(_markerLabels[_dragging]!, style: Theme.of(context).textTheme.titleSmall),
        if (_message != null) Text(_message!),
        const SizedBox(height: 12),
        Row(children: [
          Expanded(child: OutlinedButton(onPressed: _busy || _image == null ? null : _check, child: const Text('Check court'))),
          const SizedBox(width: 12),
          Expanded(child: FilledButton(onPressed: _busy || _result?.ok != true ? null : _analyze, child: const Text('Start analysis'))),
        ]),
        const SizedBox(height: 16),
        Text('The image is the court with players removed (average of the whole video).',
            style: Theme.of(context).textTheme.bodySmall),
      ]),
    );
  }
}

class _OverlayPainter extends CustomPainter {
  _OverlayPainter(this.markers, this.lines, this.active);
  final Map<String, Offset> markers;
  final List<List<double>>? lines;
  final String? active;

  @override
  void paint(Canvas canvas, Size size) {
    if (lines != null) {
      final p = Paint()..color = const Color(0xFF39FF7A)..strokeWidth = 2;
      for (final l in lines!) {
        canvas.drawLine(Offset(l[0] * size.width, l[1] * size.height), Offset(l[2] * size.width, l[3] * size.height), p);
      }
    }
    markers.forEach((k, v) {
      final c = Offset(v.dx * size.width, v.dy * size.height);
      canvas.drawCircle(c, 18, Paint()..color = Colors.black45);
      canvas.drawCircle(c, 15, Paint()..color = k == active ? Colors.yellow : const Color(0xFFD7F036)..style = PaintingStyle.stroke..strokeWidth = 3);
      canvas.drawCircle(c, 2.5, Paint()..color = Colors.white);
    });
  }

  @override
  bool shouldRepaint(covariant _OverlayPainter old) => true;
}
