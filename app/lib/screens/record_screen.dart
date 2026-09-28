import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:wakelock_plus/wakelock_plus.dart';

import '../models/session.dart';

/// Full-screen landscape recorder (1080p, no audio). Returns the recorded file.
class RecordScreen extends StatefulWidget {
  const RecordScreen({super.key});

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  CameraController? _cam;
  String? _error;
  bool _recording = false;
  DateTime? _started;
  Timer? _timer;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([DeviceOrientation.landscapeLeft, DeviceOrientation.landscapeRight]);
    WakelockPlus.enable();
    _init();
  }

  Future<void> _init() async {
    try {
      final cams = await availableCameras();
      final back = cams.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cams.first);
      final c = CameraController(back, ResolutionPreset.veryHigh, enableAudio: false);
      await c.initialize();
      await c.lockCaptureOrientation(DeviceOrientation.landscapeLeft);
      setState(() => _cam = c);
    } catch (e) {
      setState(() => _error = 'Camera unavailable: $e');
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _cam?.dispose();
    WakelockPlus.disable();
    SystemChrome.setPreferredOrientations([]);
    super.dispose();
  }

  Future<void> _toggle() async {
    final c = _cam!;
    if (!_recording) {
      await c.startVideoRecording();
      _started = DateTime.now();
      _timer = Timer.periodic(const Duration(seconds: 1), (_) => setState(() {}));
      setState(() => _recording = true);
    } else {
      final x = await c.stopVideoRecording();
      _timer?.cancel();
      if (mounted) Navigator.pop(context, File(x.path));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: _error != null
          ? Center(child: Text(_error!, style: const TextStyle(color: Colors.white)))
          : _cam == null
              ? const Center(child: CircularProgressIndicator())
              : Stack(children: [
                  Center(child: CameraPreview(_cam!)),
                  if (!_recording)
                    const Positioned(
                        left: 16, top: 16,
                        child: Text('Fit the whole court in the frame, then start.',
                            style: TextStyle(color: Colors.white, shadows: [Shadow(blurRadius: 4)]))),
                  if (_recording)
                    Positioned(
                        left: 16, top: 16,
                        child: Text('● ${formatDuration(DateTime.now().difference(_started!).inSeconds)}',
                            style: const TextStyle(color: Colors.redAccent, fontSize: 22, fontWeight: FontWeight.bold))),
                  Positioned(
                    right: 24, top: 0, bottom: 0,
                    child: Center(
                      child: FloatingActionButton.large(
                        backgroundColor: _recording ? Colors.red : Colors.white,
                        onPressed: _toggle,
                        child: Icon(_recording ? Icons.stop : Icons.fiber_manual_record, color: _recording ? Colors.white : Colors.red),
                      ),
                    ),
                  ),
                ]),
    );
  }
}
