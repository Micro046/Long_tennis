class Session {
  Session({
    required this.id,
    required this.createdAt,
    required this.nameA,
    required this.nameB,
    required this.seedASide,
    required this.handedA,
    required this.handedB,
    required this.quality,
    required this.status,
    this.progress,
    this.video,
    this.error,
  });

  final String id;
  final DateTime createdAt;
  final String nameA, nameB, seedASide, handedA, handedB, quality, status;
  final Map<String, dynamic>? progress;
  final Map<String, dynamic>? video;
  final String? error;

  double get fraction => ((progress?['fraction'] as num?) ?? 0).toDouble();
  int? get etaSeconds => (progress?['etaSeconds'] as num?)?.toInt();
  String get stage => (progress?['stage'] as String?) ?? status;
  double? get durationS => (video?['durationS'] as num?)?.toDouble();
  bool get isBusy => status == 'queued' || status == 'processing';

  factory Session.fromJson(Map<String, dynamic> j) => Session(
        id: j['id'] as String,
        createdAt: DateTime.tryParse(j['createdAt'] as String? ?? '') ?? DateTime.now(),
        nameA: j['nameA'] as String? ?? 'Player A',
        nameB: j['nameB'] as String? ?? 'Player B',
        seedASide: j['seedASide'] as String? ?? 'near',
        handedA: j['handedA'] as String? ?? 'right',
        handedB: j['handedB'] as String? ?? 'right',
        quality: j['quality'] as String? ?? 'standard',
        status: j['status'] as String? ?? 'created',
        progress: j['progress'] as Map<String, dynamic>?,
        video: j['video'] as Map<String, dynamic>?,
        error: j['error'] as String?,
      );
}

String statusLabel(String s) => switch (s) {
      'created' => 'Waiting for video',
      'uploading' => 'Upload incomplete',
      'uploaded' => 'Court setup needed',
      'calibrated' => 'Ready to analyse',
      'queued' => 'Queued',
      'processing' => 'Analysing',
      'done' => 'Report ready',
      'failed' => 'Analysis failed',
      _ => s,
    };

String formatDuration(num seconds) {
  final s = seconds.round();
  final h = s ~/ 3600, m = (s % 3600) ~/ 60, r = s % 60;
  return h > 0 ? '$h:${m.toString().padLeft(2, '0')}:${r.toString().padLeft(2, '0')}' : '$m:${r.toString().padLeft(2, '0')}';
}
