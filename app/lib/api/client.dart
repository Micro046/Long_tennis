import 'dart:io';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../models/session.dart';

/// Talks to the Long Tennis server (docs/API.md). One instance per app.
class ApiClient {
  ApiClient._(this._prefs) {
    _dio = Dio(BaseOptions(connectTimeout: const Duration(seconds: 8), receiveTimeout: const Duration(minutes: 5)));
  }

  static late ApiClient instance;
  static Future<void> init() async => instance = ApiClient._(await SharedPreferences.getInstance());

  final SharedPreferences _prefs;
  late final Dio _dio;

  String get baseUrl => _prefs.getString('baseUrl') ?? 'http://192.168.18.146:8770';
  String get token => _prefs.getString('token') ?? 'dev-token';
  Future<void> configure(String url, String tok) async {
    await _prefs.setString('baseUrl', url.trim().replaceAll(RegExp(r'/+$'), ''));
    await _prefs.setString('token', tok.trim());
  }

  String _u(String path) => '$baseUrl/api/v1$path';
  Options get _auth => Options(headers: {'Authorization': 'Bearer $token'});
  Map<String, String> get authHeaders => {'Authorization': 'Bearer $token'};

  Future<Map<String, dynamic>> health() async => (await _dio.get(_u('/health'))).data as Map<String, dynamic>;

  Future<List<Session>> sessions() async {
    final r = await _dio.get(_u('/sessions'), options: _auth);
    return (r.data as List).map((e) => Session.fromJson(e as Map<String, dynamic>)).toList();
  }

  Future<Session> create(Map<String, dynamic> body) async =>
      Session.fromJson((await _dio.post(_u('/sessions'), data: body, options: _auth)).data);

  Future<Session> get(String id) async => Session.fromJson((await _dio.get(_u('/sessions/$id'), options: _auth)).data);

  Future<Session> patch(String id, Map<String, dynamic> body) async =>
      Session.fromJson((await _dio.patch(_u('/sessions/$id'), data: body, options: _auth)).data);

  Future<void> delete(String id) async => _dio.delete(_u('/sessions/$id'), options: _auth);

  /// Resumable chunked upload. Resumes from the server's byte count; retries each chunk.
  Future<Session> upload(String id, File file,
      {void Function(int sent, int total)? onProgress, CancelToken? cancel, int chunkSize = 8 * 1024 * 1024}) async {
    final total = await file.length();
    var sent = ((await _dio.get(_u('/sessions/$id/upload'), options: _auth)).data['received'] as int?) ?? 0;
    final raf = await file.open();
    try {
      while (sent < total) {
        await raf.setPosition(sent);
        final Uint8List bytes = await raf.read(chunkSize < total - sent ? chunkSize : total - sent);
        var attempt = 0;
        while (true) {
          try {
            final r = await _dio.put(_u('/sessions/$id/upload'),
                queryParameters: {'offset': sent},
                data: Stream.fromIterable([bytes]),
                cancelToken: cancel,
                options: Options(headers: {
                  ...authHeaders,
                  Headers.contentLengthHeader: bytes.length,
                  Headers.contentTypeHeader: 'application/octet-stream',
                }, validateStatus: (s) => s == 200 || s == 409));
            sent = r.data['received'] as int;
            break;
          } on DioException catch (e) {
            if (CancelToken.isCancel(e) || ++attempt > 5) rethrow;
            await Future.delayed(Duration(seconds: 2 * attempt));
            sent = ((await _dio.get(_u('/sessions/$id/upload'), options: _auth)).data['received'] as int?) ?? sent;
            await raf.setPosition(sent);
          }
        }
        onProgress?.call(sent, total);
      }
    } finally {
      await raf.close();
    }
    final r = await _dio.post(_u('/sessions/$id/upload/complete'),
        data: {'size': total, 'filename': file.uri.pathSegments.last}, options: _auth);
    return Session.fromJson(r.data);
  }

  /// Median background image (players removed) used for court setup.
  Future<Uint8List> background(String id) async {
    final r = await _dio.get(_u('/sessions/$id/frame'),
        queryParameters: {'maxWidth': 1280}, options: _auth.copyWith(responseType: ResponseType.bytes));
    return Uint8List.fromList(r.data as List<int>);
  }

  Future<CalibrationResult> calibrate(String id, Map<String, List<double>> markers) async {
    final r = await _dio.post(_u('/sessions/$id/calibration'), data: {'markers': markers}, options: _auth);
    return CalibrationResult.fromJson(r.data as Map<String, dynamic>);
  }

  Future<Session> analyze(String id) async => Session.fromJson((await _dio.post(_u('/sessions/$id/analyze'), options: _auth)).data);

  Future<Map<String, dynamic>> report(String id) async =>
      (await _dio.get(_u('/sessions/$id/report'), options: _auth)).data as Map<String, dynamic>;

  String videoUrl(String id) => _u('/sessions/$id/video');

  Future<Map<String, dynamic>> corrections(String id) async =>
      (await _dio.get(_u('/sessions/$id/corrections'), options: _auth)).data as Map<String, dynamic>;

  /// Replaces the stored corrections; returns the corrected report.
  Future<Map<String, dynamic>> saveCorrections(String id, Map<String, dynamic> body) async =>
      (await _dio.put(_u('/sessions/$id/corrections'), data: body, options: _auth)).data as Map<String, dynamic>;
}

class CalibrationResult {
  CalibrationResult(this.ok, this.rmsPx, this.lineSupport, this.lines);
  final bool ok;
  final double? rmsPx;
  final double? lineSupport;
  final List<List<double>> lines; // normalised x1,y1,x2,y2

  factory CalibrationResult.fromJson(Map<String, dynamic> j) => CalibrationResult(
        j['ok'] == true,
        (j['rmsPx'] as num?)?.toDouble(),
        (j['lineSupport'] as num?)?.toDouble(),
        (j['lines'] as List).map((l) => (l as List).map((v) => (v as num).toDouble()).toList()).toList(),
      );
}

String describeError(Object e) {
  if (e is DioException) {
    if (e.type == DioExceptionType.connectionTimeout || e.type == DioExceptionType.connectionError) {
      return 'Cannot reach the server. Check the address in Settings and that the phone is on the same network.';
    }
    final d = e.response?.data;
    if (d is Map && d['detail'] != null) return d['detail'].toString();
    if (e.response?.statusCode == 401) return 'Server rejected the access token (Settings).';
    return e.message ?? e.toString();
  }
  return e.toString();
}
