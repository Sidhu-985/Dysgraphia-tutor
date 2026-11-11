import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;

import '../models/exercise.dart';
import '../models/attempt_result.dart';
import '../models/session_summary.dart';
import '../models/stroke_point.dart';

/// The ONLY place in the app that talks to the backend. Every ML/RL/
/// feature-extraction decision happens server-side; this class just
/// serializes requests and deserializes typed responses.
class ApiService {
  final String baseUrl;
  final http.Client _client;
  final Duration timeout;

  ApiService({required this.baseUrl, http.Client? client, this.timeout = const Duration(seconds: 15)})
      : _client = client ?? http.Client();

  Uri _u(String path) => Uri.parse('$baseUrl$path');
  static const _json = {'Content-Type': 'application/json'};

  Future<http.Response> _get(String path) => _client.get(_u(path)).timeout(timeout);
  Future<http.Response> _post(String path, [Object? body]) =>
      _client.post(_u(path), headers: _json, body: body == null ? null : jsonEncode(body)).timeout(timeout);

  Future<BackendHealth> health() async {
    final resp = await _get('/health');
    _checkOk(resp);
    return BackendHealth.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  Future<List<Exercise>> listExercises() async {
    final resp = await _get('/exercises');
    _checkOk(resp);
    return (jsonDecode(resp.body) as List<dynamic>).map((e) => Exercise.fromJson(e as Map<String, dynamic>)).toList();
  }

  /// `dataSource` is "demo" for app use. "real_child_study" is refused by
  /// the backend unless an ethics approval has been configured there.
  /// `practiceMode` ("tracing" or "free_writing") sets the session's starting
  /// mode; omitted, the backend starts in tracing mode as before.
  Future<Map<String, dynamic>> startSession(String childId,
      {String? exerciseId, String dataSource = 'demo', String? practiceMode}) async {
    final resp = await _post('/session/start', {
      'child_id': childId,
      if (exerciseId != null) 'exercise_id': exerciseId,
      'data_source': dataSource,
      if (practiceMode != null) 'practice_mode': practiceMode,
    });
    _checkOk(resp);
    return jsonDecode(resp.body) as Map<String, dynamic>;
  }

  Future<void> endSession(String sessionId) async {
    final resp = await _post('/session/$sessionId/end');
    _checkOk(resp);
  }

  Future<SessionSummary> sessionSummary(String sessionId) async {
    final resp = await _get('/session/$sessionId/summary');
    _checkOk(resp);
    return SessionSummary.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  /// Call ONLY when the child actually took the break (not when skipped):
  /// the backend applies fatigue recovery for it.
  Future<void> breakComplete(String sessionId) async {
    final resp = await _post('/session/$sessionId/break_complete');
    _checkOk(resp);
  }

  Future<Exercise> getExercise(String exerciseId) async {
    final resp = await _get('/exercise/$exerciseId');
    _checkOk(resp);
    return Exercise.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  Future<AttemptResult> submitAttempt({
    required String sessionId,
    required String exerciseId,
    required String attemptId,
    required List<StrokePoint> points,
    Map<String, dynamic>? capture,
  }) async {
    final resp = await _post('/attempt/submit', {
      'session_id': sessionId,
      'exercise_id': exerciseId,
      'attempt_id': attemptId,
      'points': points.map((p) => p.toJson()).toList(),
      if (capture != null) 'capture': capture,
    });
    _checkOk(resp);
    return AttemptResult.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
  }

  void _checkOk(http.Response resp) {
    if (resp.statusCode < 200 || resp.statusCode >= 300) {
      throw ApiException(resp.statusCode, resp.body);
    }
  }
}

class ApiException implements Exception {
  final int statusCode;
  final String body;
  ApiException(this.statusCode, this.body);

  /// Short, non-technical text that can be shown to an adult supervisor.
  String get friendlyMessage {
    String? detail;
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map && decoded['detail'] is String) detail = decoded['detail'] as String;
    } catch (_) {}
    switch (statusCode) {
      case 403:
        return detail ?? 'This action is not allowed on this server.';
      case 404:
        return 'The session was not found. Please start a new session.';
      case 409:
        return 'This session has already ended. Please start a new one.';
      case 422:
        return detail ?? 'That attempt could not be read. Please try again.';
      default:
        return 'The tutor server had a problem (code $statusCode). Please try again.';
    }
  }

  @override
  String toString() => 'ApiException($statusCode): $body';
}

/// Converts any error from [ApiService] into text for the supervisor.
String describeError(Object error) {
  if (error is ApiException) return error.friendlyMessage;
  if (error is TimeoutException) return 'The tutor server did not answer in time. Check the connection.';
  return 'Could not reach the tutor server. Check that it is running and on the same network.';
}
