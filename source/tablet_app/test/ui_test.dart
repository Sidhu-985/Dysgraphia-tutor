// Widget and unit tests for the Phase B UI and capture changes.
//
// These tests have NOT been run in the environment where they were written.
// No Flutter SDK was available there: storage.googleapis.com and pub.dev were
// blocked by the network policy. Run them with:
//   flutter pub get && flutter analyze && flutter test

import 'dart:convert';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:dysgraphia_tutor_tablet/main.dart';
import 'package:dysgraphia_tutor_tablet/models/attempt_result.dart';
import 'package:dysgraphia_tutor_tablet/models/session_summary.dart';
import 'package:dysgraphia_tutor_tablet/models/stroke_point.dart';
import 'package:dysgraphia_tutor_tablet/screens/break_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/progress_screen.dart';
import 'package:dysgraphia_tutor_tablet/services/api_service.dart';
import 'package:dysgraphia_tutor_tablet/theme/app_theme.dart';
import 'package:dysgraphia_tutor_tablet/widgets/feedback_overlay.dart';
import 'package:dysgraphia_tutor_tablet/widgets/handwriting_canvas.dart';
import 'package:dysgraphia_tutor_tablet/widgets/score_chart.dart';
import 'package:dysgraphia_tutor_tablet/widgets/template_geometry.dart';

Widget _host(Widget child) => MaterialApp(theme: buildAppTheme(), home: Scaffold(body: child));

Map<String, dynamic> _attemptJson({double quality = 0.82, String type = 'continue'}) => {
      'session_id': 's',
      'attempt_id': 'a',
      'exercise_id': 'line_easy',
      'quality_score': quality,
      'static_features': <String, dynamic>{},
      'dynamic_features': <String, dynamic>{},
      'state_vector': List<double>.filled(12, 0.5),
      'safety': {
        'raw_action_id': 9,
        'raw_action_name': 'CONTINUE_NO_CHANGE',
        'safe_action_id': 9,
        'safe_action_name': 'CONTINUE_NO_CHANGE',
        'overridden': false,
        'reasons': <String>[],
      },
      'intervention': {
        'action_id': 9,
        'action_name': 'CONTINUE_NO_CHANGE',
        'type': type,
        'message': 'Keep going!',
        'params': <String, dynamic>{},
      },
      'reward': 1.0,
      'reward_components': <String, dynamic>{},
      'next_exercise': {
        'exercise_id': 'line_easy',
        'prompt': 'Trace the straight line',
        'difficulty': 0,
        'reference_template': [
          [0.0, 50.0],
          [100.0, 50.0]
        ],
      },
      'session_summary': <String, dynamic>{},
      'presentation': {'difficulty_level': 0, 'template_scale': 1.25, 'mode': 'free_writing'},
    };

void main() {
  group('HandwritingCanvas capture', () {
    testWidgets('uses device event timestamps and null pressure for touch', (tester) async {
      List<List<StrokePoint>> captured = [];
      await tester.pumpWidget(_host(SizedBox(
        width: 400,
        height: 300,
        child: HandwritingCanvas(onStrokesChanged: (s) => captured = s),
      )));
      final origin = tester.getTopLeft(find.byType(HandwritingCanvas));
      final g = await tester.startGesture(origin + const Offset(10, 10), kind: PointerDeviceKind.touch);
      await g.moveTo(origin + const Offset(20, 12), timeStamp: const Duration(milliseconds: 16));
      await g.moveTo(origin + const Offset(30, 14), timeStamp: const Duration(milliseconds: 32));
      await g.up(timeStamp: const Duration(milliseconds: 40));
      await tester.pump();

      expect(captured.length, 1);
      final pts = captured.first;
      expect(pts.length, 3);
      expect(pts.map((p) => p.t).toList(), [0.0, 0.016, 0.032]);
      expect(pts.every((p) => p.pressure == null), isTrue);
      expect(pts.first.isStrokeStart, isTrue);
      expect(pts.last.isStrokeEnd, isTrue);
      expect(pts.first.x, closeTo(10, 1e-6));
    });

    testWidgets('a tap is kept as a one-point stroke', (tester) async {
      List<List<StrokePoint>> captured = [];
      await tester.pumpWidget(_host(SizedBox(
        width: 400,
        height: 300,
        child: HandwritingCanvas(onStrokesChanged: (s) => captured = s),
      )));
      final g = await tester.startGesture(tester.getCenter(find.byType(HandwritingCanvas)));
      await g.up();
      await tester.pump();
      expect(captured.length, 1);
      expect(captured.first.length, 1);
    });

    testWidgets('a second finger does not draw while the first is down', (tester) async {
      List<List<StrokePoint>> captured = [];
      await tester.pumpWidget(_host(SizedBox(
        width: 400,
        height: 300,
        child: HandwritingCanvas(onStrokesChanged: (s) => captured = s),
      )));
      final o = tester.getTopLeft(find.byType(HandwritingCanvas));
      final a = await tester.startGesture(o + const Offset(10, 10), pointer: 1);
      final b = await tester.startGesture(o + const Offset(200, 200), pointer: 2);
      await a.moveTo(o + const Offset(40, 10), timeStamp: const Duration(milliseconds: 10));
      await b.moveTo(o + const Offset(220, 220), timeStamp: const Duration(milliseconds: 10));
      await b.up();
      await a.up();
      await tester.pump();
      expect(captured.length, 1);
      expect(captured.first.every((p) => p.x < 100), isTrue);
    });

    test('pressure is normalised only when the device reports a range', () {
      const withRange = PointerDownEvent(pressure: 0.6, pressureMin: 0.0, pressureMax: 1.0);
      const noRange = PointerDownEvent(pressure: 1.0, pressureMin: 1.0, pressureMax: 1.0);
      const wide = PointerDownEvent(pressure: 2.0, pressureMin: 0.0, pressureMax: 4.0);
      expect(HandwritingCanvasState.pressureOf(withRange), closeTo(0.6, 1e-9));
      expect(HandwritingCanvasState.pressureOf(noRange), isNull);
      expect(HandwritingCanvasState.pressureOf(wide), closeTo(0.5, 1e-9));
    });
  });

  group('TemplateFit', () {
    final line = [
      [0.0, 50.0],
      [100.0, 50.0]
    ];

    test('centres the template and scales it with the canvas', () {
      final fit = TemplateFit.compute(line, const Size(800, 600));
      final a = fit.map(line.first), b = fit.map(line.last);
      expect(b.dx - a.dx, closeTo(600 * TemplateFit.baseFraction, 1e-6));
      expect((a.dx + b.dx) / 2, closeTo(400, 1e-6));
      expect(a.dy, closeTo(300, 1e-6));
    });

    test('template_scale enlarges the guide but never past the canvas', () {
      final big = TemplateFit.compute(line, const Size(800, 600), templateScale: 2.0);
      final huge = TemplateFit.compute(line, const Size(800, 600), templateScale: 5.0);
      expect(big.map(line.last).dx - big.map(line.first).dx, closeTo(660, 1e-6));
      expect(huge.map(line.last).dx - huge.map(line.first).dx, closeTo(800 * TemplateFit.maxFraction, 1e-6));
    });
  });

  testWidgets('feedback labels the score as a practice estimate', (tester) async {
    final result = AttemptResult.fromJson(_attemptJson());
    await tester.pumpWidget(_host(FeedbackOverlay(result: result, onDismiss: () {})));
    expect(find.text('Practice score (shape match): 82%'), findsOneWidget);
    expect(find.textContaining('not a test result'), findsOneWidget);
    expect(find.textContaining('Quality score'), findsNothing);
  });

  test('attempt result exposes presentation from the backend', () {
    final r = AttemptResult.fromJson(_attemptJson());
    expect(r.templateScale, 1.25);
    expect(r.mode, 'free_writing');
  });

  group('ProgressBody shows only recorded data', () {
    SessionSummary summary(List<double> scores) => SessionSummary(
        sessionId: 's',
        attemptCount: scores.length,
        scoreHistory: scores,
        difficultyLevel: 0,
        breakCount: 1,
        ended: false);

    testWidgets('no attempts', (tester) async {
      await tester.pumpWidget(_host(ProgressBody(summary: summary([]))));
      expect(find.textContaining('No attempts yet'), findsOneWidget);
      expect(find.byType(ScoreChart), findsNothing);
    });

    testWidgets('one attempt has no trend', (tester) async {
      await tester.pumpWidget(_host(ProgressBody(summary: summary([0.4]))));
      expect(find.textContaining('One attempt so far'), findsOneWidget);
      expect(find.byType(ScoreChart), findsNothing);
    });

    testWidgets('several attempts are charted', (tester) async {
      await tester.pumpWidget(_host(ProgressBody(summary: summary([0.4, 0.5, 0.45]))));
      expect(find.byType(ScoreChart), findsOneWidget);
      expect(find.text('3'), findsOneWidget); // attempts stat
    });
  });

  group('BreakScreen', () {
    Future<bool?> openBreak(WidgetTester tester, Future<void> Function() act) async {
      bool? result;
      await tester.pumpWidget(MaterialApp(
        theme: buildAppTheme(),
        home: Builder(
          builder: (ctx) => ElevatedButton(
            onPressed: () async {
              result = await Navigator.of(ctx)
                  .push<bool>(MaterialPageRoute(builder: (_) => const BreakScreen(durationSeconds: 3)));
            },
            child: const Text('open'),
          ),
        ),
      ));
      await tester.tap(find.text('open'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));
      await act();
      return result;
    }

    testWidgets('skip returns false (no break_complete should be sent)', (tester) async {
      final r = await openBreak(tester, () async {
        await tester.tap(find.text('Skip break'));
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 400));
      });
      expect(r, isFalse);
    });

    testWidgets("I'm ready is disabled until the countdown ends, then returns true", (tester) async {
      final r = await openBreak(tester, () async {
        final ready = find.widgetWithText(ElevatedButton, "I'm ready");
        expect(tester.widget<ElevatedButton>(ready).onPressed, isNull);
        await tester.pump(const Duration(seconds: 4));
        expect(tester.widget<ElevatedButton>(ready).onPressed, isNotNull);
        await tester.tap(ready);
        await tester.pump();
        await tester.pump(const Duration(milliseconds: 400));
      });
      expect(r, isTrue);
    });
  });

  testWidgets('home rejects a real-looking name as the learner code', (tester) async {
    await tester.pumpWidget(const DysgraphiaTutorApp());
    await tester.enterText(find.widgetWithText(TextField, 'Child ID'), 'Jane Doe');
    await tester.tap(find.text('Start Session'));
    await tester.pump();
    expect(find.textContaining('not a name'), findsWidgets);
    expect(find.text('Choose an exercise'), findsNothing);
  });

  group('ApiService wire format', () {
    test('submitAttempt sends null pressure and capture metadata', () async {
      late Map<String, dynamic> sent;
      final api = ApiService(
        baseUrl: 'http://test',
        client: MockClient((req) async {
          sent = jsonDecode(req.body) as Map<String, dynamic>;
          return http.Response(jsonEncode(_attemptJson()), 200);
        }),
      );
      await api.submitAttempt(
        sessionId: 's',
        exerciseId: 'line_easy',
        attemptId: 'a',
        points: const [StrokePoint(x: 1, y: 2, t: 0.0, strokeId: 0)],
        capture: {'pointer_kind': 'touch', 'pressure_supported': false},
      );
      final p = (sent['points'] as List).first as Map<String, dynamic>;
      expect(p.containsKey('pressure'), isTrue);
      expect(p['pressure'], isNull);
      expect(sent['capture']['pointer_kind'], 'touch');
    });

    test('breakComplete posts to the session endpoint', () async {
      Uri? hit;
      final api = ApiService(
        baseUrl: 'http://test',
        client: MockClient((req) async {
          hit = req.url;
          return http.Response('{}', 200);
        }),
      );
      await api.breakComplete('abc');
      expect(hit.toString(), 'http://test/session/abc/break_complete');
    });

    test('server errors become friendly messages', () {
      final e = ApiException(422, jsonEncode({'detail': 'child_id must be a pseudonymous code'}));
      expect(describeError(e), contains('pseudonymous'));
      expect(describeError(Exception('socket')), contains('Could not reach'));
    });
  });
}
