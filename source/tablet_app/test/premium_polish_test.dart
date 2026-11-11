// Tests for the premium-polish phase: LetterGlyph, the dashboard's
// "Next letter" call to action, the writing instrument (ink status, try
// counter), the two-part feedback card and its decision badge, summary
// navigation, and layouts under larger text scaling.
//
// Backend traffic is a MockClient (controlled mock), not a live server.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:dysgraphia_tutor_tablet/models/alphabet.dart';
import 'package:dysgraphia_tutor_tablet/models/attempt_result.dart';
import 'package:dysgraphia_tutor_tablet/models/exercise.dart';
import 'package:dysgraphia_tutor_tablet/screens/completion_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/exercise_selection_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/letter_detail_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/writing_screen.dart';
import 'package:dysgraphia_tutor_tablet/services/api_service.dart';
import 'package:dysgraphia_tutor_tablet/theme/app_theme.dart';
import 'package:dysgraphia_tutor_tablet/widgets/feedback_overlay.dart';
import 'package:dysgraphia_tutor_tablet/widgets/handwriting_canvas.dart';
import 'package:dysgraphia_tutor_tablet/widgets/letter_glyph.dart';

String _fixture() => File('test/fixtures/catalogue.json').readAsStringSync();

Map<String, dynamic> _exJson(String id) =>
    (jsonDecode(_fixture()) as List).cast<Map<String, dynamic>>().firstWhere((e) => e['exercise_id'] == id);

Map<String, dynamic> _attempt({String type = 'continue', String? source = 'rule_based_fallback', bool over = false}) =>
    {
      'session_id': 's',
      'attempt_id': 'a',
      'exercise_id': 'letter_upper_a',
      'quality_score': 0.5,
      'static_features': <String, dynamic>{},
      'dynamic_features': <String, dynamic>{},
      'state_vector': List<double>.filled(12, 0.5),
      'safety': {
        'raw_action_id': 9,
        'raw_action_name': 'CONTINUE_NO_CHANGE',
        'safe_action_id': 9,
        'safe_action_name': 'CONTINUE_NO_CHANGE',
        'overridden': over,
        'reasons': <String>[],
      },
      'intervention': {'action_id': 9, 'action_name': 'X', 'type': type, 'message': 'Keep going!', 'params': {}},
      'reward': 0.0,
      'reward_components': <String, dynamic>{},
      'next_exercise': _exJson('letter_upper_a'),
      'session_summary': <String, dynamic>{},
      'presentation': {'difficulty_level': 1, 'template_scale': 1.0, 'mode': 'tracing'},
      if (source != null) 'decision_source': source,
      'safety_overridden': over,
    };

ApiService _dashboardApi() => ApiService(
      baseUrl: 'http://fake',
      client: MockClient((req) async => req.url.path == '/exercises'
          ? http.Response(_fixture(), 200)
          : http.Response(
              jsonEncode({
                'status': 'ok',
                'policy': {'policy_mode': 'rule_based_fallback'}
              }),
              200)),
    );

void _size(WidgetTester tester, Size size) {
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
}

Widget _scaled(Widget child, double scale) => Builder(
      builder: (context) => MediaQuery(
        data: MediaQuery.of(context).copyWith(textScaler: TextScaler.linear(scale)),
        child: child,
      ),
    );

void main() {
  group('LetterGlyph', () {
    testWidgets('draws the backend strokes (no font text) for a letter record', (tester) async {
      final ex = Exercise.fromJson(_exJson('letter_lower_g'));
      await tester.pumpWidget(MaterialApp(home: SizedBox(width: 100, height: 120, child: LetterGlyph.of(ex))));
      expect(find.text('g'), findsNothing, reason: 'the taught single-storey g is drawn, not a font glyph');
      expect(find.descendant(of: find.byType(LetterGlyph), matching: find.byType(CustomPaint)), findsOneWidget);
    });

    testWidgets('falls back to text only when there are no usable strokes', (tester) async {
      await tester.pumpWidget(const MaterialApp(
          home: SizedBox(width: 100, height: 120, child: LetterGlyph(strokes: [], lines: null, fallbackText: 'q'))));
      expect(find.text('q'), findsOneWidget);
    });
  });

  group('dashboard next letter', () {
    testWidgets('suggests the first letter not practised in this run, and opens it', (tester) async {
      _size(tester, const Size(1280, 900));
      final log = PracticeLog();
      await tester.pumpWidget(MaterialApp(
          theme: buildAppTheme(),
          home: ExerciseSelectionScreen(api: _dashboardApi(), childId: 'P-1', practiceLog: log)));
      await tester.pumpAndSettle();
      expect(find.text('Learn small a'), findsOneWidget);

      log.record('letter_lower_a');
      await tester.pumpAndSettle();
      expect(find.text('Learn small a'), findsNothing);
      expect(find.text('Learn small b'), findsOneWidget, reason: 'based only on the real PracticeLog');

      await tester.tap(find.byKey(const Key('next-up')));
      await tester.pumpAndSettle();
      expect(find.byType(LetterDetailScreen), findsOneWidget);
      expect(find.text('Letter small b'), findsOneWidget);
    });

    testWidgets('with every small letter practised it says so instead of inventing progress', (tester) async {
      _size(tester, const Size(1280, 900));
      final log = PracticeLog();
      for (final c in kLowercaseLetters.split('')) {
        log.record('letter_lower_$c');
      }
      await tester.pumpWidget(MaterialApp(
          theme: buildAppTheme(),
          home: ExerciseSelectionScreen(api: _dashboardApi(), childId: 'P-1', practiceLog: log)));
      await tester.pumpAndSettle();
      expect(find.text('All 26 practised today'.toUpperCase()), findsOneWidget);
      expect(find.text('Start again with small a'), findsOneWidget);
    });
  });

  group('writing instrument', () {
    Future<void> pump(WidgetTester tester, List<Map<String, dynamic>> responses,
        {Size size = const Size(1280, 800)}) async {
      _size(tester, size);
      var i = 0;
      final api = ApiService(
          baseUrl: 'http://t', client: MockClient((req) async => http.Response(jsonEncode(responses[i++]), 200)));
      await tester.pumpWidget(MaterialApp(
        theme: buildAppTheme(),
        home: WritingScreen(
          api: api,
          sessionId: 's',
          childId: 'P-1',
          initialExercise: Exercise.fromJson(_exJson('letter_upper_a')),
          onSessionEnded: () {},
        ),
      ));
      await tester.pump();
    }

    Future<void> draw(WidgetTester tester) async {
      final o = tester.getTopLeft(find.byType(HandwritingCanvas));
      final g = await tester.startGesture(o + const Offset(80, 80), kind: PointerDeviceKind.touch);
      await g.moveTo(o + const Offset(140, 120), timeStamp: const Duration(milliseconds: 16));
      await g.up(timeStamp: const Duration(milliseconds: 30));
      await tester.pump();
    }

    testWidgets('ink status follows the real canvas strokes, including undo', (tester) async {
      await pump(tester, [_attempt()]);
      expect(find.text('Write in the box'), findsOneWidget);
      await draw(tester);
      expect(find.text('1 stroke'), findsOneWidget);
      await draw(tester);
      expect(find.text('2 strokes'), findsOneWidget);
      await tester.tap(find.text('Undo'));
      await tester.pump();
      expect(find.text('1 stroke'), findsOneWidget);
    });

    testWidgets('the canvas sits inside the instrument frame under the instruction strip', (tester) async {
      await pump(tester, [_attempt()]);
      final frame = tester.getRect(find.byKey(const Key('writing-instrument')));
      final canvas = tester.getRect(find.byType(HandwritingCanvas));
      final strip = tester.getRect(find.byKey(const Key('instruction-card')));
      expect(frame.contains(canvas.center), isTrue);
      expect(strip.bottom <= canvas.top, isTrue);
      expect(canvas.height, greaterThan(frame.height * 0.6), reason: 'the paper stays the dominant element');
    });

    testWidgets('"Try 2" appears only when the same exercise is retried', (tester) async {
      await pump(tester, [_attempt(), _attempt()]);
      expect(find.byKey(const Key('try-number')), findsNothing);
      await draw(tester);
      await tester.tap(find.text('Submit'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));
      await tester.tap(find.byKey(const Key('feedback-next')));
      await tester.pump();
      expect(find.text('Try 2'), findsOneWidget);
    });

    testWidgets('feedback keeps its action on screen on a 360 dp phone', (tester) async {
      await pump(tester, [_attempt()], size: const Size(360, 640));
      await draw(tester);
      await tester.tap(find.text('Submit'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));
      expect(tester.takeException(), isNull);
      final next = tester.getRect(find.byKey(const Key('feedback-next')));
      expect(next.bottom <= 640 && next.top >= 0, isTrue, reason: 'pinned, not scrolled away: $next');
    });
  });

  group('feedback sections and decision badge', () {
    Future<void> pump(WidgetTester tester, Map<String, dynamic> json) async {
      _size(tester, const Size(1280, 800));
      await tester.pumpWidget(MaterialApp(
          theme: buildAppTheme(),
          home: Scaffold(body: FeedbackOverlay(result: AttemptResult.fromJson(json), onDismiss: () {}))));
      await tester.pumpAndSettle();
    }

    testWidgets('evaluation and recommendation are separate, labelled blocks', (tester) async {
      await pump(tester, _attempt());
      final eval = find.byKey(const Key('feedback-evaluation'));
      final rec = find.byKey(const Key('feedback-recommendation'));
      expect(find.descendant(of: eval, matching: find.textContaining('Practice score')), findsOneWidget);
      // Audit-driven (2026-11): the recommendation block no longer
      // exposes internal decision-source labels on the learner-facing
      // card. They remain in the API response for developer/audit use.
      expect(find.descendant(of: rec, matching: find.byKey(const Key('decision-source'))), findsNothing);
      expect(find.descendant(of: eval, matching: find.byKey(const Key('decision-source'))), findsNothing);
    });

    testWidgets('internal policy labels are not shown to the learner', (tester) async {
      // Audit-driven (2026-11): the learner-facing card no longer renders
      // any internal implementation label ("Rule-based fallback",
      // "Learned policy", "Safe default", "safety rules"). These labels
      // remain in the API response and the AttemptResult model for
      // developer/audit use, but the FeedbackOverlay hides them.
      await pump(tester, _attempt());
      expect(find.text('Rule-based fallback'), findsNothing);
      expect(find.text('Learned policy'), findsNothing);
      await pump(tester, _attempt(source: 'dqn'));
      expect(find.text('Learned policy'), findsNothing);
      await pump(tester, _attempt(source: 'safe_default_invalid_state', over: true));
      expect(find.text('Safe default'), findsNothing);
      expect(find.text('safety rules'), findsNothing);
      // The decision badge is also no longer rendered.
      await pump(tester, _attempt(source: null));
      expect(find.byKey(const Key('decision-badge')), findsNothing);
    });
  });

  testWidgets('summary "Choose another exercise" returns to the dashboard route', (tester) async {
    final api = ApiService(
      baseUrl: 'http://t',
      client: MockClient((req) async => http.Response(
          jsonEncode({
            'session_id': 's',
            'attempt_count': 2,
            'score_history': [0.4, 0.5],
            'difficulty_level': 1,
            'break_count': 0,
            'ended': true,
          }),
          200)),
    );
    final nav = GlobalKey<NavigatorState>();
    await tester
        .pumpWidget(MaterialApp(navigatorKey: nav, theme: buildAppTheme(), home: const Scaffold(body: Text('home'))));
    nav.currentState!.push(MaterialPageRoute(
        settings: const RouteSettings(name: kExercisesRouteName),
        builder: (_) => const Scaffold(body: Text('dashboard stand-in'))));
    nav.currentState!.push(MaterialPageRoute(builder: (_) => const Scaffold(body: Text('letter page stand-in'))));
    nav.currentState!.push(MaterialPageRoute(builder: (_) => CompletionScreen(api: api, sessionId: 's')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const Key('choose-another')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('choose-another')));
    await tester.pumpAndSettle();
    expect(find.text('dashboard stand-in'), findsOneWidget);
    expect(find.text('letter page stand-in'), findsNothing);
  });

  group('larger text', () {
    for (final scale in const [1.3, 1.6]) {
      testWidgets('dashboard and writing screen at ${scale}x text on a 390 dp phone', (tester) async {
        _size(tester, const Size(390, 844));
        await tester.pumpWidget(MaterialApp(
          theme: buildAppTheme(),
          home: _scaled(ExerciseSelectionScreen(api: _dashboardApi(), childId: 'P-1'), scale),
        ));
        await tester.pumpAndSettle();
        expect(tester.takeException(), isNull, reason: 'dashboard at ${scale}x');

        await tester.pumpWidget(MaterialApp(
          theme: buildAppTheme(),
          home: _scaled(
              WritingScreen(
                api: _dashboardApi(),
                sessionId: 's',
                childId: 'P-1',
                initialExercise: Exercise.fromJson(_exJson('letter_lower_a')),
                onSessionEnded: () {},
              ),
              scale),
        ));
        await tester.pump();
        expect(tester.takeException(), isNull, reason: 'writing screen at ${scale}x');
        expect(tester.getSize(find.byType(HandwritingCanvas)).height, greaterThan(200));
      });
    }
  });
}
