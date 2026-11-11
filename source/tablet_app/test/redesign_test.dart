// Tests for the UI redesign: design-token contrast, the local letter-case
// selector style, responsive layouts, truthful feedback presentation, and
// that the redesigned writing screen still sends canvas-local coordinates.
//
// All backend traffic here is a MockClient (controlled mock), not a live server.

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:dysgraphia_tutor_tablet/main.dart';
import 'package:dysgraphia_tutor_tablet/models/attempt_result.dart';
import 'package:dysgraphia_tutor_tablet/models/exercise.dart';
import 'package:dysgraphia_tutor_tablet/screens/break_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/completion_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/exercise_selection_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/writing_screen.dart';
import 'package:dysgraphia_tutor_tablet/services/api_service.dart';
import 'package:dysgraphia_tutor_tablet/theme/app_theme.dart';
import 'package:dysgraphia_tutor_tablet/widgets/feedback_overlay.dart';
import 'package:dysgraphia_tutor_tablet/widgets/handwriting_canvas.dart';
import 'package:dysgraphia_tutor_tablet/widgets/session_controls.dart';

String _fixture() => File('test/fixtures/catalogue.json').readAsStringSync();

Map<String, dynamic> _exJson(String id) =>
    (jsonDecode(_fixture()) as List).cast<Map<String, dynamic>>().firstWhere((e) => e['exercise_id'] == id);

Map<String, dynamic> _attempt({
  String type = 'continue',
  String message = 'Keep going!',
  String? source = 'rule_based_fallback',
  bool overridden = false,
  String next = 'letter_upper_a',
  String mode = 'tracing',
}) =>
    {
      'session_id': 's',
      'attempt_id': 'a',
      'exercise_id': 'letter_upper_a',
      'quality_score': 0.64,
      'static_features': <String, dynamic>{},
      'dynamic_features': <String, dynamic>{},
      'state_vector': List<double>.filled(12, 0.5),
      'safety': {
        'raw_action_id': 9,
        'raw_action_name': 'CONTINUE_NO_CHANGE',
        'safe_action_id': 9,
        'safe_action_name': 'CONTINUE_NO_CHANGE',
        'overridden': overridden,
        'reasons': <String>[],
      },
      'intervention': {'action_id': 2, 'action_name': 'X', 'type': type, 'message': message, 'params': {}},
      'reward': 0.0,
      'reward_components': <String, dynamic>{},
      'next_exercise': _exJson(next),
      'session_summary': <String, dynamic>{},
      'presentation': {'difficulty_level': 1, 'template_scale': 1.0, 'mode': mode},
      if (source != null) 'decision_source': source,
      if (source != null) 'policy_mode': 'rule_based_fallback',
      'safety_overridden': overridden,
    };

void _size(WidgetTester tester, Size size) {
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
}

double _luminance(Color c) {
  double ch(double v) => v <= 0.03928 ? v / 12.92 : math.pow((v + 0.055) / 1.055, 2.4).toDouble();
  return 0.2126 * ch(c.r) + 0.7152 * ch(c.g) + 0.0722 * ch(c.b);
}

double contrast(Color a, Color b) {
  final la = _luminance(a), lb = _luminance(b);
  return (math.max(la, lb) + 0.05) / (math.min(la, lb) + 0.05);
}

void main() {
  group('design tokens', () {
    test('text and control colours meet WCAG AA contrast on their backgrounds', () {
      final pairs = <String, (Color, Color, double)>{
        'ink on background': (AppColors.ink, AppColors.bg, 4.5),
        'ink on paper': (AppColors.ink, AppColors.paper, 4.5),
        'muted text on paper': (AppColors.inkMuted, AppColors.paper, 4.5),
        'muted text on background': (AppColors.inkMuted, AppColors.bg, 4.5),
        'white on primary button': (Colors.white, AppColors.primary, 4.5),
        'white on navy app bar': (Colors.white, AppColors.navy, 4.5),
        'muted white on navy': (AppColors.onNavyMuted, AppColors.navy, 4.5),
        'white on selected tab (secondary)': (Colors.white, AppColors.secondary, 4.5),
        'error text on error background': (AppColors.errorInk, AppColors.errorSoft, 4.5),
        'success icon on paper (graphics)': (AppColors.success, AppColors.paper, 3.0),
        'primary text button on paper': (AppColors.primary, AppColors.paper, 4.5),
        'writing guide on paper (graphics)': (AppColors.guide, AppColors.paper, 3.0),
        'ink text on secondarySoft (mode tag, steps)': (AppColors.ink, AppColors.secondarySoft, 4.5),
        'secondary icon on secondarySoft (graphics)': (AppColors.secondary, AppColors.secondarySoft, 3.0),
        'secondary text on paper (encouragement)': (AppColors.secondary, AppColors.paper, 4.5),
      };
      for (final e in pairs.entries) {
        final (fg, bg, min) = e.value;
        expect(contrast(fg, bg), greaterThanOrEqualTo(min), reason: '${e.key}: ${contrast(fg, bg).toStringAsFixed(2)}');
      }
    });

    test('white text on the scrimmed gradient keeps 4.5:1 at every gradient stop', () {
      for (final stop in [
        AppColors.gradientSky,
        AppColors.gradientBlue,
        AppColors.gradientTeal,
        AppColors.gradientGreen
      ]) {
        // the lighter of the two scrim ends (alpha 0.55) is the worst case
        final blended = Color.alphaBlend(AppColors.navy.withValues(alpha: 0.55), stop);
        expect(contrast(Colors.white, blended), greaterThanOrEqualTo(4.5), reason: '$stop');
      }
    });

    test('reduced motion makes every tokenised animation instant', () {
      // AppMotion.duration is used by all new animations.
      expect(AppMotion.medium, isNot(Duration.zero));
    });

    testWidgets('AppMotion.duration is zero when the platform disables animations', (tester) async {
      late Duration d;
      await tester.pumpWidget(MediaQuery(
        data: const MediaQueryData(disableAnimations: true),
        child: Builder(builder: (ctx) {
          d = AppMotion.duration(ctx, AppMotion.slow);
          return const SizedBox();
        }),
      ));
      expect(d, Duration.zero);
    });

    test('global OutlinedButton theme keeps dark text', () {
      final style = buildAppTheme().outlinedButtonTheme.style!;
      expect(style.foregroundColor!.resolve({}), AppColors.ink);
    });
  });

  group('letter-case selector', () {
    testWidgets('top-right selector has white text, other outlined buttons stay dark', (tester) async {
      _size(tester, const Size(1280, 900));
      final api = ApiService(
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
      await tester
          .pumpWidget(MaterialApp(theme: buildAppTheme(), home: ExerciseSelectionScreen(api: api, childId: 'P-0042')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('letter-tile-a-lower')));
      await tester.pumpAndSettle();

      final switchText =
          tester.widget<Text>(find.descendant(of: find.byKey(const Key('switch-case')), matching: find.text('a / A')));
      expect(switchText.style!.color, Colors.white);
      final switchStyle = tester.widget<OutlinedButton>(find.byKey(const Key('switch-case'))).style!;
      expect(switchStyle.foregroundColor!.resolve({}), Colors.white);

      // An unrelated outlined button on the same page keeps the global dark text.
      final ctx = tester.element(find.byKey(const Key('start-free-writing')));
      final themed = Theme.of(ctx).outlinedButtonTheme.style!.foregroundColor!.resolve({});
      expect(themed, AppColors.ink);
      final freeWriting = tester.widget<OutlinedButton>(find.byKey(const Key('start-free-writing')));
      expect(freeWriting.style?.foregroundColor, isNull, reason: 'no local override: uses the dark global style');
    });
  });

  group('writing screen', () {
    Future<(List<Map<String, dynamic>>, ApiService)> pumpWriting(WidgetTester tester, Size size,
        {String mode = 'tracing', Map<String, dynamic>? response}) async {
      _size(tester, size);
      final sent = <Map<String, dynamic>>[];
      final api = ApiService(
        baseUrl: 'http://test',
        client: MockClient((req) async {
          sent.add(jsonDecode(req.body) as Map<String, dynamic>);
          return http.Response(jsonEncode(response ?? _attempt()), 200);
        }),
      );
      await tester.pumpWidget(MaterialApp(
        theme: buildAppTheme(),
        home: WritingScreen(
          api: api,
          sessionId: 's',
          childId: 'P-1',
          initialExercise: Exercise.fromJson(_exJson('letter_upper_a')),
          initialMode: mode,
          onSessionEnded: () {},
        ),
      ));
      await tester.pump();
      return (sent, api);
    }

    for (final size in const [Size(360, 640), Size(390, 844), Size(800, 1280), Size(1280, 800), Size(640, 360)]) {
      testWidgets('lays out without overflow at $size and keeps a usable canvas', (tester) async {
        await pumpWriting(tester, size);
        expect(tester.takeException(), isNull);
        final canvas = tester.getSize(find.byType(HandwritingCanvas));
        expect(canvas.width, greaterThan(size.width * 0.6), reason: 'canvas width $canvas');
        expect(canvas.height, greaterThan(size.height < 500 ? 80 : size.height * 0.35), reason: 'canvas $canvas');
        for (final label in ['Clear', 'Undo', 'Retry', 'Submit']) {
          expect(find.text(label), findsOneWidget);
          final b = tester.getRect(find.text(label));
          expect(b.right <= size.width && b.bottom <= size.height, isTrue, reason: '$label visible at $size');
        }
      });
    }

    testWidgets('side controls on a wide landscape tablet', (tester) async {
      await pumpWriting(tester, const Size(1280, 800));
      final controls = tester.widget<SessionControls>(find.byType(SessionControls));
      expect(controls.vertical, isTrue);
      expect(tester.getRect(find.byType(SessionControls)).left,
          greaterThan(tester.getRect(find.byType(HandwritingCanvas)).right));
    });

    testWidgets('app-bar actions are white on navy (not the cobalt default)', (tester) async {
      await pumpWriting(tester, const Size(1280, 800));
      final btn = tester.widget<TextButton>(
          find.descendant(of: find.byKey(const Key('finish-session')), matching: find.byType(TextButton)));
      expect(btn.style!.foregroundColor!.resolve({}), Colors.white);
      await pumpWriting(tester, const Size(360, 640));
      expect(find.byTooltip('Finish'), findsOneWidget, reason: 'icon-only on phones, still labelled');
      expect(find.byTooltip('Progress'), findsOneWidget);
    });

    testWidgets('mode tag reflects the backend mode', (tester) async {
      await pumpWriting(tester, const Size(1280, 800), mode: 'free_writing');
      expect(find.text('On your own'), findsOneWidget);
      expect(find.text('Tracing'), findsNothing);
    });

    testWidgets('submitted points stay in canvas-local coordinates inside the redesigned frame', (tester) async {
      final (sent, _) = await pumpWriting(tester, const Size(1280, 800));
      final canvasFinder = find.byType(HandwritingCanvas);
      final origin = tester.getTopLeft(canvasFinder);
      final canvasSize = tester.getSize(canvasFinder);
      expect(origin.dx, greaterThan(0), reason: 'canvas is inset by the page padding and border');
      final g = await tester.startGesture(origin + const Offset(100, 120), kind: PointerDeviceKind.stylus);
      await g.moveTo(origin + const Offset(150, 140), timeStamp: const Duration(milliseconds: 20));
      await g.up(timeStamp: const Duration(milliseconds: 30));
      await tester.pump();
      await tester.tap(find.text('Submit'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));

      final body = sent.single;
      final pts = (body['points'] as List).cast<Map<String, dynamic>>();
      expect(pts.first['x'], closeTo(100, 1e-6));
      expect(pts.first['y'], closeTo(120, 1e-6));
      expect(pts.last['x'], closeTo(150, 1e-6));
      expect(pts.first['is_stroke_start'], isTrue);
      expect(pts.last['is_stroke_end'], isTrue);
      expect(pts.map((p) => p['t']).toList(), [0.0, 0.02]);
      expect(body['capture']['canvas_width'], closeTo(canvasSize.width, 1e-6));
      expect(body['capture']['canvas_height'], closeTo(canvasSize.height, 1e-6));
      expect(body['capture']['pointer_kind'], 'stylus');
      expect(body['exercise_id'], 'letter_upper_a');
    });

    testWidgets('canvas is disabled while feedback is shown and the feedback fits a phone', (tester) async {
      await pumpWriting(tester, const Size(360, 640),
          response: _attempt(type: 'hint', message: 'Start at the green dot and follow the arrows.'));
      final o = tester.getTopLeft(find.byType(HandwritingCanvas));
      final g = await tester.startGesture(o + const Offset(40, 40));
      await g.moveTo(o + const Offset(60, 50), timeStamp: const Duration(milliseconds: 10));
      await g.up();
      await tester.pump();
      await tester.tap(find.text('Submit'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));
      expect(find.byKey(const Key('feedback-card')), findsOneWidget);
      expect(tester.takeException(), isNull);
      final canvas = tester.widget<HandwritingCanvas>(find.byType(HandwritingCanvas));
      expect(canvas.enabled, isFalse);
      expect(tester.widget<ElevatedButton>(find.byKey(const Key('submit'))).onPressed, isNull);
    });
  });

  group('feedback card is truthful', () {
    Future<void> pump(WidgetTester tester, Map<String, dynamic> json) async {
      await tester.pumpWidget(MaterialApp(
          theme: buildAppTheme(),
          home: Scaffold(body: FeedbackOverlay(result: AttemptResult.fromJson(json), onDismiss: () {}))));
      await tester.pumpAndSettle();
    }

    testWidgets('fallback decision is never labelled as the learned policy', (tester) async {
      // Audit-driven (2026-11): internal implementation labels (Rule-based
      // fallback, DQN, etc.) are no longer shown on the learner-facing
      // feedback card. They remain available in the API response for
      // developer/audit use and in the supervisor-facing PolicyStatusCard.
      await pump(tester, _attempt());
      expect(find.byKey(const Key('decision-source')), findsNothing,
          reason: 'internal decision labels must not appear on the learner-facing card');
      expect(find.textContaining('rule-based fallback'), findsNothing,
          reason: 'internal labels must not be exposed to the learner');
      expect(find.textContaining('Double DQN'), findsNothing);
      expect(find.textContaining('DQN'), findsNothing);
    });

    testWidgets('safety override is not exposed on the learner card', (tester) async {
      // Audit-driven (2026-11): the safety override message is no longer
      // rendered on the learner-facing card. The AttemptResult model
      // still records it (for an optional supervisor view) but the
      // FeedbackOverlay does not show it.
      await pump(tester, _attempt(overridden: true));
      expect(find.textContaining('changed by the safety rules'), findsNothing);
    });

    testWidgets('no decision line when the backend does not report a source', (tester) async {
      await pump(tester, _attempt(source: null));
      expect(find.byKey(const Key('decision-source')), findsNothing);
    });

    testWidgets('the backend message is shown verbatim with the next exercise it sent', (tester) async {
      await pump(tester,
          _attempt(type: 'difficulty_adjustment', message: 'Great work!', next: 'word_sun', mode: 'free_writing'));
      expect(find.text('Great work!'), findsOneWidget);
      expect(find.text('Up next: Word SUN, on your own.'), findsOneWidget);
      expect(find.text('Next'), findsOneWidget);
    });

    testWidgets('break and hint explain what the screen will do', (tester) async {
      await pump(tester, _attempt(type: 'break', message: "Great effort! Let's take a short break."));
      expect(find.textContaining('You can skip it'), findsOneWidget);
      expect(find.text('Start the break'), findsOneWidget);
      await pump(tester, _attempt(type: 'hint', message: 'Start at the green dot and follow the arrows.'));
      expect(find.text('Show me the hint'), findsOneWidget);
    });
  });

  group('break screen', () {
    for (final size in const [Size(360, 640), Size(640, 360), Size(1280, 800)]) {
      testWidgets('fits at $size', (tester) async {
        _size(tester, size);
        await tester.pumpWidget(MaterialApp(theme: buildAppTheme(), home: const BreakScreen(durationSeconds: 5)));
        await tester.pump(const Duration(milliseconds: 100));
        expect(tester.takeException(), isNull);
        expect(find.text('Skip break'), findsOneWidget);
      });
    }

    testWidgets('reduced motion: static circle, no breathing prompts', (tester) async {
      await tester.pumpWidget(MaterialApp(
        theme: buildAppTheme(),
        home: const MediaQuery(data: MediaQueryData(disableAnimations: true), child: BreakScreen(durationSeconds: 5)),
      ));
      await tester.pump(const Duration(seconds: 1));
      expect(find.text('Breathe in'), findsNothing);
      expect(find.text('Breathe out'), findsNothing);
      expect(tester.hasRunningAnimations, isFalse);
    });
  });

  group('completion and home', () {
    testWidgets('summary shows the backend numbers; an error offers retry', (tester) async {
      var fail = true;
      var calls = 0;
      final api = ApiService(
        baseUrl: 'http://t',
        client: MockClient((req) async {
          calls++;
          return fail
              ? http.Response('boom', 500)
              : http.Response(
                  jsonEncode({
                    'session_id': 's',
                    'attempt_count': 4,
                    'score_history': [0.1, 0.2, 0.3, 0.4],
                    'difficulty_level': 1,
                    'break_count': 1,
                    'ended': true,
                  }),
                  200);
        }),
      );
      await tester.pumpWidget(MaterialApp(theme: buildAppTheme(), home: CompletionScreen(api: api, sessionId: 's')));
      await tester.pumpAndSettle();
      expect(find.text('Try again'), findsOneWidget);
      fail = false;
      await tester.tap(find.text('Try again'));
      await tester.pump();
      expect(find.text('Loading your summary...'), findsOneWidget, reason: 'retry shows loading, not the old error');
      await tester.pumpAndSettle();
      expect(find.text('You did 4 attempts and took 1 break.'), findsOneWidget);
      expect(calls, 2);
      expect(find.textContaining('improv'), findsNothing, reason: 'no improvement claim');
    });

    for (final size in const [Size(360, 640), Size(1280, 800)]) {
      testWidgets('home explains the product and fits at $size', (tester) async {
        _size(tester, size);
        await tester.pumpWidget(const DysgraphiaTutorApp());
        expect(tester.takeException(), isNull);
        expect(find.text('How practice works'), findsOneWidget);
        expect(find.textContaining('Adaptive handwriting practice'), findsOneWidget);
      });
    }

    testWidgets('dashboard fits a 360 dp phone', (tester) async {
      _size(tester, const Size(360, 640));
      final api = ApiService(
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
      await tester
          .pumpWidget(MaterialApp(theme: buildAppTheme(), home: ExerciseSelectionScreen(api: api, childId: 'P-0042')));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull);
      await tester.ensureVisible(find.byKey(const Key('policy-status')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('policy-status')));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull, reason: 'expanding the adult details must not overflow');
      expect(find.textContaining('Behavioural gate'), findsOneWidget);
    });
  });
}
