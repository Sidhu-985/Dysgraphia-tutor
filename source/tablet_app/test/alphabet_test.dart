// Tests for the A-Z alphabet dashboard, the letter learning page and the
// supervisor policy status.
//
// NOT RUN where they were written: no Flutter SDK was available there
// (storage.googleapis.com and pub.dev were blocked by the network policy).
// Run them with:  flutter pub get && flutter analyze && flutter test
//
// test/fixtures/catalogue.json is the backend's real /exercises response
// (backend/tests/test_exercise_catalogue.py::test_flutter_fixture_matches_the_api).

import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:dysgraphia_tutor_tablet/models/alphabet.dart';
import 'package:dysgraphia_tutor_tablet/models/attempt_result.dart';
import 'package:dysgraphia_tutor_tablet/models/exercise.dart';
import 'package:dysgraphia_tutor_tablet/models/session_summary.dart';
import 'package:dysgraphia_tutor_tablet/screens/exercise_selection_screen.dart';
import 'package:dysgraphia_tutor_tablet/screens/letter_detail_screen.dart';
import 'package:dysgraphia_tutor_tablet/services/api_service.dart';
import 'package:dysgraphia_tutor_tablet/theme/app_theme.dart';
import 'package:dysgraphia_tutor_tablet/widgets/alphabet_section.dart';
import 'package:dysgraphia_tutor_tablet/widgets/letter_guide.dart';
import 'package:dysgraphia_tutor_tablet/widgets/policy_status_card.dart';

String _fixture() => File('test/fixtures/catalogue.json').readAsStringSync();

List<Exercise> _catalogue() =>
    (jsonDecode(_fixture()) as List<dynamic>).map((e) => Exercise.fromJson(e as Map<String, dynamic>)).toList();

Map<String, dynamic> _health({String mode = 'rule_based_fallback', bool dqn = false, bool? gate = false}) => {
      'status': 'ok',
      'policy': {
        'policy_mode': mode,
        'dqn_making_decisions': dqn,
        'behavioral_gate_passed': gate,
        'behavioral_gate_failures': gate == false ? ['G4: requests SUGGEST_BREAK in 34% of low-fatigue states'] : [],
        'fallback_reasons': gate == false ? ['G4: requests SUGGEST_BREAK in 34% of low-fatigue states'] : [],
        'checkpoint_sha256': '7b17f6f3e1a2c3e98e4e4153d792b1db9ccd219146f01ead3e0fc4b7f59a0c43',
      },
    };

/// Fake backend: /exercises from the real fixture, /health as given,
/// /session/start echoes the requested exercise and practice mode.
class _FakeBackend {
  final List<Map<String, dynamic>> started = [];
  Map<String, dynamic> health;
  bool failExercises;

  _FakeBackend({Map<String, dynamic>? health, this.failExercises = false}) : health = health ?? _health();

  ApiService api() => ApiService(
        baseUrl: 'http://fake',
        client: MockClient((req) async {
          if (req.url.path == '/exercises') {
            return failExercises ? http.Response('boom', 500) : http.Response(_fixture(), 200);
          }
          if (req.url.path == '/health') return http.Response(jsonEncode(health), 200);
          if (req.url.path == '/session/start') {
            final body = jsonDecode(req.body) as Map<String, dynamic>;
            started.add(body);
            final ex = (jsonDecode(_fixture()) as List).firstWhere((e) => e['exercise_id'] == body['exercise_id']);
            return http.Response(
                jsonEncode({
                  'session_id': 's-1',
                  'child_id': body['child_id'],
                  'exercise': ex,
                  'data_source': 'demo',
                  'policy_mode': 'rule_based_fallback',
                  'practice_mode': body['practice_mode'] ?? 'tracing',
                }),
                200);
          }
          return http.Response('not found', 404);
        }),
      );
}

Future<void> _pumpDashboard(WidgetTester tester, _FakeBackend backend, {Size size = const Size(1280, 900)}) async {
  tester.view.physicalSize = size;
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
  await tester.pumpWidget(MaterialApp(
    theme: buildAppTheme(),
    home: ExerciseSelectionScreen(api: backend.api(), childId: 'P-0042'),
  ));
  await tester.pumpAndSettle();
}

void main() {
  group('AlphabetIndex (catalogue fixture)', () {
    final index = AlphabetIndex.fromExercises(_catalogue());

    test('26 uppercase and 26 lowercase letters, A-Z order', () {
      expect(index.count(LetterCase.upper), 26);
      expect(index.count(LetterCase.lower), 26);
      expect(index.complete, isTrue);
      expect(index.letters(LetterCase.upper).map((l) => l.$1).join(), kUppercaseLetters);
      expect(index.letters(LetterCase.lower).map((l) => l.$1).join(), kLowercaseLetters);
    });

    test('every letter has its own exercise, target and guide', () {
      final ids = <String>{};
      for (final c in LetterCase.values) {
        for (final (ch, ex) in index.letters(c)) {
          expect(ex, isNotNull, reason: ch);
          expect(ex!.target, ch);
          expect(ex.guideStrokes, isNotEmpty);
          expect(ex.writingLines?.descender, isNotNull);
          ids.add(ex.exerciseId);
        }
      }
      expect(ids.length, 52);
    });

    test('upper and lower forms are distinct records and switch to each other', () {
      expect(index['a']!.exerciseId, isNot(index['A']!.exerciseId));
      expect(index.counterpart('a')!.target, 'A');
      expect(index.counterpart('Q')!.target, 'q');
      expect(index['g']!.letterZone, LetterZone.descender);
      expect(index['b']!.letterZone, LetterZone.ascender);
      expect(index['o']!.letterZone, LetterZone.xHeight);
      expect(index['B']!.letterZone, LetterZone.capital);
    });

    test('neighbours walk a sequence without wrapping', () {
      expect(neighbours(['a', 'b', 'c'], 'a'), (null, 'b'));
      expect(neighbours(['a', 'b', 'c'], 'c'), ('b', null));
      expect(neighbours(['a', 'b'], 'z'), (null, null));
    });

    test('a mislabelled record is not shown as a letter', () {
      final bad = Exercise.fromJson({
        'exercise_id': 'x',
        'prompt': 'p',
        'difficulty': 0,
        'reference_template': [
          [0, 0],
          [1, 1]
        ],
        'category': 'letter',
        'target': 'a',
        'letter_case': 'upper',
      });
      expect(AlphabetIndex.fromExercises([bad]).count(LetterCase.lower), 0);
    });
  });

  group('PracticeLog', () {
    test('only submitted attempts count as practised', () {
      final log = PracticeLog();
      expect(log.practised('letter_lower_a'), isFalse);
      log.record('letter_lower_a');
      expect(log.practised('letter_lower_a'), isTrue);
      expect(log.practisedCount(['letter_lower_a', 'letter_lower_b']), 1);
    });
  });

  group('dashboard', () {
    testWidgets('shows the alphabet section with both tabs and 26 tiles each', (tester) async {
      final backend = _FakeBackend();
      await _pumpDashboard(tester, backend);
      expect(find.text('Learn & Write the Alphabet'), findsOneWidget);
      expect(find.byKey(const Key('tab-lowercase')), findsOneWidget);
      expect(find.byKey(const Key('tab-uppercase')), findsOneWidget);
      for (final ch in kLowercaseLetters.split('')) {
        expect(find.byKey(Key('letter-tile-$ch-lower')), findsOneWidget, reason: ch);
      }
      await tester.tap(find.byKey(const Key('tab-uppercase')));
      await tester.pumpAndSettle();
      for (final ch in kUppercaseLetters.split('')) {
        expect(find.byKey(Key('letter-tile-$ch-upper')), findsOneWidget, reason: ch);
      }
      expect(find.byKey(const Key('letter-tile-a-lower')), findsNothing);
    });

    testWidgets('no overflow on a phone and on a tablet', (tester) async {
      for (final size in const [Size(390, 844), Size(1280, 800), Size(800, 1280)]) {
        final backend = _FakeBackend();
        await _pumpDashboard(tester, backend, size: size);
        expect(tester.takeException(), isNull, reason: '$size');
        expect(find.byType(AlphabetSection), findsOneWidget);
      }
    });

    testWidgets('a group filter narrows the tiles', (tester) async {
      final backend = _FakeBackend();
      await _pumpDashboard(tester, backend);
      await tester.tap(find.text('Tail letters'));
      await tester.pumpAndSettle();
      for (final ch in 'gjpqy'.split('')) {
        expect(find.byKey(Key('letter-tile-$ch-lower')), findsOneWidget);
      }
      expect(find.byKey(const Key('letter-tile-a-lower')), findsNothing);
    });

    testWidgets('tiles are accessible buttons with a name', (tester) async {
      final handle = tester.ensureSemantics();
      final backend = _FakeBackend();
      await _pumpDashboard(tester, backend);
      expect(find.bySemanticsLabel(RegExp(r'^small a, ')), findsOneWidget);
      final size = tester.getSize(find.byKey(const Key('letter-tile-a-lower')));
      expect(size.width, greaterThanOrEqualTo(AppSpacing.touchTarget));
      expect(size.height, greaterThanOrEqualTo(AppSpacing.touchTarget));
      handle.dispose();
    });

    testWidgets('backend error shows a retryable message', (tester) async {
      final backend = _FakeBackend(failExercises: true);
      await _pumpDashboard(tester, backend);
      expect(find.text('Try again'), findsWidgets);
      expect(find.byType(AlphabetSection), findsNothing);
    });

    testWidgets('policy status shows the fallback when the gate failed', (tester) async {
      final backend = _FakeBackend();
      await _pumpDashboard(tester, backend);
      expect(find.text('Decisions: rule-based fallback'), findsOneWidget);
      expect(find.text('Decisions: learned policy (Double DQN)'), findsNothing);
    });
  });

  group('policy status card', () {
    Future<void> pump(WidgetTester t, BackendHealth h) =>
        t.pumpWidget(MaterialApp(theme: buildAppTheme(), home: Scaffold(body: PolicyStatusCard(health: h))));

    testWidgets('DQN badge only when mode is dqn AND the DQN makes decisions', (tester) async {
      await pump(tester, BackendHealth.fromJson(_health(mode: 'dqn', dqn: true, gate: true)));
      expect(find.text('Decisions: learned policy (Double DQN)'), findsOneWidget);
      // inconsistent report: never upgraded to the DQN badge
      await pump(tester, BackendHealth.fromJson(_health(mode: 'dqn', dqn: false, gate: false)));
      expect(find.text('Decisions: learned policy (Double DQN)'), findsNothing);
      expect(find.text('Decisions: rule-based fallback'), findsOneWidget);
    });
  });

  group('attempt decision source', () {
    test('reports the real source and safety override', () {
      final r = AttemptResult.fromJson({
        'session_id': 's',
        'attempt_id': 'a',
        'exercise_id': 'letter_lower_a',
        'quality_score': 0.7,
        'static_features': <String, dynamic>{},
        'dynamic_features': <String, dynamic>{},
        'state_vector': List<double>.filled(12, 0.5),
        'safety': {
          'raw_action_id': 1,
          'raw_action_name': 'INCREASE_DIFFICULTY',
          'safe_action_id': 2,
          'safe_action_name': 'SHOW_HINT',
          'overridden': true,
          'reasons': ['x'],
        },
        'intervention': {'action_id': 2, 'action_name': 'SHOW_HINT', 'type': 'hint', 'message': 'm', 'params': {}},
        'reward': 0.0,
        'reward_components': <String, dynamic>{},
        'next_exercise': (jsonDecode(_fixture()) as List).firstWhere((e) => e['exercise_id'] == 'letter_lower_a'),
        'session_summary': <String, dynamic>{},
        'policy_mode': 'rule_based_fallback',
        'decision_source': 'rule_based_fallback',
        'safety_overridden': true,
      });
      expect(r.decisionSource, 'rule_based_fallback');
      expect(r.decisionLabel, 'rule-based fallback, changed by the safety rules');
    });
  });

  group('letter page', () {
    Future<_FakeBackend> openLetter(WidgetTester tester, String char, {Size size = const Size(1280, 900)}) async {
      final backend = _FakeBackend();
      await _pumpDashboard(tester, backend, size: size);
      if (char.toUpperCase() == char) {
        await tester.tap(find.byKey(const Key('tab-uppercase')));
        await tester.pumpAndSettle();
      }
      final key = Key('letter-tile-$char${char.toUpperCase() == char ? '-upper' : '-lower'}');
      await tester.ensureVisible(find.byKey(key));
      await tester.pumpAndSettle(); // ensureVisible scrolls but does not lay out; tap needs the new position
      await tester.tap(find.byKey(key));
      await tester.pumpAndSettle();
      return backend;
    }

    testWidgets('shows the reference letter, guide, stroke chips and both start buttons', (tester) async {
      await openLetter(tester, 'b');
      expect(find.byType(LetterDetailScreen), findsOneWidget);
      expect(find.byKey(const Key('reference-letter')), findsOneWidget);
      expect(find.byType(LetterGuide), findsOneWidget);
      expect(find.text('Stroke 1'), findsOneWidget);
      expect(find.text('Stroke 2'), findsOneWidget);
      expect(find.byKey(const Key('start-tracing')), findsOneWidget);
      expect(find.byKey(const Key('start-free-writing')), findsOneWidget);
      expect(tester.takeException(), isNull);
    });

    testWidgets('switching case keeps the same letter', (tester) async {
      await openLetter(tester, 'g');
      expect(find.text('Letter small g'), findsOneWidget);
      await tester.tap(find.byKey(const Key('switch-case')));
      await tester.pumpAndSettle();
      expect(find.text('Letter capital G'), findsOneWidget);
      await tester.tap(find.byKey(const Key('switch-case')));
      await tester.pumpAndSettle();
      expect(find.text('Letter small g'), findsOneWidget);
    });

    testWidgets('next and previous follow A-Z without forcing it', (tester) async {
      await openLetter(tester, 'a');
      expect(find.text('Previous'), findsOneWidget); // nothing before a
      await tester.tap(find.byKey(const Key('next-letter')));
      await tester.pumpAndSettle();
      expect(find.text('Letter small b'), findsOneWidget);
    });

    testWidgets('starting a session sends the exact letter and practice mode', (tester) async {
      final backend = await openLetter(tester, 'Q');
      await tester.ensureVisible(find.byKey(const Key('start-free-writing')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('start-free-writing')));
      await tester.pumpAndSettle();
      expect(backend.started.single['exercise_id'], 'letter_upper_q');
      expect(backend.started.single['practice_mode'], 'free_writing');
    });

    testWidgets('letter page fits on a phone', (tester) async {
      await openLetter(tester, 'y', size: const Size(390, 844));
      expect(tester.takeException(), isNull);
      expect(find.byType(LetterGuide), findsOneWidget);
    });
  });
}
