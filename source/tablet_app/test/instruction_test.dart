// Tests for the child-facing instruction feature (letters, words, shapes).
//
// NOT executed in the environment where they were written: no Flutter SDK
// was available there (storage.googleapis.com is blocked by its network
// policy). Run with:  flutter pub get && flutter analyze && flutter test
//
// test/fixtures/catalogue.json is the backend's real /exercises response; a
// backend test (test_flutter_fixture_matches_the_api) keeps it in sync.

import 'dart:convert';
import 'dart:io';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:dysgraphia_tutor_tablet/models/exercise.dart';
import 'package:dysgraphia_tutor_tablet/models/exercise_instruction.dart';
import 'package:dysgraphia_tutor_tablet/screens/writing_screen.dart';
import 'package:dysgraphia_tutor_tablet/services/api_service.dart';
import 'package:dysgraphia_tutor_tablet/theme/app_theme.dart';
import 'package:dysgraphia_tutor_tablet/widgets/exercise_display.dart';
import 'package:dysgraphia_tutor_tablet/widgets/handwriting_canvas.dart';

const _line = [
  [0.0, 50.0],
  [100.0, 50.0]
];

Exercise _ex(ExerciseCategory c, String? target,
        {String id = 'x', String prompt = 'Backend prompt', List<List<List<double>>> strokes = const []}) =>
    Exercise(
        exerciseId: id,
        prompt: prompt,
        difficulty: 1,
        referenceTemplate: _line,
        category: c,
        target: target,
        templateStrokes: strokes);

List<Exercise> _catalogue() {
  final raw = jsonDecode(File('test/fixtures/catalogue.json').readAsStringSync()) as List<dynamic>;
  return raw.map((e) => Exercise.fromJson(e as Map<String, dynamic>)).toList();
}

Map<String, dynamic> _catalogueJson(String id) {
  final raw = jsonDecode(File('test/fixtures/catalogue.json').readAsStringSync()) as List<dynamic>;
  return raw.cast<Map<String, dynamic>>().firstWhere((e) => e['exercise_id'] == id);
}

Widget _host(Widget child) => MaterialApp(theme: buildAppTheme(), home: Scaffold(body: child));

HandwritingPainter _painter(WidgetTester tester) => tester
    .widgetList<CustomPaint>(find.descendant(of: find.byType(HandwritingCanvas), matching: find.byType(CustomPaint)))
    .map((c) => c.painter)
    .whereType<HandwritingPainter>()
    .single;

void main() {
  // ------------------------------------------------------------------ A. generation
  group('instructionFor', () {
    test('uppercase and lowercase letters keep their case', () {
      expect(instructionFor(_ex(ExerciseCategory.letter, 'A'), 'tracing').headline, 'Trace the letter A.');
      expect(instructionFor(_ex(ExerciseCategory.letter, 'm'), 'free_writing').headline, "Let's write the letter m!");
    });

    test('words keep their exact spelling', () {
      expect(instructionFor(_ex(ExerciseCategory.word, 'CAT'), 'tracing').headline, 'Trace the word CAT.');
      expect(instructionFor(_ex(ExerciseCategory.word, 'SUN'), 'free_writing').headline, "Let's write the word SUN!");
      expect(instructionFor(_ex(ExerciseCategory.word, 'cat'), 'tracing').headline, 'Trace the word cat.');
    });

    test('shapes use their name, or a generic sentence when it is missing', () {
      expect(
          instructionFor(_ex(ExerciseCategory.shape, 'straight line'), 'tracing').headline, 'Trace the straight line.');
      expect(instructionFor(_ex(ExerciseCategory.shape, 'zigzag'), 'free_writing').headline,
          'Draw the zigzag in the writing area.');
      expect(instructionFor(_ex(ExerciseCategory.shape, null), 'tracing').headline, 'Trace the shape carefully.');
      expect(instructionFor(_ex(ExerciseCategory.shape, ''), 'free_writing').headline,
          'Draw the shape in the writing area.');
    });

    test('tracing vs independent guidance follows the canvas rule', () {
      final t = instructionFor(_ex(ExerciseCategory.letter, 'A'), 'tracing');
      final f = instructionFor(_ex(ExerciseCategory.letter, 'A'), 'free_writing');
      expect(t.guidance, contains('Follow the guide'));
      expect(f.guidance, contains('No guide this time'));
      expect(guideVisibleFor('tracing'), isTrue);
      expect(guideVisibleFor('free_writing'), isFalse);
      // any other value shows the guide on the canvas, so it must read as tracing too
      expect(guideVisibleFor('unexpected'), isTrue);
      expect(instructionFor(_ex(ExerciseCategory.letter, 'A'), 'unexpected').headline, 'Trace the letter A.');
    });

    test('whitespace around a target is ignored, inner spelling is not changed', () {
      expect(instructionFor(_ex(ExerciseCategory.letter, ' A '), 'tracing').headline, 'Trace the letter A.');
      expect(instructionFor(_ex(ExerciseCategory.word, '  SUN\n'), 'tracing').headline, 'Trace the word SUN.');
      expect(instructionFor(_ex(ExerciseCategory.shape, '  figure   eight '), 'tracing').headline,
          'Trace the figure eight.');
    });

    test('malformed letter/word targets are never shown as a letter or word', () {
      for (final bad in ['', 'AB', '1', '?', 'é', 'A B']) {
        final i = instructionFor(_ex(ExerciseCategory.letter, bad), 'tracing');
        expect(i.isFallback, isTrue, reason: bad);
        expect(i.headline, 'Backend prompt');
        expect(i.headline.contains('letter'), isFalse);
      }
      for (final bad in ['C', 'CAT DOG', 'C4T', '<b>', '']) {
        final i = instructionFor(_ex(ExerciseCategory.word, bad), 'tracing');
        expect(i.isFallback, isTrue, reason: bad);
        expect(i.headline.contains('word'), isFalse);
      }
      expect(instructionFor(_ex(ExerciseCategory.letter, null), 'tracing').isFallback, isTrue);
    });

    test('unknown category is not guessed from the display text', () {
      final i = instructionFor(_ex(ExerciseCategory.unknown, 'A', prompt: 'Trace the loop'), 'tracing');
      expect(i.isFallback, isTrue);
      expect(i.headline, 'Trace the loop');
      expect(i.label, '');
      final empty = instructionFor(_ex(ExerciseCategory.unknown, null, prompt: '  '), 'tracing');
      expect(empty.headline, 'Look at the model, then write in the writing area.');
    });

    test('no internal identifiers or field names are shown', () {
      for (final ex in _catalogue()) {
        for (final mode in ['tracing', 'free_writing']) {
          final i = instructionFor(ex, mode);
          final all = '${i.headline} ${i.guidance} ${i.label}';
          expect(all.contains(ex.exerciseId), isFalse, reason: ex.exerciseId);
          expect(all.contains('_'), isFalse);
          expect(all.toLowerCase().contains('category'), isFalse);
          expect(all.toLowerCase().contains('target'), isFalse);
        }
      }
    });

    test('a retry adds encouragement', () {
      expect(instructionFor(_ex(ExerciseCategory.letter, 'A'), 'tracing').encouragement, isNull);
      expect(instructionFor(_ex(ExerciseCategory.letter, 'A'), 'tracing', retryIndex: 2).encouragement, tryAgainText);
    });
  });

  // ------------------------------------------------------------------ parsing
  group('Exercise.fromJson', () {
    test('reads category, target and strokes', () {
      final ex = Exercise.fromJson(_catalogueJson('letter_upper_a'));
      expect(ex.category, ExerciseCategory.letter);
      expect(ex.target, 'A');
      expect(ex.guideStrokes.length, 3);
      expect(ex.writingLines!.ascender, 10);
      // Four-line paper since the A-Z upgrade (ALPHABET_UPGRADE.md): 10 / 40 / 70 / 100.
      expect(ex.writingLines!.xHeight, 40);
      expect(ex.writingLines!.baseline, 70);
      expect(ex.writingLines!.descender, 100);
      expect(Exercise.fromJson(_catalogueJson('loop_medium')).writingLines, isNull);
      expect(WritingLines.tryParse({'ascender': 90, 'x_height': 45, 'baseline': 10}), isNull,
          reason: 'lines in the wrong order are rejected');
    });

    test('older backend without the new fields still works', () {
      final ex = Exercise.fromJson({
        'exercise_id': 'line_easy',
        'prompt': 'Trace the straight line',
        'difficulty': 0,
        'reference_template': _line,
      });
      expect(ex.category, ExerciseCategory.unknown);
      expect(ex.guideStrokes, [_line]);
      expect(instructionFor(ex, 'tracing').headline, 'Trace the straight line');
    });

    test('malformed strokes fall back to reference_template; unknown category stays unknown', () {
      final ex = Exercise.fromJson({
        'exercise_id': 'x',
        'prompt': 'p',
        'difficulty': 0,
        'reference_template': _line,
        'category': 'picture',
        'target': 5,
        'template_strokes': [
          [
            [1, 2]
          ]
        ],
      });
      expect(ex.category, ExerciseCategory.unknown);
      expect(ex.target, isNull);
      expect(ex.guideStrokes, [_line]);
    });
  });

  // ------------------------------------------------------------------ B. consistency
  group('target, instruction and guide come from the same record', () {
    final catalogue = _catalogue();

    test('catalogue has real letters, words and shapes', () {
      expect(catalogue.where((e) => e.category == ExerciseCategory.letter).length, 52);
      expect(catalogue.where((e) => e.category == ExerciseCategory.word).map((e) => e.target).toList(),
          ['TO', 'CAT', 'SUN']);
      expect(catalogue.where((e) => e.category == ExerciseCategory.shape).length, 5);
      expect(catalogue.any((e) => e.category == ExerciseCategory.unknown), isFalse);
    });

    test('instruction names the exact target; guide is that record\'s strokes', () {
      for (final ex in catalogue) {
        final json = _catalogueJson(ex.exerciseId);
        for (final mode in ['tracing', 'free_writing']) {
          final i = instructionFor(ex, mode);
          expect(i.isFallback, isFalse, reason: ex.exerciseId);
          switch (ex.category) {
            case ExerciseCategory.letter:
              expect(i.headline, contains('the letter ${ex.target}'));
              expect(i.label, 'Letter ${ex.target}');
            case ExerciseCategory.word:
              expect(i.headline, contains('the word ${ex.target}'));
              expect(i.label, 'Word ${ex.target}');
            case ExerciseCategory.shape:
              expect(i.headline, contains(ex.target!));
            case ExerciseCategory.unknown:
              fail('unexpected');
          }
        }
        final strokes = (json['template_strokes'] as List).length;
        expect(ex.guideStrokes.length, strokes);
        expect(ex.guidePoints, ex.referenceTemplate, reason: 'strokes concatenate to reference_template');
      }
    });

    test('a letter never resolves to a shape guide; a word never to another word', () {
      final shapes = catalogue.where((e) => e.category == ExerciseCategory.shape).map((e) => e.guidePoints).toList();
      for (final ex in catalogue.where((e) => e.category == ExerciseCategory.letter)) {
        expect(shapes.any((s) => s.toString() == ex.guidePoints.toString()), isFalse, reason: ex.target);
        expect(ex.exerciseId.endsWith(ex.target!.toLowerCase()), isTrue);
      }
      final words = catalogue.where((e) => e.category == ExerciseCategory.word).toList();
      const wordAdvance = 90.0; // backend letter_templates.WORD_ADVANCE
      for (final a in words) {
        // a word's guide is the strokes of ITS OWN letters, in order, each
        // letter shifted one advance to the right
        var k = 0;
        final letters = a.target!.split('');
        for (var i = 0; i < letters.length; i++) {
          final glyph =
              catalogue.firstWhere((e) => e.category == ExerciseCategory.letter && e.target == letters[i]).guideStrokes;
          for (final s in glyph) {
            final w = a.guideStrokes[k++];
            expect(w.length, s.length, reason: '${a.target} letter ${letters[i]}');
            for (var j = 0; j < s.length; j++) {
              expect(w[j][0], closeTo(s[j][0] + i * wordAdvance, 0.011));
              expect(w[j][1], closeTo(s[j][1], 1e-9));
            }
          }
        }
        expect(k, a.guideStrokes.length);
        for (final b in words.where((w) => w != a)) {
          expect(a.guidePoints.toString() == b.guidePoints.toString(), isFalse);
        }
      }
    });

    testWidgets('independent mode hides the canvas guide; tracing shows every stroke', (tester) async {
      final ex = Exercise.fromJson(_catalogueJson('letter_upper_e'));
      for (final mode in ['tracing', 'free_writing']) {
        await tester.pumpWidget(_host(SizedBox(
            width: 600,
            height: 400,
            child: HandwritingCanvas(
                key: ValueKey(mode),
                referenceTemplate: ex.referenceTemplate,
                templateStrokes: ex.guideStrokes,
                mode: mode,
                onStrokesChanged: (_) {}))));
        final p = _painter(tester);
        expect(p.showGuide, mode == 'tracing');
        expect(p.guideStrokes!.length, 4, reason: 'E is drawn as 4 separate strokes, not one joined line');
      }
    });

    testWidgets('lowercase c is drawn smaller than capital C, on writing lines', (tester) async {
      double guideHeight(HandwritingPainter p) {
        final ys = [for (final s in p.guideStrokes!) ...s.map((o) => o.dy)];
        return ys.reduce((a, b) => a > b ? a : b) - ys.reduce((a, b) => a < b ? a : b);
      }

      final heights = <String, double>{};
      for (final id in ['letter_lower_c', 'letter_upper_c', 'loop_medium']) {
        final ex = Exercise.fromJson(_catalogueJson(id));
        for (final mode in ['tracing', 'free_writing']) {
          await tester.pumpWidget(_host(SizedBox(
              width: 800,
              height: 500,
              child: HandwritingCanvas(
                  key: ValueKey('$id$mode'),
                  referenceTemplate: ex.referenceTemplate,
                  templateStrokes: ex.guideStrokes,
                  writingLines: ex.writingLines,
                  mode: mode,
                  onStrokesChanged: (_) {}))));
          final p = _painter(tester);
          heights[id] = guideHeight(p);
          if (id == 'loop_medium') {
            expect(p.writingLineYs, isNull, reason: 'shapes have no writing lines');
          } else {
            // lines stay in free writing too: they are paper, not the tracing guide
            // top, middle, baseline and descender line (four-line paper)
            expect(p.writingLineYs, hasLength(4));
            final lines = p.writingLineYs!;
            expect(lines[0] < lines[1] && lines[1] < lines[2] && lines[2] < lines[3], isTrue);
          }
        }
      }
      expect(heights['letter_lower_c']! < 0.65 * heights['letter_upper_c']!, isTrue,
          reason: 'c must not look like C: ${heights['letter_lower_c']} vs ${heights['letter_upper_c']}');
    });

    testWidgets('instruction card shows headline and guidance without the internal id', (tester) async {
      final ex = Exercise.fromJson(_catalogueJson('word_cat'));
      await tester.pumpWidget(_host(SizedBox(width: 900, child: ExerciseDisplay(exercise: ex, mode: 'tracing'))));
      expect(find.text('Trace the word CAT.'), findsOneWidget);
      expect(find.text('Follow the guide in the writing area.'), findsOneWidget);
      expect(find.textContaining('word_cat'), findsNothing);
      expect(find.text('Write the word CAT'), findsNothing, reason: 'no duplicate title');
    });

    testWidgets('instruction card does not overflow on a narrow tablet in portrait', (tester) async {
      tester.view.physicalSize = const Size(600, 1000);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      final ex = Exercise.fromJson(_catalogueJson('letter_lower_m'));
      await tester.pumpWidget(_host(ExerciseDisplay(exercise: ex, mode: 'free_writing', retryIndex: 1)));
      expect(tester.takeException(), isNull);
      expect(find.text("Let's write the letter m!"), findsOneWidget);
      expect(find.text(tryAgainText), findsOneWidget);
    });
  });

  // ------------------------------------------------------------------ C. transitions
  group('exercise transition after an intervention', () {
    Map<String, dynamic> attemptResponse(Map<String, dynamic> next, String mode, String type) => {
          'session_id': 's',
          'attempt_id': 'a',
          'exercise_id': 'letter_upper_a',
          'quality_score': 0.8,
          'static_features': <String, dynamic>{},
          'dynamic_features': <String, dynamic>{},
          'state_vector': List<double>.filled(12, 0.5),
          'safety': {
            'raw_action_id': 1,
            'raw_action_name': 'INCREASE_DIFFICULTY',
            'safe_action_id': 1,
            'safe_action_name': 'INCREASE_DIFFICULTY',
            'overridden': false,
            'reasons': <String>[],
          },
          'intervention': {
            'action_id': 1,
            'action_name': 'INCREASE_DIFFICULTY',
            'type': type,
            'message': 'Great work!',
            'params': <String, dynamic>{},
          },
          'reward': 1.0,
          'reward_components': <String, dynamic>{},
          'next_exercise': next,
          'session_summary': <String, dynamic>{},
          'presentation': {'difficulty_level': next['difficulty'], 'template_scale': 1.0, 'mode': mode},
        };

    Future<void> drawAndSubmit(WidgetTester tester) async {
      final canvas = find.byType(HandwritingCanvas);
      final o = tester.getTopLeft(canvas);
      final g = await tester.startGesture(o + const Offset(60, 60), kind: PointerDeviceKind.touch);
      await g.moveTo(o + const Offset(120, 90), timeStamp: const Duration(milliseconds: 16));
      await g.up(timeStamp: const Duration(milliseconds: 32));
      await tester.pump();
      await tester.tap(find.text('Submit'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 50));
    }

    testWidgets('instruction, target, guide and mode all switch to the next exercise', (tester) async {
      tester.view.physicalSize = const Size(1280, 900);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);
      final responses = [
        attemptResponse(_catalogueJson('word_sun'), 'free_writing', 'difficulty_adjustment'),
        attemptResponse(_catalogueJson('word_sun'), 'free_writing', 'continue'),
      ];
      var calls = 0;
      final api = ApiService(
          baseUrl: 'http://test',
          client: MockClient((req) async => http.Response(jsonEncode(responses[calls++]), 200)));
      await tester.pumpWidget(MaterialApp(
        theme: buildAppTheme(),
        home: WritingScreen(
          api: api,
          sessionId: 's',
          childId: 'P-0001',
          initialExercise: Exercise.fromJson(_catalogueJson('letter_upper_a')),
          onSessionEnded: () {},
        ),
      ));
      expect(find.text('Trace the letter A.'), findsOneWidget);
      expect(_painter(tester).guideStrokes!.length, 3);
      expect(_painter(tester).showGuide, isTrue);

      await drawAndSubmit(tester);
      expect(find.text('Great work!'), findsOneWidget); // existing feedback still shown
      await tester.tap(find.text('Next'));
      await tester.pump();

      expect(find.text('Trace the letter A.'), findsNothing, reason: 'no stale instruction');
      expect(find.text("Let's write the word SUN!"), findsOneWidget);
      expect(find.text(tryAgainText), findsNothing, reason: 'new exercise: not a retry');
      final p = _painter(tester);
      expect(p.showGuide, isFalse, reason: 'free writing hides the guide');
      // word_sun has 4 pen strokes in the catalogue (S, u, n); A has 3.
      expect(p.guideStrokes!.length, 4, reason: 'S, U, N strokes, not the 3 strokes of A');
      expect(p.guideStrokes!.length, Exercise.fromJson(_catalogueJson('word_sun')).guideStrokes.length);
      expect(p.strokes, isEmpty, reason: 'canvas cleared for the new exercise');

      // same exercise again -> encouragement
      await drawAndSubmit(tester);
      await tester.tap(find.text('Next'));
      await tester.pump();
      expect(find.text("Let's write the word SUN!"), findsOneWidget);
      expect(find.text(tryAgainText), findsOneWidget);
      expect(calls, 2);
    });
  });
}
