// Basic widget smoke tests for the tablet app.
//
// NOTE: these were written as source but NOT executed in the environment
// this project was built in (no Flutter SDK available there -- see
// project README "Known limitations"). Run them yourself after
// `flutter pub get`:
//
//   flutter test

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:dysgraphia_tutor_tablet/main.dart';
import 'package:dysgraphia_tutor_tablet/widgets/session_controls.dart';
import 'package:dysgraphia_tutor_tablet/widgets/exercise_display.dart';
import 'package:dysgraphia_tutor_tablet/models/exercise.dart';

void main() {
  testWidgets('App starts on the session screen with a Child ID field', (WidgetTester tester) async {
    await tester.pumpWidget(const DysgraphiaTutorApp());
    expect(find.text('Dysgraphia Tutor'), findsOneWidget);
    expect(find.widgetWithText(TextField, 'Child ID'), findsOneWidget);
    expect(find.text('Start Session'), findsOneWidget);
  });

  testWidgets('SessionControls renders all four action buttons', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: SessionControls(
          onClear: () {},
          onUndo: () {},
          onRetry: () {},
          onSubmit: () {},
        ),
      ),
    );
    expect(find.text('Clear'), findsOneWidget);
    expect(find.text('Undo'), findsOneWidget);
    expect(find.text('Retry'), findsOneWidget);
    expect(find.text('Submit'), findsOneWidget);
  });

  testWidgets('SessionControls submit button disabled while submitting', (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: SessionControls(
          onClear: () {},
          onUndo: () {},
          onRetry: () {},
          onSubmit: () {},
          isSubmitting: true,
        ),
      ),
    );
    final button = tester.widget<ElevatedButton>(find.byType(ElevatedButton));
    expect(button.onPressed, isNull);
  });

  testWidgets('ExerciseDisplay shows the prompt and difficulty level', (WidgetTester tester) async {
    const exercise = Exercise(
      exerciseId: 'line_easy',
      prompt: 'Trace the straight line',
      difficulty: 0,
      referenceTemplate: [
        [0, 50],
        [100, 50],
      ],
    );
    await tester.pumpWidget(
      const MaterialApp(home: Scaffold(body: ExerciseDisplay(exercise: exercise))),
    );
    expect(find.text('Trace the straight line'), findsOneWidget);
    expect(find.text('Level 0'), findsOneWidget);
  });
}
