import 'exercise.dart';

/// Child-facing instruction for the writing screen, built ONLY from the
/// current exercise record (category + target) and the practice mode the
/// backend reported. Nothing here is stored separately, so the instruction
/// cannot drift from the exercise or the guide on the canvas.
///
/// Practice mode follows the canvas rule exactly: "free_writing" hides the
/// guide (independent writing); every other value shows it (tracing).
enum PracticeMode { tracing, independent }

PracticeMode practiceModeOf(String mode) => mode == 'free_writing' ? PracticeMode.independent : PracticeMode.tracing;

/// Whether the canvas draws the tracing guide for this mode. The canvas
/// and the instruction both use this, so the wording ("Trace ..." vs
/// "Write ... on your own") always matches what is on the paper.
bool guideVisibleFor(String mode) => practiceModeOf(mode) == PracticeMode.tracing;

class ExerciseInstruction {
  /// What to write, e.g. "Trace the letter A." (shown large).
  final String headline;

  /// How and where, e.g. "Follow the guide in the writing area."
  final String guidance;

  /// Shown when the child is trying the same exercise again.
  final String? encouragement;

  /// Short name for menus and screen readers, e.g. "Letter A", "Word CAT",
  /// "Straight line". Empty when the target is unknown.
  final String label;

  /// True when category or target could not be determined and the backend's
  /// prompt (or a neutral sentence) is shown instead.
  final bool isFallback;

  const ExerciseInstruction({
    required this.headline,
    required this.guidance,
    required this.label,
    this.encouragement,
    this.isFallback = false,
  });
}

final _letterRe = RegExp(r'^[A-Za-z]$');
final _wordRe = RegExp(r'^[A-Za-z]{2,12}$');
final _shapeNameRe = RegExp(r"^[A-Za-z][A-Za-z '\-]{0,30}$");

const tryAgainText = "Try again! You've got this.";

String? _cleanLetter(String? t) {
  final v = t?.trim();
  return v != null && _letterRe.hasMatch(v) ? v : null;
}

String? _cleanWord(String? t) {
  final v = t?.trim();
  return v != null && _wordRe.hasMatch(v) ? v : null; // spelling and case kept exactly
}

String? _cleanShape(String? t) {
  final v = t?.trim().replaceAll(RegExp(r'\s+'), ' ');
  return v != null && _shapeNameRe.hasMatch(v) ? v.toLowerCase() : null;
}

/// Short label for an exercise ("Letter A", "Word CAT", "Straight line"), or
/// '' when the exercise cannot be identified.
String exerciseLabel(Exercise ex) {
  final letter = _cleanLetter(ex.target), word = _cleanWord(ex.target), shape = _cleanShape(ex.target);
  return switch (ex.category) {
    ExerciseCategory.letter => letter == null ? '' : 'Letter $letter',
    ExerciseCategory.word => word == null ? '' : 'Word $word',
    ExerciseCategory.shape => shape == null ? 'Shape' : '${shape[0].toUpperCase()}${shape.substring(1)}',
    ExerciseCategory.unknown => '',
  };
}

ExerciseInstruction instructionFor(Exercise ex, String mode, {int retryIndex = 0}) {
  final independent = practiceModeOf(mode) == PracticeMode.independent;
  final encouragement = retryIndex > 0 ? tryAgainText : null;
  const traceHow = 'Follow the guide in the writing area.';
  const ownHow = 'No guide this time. Write it in the writing area.';

  switch (ex.category) {
    case ExerciseCategory.letter:
      final c = _cleanLetter(ex.target);
      if (c != null) {
        return ExerciseInstruction(
          headline: independent ? "Let's write the letter $c!" : 'Trace the letter $c.',
          guidance: independent ? ownHow : traceHow,
          label: 'Letter $c',
          encouragement: encouragement,
        );
      }
      break; // malformed letter target: never shown as a letter
    case ExerciseCategory.word:
      final w = _cleanWord(ex.target);
      if (w != null) {
        return ExerciseInstruction(
          headline: independent ? "Let's write the word $w!" : 'Trace the word $w.',
          guidance: independent ? ownHow : traceHow,
          label: 'Word $w',
          encouragement: encouragement,
        );
      }
      break;
    case ExerciseCategory.shape:
      final s = _cleanShape(ex.target);
      return ExerciseInstruction(
        headline: s == null
            ? (independent ? 'Draw the shape in the writing area.' : 'Trace the shape carefully.')
            : (independent ? 'Draw the $s in the writing area.' : 'Trace the $s.'),
        guidance: independent ? 'No guide this time. You can do it!' : traceHow,
        label: exerciseLabel(ex),
        encouragement: encouragement,
      );
    case ExerciseCategory.unknown:
      break;
  }

  // Category or target unknown (e.g. an older backend): show the backend's
  // own sentence rather than guessing what the exercise is.
  final prompt = ex.prompt.trim();
  return ExerciseInstruction(
    headline: prompt.isNotEmpty ? prompt : 'Look at the model, then write in the writing area.',
    guidance: independent ? 'No guide this time. Write it in the writing area.' : traceHow,
    label: '',
    encouragement: encouragement,
    isFallback: true,
  );
}
