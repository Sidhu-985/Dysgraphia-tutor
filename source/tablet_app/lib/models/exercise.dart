/// Mirrors the backend's `ExerciseOut` schema. The UI renders the guide
/// strokes and builds the child-facing instruction from `category`,
/// `target` and the session's practice mode (see exercise_instruction.dart).
/// It does not interpret or score the template (the backend does that).
///
/// `category`, `target` and `templateStrokes` were added for the
/// instruction feature. They are optional: an older backend that does not
/// send them yields [ExerciseCategory.unknown], and the guide falls back to
/// the single `referenceTemplate` polyline.
enum ExerciseCategory { shape, letter, word, unknown }

ExerciseCategory parseExerciseCategory(Object? raw) {
  switch (raw) {
    case 'shape':
      return ExerciseCategory.shape;
    case 'letter':
      return ExerciseCategory.letter;
    case 'word':
      return ExerciseCategory.word;
    default:
      return ExerciseCategory.unknown; // missing or unsupported: never guessed
  }
}

/// Lined-paper guide lines for letters and words, in template units
/// (y grows downward): ascender (top of capitals and tall letters), x-height
/// (top of short lowercase letters), baseline and, from backends that send
/// it, the descender line (bottom of the tails of g, j, p, q, y).
class WritingLines {
  final double ascender;
  final double xHeight;
  final double baseline;

  /// Null for older backends that send only three lines.
  final double? descender;

  const WritingLines({required this.ascender, required this.xHeight, required this.baseline, this.descender});

  /// Null unless the first three are finite numbers in the order
  /// ascender < x-height < baseline. A descender that is missing, not finite
  /// or not below the baseline is ignored (older backends do not send it).
  static WritingLines? tryParse(Object? raw) {
    if (raw is! Map) return null;
    final a = raw['ascender'], x = raw['x_height'], b = raw['baseline'], d = raw['descender'];
    if (a is! num || x is! num || b is! num) return null;
    final dv = d is num && d.toDouble().isFinite && d.toDouble() > b.toDouble() ? d.toDouble() : null;
    final lines = WritingLines(ascender: a.toDouble(), xHeight: x.toDouble(), baseline: b.toDouble(), descender: dv);
    final ok = [lines.ascender, lines.xHeight, lines.baseline].every((v) => v.isFinite) &&
        lines.ascender < lines.xHeight &&
        lines.xHeight < lines.baseline;
    return ok ? lines : null;
  }

  /// Every line to draw, top to bottom.
  List<double> get all => [ascender, xHeight, baseline, if (descender != null) descender!];

  /// The lowest line used to size the guide (descender when known).
  double get bottom => descender ?? baseline;
}

/// Where a letter sits on the lines (sent by the backend for letters).
enum LetterZone { capital, ascender, xHeight, descender, unknown }

LetterZone parseLetterZone(Object? raw) {
  switch (raw) {
    case 'capital':
      return LetterZone.capital;
    case 'ascender':
      return LetterZone.ascender;
    case 'x_height':
      return LetterZone.xHeight;
    case 'descender':
      return LetterZone.descender;
    default:
      return LetterZone.unknown;
  }
}

class Exercise {
  final String exerciseId;
  final String prompt;
  final int difficulty;
  final List<List<double>> referenceTemplate; // [[x, y], ...]
  final ExerciseCategory category;
  final String? target;

  /// The guide as separate pen strokes, in drawing order. Empty when the
  /// backend did not send them.
  final List<List<List<double>>> templateStrokes;

  /// Letters and words only. The guide is sized against these lines rather
  /// than its own ink, so a lowercase "c" is drawn smaller than a capital "C".
  final WritingLines? writingLines;

  /// Letters only: "upper" or "lower" (null for shapes, words, old backends).
  final String? letterCase;

  /// Letters only: where the letter sits on the lines.
  final LetterZone letterZone;

  /// Letters only: the backend's child-facing sentence about the letter.
  final String? letterInstruction;

  bool get isLetter => category == ExerciseCategory.letter && target != null && target!.length == 1;
  bool get isUppercase => letterCase == 'upper';
  bool get isLowercase => letterCase == 'lower';

  const Exercise({
    required this.exerciseId,
    required this.prompt,
    required this.difficulty,
    required this.referenceTemplate,
    this.category = ExerciseCategory.unknown,
    this.target,
    this.templateStrokes = const [],
    this.writingLines,
    this.letterCase,
    this.letterZone = LetterZone.unknown,
    this.letterInstruction,
  });

  factory Exercise.fromJson(Map<String, dynamic> json) {
    return Exercise(
      exerciseId: json['exercise_id'] as String,
      prompt: json['prompt'] as String,
      difficulty: json['difficulty'] as int,
      referenceTemplate: _points(json['reference_template']) ?? const [],
      category: parseExerciseCategory(json['category']),
      target: json['target'] is String ? json['target'] as String : null,
      templateStrokes: _strokes(json['template_strokes']),
      writingLines: WritingLines.tryParse(json['writing_lines']),
      letterCase:
          (json['letter_case'] == 'upper' || json['letter_case'] == 'lower') ? json['letter_case'] as String : null,
      letterZone: parseLetterZone(json['letter_zone']),
      letterInstruction: json['instruction'] is String ? json['instruction'] as String : null,
    );
  }

  /// Strokes to draw as the guide. The backend's `template_strokes` when they
  /// are well formed (every stroke has at least 2 finite points); otherwise
  /// the single `referenceTemplate` polyline (older backends); otherwise
  /// nothing. Both the instruction and the guide come from this one record,
  /// so they cannot describe different exercises.
  List<List<List<double>>> get guideStrokes {
    if (templateStrokes.isNotEmpty) return templateStrokes;
    if (referenceTemplate.length >= 2) return [referenceTemplate];
    return const [];
  }

  /// All guide points in drawing order.
  List<List<double>> get guidePoints => [for (final s in guideStrokes) ...s];

  static List<List<double>>? _points(Object? raw) {
    if (raw is! List) return null;
    final out = <List<double>>[];
    for (final p in raw) {
      if (p is! List || p.length < 2 || p[0] is! num || p[1] is! num) return null;
      final x = (p[0] as num).toDouble(), y = (p[1] as num).toDouble();
      if (!x.isFinite || !y.isFinite) return null;
      out.add([x, y]);
    }
    return out;
  }

  static List<List<List<double>>> _strokes(Object? raw) {
    if (raw is! List || raw.isEmpty) return const [];
    final out = <List<List<double>>>[];
    for (final s in raw) {
      final pts = _points(s);
      if (pts == null || pts.length < 2) return const []; // malformed: use referenceTemplate instead
      out.add(pts);
    }
    return out;
  }
}
