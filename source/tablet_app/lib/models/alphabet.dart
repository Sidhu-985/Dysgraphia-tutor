import 'package:flutter/foundation.dart';

import 'exercise.dart';

/// The 52 letters in alphabetical order. The app always shows letters in this
/// order; exercises come from the backend's `/exercises` list (nothing is
/// invented on the client).
const String kUppercaseLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
const String kLowercaseLetters = 'abcdefghijklmnopqrstuvwxyz';

enum LetterCase { lower, upper }

String lettersOf(LetterCase c) => c == LetterCase.upper ? kUppercaseLetters : kLowercaseLetters;

/// Child-facing name of a letter: "capital A" or "small a".
String letterName(String char) => char.toUpperCase() == char ? 'capital $char' : 'small $char';

/// Short description of where a letter sits on the lines.
String zoneDescription(LetterZone zone) {
  switch (zone) {
    case LetterZone.capital:
      return 'Capital letter: top line to baseline';
    case LetterZone.ascender:
      return 'Tall letter: reaches the top line';
    case LetterZone.xHeight:
      return 'Small letter: middle line to baseline';
    case LetterZone.descender:
      return 'Tail letter: goes below the baseline';
    case LetterZone.unknown:
      return '';
  }
}

/// Optional practice groups. A design choice for practice, not a clinical
/// grouping. Each group lists letters of both cases; a tab shows only the
/// letters of its own case.
class PracticeGroup {
  final String id;
  final String title;
  final String letters;

  const PracticeGroup(this.id, this.title, this.letters);

  List<String> lettersFor(LetterCase c) => [
        for (final ch in lettersOf(c).split(''))
          if (letters.contains(ch)) ch
      ];
}

const List<PracticeGroup> kPracticeGroups = [
  PracticeGroup('straight', 'Straight lines', 'EFHILTAKMNVWXYZiklvwxz'),
  PracticeGroup('round', 'Round letters', 'COQGSacdegoqs'),
  PracticeGroup('tall', 'Tall letters', 'bdfhklt'),
  PracticeGroup('tails', 'Tail letters', 'gjpqy'),
  PracticeGroup('humps', 'Humps and bumps', 'BDPRUhmnru'),
];

/// Letter exercises from the backend, indexed by character.
class AlphabetIndex {
  final Map<String, Exercise> byChar;

  AlphabetIndex._(this.byChar);

  /// Uses only well-formed letter records whose target matches the case the
  /// backend states, so a mislabelled record is never shown as a letter.
  factory AlphabetIndex.fromExercises(Iterable<Exercise> exercises) {
    final map = <String, Exercise>{};
    for (final e in exercises) {
      if (!e.isLetter) continue;
      final ch = e.target!;
      final isUpper = ch.toUpperCase() == ch && ch.toLowerCase() != ch;
      if ((e.isUppercase && !isUpper) || (e.isLowercase && isUpper)) continue;
      map.putIfAbsent(ch, () => e);
    }
    return AlphabetIndex._(map);
  }

  Exercise? operator [](String char) => byChar[char];

  /// The letters of one case in A-Z order; null where the backend has none.
  List<(String, Exercise?)> letters(LetterCase c) => [for (final ch in lettersOf(c).split('')) (ch, byChar[ch])];

  int count(LetterCase c) => letters(c).where((l) => l.$2 != null).length;

  bool get complete => count(LetterCase.upper) == 26 && count(LetterCase.lower) == 26;

  /// The same letter in the other case ("a" <-> "A"), if the backend has it.
  Exercise? counterpart(String char) {
    final other = char.toUpperCase() == char ? char.toLowerCase() : char.toUpperCase();
    return byChar[other];
  }
}

/// Neighbours of [current] in a practice sequence (A-Z or a group).
(String?, String?) neighbours(List<String> sequence, String current) {
  final i = sequence.indexOf(current);
  if (i < 0) return (null, null);
  return (i > 0 ? sequence[i - 1] : null, i < sequence.length - 1 ? sequence[i + 1] : null);
}

/// Which letters were practised in THIS run of the app: an attempt was
/// submitted and the backend accepted it. Kept in memory only (it is not a
/// stored learning record), so a tile shows "practised" only when that is
/// known for certain.
class PracticeLog extends ChangeNotifier {
  final Map<String, int> _attempts = {};

  void record(String exerciseId) {
    _attempts[exerciseId] = (_attempts[exerciseId] ?? 0) + 1;
    notifyListeners();
  }

  int attempts(String exerciseId) => _attempts[exerciseId] ?? 0;

  bool practised(String exerciseId) => attempts(exerciseId) > 0;

  int practisedCount(Iterable<String> exerciseIds) => exerciseIds.where(practised).length;
}
