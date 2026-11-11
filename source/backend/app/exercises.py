"""
Exercise catalogue: geometric shapes, letters and words.

Every exercise states WHAT the child writes explicitly:
  category   "shape" | "letter" | "word"
  target     the learner-facing target: the shape's name ("straight line"),
             the exact character ("A", "m") or the exact word ("CAT")
  strokes    the guide, one polyline per pen stroke, in drawing order
  writing_lines  letters/words only: ascender, x-height and baseline
             (shown as lined paper so "c" and "C" look different)
  reference_template   all strokes concatenated in order (the original
             single-polyline field: kept for older clients and used by the
             static feature extractor exactly as before)
The tablet app builds its instruction ("Trace the letter A.") from category,
target and the session's practice mode, never from the prompt text. The
`prompt` field stays as a plain-language fallback for older clients.

`EXERCISES` keeps its original meaning: the five geometric shapes. The
quality-proxy perturbation suite and the discrimination script iterate over
it, so their predeclared evaluation scope is unchanged. `CATALOGUE` holds
every exercise the API serves.

The shape templates are synthetic placeholder shapes. Letter and word
guides are hand-authored (letter_templates.py) and checked automatically
against a real font, but they are NOT clinically validated, and the quality
proxy was calibrated on the shapes only (docs/EXERCISE_INSTRUCTIONS.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .letter_templates import (ASCENDER_TOP, BASELINE, DESCENDER, LOWERCASE, UPPERCASE, X_HEIGHT, flatten,
                               glyph_strokes, letter_zone, word_strokes)

CATEGORIES = ("shape", "letter", "word")


@dataclass
class Exercise:
    exercise_id: str
    prompt: str
    difficulty: int
    reference_template: list[list[float]] = field(default_factory=list)  # [[x, y], ...]
    category: str = "shape"
    target: str = ""
    strokes: list[list[list[float]]] = field(default_factory=list)
    # Letters and words: the lined-paper guide lines (template y units). The
    # app fits the guide to these lines, not to the letter's own ink, so a
    # lowercase "c" stays smaller than a capital "C" on screen. None for shapes.
    writing_lines: dict | None = None
    # Letters only (backward-compatible additions; None for shapes/words):
    # letter_case "upper" | "lower"; letter_zone "capital" | "ascender" |
    # "x_height" | "descender"; instruction: a child-facing sentence about
    # where the letter sits and how to start it.
    letter_case: str | None = None
    letter_zone: str | None = None
    instruction: str | None = None

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown exercise category {self.category!r}")
        if not self.strokes:
            self.strokes = [self.reference_template]
        if not self.reference_template:
            self.reference_template = flatten(self.strokes)

    def reference_array(self) -> np.ndarray:
        return np.array(self.reference_template, dtype=np.float64)


def _line_template(n: int = 20) -> list[list[float]]:
    xs = np.linspace(0.0, 100.0, n)
    ys = np.full(n, 50.0)
    return np.stack([xs, ys], axis=1).tolist()


def _loop_template(n: int = 40) -> list[list[float]]:
    theta = np.linspace(0, 2 * np.pi, n)
    xs = 50 + 40 * np.sin(theta)
    ys = 50 + 40 * (1 - np.cos(theta))
    return np.stack([xs, ys], axis=1).tolist()


def _zigzag_template(n: int = 30) -> list[list[float]]:
    xs = np.linspace(0.0, 100.0, n)
    ys = 50 + 25 * np.sign(np.sin(xs / 100 * 6 * np.pi))
    return np.stack([xs, ys], axis=1).tolist()


def _wave_template(n: int = 40) -> list[list[float]]:
    xs = np.linspace(0.0, 100.0, n)
    ys = 50 + 15 * np.sin(xs / 100 * 2 * np.pi)
    return np.stack([xs, ys], axis=1).tolist()


def _figure_eight_template(n: int = 60) -> list[list[float]]:
    theta = np.linspace(0, 2 * np.pi, n)
    xs = 50 + 30 * np.sin(2 * theta)
    ys = 50 - 40 * np.sin(theta)
    return np.stack([xs, ys], axis=1).tolist()


# One exercise per difficulty level 0-4, so every DECREASE/INCREASE
# action changes what the child sees (previously only levels 0, 2 and 4
# existed, so a +1 step from 2 mapped back to the same exercise).
# Unchanged except for the explicit category/target (exercise ids, prompts,
# levels and templates are exactly as before).
EXERCISES: dict[str, Exercise] = {
    "line_easy": Exercise("line_easy", "Trace the straight line", 0, _line_template(),
                          category="shape", target="straight line"),
    "wave_easy": Exercise("wave_easy", "Trace the gentle wave", 1, _wave_template(),
                          category="shape", target="gentle wave"),
    "loop_medium": Exercise("loop_medium", "Trace the loop (like the letter 'o')", 2, _loop_template(),
                            category="shape", target="loop"),
    "eight_medium": Exercise("eight_medium", "Trace the figure eight", 3, _figure_eight_template(),
                             category="shape", target="figure eight"),
    "zigzag_hard": Exercise("zigzag_hard", "Trace the zigzag pattern", 4, _zigzag_template(),
                            category="shape", target="zigzag"),
}


LINES = {"ascender": ASCENDER_TOP, "x_height": X_HEIGHT, "baseline": BASELINE, "descender": DESCENDER}

_ZONE_SENTENCE = {
    "capital": "It is a capital letter: it goes from the top line down to the baseline.",
    "ascender": "It is a tall letter: it starts at the top line and sits on the baseline.",
    "x_height": "It is a small letter: it fits between the middle line and the baseline.",
    "descender": "It has a tail: the tail goes below the baseline.",
}


def letter_instruction(char: str) -> str:
    n = len(glyph_strokes(char))
    strokes = "one stroke" if n == 1 else f"{n} strokes"
    name = f"capital {char}" if char.isupper() else f"small {char}"
    return (f"This is {name}. {_ZONE_SENTENCE[letter_zone(char)]} "
            f"Write it in {strokes}: start at dot 1 and follow the arrows.")


def _letter(char: str, level: int) -> Exercise:
    kind = "upper" if char.isupper() else "lower"
    return Exercise(f"letter_{kind}_{char.lower()}", f"Write the letter {char}", level,
                    category="letter", target=char, strokes=glyph_strokes(char), writing_lines=dict(LINES),
                    letter_case=kind, letter_zone=letter_zone(char), instruction=letter_instruction(char))


def _word(word: str, level: int) -> Exercise:
    return Exercise(f"word_{word.lower()}", f"Write the word {word}", level,
                    category="word", target=word, strokes=word_strokes(word), writing_lines=dict(LINES))


# Letter ladder (levels are a design choice, not a clinical scale):
#   0 one straight stroke (plus a dot)
#   1 several straight strokes, only up-down and across
#   2 one curve
#   3 diagonals, several strokes, a change of direction, or a tail
#   4 a curve joined to a stem, humps, or several curves
# The 14 letters that existed before keep their levels. Within a level the
# order is alphabetical. A difficulty step stays in the same CASE: a child
# practising lowercase letters is never switched to a capital (and vice versa).
LETTER_LEVELS: dict[str, int] = {
    **{c: 0 for c in "Lli"},
    **{c: 1 for c in "FHITt"},
    **{c: 2 for c in "COco"},
    **{c: 3 for c in "ADEJKMNPSUVWXYZejksuvwxyz"},
    **{c: 4 for c in "BGQRabdfghmnpqr"},
}
assert sorted(LETTER_LEVELS) == sorted(UPPERCASE + LOWERCASE)

LETTER_EXERCISES: dict[str, Exercise] = {e.exercise_id: e for e in (
    [_letter(c, LETTER_LEVELS[c]) for c in UPPERCASE] + [_letter(c, LETTER_LEVELS[c]) for c in LOWERCASE]
)}
UPPERCASE_IDS: list[str] = [f"letter_upper_{c.lower()}" for c in UPPERCASE]
LOWERCASE_IDS: list[str] = [f"letter_lower_{c}" for c in LOWERCASE]

WORD_EXERCISES: dict[str, Exercise] = {e.exercise_id: e for e in (
    _word("TO", 1), _word("CAT", 2), _word("SUN", 3),
)}

CATALOGUE: dict[str, Exercise] = {**EXERCISES, **LETTER_EXERCISES, **WORD_EXERCISES}
assert len(CATALOGUE) == len(EXERCISES) + len(LETTER_EXERCISES) + len(WORD_EXERCISES), "duplicate id"

DEFAULT_EXERCISE_ID = "line_easy"


def get_exercise(exercise_id: str) -> Exercise:
    if exercise_id not in CATALOGUE:
        raise KeyError(f"Unknown exercise_id: {exercise_id}")
    return CATALOGUE[exercise_id]


def category_of(exercise_id: str | None) -> str:
    """Category of an exercise id; "shape" for None/unknown (the original track)."""
    ex = CATALOGUE.get(exercise_id or "")
    return ex.category if ex is not None else "shape"


TRACKS = ("shape", "word", "letter_upper", "letter_lower")


def track_of(exercise_id: str | None) -> str:
    """Difficulty track of an exercise: its category, with letters split by
    case. Difficulty actions move along one track only."""
    ex = CATALOGUE.get(exercise_id or "")
    if ex is None:
        return "shape"
    if ex.category == "letter":
        return f"letter_{ex.letter_case}"
    return ex.category


def exercises_in(category: str = "shape") -> list[Exercise]:
    """Exercises of a category ("shape", "letter", "word") or of a track
    ("letter_upper", "letter_lower")."""
    if category in ("letter_upper", "letter_lower"):
        case = category.split("_")[1]
        return [e for e in CATALOGUE.values() if e.category == "letter" and e.letter_case == case]
    return [e for e in CATALOGUE.values() if e.category == category]


def _ordered(category: str) -> list[Exercise]:
    exs = exercises_in(category)
    if category.startswith("letter"):
        exs = sorted(exs, key=lambda e: e.target)          # alphabetical within a level
    return sorted(exs, key=lambda e: e.difficulty)         # stable


def exercise_for_difficulty(difficulty_level: int, category: str = "shape") -> Exercise:
    """Exercise for a difficulty level WITHIN one category or track, so a
    difficulty action never turns a letter exercise into a shape (or the
    reverse), nor a lowercase letter into a capital. With the default
    category this is exactly the original behaviour (the five shapes, one
    per level)."""
    return min(_ordered(category), key=lambda e: abs(e.difficulty - difficulty_level))


def levels_in(category: str) -> list[int]:
    return sorted({e.difficulty for e in exercises_in(category)})
