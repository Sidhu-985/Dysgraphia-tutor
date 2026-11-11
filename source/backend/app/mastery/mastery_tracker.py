"""
Per-letter mastery tracking and same-level letter rotation
=============================================================

Implements the Level 4 mastery / rotation behaviour required by the audit:

* Level 4 (and every other level) is treated as a difficulty level that
  contains MULTIPLE exercises (multiple letters), not as a single letter.
* The system tracks per-letter mastery independently: how many recent
  attempts the learner has made on that letter, what scores they earned,
  and how many consecutive successes they have.
* A learner achieves per-letter mastery only after a configurable number
  of consecutive successes at or above a configurable score threshold
  (``MasteryConfig.mastery_score_threshold`` and
  ``mastery_consecutive_successes_required``). One good score is NOT
  enough.
* When the policy chooses ``INCREASE_DIFFICULTY`` but the learner has not
  yet mastered the current letter, the intervention controller ignores
  the escalation and stays on the same letter (effectively a CONTINUE).
* When the learner has mastered the current letter AND other eligible
  same-level letters remain, the intervention controller rotates to a
  different same-level letter (the difficulty level does not change).
* Only when the learner has mastered EVERY eligible same-level letter
  does ``INCREASE_DIFFICULTY`` escalate the difficulty by one level.

This module is the single source of truth for the per-letter state. The
session context owns a ``letter_mastery`` dict; this module provides the
helpers that mutate and read it. The exercise selector and intervention
controller both go through here, so the rules can never drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import MasteryConfig, settings
from ..exercises import (
    CATALOGUE, Exercise, exercises_in, get_exercise, track_of,
)
from ..state.session_context import SessionContext


@dataclass
class LetterMastery:
    """Per-exercise mastery state, accumulated across a single session.

    Kept deliberately small and JSON-serializable so it can live inside
    :class:`SessionContext.letter_mastery` and survive
    serialization/deserialization through the session store.
    """
    exercise_id: str
    attempts: int = 0
    consecutive_successes: int = 0
    best_score: float = 0.0
    recent_scores: list[float] = field(default_factory=list)
    mastered: bool = False

    def to_dict(self) -> dict:
        return {
            "exercise_id": self.exercise_id,
            "attempts": self.attempts,
            "consecutive_successes": self.consecutive_successes,
            "best_score": self.best_score,
            "recent_scores": list(self.recent_scores),
            "mastered": self.mastered,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LetterMastery":
        return cls(
            exercise_id=d["exercise_id"],
            attempts=int(d.get("attempts", 0)),
            consecutive_successes=int(d.get("consecutive_successes", 0)),
            best_score=float(d.get("best_score", 0.0)),
            recent_scores=[float(s) for s in d.get("recent_scores", [])],
            mastered=bool(d.get("mastered", False)),
        )


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------
def _ensure_letter_mastery(value) -> LetterMastery:
    """Coerce a raw dict (from SessionContext.from_dict) or a
    LetterMastery instance into a LetterMastery. Tolerant of older
    serialized session data that predates the mastery-tracker patch."""
    if isinstance(value, LetterMastery):
        return value
    if isinstance(value, dict):
        return LetterMastery.from_dict(value)
    return LetterMastery(exercise_id=str(value) if value else "")


def record_attempt(
    session: SessionContext,
    exercise_id: str,
    quality: float,
    cfg: Optional[MasteryConfig] = None,
) -> LetterMastery:
    """Update the per-letter mastery entry for ``exercise_id`` after one
    attempt with the given normalized quality score (in [0, 1]).

    Mutates ``session.letter_mastery`` in place and returns the updated
    entry. Idempotent within the same call (calling twice would record
    the attempt twice -- callers should call exactly once per attempt).
    """
    cfg = cfg or settings.mastery
    raw = session.letter_mastery.get(exercise_id)
    entry = _ensure_letter_mastery(raw) if raw is not None else LetterMastery(exercise_id=exercise_id)
    if not entry.exercise_id:
        entry.exercise_id = exercise_id

    entry.attempts += 1
    entry.recent_scores.append(float(quality))
    # Bound the recent-score window to the larger of the consistency
    # window and the mastery requirement so the entry does not grow
    # unbounded across a long session.
    keep = max(settings.session_dynamics.consistency_window,
               cfg.mastery_consecutive_successes_required + 2)
    if len(entry.recent_scores) > keep:
        entry.recent_scores = entry.recent_scores[-keep:]
    entry.best_score = max(entry.best_score, float(quality))

    if quality >= cfg.mastery_score_threshold:
        entry.consecutive_successes += 1
    else:
        entry.consecutive_successes = 0
    # Mastery is sticky for the session: once mastered, stays mastered
    # so a single bad attempt after mastery does not un-master the
    # letter (the rotation rule has already moved on).
    if entry.consecutive_successes >= cfg.mastery_consecutive_successes_required:
        entry.mastered = True

    session.letter_mastery[exercise_id] = entry
    return entry


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def is_letter_mastered(
    session: SessionContext,
    exercise_id: str,
    cfg: Optional[MasteryConfig] = None,
) -> bool:
    """Whether the learner has demonstrated mastery of ``exercise_id`` in
    this session (sticky once achieved)."""
    cfg = cfg or settings.mastery
    raw = session.letter_mastery.get(exercise_id)
    if raw is None:
        return False
    entry = _ensure_letter_mastery(raw)
    return bool(entry.mastered)


def eligible_same_level_exercises(
    current_exercise_id: str,
    level: Optional[int] = None,
) -> list[Exercise]:
    """All exercises in the SAME track and at the SAME difficulty level
    as ``current_exercise_id``. Excludes the current exercise itself.

    The track is preserved: a lowercase letter stays lowercase; an
    uppercase letter stays uppercase; shapes and words stay on their own
    tracks. This matches the existing difficulty-action contract (see
    ``exercises.track_of``).
    """
    current = CATALOGUE.get(current_exercise_id)
    if current is None:
        return []
    if level is None:
        level = current.difficulty
    track = track_of(current_exercise_id)
    candidates = [
        e for e in exercises_in(track)
        if e.difficulty == level and e.exercise_id != current_exercise_id
    ]
    # Stable alphabetical order by target so rotation is deterministic
    # (the audit requires deterministic behaviour for identical input).
    candidates.sort(key=lambda e: (e.target or "", e.exercise_id))
    return candidates


def level_mastered_count(
    session: SessionContext,
    level: int,
    track: str,
    cfg: Optional[MasteryConfig] = None,
) -> tuple[int, int]:
    """How many exercises at (level, track) are mastered, and how many
    total exercises are at (level, track)."""
    cfg = cfg or settings.mastery
    same_level = [e for e in exercises_in(track) if e.difficulty == level]
    if not same_level:
        return (0, 0)
    mastered = sum(1 for e in same_level if is_letter_mastered(session, e.exercise_id, cfg))
    return (mastered, len(same_level))


def all_same_level_mastered(
    session: SessionContext,
    current_exercise_id: str,
    cfg: Optional[MasteryConfig] = None,
) -> bool:
    """Whether EVERY exercise at the same level and track as the current
    exercise has been mastered. Used by the intervention controller to
    decide whether INCREASE_DIFFICULTY may finally escalate the level."""
    current = CATALOGUE.get(current_exercise_id)
    if current is None:
        return False
    track = track_of(current_exercise_id)
    same_level = [e for e in exercises_in(track) if e.difficulty == current.difficulty]
    if not same_level:
        return False
    return all(is_letter_mastered(session, e.exercise_id, cfg) for e in same_level)


# ---------------------------------------------------------------------------
# Rotation / next-exercise selection
# ---------------------------------------------------------------------------
def select_next_exercise_with_rotation(
    session: SessionContext,
    current_exercise_id: str,
    *,
    current_letter_mastered: Optional[bool] = None,
    cfg: Optional[MasteryConfig] = None,
) -> tuple[str, str]:
    """Pick the next exercise id, applying the Level 4 rotation rule.

    Returns ``(next_exercise_id, reason)`` where ``reason`` is one of:

      - ``"continue_current"`` -- the current letter is not yet mastered;
        keep practising it.
      - ``"rotate_same_level"`` -- the current letter is mastered and at
        least one other same-level letter is available; rotate to it.
      - ``"all_mastered"`` -- every same-level letter is mastered;
        callers (the intervention controller) may escalate the level.
      - ``"no_alternatives"`` -- the current letter is mastered but no
        other same-level letters exist; callers should stay at this
        level (e.g. CONTINUE_NO_CHANGE) until the curriculum moves on.

    The rotation avoids the most recently practised same-level exercises
    (``rotation_avoid_recent``), recorded in
    ``session.recent_same_level_exercises``. This prevents the learner
    from bouncing between just two letters.
    """
    cfg = cfg or settings.mastery
    if current_letter_mastered is None:
        current_letter_mastered = is_letter_mastered(session, current_exercise_id, cfg)

    if not current_letter_mastered:
        # Per the spec: "If the learner has not mastered the current
        # exercise, continue targeted practice at the appropriate level."
        return current_exercise_id, "continue_current"

    # Current letter mastered: look for another eligible same-level letter.
    candidates = eligible_same_level_exercises(current_exercise_id)
    if not candidates:
        return current_exercise_id, "no_alternatives"

    # If EVERY same-level letter (including the current one) is mastered,
    # signal "all_mastered" so the intervention controller can escalate
    # the difficulty. This is the audit's requirement: "Only consider
    # moving to Level 5 when the configured Level 4 mastery requirements
    # have been satisfied."
    same_level_all = [e for e in exercises_in(track_of(current_exercise_id))
                      if e.difficulty == CATALOGUE[current_exercise_id].difficulty]
    if same_level_all and all(is_letter_mastered(session, e.exercise_id, cfg)
                              for e in same_level_all):
        return current_exercise_id, "all_mastered"

    # Avoid the N most recently practised same-level exercises (which
    # include the current one) so the rotation spreads practice.
    avoid = set(session.recent_same_level_exercises[-cfg.rotation_avoid_recent:])
    avoid.add(current_exercise_id)
    preferred = [e for e in candidates if e.exercise_id not in avoid]
    if not preferred:
        # All alternatives were recently practised; fall back to the
        # full candidate list so the learner is not stuck on the current
        # letter indefinitely.
        preferred = candidates

    # Prefer letters that have not been mastered yet, then letters with
    # the fewest attempts (so practice is spread evenly across the
    # level). Stable sort: mastered first alphabetically, then by attempts.
    preferred.sort(key=lambda e: (
        is_letter_mastered(session, e.exercise_id, cfg),
        (_ensure_letter_mastery(session.letter_mastery[e.exercise_id]).attempts
         if e.exercise_id in session.letter_mastery else 0),
        e.target or "",
    ))
    next_ex = preferred[0]
    return next_ex.exercise_id, "rotate_same_level"
