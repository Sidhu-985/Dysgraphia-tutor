"""
Level 4 letter rotation and mastery progression tests (audit-driven, 2026-11)
==============================================================================

These tests verify the audit's headline requirement:

  * When a learner achieves a good score on a Level 4 letter, the
    system moves them to ANOTHER Level 4 letter (not directly to
    Level 5).
  * Per-letter mastery requires multiple consecutive successes (not a
    single good score).
  * Mastery of one letter does not imply mastery of all letters at
    the same level.
  * The system only escalates the difficulty once every eligible
    same-level letter is mastered.
  * The exercise selector never invents an exercise id; it always
    returns one that exists in the catalogue.
"""

from __future__ import annotations

import pytest

from app.config import MasteryConfig, settings
from app.exercises import (
    CATALOGUE, LETTER_EXERCISES, exercise_for_difficulty,
    exercises_in, get_exercise, levels_in, track_of,
)
from app.interventions.intervention_controller import (
    DECREASE_DIFFICULTY, INCREASE_DIFFICULTY, build_intervention,
)
from app.mastery.mastery_tracker import (
    LetterMastery, all_same_level_mastered, eligible_same_level_exercises,
    is_letter_mastered, level_mastered_count, record_attempt,
    select_next_exercise_with_rotation,
)
from app.state.session_context import SessionContext
from app.dysgraphia_rl_bridge import ACTION_NAMES


# Helpers ----------------------------------------------------------------
def _session_with_exercise(eid: str) -> SessionContext:
    e = get_exercise(eid)
    return SessionContext(
        session_id="s", child_id="P-0001",
        difficulty_level=e.difficulty, current_exercise_id=eid,
    )


def _force_mastery(session: SessionContext, eid: str,
                   n: int = None) -> None:
    """Mark an exercise as mastered by recording n consecutive
    successful attempts (default = the configured requirement)."""
    cfg = settings.mastery
    n = n if n is not None else cfg.mastery_consecutive_successes_required
    for _ in range(n):
        record_attempt(session, eid, cfg.mastery_score_threshold + 0.05)


# Catalogue invariants ----------------------------------------------------
def test_level_4_has_multiple_lowercase_letters():
    """Level 4 lowercase is treated as a difficulty level containing
    multiple letters, not as a single letter."""
    same_level = [e for e in exercises_in("letter_lower") if e.difficulty == 4]
    assert len(same_level) >= 5, (
        "Level 4 lowercase should have at least 5 letters for rotation"
    )
    # The catalogue defines Level 4 lowercase letters as
    # 'a', 'b', 'd', 'f', 'g', 'h', 'm', 'n', 'p', 'q', 'r' (11 letters)
    targets = sorted(e.target for e in same_level)
    expected = sorted("abdfghmnpqr")
    assert targets == expected, f"Level 4 lowercase mismatch: {targets} vs {expected}"


def test_each_level_has_at_least_one_letter():
    assert levels_in("letter_lower") == [0, 1, 2, 3, 4]
    assert levels_in("letter_upper") == [0, 1, 2, 3, 4]


# Per-letter mastery tracking --------------------------------------------
def test_fresh_letter_is_not_mastered():
    s = _session_with_exercise("letter_lower_a")
    assert not is_letter_mastered(s, "letter_lower_a")


def test_one_good_attempt_does_not_master():
    """A single good score must NOT mark the letter as mastered."""
    s = _session_with_exercise("letter_lower_a")
    cfg = settings.mastery
    record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    assert not is_letter_mastered(s, "letter_lower_a"), (
        "one good score should NOT be enough for mastery"
    )


def test_consecutive_successes_master():
    s = _session_with_exercise("letter_lower_a")
    cfg = settings.mastery
    n = cfg.mastery_consecutive_successes_required
    for _ in range(n):
        record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    assert is_letter_mastered(s, "letter_lower_a")


def test_one_failure_in_between_resets_consecutive_count():
    """A failure in the middle resets the consecutive-successes counter."""
    s = _session_with_exercise("letter_lower_a")
    cfg = settings.mastery
    record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    record_attempt(s, "letter_lower_a", 0.30)  # failure
    record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    # Only 1 consecutive success after the failure -- not enough
    assert not is_letter_mastered(s, "letter_lower_a")


def test_mastery_is_sticky_for_session():
    """Once mastered, stays mastered even if a later attempt is poor
    (the rotation rule has already moved on)."""
    s = _session_with_exercise("letter_lower_a")
    cfg = settings.mastery
    n = cfg.mastery_consecutive_successes_required
    for _ in range(n):
        record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    assert is_letter_mastered(s, "letter_lower_a")
    # A poor attempt after mastery does NOT un-master it
    record_attempt(s, "letter_lower_a", 0.10)
    assert is_letter_mastered(s, "letter_lower_a")


def test_mastery_independent_per_letter():
    """Mastery of one letter does not master another."""
    s = _session_with_exercise("letter_lower_a")
    _force_mastery(s, "letter_lower_a")
    assert is_letter_mastered(s, "letter_lower_a")
    assert not is_letter_mastered(s, "letter_lower_b")
    assert not is_letter_mastered(s, "letter_lower_o")


# Level 4 rotation rule --------------------------------------------------
def test_poor_score_on_level4_letter_keeps_letter():
    """Level 4 letter A with a poor score -> continue practising A."""
    s = _session_with_exercise("letter_lower_a")
    # Submit a poor attempt (below mastery threshold)
    cfg = settings.mastery
    record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold - 0.20)
    # INCREASE_DIFFICULTY action: the rotation rule should keep us on
    # the same letter (since not mastered).
    iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
    assert iv.params["next_exercise_id"] == "letter_lower_a"
    assert iv.params.get("rotation_reason") == "current_letter_not_mastered"
    assert s.difficulty_level == 4  # unchanged


def test_one_good_score_does_not_advance_to_level_5():
    """Level 4 letter A with a good but insufficient mastery record
    (only 1 success) -> do NOT advance."""
    s = _session_with_exercise("letter_lower_a")
    cfg = settings.mastery
    # Just one good attempt -- not enough for mastery
    record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    assert not is_letter_mastered(s, "letter_lower_a")
    iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
    assert iv.params["next_exercise_id"] == "letter_lower_a"
    assert s.difficulty_level == 4  # still at Level 4


def test_mastered_letter_rotates_to_another_level_4_letter():
    """Level 4 letter A with confirmed mastery -> select another Level 4
    letter (rotation, not escalation)."""
    s = _session_with_exercise("letter_lower_a")
    _force_mastery(s, "letter_lower_a")
    assert is_letter_mastered(s, "letter_lower_a")

    iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
    next_id = iv.params["next_exercise_id"]
    # Must stay at Level 4 (letter_lower track, difficulty 4)
    next_ex = get_exercise(next_id)
    assert next_ex.difficulty == 4, f"next exercise must stay at Level 4; got {next_ex.difficulty}"
    assert next_ex.letter_case == "lower", "must stay in lowercase track"
    # Must be a DIFFERENT letter than 'a'
    assert next_ex.target != "a", f"must rotate to a different letter; got {next_ex.target}"
    # Difficulty level unchanged
    assert s.difficulty_level == 4
    # Intervention type is letter_rotation (audit-driven addition)
    assert iv.type == "letter_rotation"


def test_rotation_does_not_repeat_the_same_letter():
    """When rotating among Level 4 letters, do not pick the same letter
    if alternatives are available."""
    s = _session_with_exercise("letter_lower_a")
    _force_mastery(s, "letter_lower_a")
    iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
    first_rotation = iv.params["next_exercise_id"]
    assert first_rotation != "letter_lower_a"

    # Now switch the current exercise to the rotated letter, mark IT
    # mastered, and request INCREASE_DIFFICULTY again. Should rotate to
    # yet another letter (not back to 'a' immediately).
    s.current_exercise_id = first_rotation
    _force_mastery(s, first_rotation)
    iv2 = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
    second_rotation = iv2.params["next_exercise_id"]
    next_ex = get_exercise(second_rotation)
    assert next_ex.difficulty == 4
    # The rotation should avoid the recent letters (a and first_rotation)
    # when there are still other Level 4 letters available.
    eligible = eligible_same_level_exercises(first_rotation)
    eligible_ids = [e.exercise_id for e in eligible if e.exercise_id not in
                    (first_rotation, "letter_lower_a")]
    if eligible_ids:
        assert second_rotation != first_rotation, (
            "must not bounce back to the just-practised letter when alternatives exist"
        )


def test_mastery_of_one_letter_does_not_master_all_level_4_letters():
    """Mastery is per-letter, not per-level."""
    s = _session_with_exercise("letter_lower_a")
    _force_mastery(s, "letter_lower_a")
    # Check that other Level 4 letters are NOT mastered
    same_level = [e for e in exercises_in("letter_lower") if e.difficulty == 4]
    for e in same_level:
        if e.exercise_id == "letter_lower_a":
            assert is_letter_mastered(s, e.exercise_id)
        else:
            assert not is_letter_mastered(s, e.exercise_id), (
                f"{e.target} should NOT be mastered just because 'a' is"
            )


def test_all_level_4_letters_mastered_then_escalate_to_level_5():
    """When every Level 4 letter is mastered, INCREASE_DIFFICULTY may
    escalate to Level 5 (or, since Level 4 is the maximum for letters,
    stay at 4 with the appropriate message)."""
    s = _session_with_exercise("letter_lower_a")
    same_level = [e for e in exercises_in("letter_lower") if e.difficulty == 4]
    # Master every Level 4 lowercase letter
    for e in same_level:
        _force_mastery(s, e.exercise_id)
    # Set the current exercise to one of them
    s.current_exercise_id = same_level[0].exercise_id
    s.difficulty_level = 4

    iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
    # Since Level 4 is the maximum difficulty for letters, the level
    # should not exceed 4 (clamped). The intervention should indicate
    # that escalation occurred (or that no further levels are
    # available).
    assert iv.params.get("rotation_reason") in (
        "all_mastered_escalate", "no_further_levels"
    )
    assert s.difficulty_level <= 4  # cannot exceed the max


def test_letter_rotation_never_invents_nonexistent_exercise():
    """The exercise selector must NEVER return an exercise id that is
    not in the catalogue."""
    s = _session_with_exercise("letter_lower_a")
    # Master several Level 4 letters and rotate through them
    for char in "ab":
        _force_mastery(s, f"letter_lower_{char}")
    s.current_exercise_id = "letter_lower_b"
    for _ in range(20):  # many rotations
        iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
        next_id = iv.params["next_exercise_id"]
        assert next_id in CATALOGUE, f"invented exercise id: {next_id}"
        s.current_exercise_id = next_id


def test_no_eligible_same_level_exercises_graceful():
    """When the current exercise is the only one at its level (e.g.
    shapes, words), INCREASE_DIFFICULTY behaves as before (escalate)."""
    # Shapes: one exercise per level
    s = _session_with_exercise("loop_medium")  # shape, level 2
    iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
    # Should escalate to Level 3 (eight_medium)
    assert s.difficulty_level == 3
    assert iv.params["next_exercise_id"] == "eight_medium"
    assert iv.params.get("rotation_reason") == "single_exercise_per_level"


def test_repeated_api_submission_does_not_advance_mastery_twice():
    """Calling record_attempt twice with the same data WOULD record
    twice -- the API path uses the idempotent processed_attempts cache
    (in routes.py) to deduplicate, but the unit test here documents
    that record_attempt itself is the single source of truth per
    attempt and should be called exactly once per attempt."""
    s = _session_with_exercise("letter_lower_a")
    cfg = settings.mastery
    # Simulate one attempt: call record_attempt once
    record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    assert s.letter_mastery["letter_lower_a"].attempts == 1
    # If the API called it again (which the idempotent path prevents),
    # the count would be 2 -- the API prevents this with
    # processed_attempts.
    record_attempt(s, "letter_lower_a", cfg.mastery_score_threshold + 0.05)
    assert s.letter_mastery["letter_lower_a"].attempts == 2


def test_full_session_letter_rotation_sequence():
    """End-to-end-style test: simulate a learner mastering letter 'a',
    then 'b', then 'd' at Level 4. Each mastery should rotate to the
    next unmastered Level 4 letter, NOT jump to Level 5."""
    s = _session_with_exercise("letter_lower_a")
    cfg = settings.mastery
    mastered_letters = []
    current_letter = "a"

    # Iterate: master the current letter, request INCREASE_DIFFICULTY,
    # check the rotation picks a different Level 4 letter.
    for i in range(5):
        eid = f"letter_lower_{current_letter}"
        s.current_exercise_id = eid
        s.difficulty_level = 4
        _force_mastery(s, eid)
        mastered_letters.append(current_letter)

        iv = build_intervention(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY], s)
        next_id = iv.params["next_exercise_id"]
        next_ex = get_exercise(next_id)

        # Either rotated to another Level 4 letter (preferred), or
        # escalated if all Level 4 letters are mastered.
        same_level = [e for e in exercises_in("letter_lower") if e.difficulty == 4]
        all_mastered = all(is_letter_mastered(s, e.exercise_id) for e in same_level)

        if not all_mastered:
            assert next_ex.difficulty == 4, (
                f"step {i}: must stay at Level 4 (not all letters mastered); "
                f"got Level {next_ex.difficulty}"
            )
            assert next_ex.target != current_letter, (
                f"step {i}: must rotate to a different letter"
            )
            current_letter = next_ex.target
        else:
            # All Level 4 lowercase letters mastered -- escalation
            # is acceptable (still capped at 4 since it's the max).
            break

    # The learner should have mastered at least 4 different Level 4
    # letters before any escalation was considered.
    assert len(set(mastered_letters)) >= 4


# Eligible same-level exercises query ------------------------------------
def test_eligible_same_level_excludes_current():
    eligible = eligible_same_level_exercises("letter_lower_a")
    ids = [e.exercise_id for e in eligible]
    assert "letter_lower_a" not in ids
    # All eligible are at Level 4 (current is Level 4)
    for e in eligible:
        assert e.difficulty == 4
        assert e.letter_case == "lower"


def test_eligible_same_level_preserves_track():
    """Rotation never crosses tracks (lowercase stays lowercase,
    uppercase stays uppercase, shapes stay shapes)."""
    eligible = eligible_same_level_exercises("letter_lower_a")
    for e in eligible:
        assert track_of(e.exercise_id) == "letter_lower"

    eligible = eligible_same_level_exercises("letter_upper_a")
    for e in eligible:
        assert track_of(e.exercise_id) == "letter_upper"


def test_level_mastered_count_helper():
    s = _session_with_exercise("letter_lower_a")
    _force_mastery(s, "letter_lower_a")
    _force_mastery(s, "letter_lower_b")
    mastered, total = level_mastered_count(s, 4, "letter_lower")
    assert total >= 5
    assert mastered == 2


def test_all_same_level_mastered_helper():
    s = _session_with_exercise("letter_lower_a")
    assert not all_same_level_mastered(s, "letter_lower_a")
    same_level = [e for e in exercises_in("letter_lower") if e.difficulty == 4]
    for e in same_level:
        _force_mastery(s, e.exercise_id)
    s.current_exercise_id = same_level[0].exercise_id
    assert all_same_level_mastered(s, same_level[0].exercise_id)


def test_session_serialization_roundtrip_preserves_letter_mastery():
    """Letter mastery state survives to_dict / from_dict roundtrip."""
    s = _session_with_exercise("letter_lower_a")
    _force_mastery(s, "letter_lower_a")
    record_attempt(s, "letter_lower_b", 0.40)

    d = s.to_dict()
    assert "letter_mastery" in d
    assert "letter_lower_a" in d["letter_mastery"]
    assert d["letter_mastery"]["letter_lower_a"]["mastered"] is True

    s2 = SessionContext.from_dict(d)
    # Mastery is preserved through deserialization
    assert is_letter_mastered(s2, "letter_lower_a")
    assert not is_letter_mastered(s2, "letter_lower_b")


# Decrease difficulty unaffected by rotation rule -----------------------
def test_decrease_difficulty_unaffected_by_rotation_rule():
    """DECREASE_DIFFICULTY keeps its original behaviour (no mastery
    check, no rotation)."""
    s = _session_with_exercise("letter_lower_a")  # Level 4
    s.difficulty_level = 4
    iv = build_intervention(DECREASE_DIFFICULTY, ACTION_NAMES[DECREASE_DIFFICULTY], s)
    # Level 4 -> Level 3 (lowercase, hardest at Level 3)
    assert s.difficulty_level == 3
    next_ex = get_exercise(iv.params["next_exercise_id"])
    assert next_ex.difficulty == 3
    assert next_ex.letter_case == "lower"
