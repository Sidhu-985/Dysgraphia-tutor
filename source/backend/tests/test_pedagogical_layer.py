"""
Pedagogical layer tests (audit-driven, 2026-11)
================================================

These tests verify the new pedagogical layer that sits between the
RL policy decision and the safety layer. The layer enforces three
session-level tutoring rules:

  1. Action diversity safeguard — if the policy picks the same
     action K times in the last N decisions, override to a
     pedagogically reasonable alternative.
  2. Repeated-failure de-escalation — if the learner has had >= N
     failed attempts at the current Level 4 letter, override to
     DECREASE_DIFFICULTY.
  3. Post-break progression — if the previous action was
     SUGGEST_BREAK and the current attempt's quality_score >=
     mastery threshold, override to INCREASE_DIFFICULTY (which the
     Level 4 rotation rule turns into a same-level rotation if
     other letters remain).

All tests use SYNTHETIC test traffic. No real-child data.
"""

from __future__ import annotations

import pytest

from app.config import PedagogicalLayerConfig, settings
from app.dysgraphia_rl_bridge import ACTION_NAMES
from app.rl_integration.pedagogical_layer import (
    CONTINUE_NO_CHANGE, DECREASE_DIFFICULTY, INCREASE_DIFFICULTY,
    SHOW_HINT, REPLAY_STROKE_ANIMATION, ENLARGE_TEMPLATE,
    SUGGEST_BREAK, SWITCH_TO_FREE_WRITING_MODE, SWITCH_TO_TRACING_MODE,
    PedagogicalDecision, apply_pedagogical_layer,
)
from app.state.session_context import SessionContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _session(current_eid: str = "letter_lower_a",
             difficulty: int = 4,
             last_action: int | None = None,
             action_log: list[int] | None = None,
             retries_on_current_exercise: int = 0) -> SessionContext:
    s = SessionContext(session_id="s", child_id="P-0001",
                       current_exercise_id=current_eid,
                       difficulty_level=difficulty,
                       last_action=last_action,
                       retries_on_current_exercise=retries_on_current_exercise)
    s.action_log = list(action_log or [])
    return s


def _features(prev_score: float = 0.5, fatigue: float = 0.0,
              frustration: float = 0.0) -> dict:
    return {"prev_score": prev_score, "fatigue_proxy": fatigue,
            "frustration": frustration}


# ---------------------------------------------------------------------------
# Action diversity safeguard
# ---------------------------------------------------------------------------
def test_diversity_safeguard_overrides_repeated_same_action():
    """If the policy has picked the same action K times in the last N
    decisions and is about to pick it again, override to a different
    pedagogically reasonable action."""
    cfg = PedagogicalLayerConfig(
        enable_diversity_safeguard=True,
        diversity_window=4, diversity_repeat_threshold=2,
    )
    # Session has 2 INCREASE_DIFFICULTY in the last 4. Policy proposes it again.
    s = _session(action_log=[INCREASE_DIFFICULTY, INCREASE_DIFFICULTY, INCREASE_DIFFICULTY])
    s.last_action = INCREASE_DIFFICULTY
    feat = _features(prev_score=0.95)  # learner is doing well
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert decision.overridden
    assert decision.shaped_action_id != INCREASE_DIFFICULTY
    assert decision.rule == "diversity"


def test_diversity_safeguard_does_not_fire_below_threshold():
    """If the policy has only picked the action once in the last N
    decisions, no override."""
    cfg = PedagogicalLayerConfig(
        enable_diversity_safeguard=True,
        diversity_window=4, diversity_repeat_threshold=2,
    )
    s = _session(action_log=[INCREASE_DIFFICULTY, SHOW_HINT])
    s.last_action = SHOW_HINT
    feat = _features(prev_score=0.95)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    # Only 1 INCREASE_DIFFICULTY in the last 4 — below threshold
    assert not decision.overridden


def test_diversity_safeguard_can_be_disabled():
    """When the safeguard is disabled, the layer does not override
    even if the same action is repeated."""
    cfg = PedagogicalLayerConfig(enable_diversity_safeguard=False)
    s = _session(action_log=[INCREASE_DIFFICULTY] * 5)
    s.last_action = INCREASE_DIFFICULTY
    feat = _features(prev_score=0.95)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert not decision.overridden


def test_diversity_safeguard_does_not_override_break_when_fatigue_high():
    """The diversity safeguard should NOT override SUGGEST_BREAK when
    fatigue is high — the safety layer will handle that case (the
    learner needs a break, not a different action).

    Note: this test uses a session where the previous action was NOT
    SUGGEST_BREAK (so the higher-priority post-break rule doesn't
    fire) and the policy is repeatedly picking SUGGEST_BREAK with
    high fatigue. The diversity safeguard should leave it alone.
    """
    cfg = PedagogicalLayerConfig(
        enable_diversity_safeguard=True,
        enable_post_break_progression=True,  # post-break rule must not fire here
        diversity_window=4, diversity_repeat_threshold=2,
    )
    # Previous action was SHOW_HINT (not SUGGEST_BREAK), so the post-break
    # rule won't fire (it checks session.last_action == SUGGEST_BREAK).
    # The diversity safeguard is the only relevant rule.
    s = _session(action_log=[SHOW_HINT, SUGGEST_BREAK, SUGGEST_BREAK, SUGGEST_BREAK],
                 last_action=SHOW_HINT)  # previous action was SHOW_HINT, not SUGGEST_BREAK
    feat = _features(prev_score=0.95, fatigue=0.7)  # high fatigue
    decision = apply_pedagogical_layer(SUGGEST_BREAK, s, feat, cfg=cfg)
    assert not decision.overridden, "should not override SUGGEST_BREAK when fatigue is high"


# ---------------------------------------------------------------------------
# Repeated-failure de-escalation
# ---------------------------------------------------------------------------
def test_deescalation_fires_after_max_attempts_at_letter():
    """After max_attempts_at_letter (default 4) failed attempts at the
    current Level 4 letter, override to DECREASE_DIFFICULTY."""
    cfg = PedagogicalLayerConfig(
        enable_deescalation_after_repeated_failure=True,
        max_attempts_at_letter=4,
    )
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 retries_on_current_exercise=4)
    feat = _features(prev_score=0.20)  # struggling
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert decision.overridden
    assert decision.shaped_action_id == DECREASE_DIFFICULTY
    assert decision.rule == "deescalation"


def test_deescalation_does_not_fire_below_threshold():
    """Below the threshold of failed attempts, no de-escalation."""
    cfg = PedagogicalLayerConfig(
        enable_deescalation_after_repeated_failure=True,
        max_attempts_at_letter=4,
    )
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 retries_on_current_exercise=2)
    feat = _features(prev_score=0.20)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert not decision.overridden


def test_deescalation_does_not_fire_if_letter_already_mastered():
    """If the current letter is already mastered, don't de-escalate
    (the learner is no longer stuck on it)."""
    from app.mastery.mastery_tracker import record_attempt
    cfg = PedagogicalLayerConfig(
        enable_deescalation_after_repeated_failure=True,
        max_attempts_at_letter=4,
    )
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 retries_on_current_exercise=4)
    # Master the letter
    for _ in range(settings.mastery.mastery_consecutive_successes_required):
        record_attempt(s, "letter_lower_a", settings.mastery.mastery_score_threshold + 0.05)
    feat = _features(prev_score=0.20)  # this attempt is poor, but the letter is mastered
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert not decision.overridden, "should not de-escalate when the letter is already mastered"


def test_deescalation_can_be_disabled():
    """When the rule is disabled, no de-escalation even after many
    failed attempts."""
    cfg = PedagogicalLayerConfig(enable_deescalation_after_repeated_failure=False)
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 retries_on_current_exercise=10)
    feat = _features(prev_score=0.20)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert not decision.overridden


# ---------------------------------------------------------------------------
# Post-break progression
# ---------------------------------------------------------------------------
def test_post_break_progression_fires_after_break_and_good_score():
    """If the previous action was SUGGEST_BREAK and the current
    attempt's quality_score >= mastery threshold, override to
    INCREASE_DIFFICULTY."""
    cfg = PedagogicalLayerConfig(enable_post_break_progression=True)
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 last_action=SUGGEST_BREAK)
    feat = _features(prev_score=0.95)  # good score
    decision = apply_pedagogical_layer(CONTINUE_NO_CHANGE, s, feat, cfg=cfg)
    assert decision.overridden
    assert decision.shaped_action_id == INCREASE_DIFFICULTY
    assert decision.rule == "post_break"


def test_post_break_progression_does_not_fire_after_break_and_poor_score():
    """If the previous action was SUGGEST_BREAK but the current
    attempt's quality_score is below mastery, no override (the
    learner needs more practice, not progression)."""
    cfg = PedagogicalLayerConfig(enable_post_break_progression=True)
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 last_action=SUGGEST_BREAK)
    feat = _features(prev_score=0.20)  # poor score
    decision = apply_pedagogical_layer(CONTINUE_NO_CHANGE, s, feat, cfg=cfg)
    assert not decision.overridden


def test_post_break_progression_does_not_fire_if_previous_not_break():
    """If the previous action was NOT SUGGEST_BREAK, no override."""
    cfg = PedagogicalLayerConfig(enable_post_break_progression=True)
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 last_action=SHOW_HINT)
    feat = _features(prev_score=0.95)
    decision = apply_pedagogical_layer(CONTINUE_NO_CHANGE, s, feat, cfg=cfg)
    assert not decision.overridden


def test_post_break_progression_does_not_override_if_already_increasing():
    """If the policy already chose INCREASE_DIFFICULTY after a break +
    good score, no override needed."""
    cfg = PedagogicalLayerConfig(enable_post_break_progression=True)
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 last_action=SUGGEST_BREAK)
    feat = _features(prev_score=0.95)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert not decision.overridden


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------
def test_deescalation_has_higher_priority_than_diversity():
    """When both rules would fire, de-escalation wins."""
    cfg = PedagogicalLayerConfig(
        enable_deescalation_after_repeated_failure=True,
        max_attempts_at_letter=4,
        enable_diversity_safeguard=True,
        diversity_window=4, diversity_repeat_threshold=2,
    )
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 retries_on_current_exercise=4,
                 action_log=[INCREASE_DIFFICULTY, INCREASE_DIFFICULTY])
    s.last_action = INCREASE_DIFFICULTY
    feat = _features(prev_score=0.20)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert decision.overridden
    assert decision.shaped_action_id == DECREASE_DIFFICULTY
    assert decision.rule == "deescalation"


def test_post_break_has_higher_priority_than_diversity():
    """When both post-break and diversity would fire, post-break wins."""
    cfg = PedagogicalLayerConfig(
        enable_post_break_progression=True,
        enable_diversity_safeguard=True,
        diversity_window=4, diversity_repeat_threshold=2,
    )
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 last_action=SUGGEST_BREAK,
                 action_log=[SUGGEST_BREAK, SUGGEST_BREAK])
    feat = _features(prev_score=0.95, fatigue=0.0)
    decision = apply_pedagogical_layer(CONTINUE_NO_CHANGE, s, feat, cfg=cfg)
    assert decision.overridden
    assert decision.shaped_action_id == INCREASE_DIFFICULTY
    assert decision.rule == "post_break"


# ---------------------------------------------------------------------------
# Disabled layer (master switch)
# ---------------------------------------------------------------------------
def test_layer_disabled_does_not_override_anything():
    """When the layer is fully disabled (master switch off), it's a
    pass-through."""
    cfg = PedagogicalLayerConfig(
        enabled=False,
        enable_deescalation_after_repeated_failure=True,
        enable_post_break_progression=True,
        enable_diversity_safeguard=True,
        max_attempts_at_letter=4,
        diversity_window=4, diversity_repeat_threshold=2,
    )
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 retries_on_current_exercise=10,
                 last_action=SUGGEST_BREAK,
                 action_log=[INCREASE_DIFFICULTY] * 5)
    feat = _features(prev_score=0.95)
    # Note: even with enabled=False, individual rules would normally
    # still fire. To respect the master switch, the layer should be a
    # pass-through. Verify by checking the layer returns the input
    # action unchanged.
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    # The master switch disables all rules.
    assert not decision.overridden


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_layer_is_deterministic_for_identical_input():
    """The layer must be deterministic for identical input."""
    cfg = PedagogicalLayerConfig()
    s = _session(current_eid="letter_lower_a", difficulty=4,
                 retries_on_current_exercise=4,
                 action_log=[INCREASE_DIFFICULTY, INCREASE_DIFFICULTY])
    s.last_action = INCREASE_DIFFICULTY
    feat = _features(prev_score=0.20)
    d1 = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    d2 = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert d1.shaped_action_id == d2.shaped_action_id
    assert d1.rule == d2.rule
    assert d1.reason == d2.reason


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_action_log_does_not_crash():
    """A fresh session with no action log should not crash the layer."""
    cfg = PedagogicalLayerConfig()
    s = _session(action_log=[])
    feat = _features(prev_score=0.5)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    assert not decision.overridden


def test_no_current_exercise_does_not_crash_deescalation():
    """If the session has no current exercise (shouldn't happen in
    practice but defensively), the layer should not crash."""
    cfg = PedagogicalLayerConfig(
        enable_deescalation_after_repeated_failure=True,
        max_attempts_at_letter=4,
    )
    s = _session(current_eid="", difficulty=4, retries_on_current_exercise=10)
    feat = _features(prev_score=0.20)
    decision = apply_pedagogical_layer(INCREASE_DIFFICULTY, s, feat, cfg=cfg)
    # No current exercise -> no de-escalation
    assert not decision.overridden
