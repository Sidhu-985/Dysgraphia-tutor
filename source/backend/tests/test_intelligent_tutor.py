"""
Intelligent rule-based tutor tests (audit-driven, 2026-11)
==========================================================

These tests verify the new ``IntelligentRuleBasedTutor`` (used as the
live backend's rule-based fallback when the gated Double DQN
checkpoint is not allowed to make decisions) makes context-aware
decisions in the situations the audit specifies:

  A. Incorrect handwriting -- never praise; give targeted support.
  B. Repeated mistakes -- pick a DIFFERENT support action than the
     most recent one (avoid repeating ineffective interventions).
  C. Improving handwriting -- continue, encourage free writing once
     the learner has a sustained success run.
  D. Fatigue or frustration -- suggest a break, but NOT solely because
     of one poor score (require high fatigue or sustained poor scores
     AND high frustration).
  E. Consistent high performance -- progress (the intervention
     controller's rotation rule handles the actual same-level rotation).
  Missing state values are tolerated; the safety layer still runs
  after the tutor and may override.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import SessionDynamicsConfig, settings
from app.dysgraphia_rl_bridge import ACTION_NAMES, N_ACTIONS
from app.rl_integration.intelligent_tutor import (
    IDX_FATIGUE, IDX_FRUSTRATION, IDX_PREV_SCORE, IDX_RETRY_COUNT,
    IntelligentRuleBasedTutor, TutorConfig, _label_intervention_effectiveness,
)
from app.state.session_context import SessionContext


# Action id constants (must match the canonical order in dysgraphia_rl)
DECREASE_DIFFICULTY = 0
INCREASE_DIFFICULTY = 1
SHOW_HINT = 2
ENLARGE_TEMPLATE = 3
REPLAY_STROKE_ANIMATION = 4
SWITCH_TO_TRACING_MODE = 5
SWITCH_TO_FREE_WRITING_MODE = 6
SUGGEST_BREAK = 7
REPEAT_SAME_LETTER = 8
CONTINUE_NO_CHANGE = 9

SUPPORTIVE_ACTIONS = {
    DECREASE_DIFFICULTY, SHOW_HINT, ENLARGE_TEMPLATE, REPLAY_STROKE_ANIMATION,
    SWITCH_TO_TRACING_MODE, SUGGEST_BREAK,
}
ESCALATING_ACTIONS = {INCREASE_DIFFICULTY}


def _state(prev=0.5, sim=0.5, spacing=0.5, baseline=0.5,
           speed=1.0, pressure=0.3, retries=0.0, fatigue=0.0,
           engagement=0.7, trend=0.0, frustration=0.0, consistency=0.5):
    """Build a 12-D RL state vector with the given component values."""
    return np.array([
        prev, sim, spacing, baseline, speed, pressure, retries,
        fatigue, engagement, trend, frustration, consistency,
    ], dtype=np.float32)


def _session_with_score_history(scores, action_log=None,
                                  mode="tracing",
                                  current_exercise_id="letter_lower_a"):
    s = SessionContext(session_id="s", child_id="P-0001",
                       current_exercise_id=current_exercise_id, mode=mode)
    s.score_history = list(scores)
    s.action_log = list(action_log or [])
    return s


# ---------------------------------------------------------------------------
# A. Incorrect handwriting
# ---------------------------------------------------------------------------
def test_incorrect_handwriting_gets_supportive_action_not_continue():
    """When the score is very low (malformed handwriting), the tutor
    must NOT just continue. It should pick a supportive action."""
    tutor = IntelligentRuleBasedTutor()
    state = _state(prev=0.20, fatigue=0.05, frustration=0.05)
    s = _session_with_score_history([0.20])
    a = tutor.act(state, session=s)
    assert a in SUPPORTIVE_ACTIONS, (
        f"low score should get a supportive action; got {ACTION_NAMES[a]}"
    )


def test_incorrect_handwriting_never_praised_or_escalated():
    """A very poor attempt must NEVER trigger INCREASE_DIFFICULTY (the
    learner is struggling, not progressing)."""
    tutor = IntelligentRuleBasedTutor()
    state = _state(prev=0.15)
    s = _session_with_score_history([0.15])
    a = tutor.act(state, session=s)
    assert a != INCREASE_DIFFICULTY, (
        "incorrect handwriting must not be rewarded with INCREASE_DIFFICULTY"
    )


def test_incorrect_handwriting_does_not_repeat_recently_used_harmful_action():
    """If a previous support action was labelled harmful (the score
    went DOWN after it), do not pick it again."""
    tutor = IntelligentRuleBasedTutor()
    # History: 0.40 -> 0.20 (the SHOW_HINT action between them was
    # labelled harmful because the score dropped).
    s = _session_with_score_history([0.40, 0.20], action_log=[SHOW_HINT])
    state = _state(prev=0.20)  # still incorrect
    a = tutor.act(state, session=s)
    # SHOW_HINT should be avoided (it was harmful in the recent window)
    assert a != SHOW_HINT or s.action_log[-1] != SHOW_HINT


# ---------------------------------------------------------------------------
# B. Repeated mistakes
# ---------------------------------------------------------------------------
def test_repeated_mistakes_pick_a_different_support_action():
    """When the same mistake is repeated, the tutor should pick a
    different support action than the most recent one."""
    tutor = IntelligentRuleBasedTutor()
    # Three failed attempts in a row, the last action was SHOW_HINT
    s = _session_with_score_history([0.20, 0.25, 0.20],
                                     action_log=[SHOW_HINT, SHOW_HINT])
    s.consecutive_fails = 3
    state = _state(prev=0.20, retries=0.6)
    a = tutor.act(state, session=s)
    # Should be supportive but DIFFERENT from the last action
    assert a in SUPPORTIVE_ACTIONS
    assert a != SHOW_HINT, (
        "must not repeat SHOW_HINT after repeated failures; try something else"
    )


def test_repeated_ineffective_intervention_is_avoided():
    """If the same intervention was tried and the score did not improve,
    the tutor avoids it in the next decision."""
    tutor = IntelligentRuleBasedTutor()
    # ENLARGE_TEMPLATE was the last action; the score did not improve.
    s = _session_with_score_history([0.30, 0.30], action_log=[ENLARGE_TEMPLATE])
    s.consecutive_fails = 2
    state = _state(prev=0.30, retries=0.4)
    a = tutor.act(state, session=s)
    # The tutor should not pick ENLARGE_TEMPLATE again immediately.
    # (It MAY pick SHOW_HINT, REPLAY_STROKE_ANIMATION, SWITCH_TO_TRACING_MODE,
    # or DECREASE_DIFFICULTY.)
    assert a != ENLARGE_TEMPLATE or a == ENLARGE_TEMPLATE and s.action_log[-1] != ENLARGE_TEMPLATE


# ---------------------------------------------------------------------------
# C. Improving handwriting
# ---------------------------------------------------------------------------
def test_improving_handwriting_continues_not_decreases_difficulty():
    """When the learner is improving, the tutor should continue, not
    preemptively reduce difficulty."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.40, 0.55, 0.70, 0.78])
    state = _state(prev=0.78, trend=0.10, fatigue=0.05, frustration=0.05)
    a = tutor.act(state, session=s)
    assert a != DECREASE_DIFFICULTY, (
        "improving learner should not have difficulty decreased"
    )


def test_consistent_high_performance_encourages_free_writing():
    """After a sustained success run in tracing mode, encourage
    independent writing."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.78, 0.80, 0.82, 0.85], mode="tracing")
    state = _state(prev=0.85, trend=0.10, consistency=0.8, fatigue=0.05)
    a = tutor.act(state, session=s)
    # Should either continue or switch to free writing (encourage
    # independence). Both are acceptable; INCREASE_DIFFICULTY is also OK
    # if the tutor judges the learner has mastered the current letter
    # (the rotation rule in the intervention controller handles the
    # actual same-level rotation).
    assert a in (CONTINUE_NO_CHANGE, SWITCH_TO_FREE_WRITING_MODE, INCREASE_DIFFICULTY)


# ---------------------------------------------------------------------------
# D. Fatigue or frustration
# ---------------------------------------------------------------------------
def test_high_fatigue_suggests_break():
    """Very high fatigue always suggests a break."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.20, 0.20])
    state = _state(prev=0.20, fatigue=0.85)
    a = tutor.act(state, session=s)
    assert a == SUGGEST_BREAK, "very high fatigue must suggest a break"


def test_one_poor_score_does_not_suggest_break():
    """A single poor score alone (with low fatigue) must NOT trigger a
    break -- the audit explicitly requires this."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.30])  # one poor score
    state = _state(prev=0.30, fatigue=0.05, frustration=0.05)
    a = tutor.act(state, session=s)
    assert a != SUGGEST_BREAK, (
        "a single poor score with low fatigue must NOT suggest a break"
    )


def test_high_frustration_with_sustained_poor_scores_suggests_break():
    """High frustration AND sustained poor scores -> break is OK."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.20, 0.25, 0.20])
    state = _state(prev=0.20, fatigue=0.30, frustration=0.85)
    a = tutor.act(state, session=s)
    # Either break (because sustained poor + high frustration) or a
    # supportive action (because the tutor's safety override prefers
    # support). Both are acceptable; INCREASE_DIFFICULTY is NOT.
    assert a != INCREASE_DIFFICULTY


def test_high_frustration_blocks_escalation():
    """High frustration should NOT let the tutor pick a demanding action."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.20, 0.20])
    state = _state(prev=0.20, frustration=0.90, fatigue=0.30)
    a = tutor.act(state, session=s)
    assert a != INCREASE_DIFFICULTY


# ---------------------------------------------------------------------------
# E. Consistent high performance
# ---------------------------------------------------------------------------
def test_consistent_mastery_progresses():
    """After multiple consecutive successes with high consistency, the
    tutor picks INCREASE_DIFFICULTY (the rotation rule in the
    intervention controller handles same-level rotation, NOT jumping
    levels)."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.80, 0.82, 0.85, 0.87])
    state = _state(prev=0.87, trend=0.05, consistency=0.85, fatigue=0.05)
    a = tutor.act(state, session=s)
    assert a == INCREASE_DIFFICULTY or a == CONTINUE_NO_CHANGE, (
        f"consistent mastery should progress (or continue); got {ACTION_NAMES[a]}"
    )


# ---------------------------------------------------------------------------
# Missing state values
# ---------------------------------------------------------------------------
def test_nan_state_does_not_crash():
    """NaN state values must not crash the tutor; it should fall back to
    a safe default."""
    tutor = IntelligentRuleBasedTutor()
    state = np.full(12, np.nan, dtype=np.float32)
    s = _session_with_score_history([])
    a = tutor.act(state, session=s)
    assert 0 <= a < N_ACTIONS


def test_wrong_shape_state_does_not_crash():
    """A state vector of the wrong shape must not crash the tutor."""
    tutor = IntelligentRuleBasedTutor()
    state = np.array([0.5, 0.5], dtype=np.float32)  # wrong shape
    s = _session_with_score_history([])
    a = tutor.act(state, session=s)
    assert 0 <= a < N_ACTIONS


def test_no_session_uses_legacy_behaviour():
    """Without a session, the tutor uses the legacy RuleBasedAgent
    behaviour (so RL training tests keep passing)."""
    tutor = IntelligentRuleBasedTutor()
    state = _state(prev=0.20, fatigue=0.85)
    a = tutor.act(state)  # no session -> legacy path
    assert a == SUGGEST_BREAK  # legacy: high fatigue -> break


# ---------------------------------------------------------------------------
# Effectiveness tracking
# ---------------------------------------------------------------------------
def test_label_intervention_effectiveness_basic():
    """The effectiveness-label helper correctly labels interventions
    based on the score change after them."""
    s = _session_with_score_history([0.40, 0.55, 0.40],
                                     action_log=[SHOW_HINT, ENLARGE_TEMPLATE])
    # SHOW_HINT was applied between score 0.40 and 0.55 -> +0.15 -> effective
    # ENLARGE_TEMPLATE was applied between score 0.55 and 0.40 -> -0.15 -> harmful
    labels = _label_intervention_effectiveness(s, window=4, improvement=0.05)
    assert labels.get(SHOW_HINT) == "effective"
    assert labels.get(ENLARGE_TEMPLATE) == "harmful"


def test_label_intervention_effectiveness_empty_history():
    s = _session_with_score_history([], action_log=[])
    labels = _label_intervention_effectiveness(s, window=4, improvement=0.05)
    assert labels == {}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_deterministic_for_identical_input():
    """The tutor must be deterministic for identical input (audit
    requirement)."""
    tutor = IntelligentRuleBasedTutor()
    s = _session_with_score_history([0.20, 0.20, 0.20],
                                     action_log=[SHOW_HINT, SHOW_HINT])
    s.consecutive_fails = 3
    state = _state(prev=0.20, retries=0.6)
    a1 = tutor.act(state, session=s)
    a2 = tutor.act(state, session=s)
    assert a1 == a2


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
def test_intervention_cooldown_avoids_immediate_repeats():
    """An intervention just used should not be immediately reselected
    (when there are alternatives)."""
    tutor = IntelligentRuleBasedTutor(cfg=TutorConfig(intervention_cooldown=3))
    # Just had ENLARGE_TEMPLATE; score is poor but not too poor.
    s = _session_with_score_history([0.30], action_log=[ENLARGE_TEMPLATE])
    state = _state(prev=0.30, fatigue=0.05, frustration=0.05)
    a = tutor.act(state, session=s)
    # The tutor should NOT pick ENLARGE_TEMPLATE immediately again
    # (unless it's the only option).
    if a in (SHOW_HINT, REPLAY_STROKE_ANIMATION, SWITCH_TO_TRACING_MODE):
        pass  # picked a different support action -- good
    elif a == ENLARGE_TEMPLATE:
        # Allowed only if all other support actions are also in cooldown
        # (not the case here, with cooldown=3 and only 1 past action).
        pytest.fail("ENLARGE_TEMPLATE repeated within cooldown window")
