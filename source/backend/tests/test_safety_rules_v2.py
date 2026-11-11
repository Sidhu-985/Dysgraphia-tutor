"""Safety rules added after the Phase A audit (break over-selection).
Unit level: apply_safety_layer only, no policy and no HTTP."""
from app.config import SafetyConfig
from app.safety.safety_layer import (
    apply_safety_layer, INCREASE_DIFFICULTY, DECREASE_DIFFICULTY, SHOW_HINT, SUGGEST_BREAK,
    CONTINUE_NO_CHANGE, REPEAT_SAME_LETTER, SWITCH_TO_FREE_WRITING_MODE, SUPPORTIVE_ACTIONS,
)
from app.state.session_context import SessionContext


def _session(**kw):
    return SessionContext(session_id="s1", child_id="c_1", **kw)


def _f(fatigue=0.1, frustration=0.1):
    return {"fatigue_proxy": fatigue, "frustration": frustration}


def test_fatigue_above_0_7_always_forces_break():
    for proposed in range(10):
        d = apply_safety_layer(proposed, _session(), _f(fatigue=0.71))
        assert d.safe_action_id == SUGGEST_BREAK


def test_consecutive_break_cap_blocks_second_policy_break():
    s = _session()
    d1 = apply_safety_layer(SUGGEST_BREAK, s, _f(fatigue=0.5))
    d2 = apply_safety_layer(SUGGEST_BREAK, s, _f(fatigue=0.2))
    assert d1.safe_action_id == SUGGEST_BREAK
    assert d2.safe_action_id != SUGGEST_BREAK and d2.overridden
    assert any("consecutive_breaks" in r for r in d2.reasons)


def test_forced_break_wins_over_consecutive_break_cap():
    s = _session()
    apply_safety_layer(SUGGEST_BREAK, s, _f(fatigue=0.5))
    d = apply_safety_layer(SUGGEST_BREAK, s, _f(fatigue=0.8))
    assert d.safe_action_id == SUGGEST_BREAK


def test_break_cap_resets_after_a_non_break_action():
    s = _session()
    apply_safety_layer(SUGGEST_BREAK, s, _f())
    apply_safety_layer(CONTINUE_NO_CHANGE, s, _f())
    d = apply_safety_layer(SUGGEST_BREAK, s, _f())
    assert d.safe_action_id == SUGGEST_BREAK


def test_high_frustration_replaces_every_demanding_action_with_support():
    for proposed in (INCREASE_DIFFICULTY, SWITCH_TO_FREE_WRITING_MODE, REPEAT_SAME_LETTER, CONTINUE_NO_CHANGE):
        d = apply_safety_layer(proposed, _session(), _f(frustration=0.8))
        assert d.safe_action_id in SUPPORTIVE_ACTIONS, proposed
        assert d.safe_action_id in (SHOW_HINT, DECREASE_DIFFICULTY)


def test_high_frustration_after_a_hint_makes_the_task_easier():
    s = _session(last_action=SHOW_HINT)
    d = apply_safety_layer(CONTINUE_NO_CHANGE, s, _f(frustration=0.8))
    assert d.safe_action_id == DECREASE_DIFFICULTY


def test_repeated_hints_under_high_frustration_stay_supportive():
    s = _session()
    actions = [apply_safety_layer(SHOW_HINT, s, _f(frustration=0.9)).safe_action_id for _ in range(8)]
    assert all(a in SUPPORTIVE_ACTIONS for a in actions)
    assert CONTINUE_NO_CHANGE not in actions


def test_increase_blocked_after_a_failed_attempt():
    d = apply_safety_layer(INCREASE_DIFFICULTY, _session(consecutive_fails=1), _f())
    assert d.safe_action_id != INCREASE_DIFFICULTY and d.overridden


def test_repeated_failures_never_increase_difficulty_for_any_proposal_sequence():
    s = _session(consecutive_fails=3)
    for _ in range(10):
        d = apply_safety_layer(INCREASE_DIFFICULTY, s, _f(frustration=0.3))
        assert d.safe_action_id != INCREASE_DIFFICULTY


def test_increase_still_allowed_after_success_when_calm():
    d = apply_safety_layer(INCREASE_DIFFICULTY, _session(consecutive_fails=0), _f())
    assert d.safe_action_id == INCREASE_DIFFICULTY


def test_default_thresholds_are_the_documented_values():
    cfg = SafetyConfig()
    assert cfg.fatigue_break_threshold == 0.70
    assert cfg.max_consecutive_breaks == 1
    assert cfg.block_increase_after_consecutive_fails == 1
