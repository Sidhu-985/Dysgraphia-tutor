from app.config import SafetyConfig
from app.safety.safety_layer import apply_safety_layer, INCREASE_DIFFICULTY, SUGGEST_BREAK, CONTINUE_NO_CHANGE
from app.state.session_context import SessionContext


def _session():
    return SessionContext(session_id="s1", child_id="c1")


def _features(fatigue=0.1, frustration=0.1):
    return {"fatigue_proxy": fatigue, "frustration": frustration}


def test_forces_break_on_high_fatigue():
    session = _session()
    decision = apply_safety_layer(CONTINUE_NO_CHANGE, session, _features(fatigue=0.9))
    assert decision.safe_action_id == SUGGEST_BREAK
    assert decision.overridden
    assert any("fatigue" in r for r in decision.reasons)


def test_blocks_escalation_on_high_frustration():
    session = _session()
    decision = apply_safety_layer(INCREASE_DIFFICULTY, session, _features(frustration=0.9))
    assert decision.safe_action_id != INCREASE_DIFFICULTY
    assert decision.overridden


def test_allows_escalation_when_safe():
    session = _session()
    decision = apply_safety_layer(INCREASE_DIFFICULTY, session, _features(fatigue=0.1, frustration=0.1))
    assert decision.safe_action_id == INCREASE_DIFFICULTY
    assert not decision.overridden


def test_blocks_repeated_escalation_runs():
    cfg = SafetyConfig(max_consecutive_escalations=2)
    session = _session()
    d1 = apply_safety_layer(INCREASE_DIFFICULTY, session, _features(), cfg=cfg)
    d2 = apply_safety_layer(INCREASE_DIFFICULTY, session, _features(), cfg=cfg)
    d3 = apply_safety_layer(INCREASE_DIFFICULTY, session, _features(), cfg=cfg)
    assert d1.safe_action_id == INCREASE_DIFFICULTY
    assert d2.safe_action_id == INCREASE_DIFFICULTY
    assert d3.safe_action_id != INCREASE_DIFFICULTY  # third in a row gets blocked
    assert d3.overridden


def test_blocks_excessive_repetition_of_same_action():
    from app.safety.safety_layer import SHOW_HINT
    cfg = SafetyConfig(max_consecutive_same_action=3)
    session = _session()
    decisions = [apply_safety_layer(SHOW_HINT, session, _features(), cfg=cfg) for _ in range(5)]
    assert not decisions[2].overridden  # 3rd occurrence still allowed (count reaches threshold after this)
    assert decisions[3].overridden or decisions[4].overridden  # eventually overridden


def test_forced_break_after_max_attempts_without_break():
    cfg = SafetyConfig(max_attempts_before_forced_break=3)
    session = _session()
    for _ in range(3):
        d = apply_safety_layer(CONTINUE_NO_CHANGE, session, _features(), cfg=cfg)
    assert d.safe_action_id == SUGGEST_BREAK


def test_attempts_since_break_resets_after_break():
    cfg = SafetyConfig(max_attempts_before_forced_break=100)
    session = _session()
    apply_safety_layer(SUGGEST_BREAK, session, _features(), cfg=cfg)
    assert session.attempts_since_break == 0


def test_raw_action_always_reported_even_when_overridden():
    session = _session()
    decision = apply_safety_layer(INCREASE_DIFFICULTY, session, _features(frustration=0.99))
    assert decision.raw_action_id == INCREASE_DIFFICULTY
    assert decision.raw_action_name == "INCREASE_DIFFICULTY"
