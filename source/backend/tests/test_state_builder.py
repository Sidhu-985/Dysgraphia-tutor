import numpy as np
import pytest

from app.features.static_features import extract_static_features
from app.features.dynamic_features import extract_dynamic_features
from app.state.session_context import SessionContext
from app.state.state_builder import build_state, compute_quality_score, update_session_dynamics
from app.dysgraphia_rl_bridge import STATE_FEATURE_NAMES, STATE_DIM, validate_state_vector
from .conftest import make_line_stroke


def _session():
    return SessionContext(session_id="s1", child_id="c1")


def test_state_vector_has_canonical_shape_and_order():
    stroke = make_line_stroke()
    static = extract_static_features([stroke])
    dynamic = extract_dynamic_features([stroke])
    session = _session()

    result = build_state(static, dynamic, session)
    assert result.state_vector.shape == (STATE_DIM,)
    assert list(result.feature_dict.keys()) == STATE_FEATURE_NAMES
    validate_state_vector(result.state_vector, strict=True)  # raises on violation


def test_build_state_appends_to_score_history_and_increments_attempt_count():
    stroke = make_line_stroke()
    static = extract_static_features([stroke])
    dynamic = extract_dynamic_features([stroke])
    session = _session()

    build_state(static, dynamic, session)
    assert session.attempt_count == 1
    assert len(session.score_history) == 1

    build_state(static, dynamic, session)
    assert session.attempt_count == 2
    assert len(session.score_history) == 2


def test_break_taken_does_not_append_score_history():
    stroke = make_line_stroke()
    static = extract_static_features([stroke])
    dynamic = extract_dynamic_features([stroke])
    session = _session()
    build_state(static, dynamic, session)
    n_before = len(session.score_history)

    build_state(static, dynamic, session, break_taken=True)
    assert len(session.score_history) == n_before


def test_fatigue_increases_over_repeated_attempts():
    stroke = make_line_stroke()
    static = extract_static_features([stroke])
    dynamic = extract_dynamic_features([stroke])
    session = _session()

    fatigue_trace = []
    for _ in range(5):
        build_state(static, dynamic, session)
        fatigue_trace.append(session.fatigue)
    assert fatigue_trace[-1] > fatigue_trace[0]


def test_break_taken_reduces_fatigue():
    stroke = make_line_stroke()
    static = extract_static_features([stroke])
    dynamic = extract_dynamic_features([stroke])
    session = _session()
    for _ in range(5):
        build_state(static, dynamic, session)
    fatigue_before = session.fatigue

    build_state(static, dynamic, session, break_taken=True)
    assert session.fatigue < fatigue_before


def test_quality_score_bounded_and_monotonic_in_stroke_similarity():
    from app.features.static_features import StaticFeatures
    from app.features.dynamic_features import DynamicFeatures

    dyn = DynamicFeatures(
        writing_speed=1.0, velocity_variation=0.1, acceleration_rms=None,
        acceleration_reliable=False, movement_smoothness=0.5, pause_count=0,
        pause_total_s=0.0, pause_mean_s=0.0, stroke_duration_s=1.0,
        inter_stroke_interval_mean_s=0.0, pressure_mean=0.5,
        pressure_variance_normalized=0.2, pressure_available=True,
        retry_count_normalized=0.0, temporal_consistency=0.5,
    )
    low = StaticFeatures(0.2, 0.5, 0.5, 10, 10, 1.0, 0.5, 0.5, 0.1, 1)
    high = StaticFeatures(0.9, 0.5, 0.5, 10, 10, 1.0, 0.5, 0.5, 0.1, 1)

    q_low = compute_quality_score(low, dyn)
    q_high = compute_quality_score(high, dyn)
    assert 0.0 <= q_low <= 1.0
    assert 0.0 <= q_high <= 1.0
    assert q_high > q_low


def test_improvement_trend_positive_when_quality_rising():
    from app.features.static_features import StaticFeatures
    from app.features.dynamic_features import DynamicFeatures
    from app.dysgraphia_rl_bridge import compute_improvement_trend

    session = _session()
    session.score_history = [0.2, 0.3, 0.4, 0.5, 0.6]
    trend = compute_improvement_trend(session.score_history)
    assert trend > 0


def test_consecutive_fails_resets_on_success():
    from app.features.dynamic_features import DynamicFeatures

    dyn = DynamicFeatures(
        writing_speed=1.0, velocity_variation=0.1, acceleration_rms=None,
        acceleration_reliable=False, movement_smoothness=0.5, pause_count=0,
        pause_total_s=0.0, pause_mean_s=0.0, stroke_duration_s=1.0,
        inter_stroke_interval_mean_s=0.0, pressure_mean=0.5,
        pressure_variance_normalized=0.2, pressure_available=True,
        retry_count_normalized=0.0, temporal_consistency=0.5,
    )
    session = _session()
    update_session_dynamics(session, quality=0.1, dynamic=dyn)  # fail
    assert session.consecutive_fails == 1
    update_session_dynamics(session, quality=0.9, dynamic=dyn)  # success
    assert session.consecutive_fails == 0
