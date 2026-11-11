"""
End-to-end pipeline test (project spec Section 13):

    raw handwriting -> preprocessing -> static/dynamic features ->
    12-D state -> Double DQN -> safety -> intervention -> API response

Exercised twice: once by calling each module directly (so a failure
localizes to a specific stage), and once through the actual FastAPI HTTP
surface (test_api.py::test_full_attempt_submit_returns_well_formed_response
already covers the HTTP path; this file focuses on the direct in-process
chain plus a multi-attempt session run).
"""

import math

import numpy as np

from app.preprocessing.stroke_preprocessing import Point, Stroke, clean_strokes
from app.features.static_features import extract_static_features
from app.features.dynamic_features import extract_dynamic_features
from app.state.session_context import SessionContext
from app.state.state_builder import build_state
from app.rl_integration.policy_service import PolicyService
from app.safety.safety_layer import apply_safety_layer
from app.interventions.intervention_controller import build_intervention
from app.dysgraphia_rl_bridge import compute_reward, RewardWeights, validate_state_vector
from app.exercises import get_exercise


def _mock_tablet_attempt(n=40, noisy=True, seed=0) -> list[Stroke]:
    """Simulates realistic raw stylus telemetry -- irregular timestamps,
    sensor jitter, and pressure -- standing in for real tablet hardware
    which is unavailable in this environment (spec Section 13: mock the
    hardware input with realistic test data)."""
    rng = np.random.default_rng(seed)
    points = []
    t = 0.0
    for i in range(n):
        t += rng.uniform(0.01, 0.03)
        x = (i / (n - 1)) * 100.0 + (rng.normal(0, 0.8) if noisy else 0.0)
        y = 50.0 + 8 * math.sin(i / 6.0) + (rng.normal(0, 0.8) if noisy else 0.0)
        pressure = float(np.clip(rng.normal(0.5, 0.1), 0.0, 1.0))
        points.append(Point(x=x, y=y, t=t, pressure=pressure))
    return [Stroke(stroke_id=0, points=points)]


def test_full_pipeline_single_attempt_runs_without_error():
    exercise = get_exercise("loop_medium")
    strokes = clean_strokes(_mock_tablet_attempt())

    static_feats = extract_static_features(strokes, reference_template=exercise.reference_array())
    dynamic_feats = extract_dynamic_features(strokes, retries_so_far=0)

    session = SessionContext(session_id="e2e", child_id="mock_child")
    result = build_state(static_feats, dynamic_feats, session)
    validate_state_vector(result.state_vector, strict=True)

    reward = compute_reward(
        delta_quality=0.0, completion=1.0,
        engagement=result.feature_dict["engagement"],
        frustration=result.feature_dict["frustration"],
        consistency=result.feature_dict["consistency"],
        weights=RewardWeights(),
    )
    assert isinstance(reward, float)

    policy = PolicyService()
    decision = policy.select_action(result.state_vector)

    safety = apply_safety_layer(decision.action_id, session, result.feature_dict)
    intervention = build_intervention(safety.safe_action_id, safety.safe_action_name, session)

    assert intervention.action_id == safety.safe_action_id
    assert isinstance(intervention.message, str) and intervention.message


def test_full_pipeline_multi_attempt_session_is_stable():
    """Runs 15 simulated attempts through the entire chain and checks the
    pipeline never produces NaNs/invalid states and the safety layer stays
    within its configured bounds (no runaway escalation, fatigue stays
    in [0,1])."""
    exercise = get_exercise("loop_medium")
    session = SessionContext(session_id="e2e_multi", child_id="mock_child_2")
    policy = PolicyService()

    for i in range(15):
        strokes = clean_strokes(_mock_tablet_attempt(seed=i))
        static_feats = extract_static_features(strokes, reference_template=exercise.reference_array())
        dynamic_feats = extract_dynamic_features(strokes, retries_so_far=session.retries_on_current_exercise)

        result = build_state(static_feats, dynamic_feats, session)
        validate_state_vector(result.state_vector, strict=True)
        assert not np.any(np.isnan(result.state_vector))

        decision = policy.select_action(result.state_vector)
        safety = apply_safety_layer(decision.action_id, session, result.feature_dict)
        build_intervention(safety.safe_action_id, safety.safe_action_name, session)

        assert 0.0 <= session.fatigue <= 1.0
        assert 0.0 <= session.frustration <= 1.0
        assert 0.0 <= session.engagement <= 1.0

    assert session.attempt_count == 15
