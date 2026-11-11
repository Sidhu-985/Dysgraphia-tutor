"""Simulator and backend must encode the 12-D state the same way
(docs/STATE_SCHEMA.md). These tests pin the Round 3 Phase C corrections."""
import numpy as np

from app.config import QualityScoreConfig
from app.dysgraphia_rl_bridge import STATE_FEATURE_NAMES, rl_root  # noqa: F401  (puts dysgraphia_rl on sys.path)
from app.features.dynamic_features import DynamicFeatures
from app.features.static_features import StaticFeatures, extract_static_features
from app.state.session_context import SessionContext
from app.state.state_builder import build_state, compute_quality_score
from envs import dysgraphia_tutor_env as env_mod
from envs.dysgraphia_tutor_env import DysgraphiaTutorEnv, CONTINUE_NO_CHANGE
from tests.conftest import make_line_stroke


def _dyn(pv=0.3, available=False):
    return DynamicFeatures(writing_speed=1.0, velocity_variation=0.1, acceleration_rms=None,
                           acceleration_reliable=False, movement_smoothness=0.5, pause_count=0,
                           pause_total_s=0.0, pause_mean_s=0.0, stroke_duration_s=1.0,
                           inter_stroke_interval_mean_s=0.0, pressure_mean=0.5,
                           pressure_variance_normalized=pv, pressure_available=available,
                           retry_count_normalized=0.0, temporal_consistency=0.5)


def test_quality_weights_match():
    cfg = QualityScoreConfig()
    assert (env_mod.QUALITY_W_SIMILARITY, env_mod.QUALITY_W_SPACING, env_mod.QUALITY_W_BASELINE,
            env_mod.QUALITY_W_PRESSURE) == (cfg.w_stroke_similarity, cfg.w_spacing_accuracy,
                                            cfg.w_baseline_alignment, cfg.w_pressure_stability)


def test_placeholders_match_backend():
    single = extract_static_features([make_line_stroke()])
    assert single.spacing_accuracy == env_mod.SPACING_PLACEHOLDER
    from app.features.dynamic_features import extract_dynamic_features
    no_pressure = extract_dynamic_features([make_line_stroke(pressure=None)])
    assert not no_pressure.pressure_available
    assert no_pressure.pressure_variance_normalized == env_mod.PRESSURE_VARIANCE_PLACEHOLDER


def test_simulator_quality_equals_backend_formula_on_same_features():
    env = DysgraphiaTutorEnv(seed=3)
    for seed in range(20):
        env.reset(seed=seed)
        trial = env._simulate_trial()
        static = StaticFeatures(trial["stroke_sim"], trial["spacing"], trial["alignment"],
                                1, 1, 1, 0.5, 0.5, 0.0, 1)
        dyn = _dyn(trial["pressure_var"], env.pressure_available)
        assert abs(compute_quality_score(static, dyn) - trial["quality"]) < 1e-9


def test_simulator_emits_backend_placeholders_when_unmeasurable():
    seen_without_pressure = False
    for seed in range(40):
        env = DysgraphiaTutorEnv()
        obs, info = env.reset(seed=seed)
        assert obs[STATE_FEATURE_NAMES.index("spacing_accuracy")] == np.float32(0.5)
        if not info["pressure_available"]:
            seen_without_pressure = True
            obs, *_ = env.step(CONTINUE_NO_CHANGE)
            assert obs[STATE_FEATURE_NAMES.index("pressure_variance")] == np.float32(0.3)
    assert seen_without_pressure


def test_backend_retry_count_includes_current_attempt_like_simulator():
    session = SessionContext(session_id="s", child_id="c_1")
    bad = StaticFeatures(0.1, 0.5, 0.1, 1, 1, 1, 0.5, 0.5, 1.0, 1)
    r1 = build_state(bad, _dyn(), session)
    r2 = build_state(bad, _dyn(), session)
    assert r1.feature_dict["retry_count"] == 0.2  # 1 consecutive failure / cap 5
    assert r2.feature_dict["retry_count"] == 0.4
    env = DysgraphiaTutorEnv(seed=0)
    env.reset(seed=0)
    env.consecutive_fails = 0
    env.difficulty, env.fatigue = 4, 0.9  # make failure certain
    obs, *_ = env.step(CONTINUE_NO_CHANGE)
    assert obs[STATE_FEATURE_NAMES.index("retry_count")] == np.float32((env.consecutive_fails) / 5)
