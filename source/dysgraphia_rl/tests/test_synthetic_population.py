"""Tests for the optional broad synthetic learner population."""
from envs.dysgraphia_tutor_env import DysgraphiaTutorEnv


def test_synthetic_population_has_requested_size_and_unique_profiles():
    env = DysgraphiaTutorEnv(profile_sampling="synthetic_population",
                             synthetic_profile_count=1000, seed=123)
    assert len(env._synthetic_profiles) == 1000
    assert len({p.name for p in env._synthetic_profiles}) == 1000
    assert len({(p.base_skill, p.learning_rate, p.fatigue_rate,
                 p.frustration_sensitivity, p.spacing_weakness,
                 p.stroke_instability, p.recovery_rate)
                for p in env._synthetic_profiles}) == 1000


def test_synthetic_population_is_reproducible():
    a = DysgraphiaTutorEnv(profile_sampling="synthetic_population",
                           synthetic_profile_count=1000, seed=123)
    b = DysgraphiaTutorEnv(profile_sampling="synthetic_population",
                           synthetic_profile_count=1000, seed=123)
    assert a._synthetic_profiles == b._synthetic_profiles


def test_reset_selects_a_profile_from_synthetic_population():
    env = DysgraphiaTutorEnv(profile_sampling="synthetic_population",
                             synthetic_profile_count=1000, seed=123)
    _, _ = env.reset(seed=456)
    assert env.profile in env._synthetic_profiles
