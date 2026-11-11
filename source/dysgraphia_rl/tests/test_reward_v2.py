"""Reward version 2, break cap and generalized audit (Phase D)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dysgraphia_tutor_env import (  # noqa: E402
    DysgraphiaTutorEnv, SUGGEST_BREAK, CONTINUE_NO_CHANGE, DECREASE_DIFFICULTY, N_ACTIONS,
)
from utils.reward_utils import (  # noqa: E402
    RewardWeights, validate_weights, unnecessary_break_penalty, audit_action_distribution, compute_reward,
)
from agents.baselines import AlwaysBreakAgent, DoNothingAgent  # noqa: E402


def test_w1_quality_is_strictly_largest_weight():
    validate_weights(RewardWeights())
    validate_weights(RewardWeights.v2())
    with pytest.raises(ValueError):
        validate_weights(RewardWeights.v2(w6_unnecessary_break=3.0))
    with pytest.raises(ValueError):
        DysgraphiaTutorEnv(reward_weights={"w3_engagement": 5.0})


def test_unnecessary_break_penalty_shape():
    assert unnecessary_break_penalty(False, 0.0) == 0.0
    assert unnecessary_break_penalty(True, 0.0) == 1.0
    assert unnecessary_break_penalty(True, 0.5) == 0.0
    assert unnecessary_break_penalty(True, 0.9) == 0.0
    assert 0.0 < unnecessary_break_penalty(True, 0.25) < 1.0


def test_v2_low_fatigue_break_reward_includes_penalty():
    env = DysgraphiaTutorEnv(seed=0, reward_version="v2")
    env.reset(seed=0)
    _, r, _, _, info = env.step(SUGGEST_BREAK)
    assert info["break_taken"] and info["unnecessary_break"] > 0.5
    expected = compute_reward(info["quality_improvement"], 0.0, info["engagement"], info["frustration"],
                              _consistency(env), weights=env.reward_weights,
                              unnecessary_break=info["unnecessary_break"])
    assert abs(r - expected) < 1e-9


def _consistency(env):
    h = env.score_history[-5:]
    return float(np.clip(1.0 - np.std(h), 0, 1)) if len(env.score_history) >= 2 else 0.5


def test_consecutive_break_cap_executes_continue_instead():
    env = DysgraphiaTutorEnv(seed=1)
    env.reset(seed=1)
    _, _, _, _, i1 = env.step(SUGGEST_BREAK)
    _, _, _, _, i2 = env.step(SUGGEST_BREAK)
    assert i1["executed_action"] == SUGGEST_BREAK and not i1["break_capped"]
    assert i2["executed_action"] == CONTINUE_NO_CHANGE and i2["break_capped"]
    assert i2["requested_action"] == SUGGEST_BREAK


def test_break_step_carries_forward_observed_features_not_zeros():
    env = DysgraphiaTutorEnv(seed=2)
    obs0, _ = env.reset(seed=2)
    obs1, _, _, _, _ = env.step(SUGGEST_BREAK)
    assert np.allclose(obs1[:4], obs0[:4])  # prev_score, similarity, spacing, alignment
    assert obs1[1] > 0.0


def test_v2_first_state_comes_from_an_observed_attempt():
    env = DysgraphiaTutorEnv(seed=3)
    obs, info = env.reset(seed=3)
    assert info["start_quality"] == env.score_history[0] == pytest.approx(float(obs[0]), abs=1e-6)
    assert env.score_history[0] != env.profile.base_skill


def test_v1_mode_reproduces_original_break_hacking():
    """Reproduces the Phase A defect: under the original reward and with
    no break cap, always-break earns ~49 per 40-step episode with zero
    quality gain."""
    env = DysgraphiaTutorEnv(reward_version="v1", max_consecutive_breaks=None)
    rewards = []
    for ep in range(5):
        env.reset(seed=6000 + ep)
        start = env.score_history[0]
        total, done = 0.0, False
        while not done:
            _, r, te, tr, _ = env.step(SUGGEST_BREAK)
            total += r
            done = te or tr
        rewards.append(total)
        assert env.score_history[-1] - start == 0.0
    assert np.mean(rewards) == pytest.approx(49.25, abs=0.01)


def _episode_return(env, policy, seed):
    s, _ = env.reset(seed=seed)
    total, done = 0.0, False
    while not done:
        s, r, te, tr, _ = env.step(policy(s))
        total += r
        done = te or tr
    return total


def test_v2_always_break_no_longer_dominates_reward():
    """Under reward v2 with the break cap, always-break must earn far less
    than under v1 (49.25)."""
    env = DysgraphiaTutorEnv()
    ab = np.mean([_episode_return(env, AlwaysBreakAgent().act, 7000 + i) for i in range(20)])
    assert ab < 25.0


def test_generalized_audit_flags_break_dominance_with_no_gain():
    actions = [SUGGEST_BREAK] * 45 + [CONTINUE_NO_CHANGE] * 55
    res = audit_action_distribution(actions, N_ACTIONS, quality_gain=0.0, mean_quality=0.5)
    assert res["reward_hacking_suspected"] is True
    assert "non_writing_excess" in res["flags"] and "no_quality_gain" in res["flags"]


def test_generalized_audit_flags_majority_breaks_even_with_gain():
    actions = [SUGGEST_BREAK] * 60 + [CONTINUE_NO_CHANGE] * 40
    assert audit_action_distribution(actions, N_ACTIONS, quality_gain=0.1)["reward_hacking_suspected"]


def test_generalized_audit_passes_varied_policy_with_gain():
    actions = [CONTINUE_NO_CHANGE] * 40 + [2] * 30 + [DECREASE_DIFFICULTY] * 20 + [SUGGEST_BREAK] * 10
    res = audit_action_distribution(actions, N_ACTIONS, quality_gain=0.05, mean_quality=0.5)
    assert res["reward_hacking_suspected"] is False


def test_original_audit_missed_the_break_policy():
    """Documents the Phase A gap: the original easy-action-only rule (no
    quality_gain, default easy_actions) does not see break dominance."""
    actions = [SUGGEST_BREAK] * 45 + [CONTINUE_NO_CHANGE] * 55
    res = audit_action_distribution(actions, N_ACTIONS, mean_quality=0.5)
    assert res["easy_action_fraction"] == 0.0
    assert "non_writing_excess" in res["flags"]


def test_v3_is_the_default_reward_version():
    assert DysgraphiaTutorEnv().reward_version == "v3"


def test_v3_break_step_earns_only_the_unnecessary_break_penalty():
    env = DysgraphiaTutorEnv(seed=0)
    env.reset(seed=0)
    _, r, _, _, info = env.step(SUGGEST_BREAK)
    assert info["break_taken"]
    assert r == pytest.approx(-env.reward_weights.w6_unnecessary_break * info["unnecessary_break"])
    assert r < 0


def test_v3_writing_steps_match_v2():
    a, b = DysgraphiaTutorEnv(reward_version="v2"), DysgraphiaTutorEnv(reward_version="v3")
    a.reset(seed=11)
    b.reset(seed=11)
    for act in (CONTINUE_NO_CHANGE, DECREASE_DIFFICULTY, 2, CONTINUE_NO_CHANGE):
        _, ra, _, _, _ = a.step(act)
        _, rb, _, _, _ = b.step(act)
        assert ra == pytest.approx(rb)


def test_v3_tired_learner_break_is_not_penalized():
    env = DysgraphiaTutorEnv(seed=4)
    env.reset(seed=4)
    env.fatigue = 0.8
    _, r, _, _, info = env.step(SUGGEST_BREAK)
    assert info["unnecessary_break"] == 0.0 and r == 0.0
