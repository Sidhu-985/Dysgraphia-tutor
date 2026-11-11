"""
Basic unit/sanity tests for the core modules. Run with:
    python3 -m pytest tests/ -v
or simply:
    python3 tests/test_core.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dysgraphia_tutor_env import DysgraphiaTutorEnv, STATE_DIM, N_ACTIONS
from utils.reward_utils import compute_reward, RewardWeights, audit_action_distribution
from utils.state_utils import build_state_vector, STATE_FEATURE_NAMES
from agents.double_dqn_agent import DoubleDQNAgent
from agents.baselines import RandomAgent, RuleBasedAgent, ContextualBanditAgent


def test_env_reset_shape():
    env = DysgraphiaTutorEnv(seed=0)
    obs, info = env.reset(seed=1)
    assert obs.shape == (STATE_DIM,)
    assert "profile" in info


def test_env_step_returns_valid_types():
    env = DysgraphiaTutorEnv(seed=0)
    obs, info = env.reset(seed=1)
    obs2, reward, terminated, truncated, info2 = env.step(env.action_space.sample())
    assert obs2.shape == (STATE_DIM,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_episode_terminates_within_max_steps():
    env = DysgraphiaTutorEnv(max_steps=10, seed=0)
    env.reset(seed=2)
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated) and steps < 100:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps <= 10
    assert truncated or terminated


def test_reward_formula_matches_spec():
    w = RewardWeights()
    r = compute_reward(delta_quality=0.1, completion=1.0, engagement=0.8,
                        frustration=0.2, consistency=0.5, weights=w)
    expected = w.w1_quality * 0.1 + w.w2_completion * 1.0 + w.w3_engagement * 0.8 \
        - w.w4_frustration * 0.2 + w.w5_consistency * 0.5
    assert abs(r - expected) < 1e-9


def test_reward_hacking_audit_flags_high_easy_fraction_with_poor_quality():
    # 80% of actions are action 0 (the "easy" action) AND quality stayed low
    actions = [0] * 80 + [1] * 20
    result = audit_action_distribution(actions, n_actions=N_ACTIONS, easy_actions=(0,),
                                        mean_quality=0.15, quality_threshold=0.35)
    assert result["reward_hacking_suspected"] is True


def test_reward_hacking_audit_ok_for_balanced_actions():
    actions = list(range(N_ACTIONS)) * 10  # uniform
    result = audit_action_distribution(actions, n_actions=N_ACTIONS, easy_actions=(0,),
                                        mean_quality=0.5)
    assert result["reward_hacking_suspected"] is False


def test_reward_hacking_audit_not_flagged_when_easy_use_yields_real_quality():
    # High easy-action usage, but quality genuinely improved -- NOT hacking.
    actions = [0] * 80 + [1] * 20
    result = audit_action_distribution(actions, n_actions=N_ACTIONS, easy_actions=(0,),
                                        mean_quality=0.6, quality_threshold=0.35)
    assert result["reward_hacking_suspected"] is False


def test_state_vector_builder_defaults():
    vec = build_state_vector({})
    assert vec.shape == (len(STATE_FEATURE_NAMES),)
    assert vec.shape[0] == STATE_DIM


def test_double_dqn_agent_act_and_train():
    agent = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    state = np.random.rand(STATE_DIM).astype(np.float32)
    action = agent.act(state, epsilon=0.0)
    assert 0 <= action < N_ACTIONS

    # fill buffer enough to allow a train step
    for _ in range(100):
        s = np.random.rand(STATE_DIM).astype(np.float32)
        a = np.random.randint(N_ACTIONS)
        r = np.random.rand()
        s2 = np.random.rand(STATE_DIM).astype(np.float32)
        agent.store(s, a, r, s2, 0.0)
    loss = agent.train_step(batch_size=32)
    assert loss is not None and loss >= 0.0


def test_baseline_agents_produce_valid_actions():
    state = np.random.rand(STATE_DIM).astype(np.float32)
    for agent in [RandomAgent(), RuleBasedAgent(), ContextualBanditAgent(seed=0)]:
        a = agent.act(state)
        assert 0 <= a < N_ACTIONS


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -> {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
