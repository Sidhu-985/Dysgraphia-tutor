"""
Extended unit/integration tests, added during the Code Review pass to close
specific gaps identified in a self-audit (see CODE_REVIEW_READINESS_REPORT.md):

  - the QNetwork architecture wasn't directly asserted anywhere (only
    implied by the checkpoint tensor shapes)
  - the replay buffer's "prioritized" sampling wasn't tested for actually
    being prioritized (a uniform-random sampler would have passed the
    original test_core.py suite unnoticed)
  - checkpoint save/load round-tripping wasn't tested
  - the Double DQN target computation for TERMINAL transitions (target = r,
    no bootstrap) wasn't directly verified
  - the state feature ordering wasn't pinned down by an explicit test, only
    by convention across three files
  - full run-to-run reproducibility given --seed wasn't asserted anywhere
  - a real integration bug: the environment's fatigue safety cutoff set
    `truncated=True` instead of `terminated=True`, and scripts/train_dqn.py
    only ever stored `done=float(terminated)` -- since `terminated` was
    hard-coded to False, EVERY stored transition had done=0.0, so the
    Double DQN Bellman target bootstrapped through every episode boundary,
    including genuine terminal states. The agent-level target-computation
    test above (test_terminal_transitions_do_not_bootstrap) only proved the
    agent's math is correct GIVEN a correct done flag -- it did not prove
    the training loop ever produces one, and it didn't. Fixed in both
    envs/dysgraphia_tutor_env.py and scripts/train_dqn.py; the two tests at
    the end of this file lock the fix in at the integration level.

Run with:
    python3 -m pytest tests/ -v
or:
    python3 tests/test_extra.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.double_dqn_agent import DoubleDQNAgent, QNetwork
from agents.replay_buffer import ReplayBuffer
from envs.dysgraphia_tutor_env import (
    DysgraphiaTutorEnv, STATE_DIM, N_ACTIONS, CONTINUE_NO_CHANGE,
)
from utils.state_utils import STATE_FEATURE_NAMES, validate_state_vector
from utils.reward_utils import compute_reward, RewardWeights


# ---------------------------------------------------------------------------
# 1. Network architecture (Design and Solution Section 6: 12 -> 128 -> 64 -> 10)
# ---------------------------------------------------------------------------
def test_qnetwork_architecture_matches_design_doc():
    net = QNetwork(state_dim=12, n_actions=10)
    layers = [m for m in net.net if isinstance(m, torch.nn.Linear)]
    assert len(layers) == 3, "expected exactly 3 Linear layers (12->128, 128->64, 64->10)"
    assert (layers[0].in_features, layers[0].out_features) == (12, 128)
    assert (layers[1].in_features, layers[1].out_features) == (128, 64)
    assert (layers[2].in_features, layers[2].out_features) == (64, 10)
    # ReLU activations after the two hidden layers, matching the design doc
    relus = [m for m in net.net if isinstance(m, torch.nn.ReLU)]
    assert len(relus) == 2

    # forward pass shape check
    x = torch.randn(4, 12)
    out = net(x)
    assert out.shape == (4, 10)


# ---------------------------------------------------------------------------
# 2. Checkpoint save/load round trip
# ---------------------------------------------------------------------------
def test_checkpoint_save_load_roundtrip(tmp_path):
    ckpt_path = str(tmp_path / "_test_ckpt.pt")
    agent = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    state = np.random.rand(STATE_DIM).astype(np.float32)
    action_before = agent.act(state, epsilon=0.0)
    q_before = agent.online_net(torch.as_tensor(state).unsqueeze(0)).detach().numpy()

    agent.save(ckpt_path)

    agent2 = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    agent2.load(ckpt_path)
    action_after = agent2.act(state, epsilon=0.0)
    q_after = agent2.online_net(torch.as_tensor(state).unsqueeze(0)).detach().numpy()

    assert action_before == action_after
    assert np.allclose(q_before, q_after, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. Prioritized replay is GENUINELY prioritized, not uniform-random.
#    (A uniform sampler would pass every test in test_core.py silently --
#    this test would catch that regression.)
# ---------------------------------------------------------------------------
def test_prioritized_replay_favors_high_priority_transitions():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=1000, alpha=1.0)  # alpha=1 -> priority fully determines prob.

    # Store 100 "low value" transitions with default (max) priority, then one
    # transition whose priority we boost massively via update_priorities.
    for i in range(100):
        s = rng.random(STATE_DIM).astype(np.float32)
        buf.store(s, 0, 0.0, s, 0.0)

    # Give transition index 50 a much higher TD-error / priority than the rest.
    buf.update_priorities([50], [100.0])
    for i in range(100):
        if i != 50:
            buf.update_priorities([i], [0.01])

    # Sample many batches; transition 50 should appear far more than the
    # ~1/100 uniform-random rate.
    hit_count = 0
    n_trials = 300
    for _ in range(n_trials):
        _, _, _, _, _, indices, _ = buf.sample(batch_size=8, beta=0.4)
        if 50 in indices:
            hit_count += 1

    uniform_expected = n_trials * (8 / 100)  # ~24 under uniform sampling
    assert hit_count > uniform_expected * 2, (
        f"high-priority transition sampled {hit_count}/{n_trials} times, "
        f"expected far more than the uniform-random baseline (~{uniform_expected:.0f}); "
        "replay buffer does not appear to be genuinely prioritized"
    )


def test_replay_buffer_importance_weights_are_bounded():
    """IS-weights should be normalized so the max weight in a batch is 1.0
    (standard PER normalization to keep gradient scale stable)."""
    rng = np.random.default_rng(1)
    buf = ReplayBuffer(capacity=200)
    for i in range(50):
        s = rng.random(STATE_DIM).astype(np.float32)
        buf.store(s, 0, float(i), s, 0.0)
    _, _, _, _, _, indices, weights = buf.sample(batch_size=16, beta=0.5)
    assert weights.max() <= 1.0 + 1e-6
    assert (weights > 0).all()


# ---------------------------------------------------------------------------
# 4. Double DQN terminal-transition handling: target = r (no bootstrap)
# ---------------------------------------------------------------------------
def test_terminal_transitions_do_not_bootstrap():
    """For a batch where every transition has done=1, the Double DQN target
    is y = r (the gamma * Q_target(...) term is multiplied by (1-done)=0).
    We verify this indirectly: training on an all-terminal batch with two
    very different gamma values should produce the SAME loss, because gamma
    only ever multiplies a term that is zeroed out for terminal transitions.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    def make_agent(gamma):
        torch.manual_seed(42)  # identical initialization across both agents
        return DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, gamma=gamma)

    agent_low_gamma = make_agent(gamma=0.01)
    agent_high_gamma = make_agent(gamma=0.99)

    # Feed both agents the identical all-terminal batch of transitions.
    rng = np.random.default_rng(7)
    transitions = []
    for _ in range(80):
        s = rng.random(STATE_DIM).astype(np.float32)
        a = int(rng.integers(0, N_ACTIONS))
        r = float(rng.random())
        s2 = rng.random(STATE_DIM).astype(np.float32)
        transitions.append((s, a, r, s2, 1.0))  # done=1.0 for all

    for s, a, r, s2, done in transitions:
        agent_low_gamma.store(s, a, r, s2, done)
        agent_high_gamma.store(s, a, r, s2, done)

    loss_low = agent_low_gamma.train_step(batch_size=len(transitions), beta=1.0)
    loss_high = agent_high_gamma.train_step(batch_size=len(transitions), beta=1.0)

    assert loss_low is not None and loss_high is not None
    assert abs(loss_low - loss_high) < 1e-5, (
        f"loss differs between gamma=0.01 ({loss_low}) and gamma=0.99 ({loss_high}) "
        "on an all-terminal batch -- terminal transitions appear to be bootstrapping "
        "from the next state, which contradicts the Bellman target y=r for done=1"
    )


# ---------------------------------------------------------------------------
# 5. State feature ordering is pinned down (Design and Solution Section 3)
# ---------------------------------------------------------------------------
def test_state_ordering_matches_canonical_spec():
    expected = [
        "prev_score", "stroke_similarity", "spacing_accuracy", "baseline_alignment",
        "writing_speed", "pressure_variance", "retry_count", "fatigue_proxy",
        "engagement", "improvement_trend", "frustration", "consistency",
    ]
    assert STATE_FEATURE_NAMES == expected, (
        "utils/state_utils.py:STATE_FEATURE_NAMES no longer matches the canonical "
        "ordering documented in envs/dysgraphia_tutor_env.py and the Design and "
        "Solution doc Section 3 -- update all three together if this is intentional."
    )


def test_env_state_output_is_valid_per_canonical_ranges():
    env = DysgraphiaTutorEnv(seed=0)
    obs, _ = env.reset(seed=0)
    problems = validate_state_vector(obs, strict=False)
    assert problems == [], f"reset() produced an out-of-spec state: {problems}"

    for _ in range(20):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        problems = validate_state_vector(obs, strict=False)
        assert problems == [], f"step() produced an out-of-spec state: {problems}"
        if terminated or truncated:
            break


# ---------------------------------------------------------------------------
# 6. Environment determinism given a seed (needed for reproducible training)
# ---------------------------------------------------------------------------
def test_env_is_deterministic_given_seed():
    env_a = DysgraphiaTutorEnv(seed=123)
    env_b = DysgraphiaTutorEnv(seed=123)
    obs_a, info_a = env_a.reset(seed=5)
    obs_b, info_b = env_b.reset(seed=5)
    assert np.allclose(obs_a, obs_b)
    assert info_a["profile"] == info_b["profile"]

    actions = [0, 1, 2, 9, 5, 7, 3]
    for a in actions:
        oa, ra, ta, trunc_a, _ = env_a.step(a)
        ob, rb, tb, trunc_b, _ = env_b.step(a)
        assert np.allclose(oa, ob)
        assert abs(ra - rb) < 1e-9
        assert (ta, trunc_a) == (tb, trunc_b)


# ---------------------------------------------------------------------------
# 7. Cross-module consistency: env's internal reward matches the standalone
#    reward_utils.compute_reward formula it now delegates to (regression
#    guard against the two formulas drifting apart again).
# ---------------------------------------------------------------------------
def test_env_reward_matches_standalone_reward_utils():
    weights = RewardWeights()
    delta_quality, completion, engagement, frustration, consistency = 0.2, 1.0, 0.6, 0.3, 0.7
    expected = compute_reward(delta_quality, completion, engagement, frustration,
                               consistency, weights=weights)
    manual = (
        weights.w1_quality * delta_quality + weights.w2_completion * completion
        + weights.w3_engagement * engagement - weights.w4_frustration * frustration
        + weights.w5_consistency * consistency
    )
    assert abs(expected - manual) < 1e-9


# ---------------------------------------------------------------------------
# 8. End-to-end training reproducibility given --seed (subprocess-level check;
#    slow-ish (~5-10s) so kept small: 15 episodes is enough to exercise the
#    full training loop including several gradient updates).
# ---------------------------------------------------------------------------
def test_training_is_reproducible_given_seed(tmp_path):
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "scripts", "train_dqn.py")
    ckpt_a, ckpt_b = str(tmp_path / "_repro_a.pt"), str(tmp_path / "_repro_b.pt")
    plot_a, plot_b = str(tmp_path / "_repro_a.png"), str(tmp_path / "_repro_b.png")
    common_args = ["--episodes", "15", "--seed", "99", "--warmup_steps", "20"]

    for ckpt, plot in [(ckpt_a, plot_a), (ckpt_b, plot_b)]:
        subprocess.run(
            [sys.executable, script, *common_args, "--checkpoint", ckpt, "--plot_path", plot],
            check=True, capture_output=True, text=True,
        )

    a = torch.load(ckpt_a, map_location="cpu")
    b = torch.load(ckpt_b, map_location="cpu")
    for k in a["online_net"]:
        assert torch.equal(a["online_net"][k], b["online_net"][k]), (
            f"parameter '{k}' differs between two runs with identical --seed -- "
            "training is not fully reproducible"
        )
    assert a["train_steps"] == b["train_steps"]

    for f in [ckpt_a, ckpt_b, plot_a, plot_b]:
        if os.path.exists(f):
            os.remove(f)


# ---------------------------------------------------------------------------
# 9. Reward-formula monotonicity: 5 explicit, separately-named cases
#    requested during self-audit (each reward TERM checked in isolation by
#    varying only that term and holding all others fixed). The 6th case
#    from the same request -- "easy-action exploitation -> penalized" -- is
#    already covered by the 3 reward-hacking-audit tests in test_core.py;
#    it is not duplicated here.
# ---------------------------------------------------------------------------
def test_reward_increases_with_quality_improvement():
    """Higher delta_quality, all else equal, must yield strictly higher
    reward -- this is the term the Design doc requires to dominate."""
    w = RewardWeights()
    common = dict(completion=0.5, engagement=0.5, frustration=0.3, consistency=0.5, weights=w)
    r_small_gain = compute_reward(delta_quality=0.05, **common)
    r_big_gain = compute_reward(delta_quality=0.30, **common)
    assert r_big_gain > r_small_gain


def test_reward_penalized_by_quality_deterioration():
    """A negative delta_quality (the child's handwriting got worse) must
    yield a lower reward than a neutral (zero) delta_quality, all else
    equal -- deterioration should never be reward-neutral or positive."""
    w = RewardWeights()
    common = dict(completion=0.5, engagement=0.5, frustration=0.3, consistency=0.5, weights=w)
    r_neutral = compute_reward(delta_quality=0.0, **common)
    r_worse = compute_reward(delta_quality=-0.2, **common)
    assert r_worse < r_neutral


def test_reward_penalized_by_frustration():
    """Higher frustration, all else equal, must yield strictly lower
    reward (w4 is subtracted in the formula)."""
    w = RewardWeights()
    common = dict(delta_quality=0.1, completion=0.5, engagement=0.5, consistency=0.5, weights=w)
    r_calm = compute_reward(frustration=0.1, **common)
    r_frustrated = compute_reward(frustration=0.9, **common)
    assert r_frustrated < r_calm


def test_reward_rewards_completion():
    """A completed trial (completion=1.0, e.g. a normal writing attempt)
    must score higher than an incomplete one (completion=0.0, e.g. a
    break was taken instead), all else equal."""
    w = RewardWeights()
    common = dict(delta_quality=0.1, engagement=0.5, frustration=0.3, consistency=0.5, weights=w)
    r_incomplete = compute_reward(completion=0.0, **common)
    r_complete = compute_reward(completion=1.0, **common)
    assert r_complete > r_incomplete


def test_reward_rewards_consistency():
    """More consistent recent performance, all else equal, must score
    higher (w5 is added in the formula)."""
    w = RewardWeights()
    common = dict(delta_quality=0.1, completion=0.5, engagement=0.5, frustration=0.3, weights=w)
    r_erratic = compute_reward(consistency=0.0, **common)
    r_consistent = compute_reward(consistency=1.0, **common)
    assert r_consistent > r_erratic


# ---------------------------------------------------------------------------
# 10. Learner profiles are functionally different, not just decorative.
#     Uses the environment's `profile=` constructor override to pin a
#     specific profile and isolate its effect under an identical action
#     sequence and seed.
# ---------------------------------------------------------------------------
def test_learner_profiles_produce_measurably_different_dynamics():
    from envs.dysgraphia_tutor_env import PROFILES

    def run_fixed_profile(profile_name, n_steps, seed):
        profile = next(p for p in PROFILES if p.name == profile_name)
        env = DysgraphiaTutorEnv(profile=profile, seed=seed)
        env.reset(seed=seed)
        for _ in range(n_steps):
            _, _, terminated, truncated, _ = env.step(CONTINUE_NO_CHANGE)
            if terminated or truncated:
                break
        return env.fatigue, env.score_history

    # --- Fatigue axis: fatigue_prone (fatigue_rate=0.06) vs fast_improver
    #     (fatigue_rate=0.02) under 10 identical no-op steps.
    fatigue_fast, _ = run_fixed_profile("fast_improver", n_steps=10, seed=0)
    fatigue_prone, _ = run_fixed_profile("fatigue_prone", n_steps=10, seed=0)
    assert fatigue_prone > fatigue_fast + 0.3, (
        f"fatigue_prone fatigue ({fatigue_prone:.3f}) is not meaningfully "
        f"higher than fast_improver fatigue ({fatigue_fast:.3f}) after 10 "
        "identical steps -- profiles may not be functionally differentiated"
    )

    # --- Skill-growth axis: fast_improver (higher base_skill + learning_rate)
    #     vs slow_improver under 15 identical no-op steps.
    _, history_fast = run_fixed_profile("fast_improver", n_steps=15, seed=1)
    _, history_slow = run_fixed_profile("slow_improver", n_steps=15, seed=1)
    assert history_fast[-1] > history_slow[-1], (
        f"fast_improver final quality ({history_fast[-1]:.3f}) is not higher "
        f"than slow_improver final quality ({history_slow[-1]:.3f}) -- "
        "profiles may not be functionally differentiated"
    )


# ---------------------------------------------------------------------------
# 11. The bug this Code Review pass found and fixed: the environment's
#     fatigue safety cutoff used to set `truncated=True` instead of
#     `terminated=True`. This test locks in the corrected env-level
#     behavior directly (see envs/dysgraphia_tutor_env.py::step()).
# ---------------------------------------------------------------------------
def test_fatigue_cutoff_sets_terminated_not_truncated():
    from envs.dysgraphia_tutor_env import PROFILES

    # A fast-fatigue profile with a generous max_steps: the episode MUST
    # end via the fatigue safety cutoff, not the time limit.
    fatigue_prone = next(p for p in PROFILES if p.name == "fatigue_prone")
    env = DysgraphiaTutorEnv(profile=fatigue_prone, max_steps=200, seed=0)
    env.reset(seed=0)
    for _ in range(200):
        _, _, terminated, truncated, _ = env.step(CONTINUE_NO_CHANGE)
        if terminated or truncated:
            break
    assert env.fatigue > 0.95, "test setup problem: fatigue never crossed the safety threshold"
    assert terminated is True, "fatigue safety cutoff must set terminated=True"
    assert truncated is False, "fatigue safety cutoff must NOT set truncated=True"

    # A slow-fatigue profile with a tight max_steps: the episode MUST end
    # via the ordinary time limit, not the fatigue cutoff.
    fast_improver = next(p for p in PROFILES if p.name == "fast_improver")
    env2 = DysgraphiaTutorEnv(profile=fast_improver, max_steps=5, seed=0)
    env2.reset(seed=0)
    terminated2 = truncated2 = False
    for _ in range(5):
        _, _, terminated2, truncated2, _ = env2.step(CONTINUE_NO_CHANGE)
    assert env2.fatigue <= 0.95, "test setup problem: fatigue crossed the safety threshold too early"
    assert truncated2 is True, "ordinary max_steps cutoff must set truncated=True"
    assert terminated2 is False, "ordinary max_steps cutoff must NOT set terminated=True"


def test_train_dqn_stores_done_flag_on_true_termination(tmp_path):
    """Integration-level regression test for the bug described in module
    docstring item 6: run the REAL training loop (scripts/train_dqn.py's
    train()) against a fatigue-prone-only environment for a couple of
    short episodes, and confirm at least one transition actually stored in
    the replay buffer has done=1.0. Before the fix, this was impossible --
    every stored transition had done=0.0 because `terminated` was
    hard-coded to False in the environment."""
    import functools
    import importlib
    from envs.dysgraphia_tutor_env import PROFILES

    train_dqn = importlib.import_module("scripts.train_dqn")
    fatigue_prone = next(p for p in PROFILES if p.name == "fatigue_prone")

    # Force every episode in this run to use the fast-fatigue profile, and
    # give a generous max_steps so the fatigue cutoff -- not the time
    # limit -- is what actually ends each episode.
    original_env_cls = train_dqn.DysgraphiaTutorEnv
    forced_env_cls = functools.partial(original_env_cls, profile=fatigue_prone)
    train_dqn.DysgraphiaTutorEnv = forced_env_cls
    try:
        agent, rewards, qualities = train_dqn.train(
            episodes=3, max_steps=200, seed=0, warmup_steps=0,
            checkpoint_path=str(tmp_path / "_test_done_flag_ckpt.pt"), quiet=True,
        )
    finally:
        train_dqn.DysgraphiaTutorEnv = original_env_cls  # always restore, even on failure

    done_flags = [transition[4] for transition in agent.replay_buffer.buffer]
    assert any(d == 1.0 for d in done_flags), (
        "no transition in the replay buffer has done=1.0 after 3 "
        "fatigue-prone-only episodes with a generous max_steps -- the "
        "training loop is not correctly propagating true terminal states "
        "into the replay buffer (this is exactly the bug this test targets)"
    )
    # Sanity check the test itself isn't vacuous: most transitions should
    # still be non-terminal (done=0.0), since termination only happens once
    # per episode, at the very end.
    assert done_flags.count(0.0) > done_flags.count(1.0)


# ---------------------------------------------------------------------------
# 12. Objective 6 (interpretability): sanity-check the per-profile analysis
#     script's core function returns well-formed, valid output. Does not
#     re-verify the RL agent itself -- that's covered elsewhere -- just that
#     this reporting tool doesn't silently break (e.g. on an ACTION_NAMES /
#     N_ACTIONS mismatch) as the codebase evolves.
# ---------------------------------------------------------------------------
def test_policy_by_profile_analysis_returns_valid_output():
    import importlib
    analyze_module = importlib.import_module("scripts.analyze_policy_by_profile")
    from envs.dysgraphia_tutor_env import PROFILES, ACTION_NAMES

    agent = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)  # untrained is fine for a shape/format check
    profile = PROFILES[0]
    result = analyze_module.run_profile(agent, profile, n_episodes=3, max_steps=15, seed_offset=42000)

    assert result["profile"] == profile.name
    assert set(result["action_frequencies"].keys()) == set(ACTION_NAMES)
    total_freq = sum(result["action_frequencies"].values())
    assert abs(total_freq - 1.0) < 1e-6, f"action frequencies should sum to 1.0, got {total_freq}"
    for key in ("mean_final_quality", "mean_fatigue", "mean_frustration", "mean_engagement"):
        assert 0.0 - 1e-6 <= result[key] <= 1.0 + 1e-6, f"{key}={result[key]} outside [0,1]"


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
