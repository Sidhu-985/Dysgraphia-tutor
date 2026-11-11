import numpy as np
import pytest

from app.rl_integration.policy_service import PolicyService
from app.dysgraphia_rl_bridge import N_ACTIONS, ENV_STATE_DIM


def test_policy_service_loads_shipped_checkpoint():
    service = PolicyService()
    assert service.checkpoint_loaded is True


def test_select_action_returns_valid_action_id():
    service = PolicyService()
    state = np.zeros(ENV_STATE_DIM, dtype=np.float32)
    decision = service.select_action(state)
    assert 0 <= decision.action_id < N_ACTIONS
    assert decision.action_name
    assert len(decision.q_values) == N_ACTIONS


def test_greedy_policy_is_deterministic_given_same_state():
    service = PolicyService()
    state = np.random.default_rng(0).uniform(0, 1, size=ENV_STATE_DIM).astype(np.float32)
    d1 = service.select_action(state, epsilon=0.0)
    d2 = service.select_action(state, epsilon=0.0)
    assert d1.action_id == d2.action_id
    assert d1.q_values == d2.q_values


# As of 2026-11-10, the deployed checkpoint is childstart_R12_4_best.pt
# (Phase 5 full run, locked protocol configs/phase5.yaml v1). It PASSES the
# behavioural gate, so PolicyService serves the DQN policy and its action
# IS the DQN argmax. The xfail marker that was here for the legacy gated-
# out checkpoint has been removed (the project's own comment said: "once
# a checkpoint passes the gate, this marker must be removed"). The DQN
# inference property itself is also checked below with the gate explicitly
# disabled.
def test_q_values_correspond_to_argmax_action():
    service = PolicyService()
    # If the deployed checkpoint is gated out (e.g. an old legacy file is
    # reinstalled), the rule-based fallback is used and its action is not
    # the DQN argmax -- in that case the test is skipped rather than
    # failing, because the rule-based fallback is a valid deployment mode.
    if service.mode != "dqn":
        pytest.skip("deployed checkpoint is gated out; rule-based fallback is active")
    state = np.random.default_rng(1).uniform(0, 1, size=ENV_STATE_DIM).astype(np.float32)
    decision = service.select_action(state, epsilon=0.0)
    assert decision.action_id == int(np.argmax(decision.q_values))


def test_dqn_inference_path_returns_argmax():
    """The DQN path itself (used whenever a checkpoint passes the gate): the
    chosen action is the argmax of the reported Q-values. Called directly,
    because the gate can no longer be switched off for live decisions."""
    service = PolicyService()
    for seed in range(5):
        state = np.random.default_rng(seed).uniform(0, 1, size=ENV_STATE_DIM).astype(np.float32)
        d = service._dqn_decision(state, 0.0)
        assert d.source == "dqn" and d.action_id == int(np.argmax(d.q_values))


def test_gate_cannot_be_switched_off_for_live_decisions():
    """RL_BEHAVIORAL_GATE_ENABLED=false (or behavioral_gate=False) used to let
    a gated-out checkpoint make decisions. It is now only reported."""
    service = PolicyService(behavioral_gate=False)
    assert service.gate_report is not None, "the gate always runs on a loaded checkpoint"
    if not service.gate_report.passed:
        assert service.mode == "rule_based_fallback"
        assert service.status()["behavioral_gate_config_requested"] is False


def test_fallback_decisions_come_from_rule_based_agent():
    from agents.baselines import RuleBasedAgent
    service = PolicyService()
    if service.mode != "rule_based_fallback":
        pytest.skip("deployed checkpoint passes the gate")
    rb = RuleBasedAgent()
    for seed in range(5):
        state = np.random.default_rng(seed).uniform(0, 1, size=ENV_STATE_DIM).astype(np.float32)
        d = service.select_action(state)
        assert d.source == "rule_based_fallback" and d.action_id == rb.act(state)
