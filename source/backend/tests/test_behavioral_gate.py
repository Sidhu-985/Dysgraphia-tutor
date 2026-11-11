"""Behavioural gate + fallback (Phase D). The original shipped checkpoint
(kept unchanged as checkpoints/legacy/double_dqn_v1_original.pt) is the
known-bad case: it over-selects SUGGEST_BREAK for low-fatigue learners."""
from pathlib import Path

import numpy as np

from app.dysgraphia_rl_bridge import rl_root, ACTION_NAMES
from app.rl_integration.behavioral_gate import run_behavioral_gate, low_fatigue_population
from app.rl_integration.policy_service import PolicyService

LEGACY = Path(rl_root) / "checkpoints" / "legacy" / "double_dqn_v1_original.pt"
BREAK = ACTION_NAMES.index("SUGGEST_BREAK")


def test_original_checkpoint_is_rejected_and_fallback_serves():
    svc = PolicyService(checkpoint_path=str(LEGACY))
    st = svc.status()
    assert st["checkpoint_loaded"] is True
    assert st["behavioral_gate_passed"] is False
    assert st["policy_mode"] == "rule_based_fallback"
    assert any("G4" in r for r in st["fallback_reasons"])
    d = svc.select_action(low_fatigue_population(n=1)[0])
    assert d.source == "rule_based_fallback"


def test_missing_checkpoint_falls_back():
    svc = PolicyService(checkpoint_path="/nonexistent/ckpt.pt")
    assert svc.mode == "rule_based_fallback" and not svc.checkpoint_loaded


def test_corrupt_checkpoint_falls_back(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a checkpoint")
    svc = PolicyService(checkpoint_path=str(bad))
    assert svc.mode == "rule_based_fallback"
    assert svc.load_error


def test_gate_fails_a_policy_that_always_breaks():
    def always_break(states):
        q = np.zeros((len(states), len(ACTION_NAMES)))
        q[:, BREAK] = 1.0
        return q
    rep = run_behavioral_gate(always_break)
    assert not rep.passed
    assert any(f.startswith("G1") for f in rep.failures) and any(f.startswith("G4") for f in rep.failures)


def test_gate_passes_a_policy_that_only_continues():
    def always_continue(states):
        q = np.zeros((len(states), len(ACTION_NAMES)))
        q[:, ACTION_NAMES.index("CONTINUE_NO_CHANGE")] = 1.0
        return q
    assert run_behavioral_gate(always_continue).passed


def test_gate_rejects_non_finite_q_values():
    rep = run_behavioral_gate(lambda s: np.full((len(s), len(ACTION_NAMES)), np.nan))
    assert not rep.passed and "G5" in rep.failures[0]


def test_gate_population_contains_deployment_placeholder_states():
    pop = low_fatigue_population()
    assert np.sum((np.isclose(pop[:, 2], 0.5)) & (np.isclose(pop[:, 5], 0.3))) >= len(pop) // 2
    assert pop[:, 7].max() < 0.4 and pop[:, 10].max() < 0.4
