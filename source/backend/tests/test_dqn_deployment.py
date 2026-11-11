"""DQN deployment: checkpoint, architecture, gate, /health and decision source.

These tests hold whichever checkpoint is configured. They never assume the
gate passes: every assertion about "dqn" is conditional on the gate result
that the backend itself computed."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from app.config import settings
from app.dysgraphia_rl_bridge import ACTION_NAMES, ENV_STATE_DIM, N_ACTIONS
from app.main import app
from app.rl_integration import behavioral_gate as BG
from app.rl_integration import policy_service as PS
from app.rl_integration.behavioral_gate import GateReport
from app.rl_integration.policy_service import PolicyService, get_policy_service

client = TestClient(app)


def test_dimensions_and_action_mapping():
    assert ENV_STATE_DIM == 12 and N_ACTIONS == 10
    assert ACTION_NAMES == ["DECREASE_DIFFICULTY", "INCREASE_DIFFICULTY", "SHOW_HINT", "ENLARGE_TEMPLATE",
                            "REPLAY_STROKE_ANIMATION", "SWITCH_TO_TRACING_MODE", "SWITCH_TO_FREE_WRITING_MODE",
                            "SUGGEST_BREAK", "REPEAT_SAME_LETTER", "CONTINUE_NO_CHANGE"]


def test_configured_checkpoint_loads_with_the_double_dqn_architecture():
    s = PolicyService()
    assert s.checkpoint_loaded, s.load_error
    layers = s.architecture()["linear_layers"]
    assert layers == [{"in": 12, "out": 128}, {"in": 128, "out": 64}, {"in": 64, "out": 10}]
    ck = torch.load(s.checkpoint_path, map_location="cpu")
    assert set(ck) >= {"online_net", "target_net"}
    for k, v in ck["online_net"].items():   # online and target networks have identical shapes
        assert ck["target_net"][k].shape == v.shape


def test_status_reports_hash_path_architecture_and_gate():
    s = PolicyService()
    st = s.status()
    assert st["checkpoint_sha256"] == hashlib.sha256(Path(s.checkpoint_path).read_bytes()).hexdigest()
    assert st["architecture"]["state_dim"] == 12 and st["architecture"]["n_actions"] == 10
    assert st["behavioral_gate_enabled"] is True
    assert isinstance(st["behavioral_gate_passed"], bool)
    assert st["safety_layer_active"] is True
    assert st["dqn_making_decisions"] == (st["policy_mode"] == "dqn")


def test_policy_mode_dqn_is_impossible_while_the_gate_fails(monkeypatch):
    failing = GateReport(False, ["G4: forced failure for this test"])
    monkeypatch.setattr(PS, "run_behavioral_gate", lambda q_fn: failing)
    s = PolicyService()
    assert s.mode == "rule_based_fallback"
    st = s.status()
    assert st["policy_mode"] == "rule_based_fallback" and st["dqn_making_decisions"] is False
    assert "G4: forced failure for this test" in st["fallback_reasons"]
    s2 = PolicyService(behavioral_gate=False)       # the old off switch cannot bypass it
    assert s2.mode == "rule_based_fallback"
    d = s.select_action(np.full(ENV_STATE_DIM, 0.5, np.float32))
    assert d.source == "rule_based_fallback"


def test_passing_gate_gives_dqn_decisions_that_are_the_greedy_argmax(monkeypatch):
    monkeypatch.setattr(PS, "run_behavioral_gate", lambda q_fn: GateReport(True, []))
    s = PolicyService()
    assert s.mode == "dqn" and s.status()["dqn_making_decisions"] is True
    for seed in range(10):
        state = np.random.default_rng(seed).uniform(0, 1, ENV_STATE_DIM).astype(np.float32)
        d1, d2 = s.select_action(state, epsilon=0.0), s.select_action(state, epsilon=0.0)
        assert d1.source == "dqn" and d1.action_id == int(np.argmax(d1.q_values))
        assert (d1.action_id, d1.q_values) == (d2.action_id, d2.q_values), "deterministic greedy inference"


def test_missing_checkpoint_falls_back_and_says_why(tmp_path):
    s = PolicyService(checkpoint_path=str(tmp_path / "nope.pt"))
    assert s.mode == "rule_based_fallback" and not s.checkpoint_loaded
    assert any("not found" in r for r in s.status()["fallback_reasons"])


def test_incompatible_checkpoint_falls_back(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"online_net": {"w": torch.zeros(3)}, "target_net": {}}, bad)
    s = PolicyService(checkpoint_path=str(bad))
    assert s.mode == "rule_based_fallback" and not s.checkpoint_loaded


@pytest.mark.parametrize("state", [np.zeros(5, np.float32), np.full(12, np.nan, np.float32),
                                   np.full(12, 50.0, np.float32)])
def test_invalid_states_never_reach_the_network(state, monkeypatch):
    s = PolicyService()
    called = []
    monkeypatch.setattr(s, "_q_values_batch", lambda x: called.append(1) or np.zeros((len(x), 10)))
    d = s.select_action(state)
    assert d.source == "safe_default_invalid_state" and called == []
    assert d.action_name == "CONTINUE_NO_CHANGE"


def test_inference_failure_is_handled(monkeypatch):
    monkeypatch.setattr(PS, "run_behavioral_gate", lambda q_fn: GateReport(True, []))
    s = PolicyService()

    def boom(_):
        raise RuntimeError("simulated inference failure")
    monkeypatch.setattr(s, "_q_values_batch", boom)
    d = s.select_action(np.full(ENV_STATE_DIM, 0.5, np.float32))
    assert d.source == "rule_based_after_inference_error"
    assert "simulated inference failure" in d.notes[0]


def test_gate_probes_and_threshold_are_unchanged():
    """The gate itself is not edited to fit a model."""
    assert BG.MAX_LOW_FATIGUE_BREAK_RATE == 0.25
    src = Path(BG.__file__).read_text()
    for probe in ("G1_fresh_good_writer", "G2_mid_session_writer", "G3_struggling_writer", "G6"):
        assert probe in src
    assert len(BG.low_fatigue_population()) == 300


def test_health_matches_the_live_service():
    h = client.get("/health").json()
    st = get_policy_service().status()
    assert h["policy_mode"] == st["policy_mode"] == h["policy"]["policy_mode"]
    assert h["dqn_making_decisions"] == (h["policy_mode"] == "dqn")
    assert h["behavioral_gate_enabled"] is True
    assert h["rl_checkpoint_sha256"] == st["checkpoint_sha256"]
    if h["behavioral_gate_passed"] is False:
        assert h["policy_mode"] == "rule_based_fallback"


def _trace(strokes, scale=3.0):
    pts, t = [], 0.0
    for sid, s in enumerate(strokes):
        for i, (x, y) in enumerate(s):
            pts.append({"x": x * scale, "y": y * scale, "t": t, "pressure": None, "stroke_id": sid,
                        "is_stroke_start": i == 0, "is_stroke_end": i == len(s) - 1})
            t += 0.02
        t += 0.3
    return pts


def test_every_attempt_reports_its_real_decision_source():
    start = client.post("/session/start", json={"child_id": "P-0042", "exercise_id": "letter_lower_b",
                                                "data_source": "synthetic_test"}).json()
    ex = start["exercise"]
    for i in range(3):
        r = client.post("/attempt/submit", json={"session_id": start["session_id"], "exercise_id": ex["exercise_id"],
                                                 "attempt_id": f"src-{i}", "points": _trace(ex["template_strokes"])})
        body = r.json()
        mode = get_policy_service().mode
        assert body["policy_mode"] == mode
        expected = "dqn" if mode == "dqn" else "rule_based_fallback"
        assert body["decision_source"] == expected == body["policy"]["source"]
        assert body["safety_overridden"] == body["safety"]["overridden"]
        ex = body["next_exercise"]


def test_safety_layer_runs_in_either_mode():
    """A very fatigued learner is offered a break whatever the policy chose."""
    from app.safety.safety_layer import apply_safety_layer
    from app.state.session_context import SessionContext
    s = SessionContext(session_id="s", child_id="P-0001")
    feats = {"fatigue_proxy": 0.95, "frustration": 0.1, "prev_score": 0.6}
    d = apply_safety_layer(ACTION_NAMES.index("INCREASE_DIFFICULTY"), s, feats)
    assert d.safe_action_name == "SUGGEST_BREAK" and d.overridden


def test_session_can_start_in_free_writing_mode():
    r = client.post("/session/start", json={"child_id": "P-0042", "exercise_id": "letter_upper_k",
                                            "practice_mode": "free_writing"})
    assert r.status_code == 200 and r.json()["practice_mode"] == "free_writing"
    r = client.post("/session/start", json={"child_id": "P-0042", "exercise_id": "letter_upper_k"})
    assert r.json()["practice_mode"] == "tracing"
    assert client.post("/session/start", json={"child_id": "P-0042", "practice_mode": "x"}).status_code == 422
