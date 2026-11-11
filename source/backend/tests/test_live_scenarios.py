"""Live API scenarios (in-process TestClient). The same scenario functions
run against a real uvicorn server in Phase E (scripts/live_scenarios.py)."""
import importlib.util
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

_spec = importlib.util.spec_from_file_location(
    "live_scenarios", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "live_scenarios.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


@pytest.fixture
def client():
    return TestClient(app)


def test_s1_fresh_good_writer_gets_no_break(client):
    r = sc.s1_fresh_good_writer(client)
    assert r["pass"], r


def test_s2_fatigue_above_threshold_always_gets_break(client):
    r = sc.s2_high_fatigue(client)
    assert r["pass"], r


def test_s3_s4_struggling_writer_supported_and_never_escalated(client):
    s3, s4 = sc.s3_s4_struggling_writer(client)
    assert s3["pass"], s3
    assert s4["pass"], s4


def test_s5_sessions_are_separate(client):
    r = sc.s5_session_separation(client)
    assert r["pass"], r


def test_s6_transitions_link_decisions_to_next_observed_attempt(client):
    r = sc.s6_transition_integrity(client)
    assert r["pass"], r


def test_s7_data_governance(client):
    r = sc.s7_data_governance(client)
    assert r["pass"], r


def test_inference_failure_falls_back_safely(client, monkeypatch):
    from app.rl_integration.policy_service import get_policy_service
    svc = get_policy_service()

    def boom(_):
        raise RuntimeError("simulated inference failure")

    monkeypatch.setattr(svc, "_q_values_batch", boom)
    monkeypatch.setattr(type(svc), "mode", property(lambda self: "dqn"))
    s = client.post("/session/start", json={"child_id": "scenario_fail", "data_source": "synthetic_test"}).json()
    tpl = client.get("/exercise/line_easy").json()["reference_template"]
    r = client.post("/attempt/submit", json={"session_id": s["session_id"], "exercise_id": "line_easy",
                                             "attempt_id": "x", "points": sc.clean_trace(tpl)})
    assert r.status_code == 200
    body = r.json()
    assert body["policy"]["source"] == "rule_based_after_inference_error"
    assert 0 <= body["safety"]["safe_action_id"] < 10


def test_invalid_state_never_reaches_the_model():
    from app.rl_integration.policy_service import get_policy_service
    svc = get_policy_service()
    bad = np.full(12, np.nan, dtype=np.float32)
    d = svc.select_action(bad)
    assert d.source == "safe_default_invalid_state" and d.action_name == "CONTINUE_NO_CHANGE"
    d2 = svc.select_action(np.zeros(5, dtype=np.float32))
    assert d2.source == "safe_default_invalid_state"
    out_of_range = np.zeros(12, dtype=np.float32)
    out_of_range[7] = 7.0
    assert svc.select_action(out_of_range).source == "safe_default_invalid_state"


def test_degenerate_features_rejected_without_mutating_session(client, monkeypatch):
    import app.api.routes as routes
    from app.features.static_features import StaticFeatures
    monkeypatch.setattr(routes, "extract_static_features",
                        lambda *a, **k: StaticFeatures(float("nan"), 0.5, 0.5, 1, 1, 1, 0.5, 0.5, 0.0, 1))
    s = client.post("/session/start", json={"child_id": "scenario_nan", "data_source": "synthetic_test"}).json()
    tpl = client.get("/exercise/line_easy").json()["reference_template"]
    r = client.post("/attempt/submit", json={"session_id": s["session_id"], "exercise_id": "line_easy",
                                             "attempt_id": "x", "points": sc.clean_trace(tpl)})
    assert r.status_code == 422
    summary = client.get(f"/session/{s['session_id']}/summary").json()
    assert summary["attempt_count"] == 0 and summary["score_history"] == []


def test_health_exposes_policy_status(client):
    h = client.get("/health").json()
    assert h["policy"]["policy_mode"] in ("dqn", "rule_based_fallback")
    assert "behavioral_gate_passed" in h["policy"]
    assert h["real_child_collection_enabled"] is False


def test_non_difficulty_actions_keep_the_current_exercise(client):
    s = client.post("/session/start", json={"child_id": "scenario_repeat", "exercise_id": "loop_medium",
                                            "data_source": "synthetic_test"}).json()
    ex = "loop_medium"
    for i in range(6):
        tpl = client.get(f"/exercise/{ex}").json()["reference_template"]
        body = client.post("/attempt/submit", json={"session_id": s["session_id"], "exercise_id": ex,
                                                    "attempt_id": f"r{i}", "points": sc.clean_trace(tpl)}).json()
        nxt = body["next_exercise"]
        if body["intervention"]["type"] == "difficulty_adjustment":
            assert nxt["difficulty"] == body["presentation"]["difficulty_level"]
        else:
            assert nxt["exercise_id"] == ex
        ex = nxt["exercise_id"]


def test_nan_coordinate_gets_a_422_not_a_crash(client):
    """Phase E regression: a NaN in the body used to crash the 422 response."""
    s = client.post("/session/start", json={"child_id": "scenario_nan_body", "data_source": "synthetic_test"}).json()
    raw = ('{"session_id": "%s", "exercise_id": "line_easy", "attempt_id": "n", "points": '
           '[{"x": NaN, "y": 1.0, "t": 0.0, "pressure": null, "stroke_id": 0}]}' % s["session_id"])
    r = client.post("/attempt/submit", content=raw.encode(), headers={"Content-Type": "application/json"})
    assert r.status_code == 422
    assert "finite" in r.text


def test_concurrent_first_requests_share_one_store(monkeypatch):
    """Phase E regression: lazily created stores must be singletons even
    when the first requests arrive concurrently on several threads."""
    import threading
    import app.storage as storage
    monkeypatch.setattr(storage, "_session_store", None)
    monkeypatch.setattr(storage, "_attempt_log_store", None)
    real_init = storage.SessionStore.__init__

    def slow_init(self):
        import time
        time.sleep(0.05)  # widen the race window
        real_init(self)

    monkeypatch.setattr(storage.SessionStore, "__init__", slow_init)
    got = []
    threads = [threading.Thread(target=lambda: got.append(storage.get_session_store())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(s) for s in got}) == 1


def test_duplicate_submission_is_idempotent(client):
    """Phase G: resending the same attempt (e.g. a network retry) must not
    count it twice or add a second transition. Reusing an attempt_id for
    different points is refused."""
    s = client.post("/session/start", json={"child_id": "scenario_dup", "data_source": "synthetic_test"}).json()
    sid = s["session_id"]
    tpl = client.get("/exercise/line_easy").json()["reference_template"]
    body = {"session_id": sid, "exercise_id": "line_easy", "attempt_id": "dup-1", "points": sc.clean_trace(tpl)}
    first = client.post("/attempt/submit", json=body).json()
    again = client.post("/attempt/submit", json=body).json()
    assert again["duplicate_submission"] is True and first["duplicate_submission"] is False
    assert again["state_vector"] == first["state_vector"]
    assert client.get(f"/session/{sid}/summary").json()["attempt_count"] == 1
    other = dict(body, points=sc.poor_trace(3))
    assert client.post("/attempt/submit", json=other).status_code == 409
    second = client.post("/attempt/submit", json=dict(body, attempt_id="dup-2")).json()
    assert second["duplicate_submission"] is False
    client.post(f"/session/{sid}/end")
    rows = client.get(f"/session/{sid}/transitions").json()
    assert len(rows) == 2  # one observed transition (dup-1 -> dup-2) + one terminal
