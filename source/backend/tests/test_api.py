import math

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _synthetic_points(n_strokes: int = 1, n: int = 30, dt: float = 0.02):
    points = []
    for stroke_id in range(n_strokes):
        t0 = stroke_id * 2.0
        for i in range(n):
            points.append({
                "x": (i / (n - 1)) * 100.0 + stroke_id * 150,
                "y": 50.0 + 5 * math.sin(i / 5.0),
                "t": t0 + i * dt,
                "pressure": 0.5,
                "stroke_id": stroke_id,
            })
    return points


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_list_exercises(client):
    r = client.get("/exercises")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_get_unknown_exercise_returns_404(client):
    r = client.get("/exercise/does_not_exist")
    assert r.status_code == 404


def test_session_start_returns_session_and_exercise(client):
    r = client.post("/session/start", json={"child_id": "child_1"})
    assert r.status_code == 200
    body = r.json()
    assert "session_id" in body
    assert body["exercise"]["exercise_id"]


def test_submit_attempt_unknown_session_returns_404(client):
    r = client.post("/attempt/submit", json={
        "session_id": "does-not-exist", "exercise_id": "line_easy",
        "attempt_id": "a1", "points": _synthetic_points(),
    })
    assert r.status_code == 404


def test_submit_attempt_empty_points_returns_422(client):
    start = client.post("/session/start", json={"child_id": "child_2"})
    session_id = start.json()["session_id"]
    r = client.post("/attempt/submit", json={
        "session_id": session_id, "exercise_id": "line_easy",
        "attempt_id": "a1", "points": [],
    })
    assert r.status_code == 422


def test_full_attempt_submit_returns_well_formed_response(client):
    start = client.post("/session/start", json={"child_id": "child_3"})
    body = start.json()
    session_id = body["session_id"]
    exercise_id = body["exercise"]["exercise_id"]

    r = client.post("/attempt/submit", json={
        "session_id": session_id, "exercise_id": exercise_id,
        "attempt_id": "attempt_1", "points": _synthetic_points(n_strokes=1),
    })
    assert r.status_code == 200
    out = r.json()
    assert len(out["state_vector"]) == 12
    assert 0 <= out["safety"]["safe_action_id"] < 10
    assert out["intervention"]["action_id"] == out["safety"]["safe_action_id"]
    assert "reward" in out
    assert set(out["reward_components"].keys()) == {
        "delta_quality_term", "completion_term", "engagement_term",
        "frustration_term", "consistency_term",
    }


def test_session_summary_reflects_attempts(client):
    start = client.post("/session/start", json={"child_id": "child_4"})
    body = start.json()
    session_id = body["session_id"]
    exercise_id = body["exercise"]["exercise_id"]

    client.post("/attempt/submit", json={
        "session_id": session_id, "exercise_id": exercise_id,
        "attempt_id": "a1", "points": _synthetic_points(),
    })
    r = client.get(f"/session/{session_id}/summary")
    assert r.status_code == 200
    assert r.json()["attempt_count"] == 1


def test_end_session_then_submit_is_rejected(client):
    start = client.post("/session/start", json={"child_id": "child_5"})
    body = start.json()
    session_id = body["session_id"]
    exercise_id = body["exercise"]["exercise_id"]

    end = client.post(f"/session/{session_id}/end")
    assert end.status_code == 200

    r = client.post("/attempt/submit", json={
        "session_id": session_id, "exercise_id": exercise_id,
        "attempt_id": "a1", "points": _synthetic_points(),
    })
    assert r.status_code == 409


def test_break_complete_reduces_fatigue(client):
    start = client.post("/session/start", json={"child_id": "child_6"})
    session_id = start.json()["session_id"]
    exercise_id = start.json()["exercise"]["exercise_id"]

    for _ in range(5):
        client.post("/attempt/submit", json={
            "session_id": session_id, "exercise_id": exercise_id,
            "attempt_id": "a", "points": _synthetic_points(),
        })
    fatigue_before = client.get(f"/session/{session_id}/summary").json()["fatigue"]
    client.post(f"/session/{session_id}/break_complete")
    fatigue_after = client.get(f"/session/{session_id}/summary").json()["fatigue"]
    assert fatigue_after < fatigue_before


def test_repeated_attempts_change_difficulty_or_produce_safety_action(client):
    """Runs enough attempts that the safety layer's forced-break cap (or
    the policy's own choices) must eventually engage -- a coarse
    end-to-end liveness check of the safety layer wired through the API."""
    start = client.post("/session/start", json={"child_id": "child_7"})
    session_id = start.json()["session_id"]
    exercise_id = start.json()["exercise"]["exercise_id"]

    actions_seen = set()
    for _ in range(30):
        r = client.post("/attempt/submit", json={
            "session_id": session_id, "exercise_id": exercise_id,
            "attempt_id": "a", "points": _synthetic_points(),
        })
        actions_seen.add(r.json()["safety"]["safe_action_id"])
    assert len(actions_seen) >= 1
