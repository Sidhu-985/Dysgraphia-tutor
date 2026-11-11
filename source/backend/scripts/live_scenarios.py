"""
Live API behaviour scenarios (Phase D/E). SYNTHETIC test traffic only.

Each scenario drives the HTTP API with synthetic pen traces that are
labelled data_source="synthetic_test" and returns a result dict. The same
functions run in two places:
  * in-process, through FastAPI's TestClient (tests/test_live_scenarios.py);
  * against a real running uvicorn server over the network:
        python scripts/live_scenarios.py --base-url http://127.0.0.1:8000 [--out file.json]

Scenarios:
  S1 fresh good writer: clean traces. No break is delivered while the
     fatigue proxy is low. Raw policy break requests are also reported.
  S2 fatigue > 0.7: attempts with long in-stroke pauses push the fatigue
     proxy up. Every decision taken at fatigue >= 0.7 must be SUGGEST_BREAK.
  S3 high frustration: repeated poor traces. Every decision at
     frustration >= 0.75 must be supportive (hint, easier task, larger
     template, demo, tracing or break).
  S4 repeated failures: difficulty never increases while the learner is
     failing, and INCREASE_DIFFICULTY is never delivered after a failure.
  S5 session separation: two interleaved sessions keep separate histories,
     and ending one does not affect the other.
  S6 transition integrity: every logged transition (s, a, r, s') links a
     decision to the NEXT observed attempt. The final decision is stored
     with reward null.
  S7 data governance: non-pseudonymous ids are rejected, and
     real_child_study is refused while collection is not approved.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUGGEST_BREAK = 7
INCREASE_DIFFICULTY = 1
SUPPORTIVE = {0, 2, 3, 4, 5, 7}


# --------------------------------------------------------------------------
# synthetic traces (no pressure fabricated: pressure is null throughout)
# --------------------------------------------------------------------------
def _resample(xy, n):
    xy = np.asarray(xy, float)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], axis=1)


# Synthetic traces take about 3 s per shape at 120 Hz, a plausible tracing
# pace. The earlier 0.75 s traces pushed writing_speed to its 2.0 cap
# (evidence/phase_d/state_distribution_comparison.json).
def clean_trace(template, n=360, rate_hz=120.0):
    xy = _resample(template, n)
    return [{"x": float(x), "y": float(y), "t": i / rate_hz, "pressure": None, "stroke_id": 0}
            for i, (x, y) in enumerate(xy)]


def paused_trace(template, n_segments=8, pause_s=2.5, rate_hz=120.0):
    """Clean shape, but the pen hesitates n_segments times for pause_s
    seconds each (pauses inside the stroke raise the fatigue proxy)."""
    xy = _resample(template, 360)
    pts, t = [], 0.0
    seg_len = len(xy) // n_segments
    for i, (x, y) in enumerate(xy):
        if i > 0 and i % seg_len == 0:
            t += pause_s
        pts.append({"x": float(x), "y": float(y), "t": t, "pressure": None, "stroke_id": 0})
        t += 1.0 / rate_hz
    return pts


def poor_trace(seed, n=360, rate_hz=120.0):
    """A wandering scribble unrelated to the template (low quality proxy)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 3.0, size=(n, 2))
    xy = np.cumsum(steps, axis=0) + 50
    return [{"x": float(x), "y": float(y), "t": i / rate_hz, "pressure": None, "stroke_id": 0}
            for i, (x, y) in enumerate(xy)]


# --------------------------------------------------------------------------
def _ok(r):
    if r.status_code != 200:
        raise AssertionError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def _start(client, child_id, exercise_id="line_easy"):
    return _ok(client.post("/session/start", json={"child_id": child_id, "exercise_id": exercise_id,
                                                   "data_source": "synthetic_test"}))


def _template(client, exercise_id):
    return _ok(client.get(f"/exercise/{exercise_id}"))["reference_template"]


def _submit(client, session_id, exercise_id, points, attempt_id):
    return _ok(client.post("/attempt/submit", json={
        "session_id": session_id, "exercise_id": exercise_id, "attempt_id": attempt_id,
        "points": points, "capture": {"pointer_kind": "unknown", "pressure_supported": False}}))


def _state(resp, name):
    return resp["state_vector"][resp["state_feature_names"].index(name)]


# --------------------------------------------------------------------------
def s1_fresh_good_writer(client, n_attempts=6):
    sess = _start(client, "scenario_good_writer")
    sid, ex = sess["session_id"], sess["exercise"]["exercise_id"]
    log = []
    for i in range(n_attempts):
        tpl = _template(client, ex)
        resp = _submit(client, sid, ex, clean_trace(tpl), f"s1_{i}")
        log.append({"quality": resp["quality_score"], "fatigue": _state(resp, "fatigue_proxy"),
                    "frustration": _state(resp, "frustration"),
                    "raw_action": resp["safety"]["raw_action_name"],
                    "final_action": resp["safety"]["safe_action_name"],
                    "policy_source": resp["policy"]["source"]})
        if resp["safety"]["safe_action_id"] == SUGGEST_BREAK:
            _ok(client.post(f"/session/{sid}/break_complete"))
        ex = resp["next_exercise"]["exercise_id"]
    _ok(client.post(f"/session/{sid}/end"))
    low = [e for e in log if e["fatigue"] < 0.4]
    delivered_breaks = sum(e["final_action"] == "SUGGEST_BREAK" for e in low)
    raw_breaks = sum(e["raw_action"] == "SUGGEST_BREAK" for e in log)
    return {"scenario": "S1_fresh_good_writer", "attempts": log,
            "mean_quality": float(np.mean([e["quality"] for e in log])),
            "delivered_breaks_at_low_fatigue": delivered_breaks,
            "raw_policy_break_requests": raw_breaks,
            "pass": delivered_breaks == 0 and all(e["quality"] > 0.7 for e in log)}


def s2_high_fatigue(client, n_attempts=8):
    sess = _start(client, "scenario_tired_writer")
    sid, ex = sess["session_id"], sess["exercise"]["exercise_id"]
    log, violations = [], []
    for i in range(n_attempts):
        resp = _submit(client, sid, ex, paused_trace(_template(client, ex)), f"s2_{i}")
        fat = _state(resp, "fatigue_proxy")
        log.append({"fatigue": fat, "final_action": resp["safety"]["safe_action_name"],
                    "reasons": resp["safety"]["reasons"]})
        if fat >= 0.7 and resp["safety"]["safe_action_id"] != SUGGEST_BREAK:
            violations.append(i)
        ex = resp["next_exercise"]["exercise_id"]
    _ok(client.post(f"/session/{sid}/end"))
    reached = any(e["fatigue"] >= 0.7 for e in log)
    return {"scenario": "S2_fatigue_above_0.7", "attempts": log, "fatigue_threshold_reached": reached,
            "violations": violations, "pass": reached and not violations}


def s3_s4_struggling_writer(client, n_attempts=12):
    sess = _start(client, "scenario_struggling_writer", exercise_id="loop_medium")
    sid, ex = sess["session_id"], sess["exercise"]["exercise_id"]
    log, s3_viol, s4_viol = [], [], []
    prev_difficulty = _ok(client.get(f"/session/{sid}/summary"))["difficulty_level"]
    for i in range(n_attempts):
        resp = _submit(client, sid, ex, poor_trace(seed=100 + i), f"s3_{i}")
        frus = _state(resp, "frustration")
        failing = resp["session_summary"]["consecutive_fails"] >= 1
        a = resp["safety"]["safe_action_id"]
        diff = resp["presentation"]["difficulty_level"]
        log.append({"quality": resp["quality_score"], "frustration": frus, "failing": failing,
                    "raw_action": resp["safety"]["raw_action_name"],
                    "final_action": resp["safety"]["safe_action_name"], "difficulty": diff})
        if frus >= 0.75 and a not in SUPPORTIVE:
            s3_viol.append(i)
        if failing and (a == INCREASE_DIFFICULTY or diff > prev_difficulty):
            s4_viol.append(i)
        prev_difficulty = diff
        if a == SUGGEST_BREAK:
            _ok(client.post(f"/session/{sid}/break_complete"))
        ex = resp["next_exercise"]["exercise_id"]
    _ok(client.post(f"/session/{sid}/end"))
    reached = any(e["frustration"] >= 0.75 for e in log)
    return [
        {"scenario": "S3_high_frustration_gets_support", "attempts": log,
         "frustration_threshold_reached": reached, "violations": s3_viol, "pass": reached and not s3_viol},
        {"scenario": "S4_failures_never_increase_difficulty",
         "all_attempts_failing": all(e["failing"] for e in log), "violations": s4_viol,
         "pass": not s4_viol},
    ]


def s5_session_separation(client):
    a = _start(client, "scenario_sep_a", "line_easy")
    b = _start(client, "scenario_sep_b", "zigzag_hard")
    b_start = _ok(client.get(f"/session/{b['session_id']}/summary"))
    ta, tb = _template(client, "line_easy"), _template(client, "zigzag_hard")
    _submit(client, a["session_id"], "line_easy", clean_trace(ta), "a1")
    _submit(client, b["session_id"], "zigzag_hard", poor_trace(5), "b1")
    _submit(client, a["session_id"], "line_easy", clean_trace(ta), "a2")
    _ok(client.post(f"/session/{b['session_id']}/end"))
    sa = _ok(client.get(f"/session/{a['session_id']}/summary"))
    sb = _ok(client.get(f"/session/{b['session_id']}/summary"))
    late = client.post("/attempt/submit", json={"session_id": a["session_id"], "exercise_id": "line_easy",
                                                "attempt_id": "a3", "points": clean_trace(ta)})
    checks = {
        "a_has_2_attempts": sa["attempt_count"] == 2, "b_has_1_attempt": sb["attempt_count"] == 1,
        "b_ended_a_not": sb["ended"] and not sa["ended"], "a_still_accepts_attempts": late.status_code == 200,
        "difficulty_aligned_with_start_exercise": b_start["difficulty_level"] == 4,
    }
    _ok(client.post(f"/session/{a['session_id']}/end"))
    return {"scenario": "S5_session_separation", "checks": checks, "pass": all(checks.values())}


def s6_transition_integrity(client, n_attempts=4):
    sess = _start(client, "scenario_transitions")
    sid, ex = sess["session_id"], sess["exercise"]["exercise_id"]
    responses = []
    for i in range(n_attempts):
        r = _submit(client, sid, ex, clean_trace(_template(client, ex)), f"s6_{i}")
        responses.append(r)
        ex = r["next_exercise"]["exercise_id"]
    _ok(client.post(f"/session/{sid}/end"))
    rows = _ok(client.get(f"/session/{sid}/transitions"))
    checks = {"n_rows": len(rows) == n_attempts}
    for k, row in enumerate(rows):
        state = json.loads(row["state"])
        checks[f"t{k}_state_is_decision_state"] = np.allclose(state, responses[k]["state_vector"])
        checks[f"t{k}_action_is_delivered_action"] = row["action"] == responses[k]["safety"]["safe_action_id"]
        if k < n_attempts - 1:
            nxt = json.loads(row["next_state"])
            checks[f"t{k}_next_state_is_next_attempt"] = np.allclose(nxt, responses[k + 1]["state_vector"])
            checks[f"t{k}_reward_matches_response"] = math.isclose(
                row["reward"], responses[k + 1]["transition"]["reward"], rel_tol=1e-9, abs_tol=1e-9)
            checks[f"t{k}_not_done"] = row["done"] == 0 and row["outcome_observed"] == 1
        else:
            checks["last_reward_null"] = row["reward"] is None
            checks["last_done_and_unobserved"] = row["done"] == 1 and row["outcome_observed"] == 0
    checks["first_response_has_no_transition"] = responses[0]["transition"] is None
    return {"scenario": "S6_transition_integrity", "checks": checks, "pass": all(checks.values())}


def s7_data_governance(client):
    bad_ids = ["Jane Doe", "jane.doe@example.com", "x"]
    codes = {b: client.post("/session/start", json={"child_id": b}).status_code for b in bad_ids}
    real = client.post("/session/start", json={"child_id": "P-0001", "data_source": "real_child_study"})
    ok = client.post("/session/start", json={"child_id": "P-0001", "data_source": "synthetic_test"})
    checks = {"non_pseudonymous_ids_rejected": all(c == 422 for c in codes.values()),
              "real_child_study_refused": real.status_code == 403,
              "synthetic_session_allowed": ok.status_code == 200}
    if ok.status_code == 200:
        client.post(f"/session/{ok.json()['session_id']}/end")
    return {"scenario": "S7_data_governance", "status_codes": codes, "real_child_status": real.status_code,
            "checks": checks, "pass": all(checks.values())}


def run_all(client) -> dict:
    health = _ok(client.get("/health"))
    results = [s1_fresh_good_writer(client), s2_high_fatigue(client)]
    results += s3_s4_struggling_writer(client)
    results += [s5_session_separation(client), s6_transition_integrity(client), s7_data_governance(client)]
    return {"health": health, "results": results, "all_pass": all(r["pass"] for r in results),
            "data_label": "SYNTHETIC test traffic (data_source=synthetic_test) -- not real-child data"}


if __name__ == "__main__":
    import httpx
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    with httpx.Client(base_url=args.base_url, timeout=60.0) as c:
        out = run_all(c)
    for r in out["results"]:
        print(f"{r['scenario']:<45} {'PASS' if r['pass'] else 'FAIL'}")
    print("policy mode:", out["health"]["policy"]["policy_mode"])
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
        print("saved", args.out)
    sys.exit(0 if out["all_pass"] else 1)
