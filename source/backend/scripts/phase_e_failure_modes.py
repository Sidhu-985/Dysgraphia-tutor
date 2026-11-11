"""
Phase E failure-mode checks against a REAL running server (over HTTP).
SYNTHETIC traffic only.

  F1 malformed JSON body           -> 422, server keeps serving
  F2 NaN / negative timestamp      -> 422 (validators), session untouched
  F3 unknown session               -> 404
  F4 submit after session end      -> 409
  F5 empty points                  -> 422
  F6 8 concurrent sessions x 5 attempts (threads) -> every session ends
     with exactly 5 attempts, and no history is shared between sessions
  F8 the same attempt_id resent with the same points -> replayed response,
     counted once. The same attempt_id with different points -> 409.
  F7 restart (orchestrated by the caller): `--phase before` creates a
     session and saves its id. `--phase after` runs after the server was
     killed and restarted. The in-memory session is gone (404, documented
     limitation) but its attempt rows are still in SQLite.

Usage: python scripts/phase_e_failure_modes.py --base-url URL --out f.json [--phase all|before|after]
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx

_spec = importlib.util.spec_from_file_location(
    "live_scenarios", os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_scenarios.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def _start(c, cid, ex="line_easy"):
    r = c.post("/session/start", json={"child_id": cid, "exercise_id": ex, "data_source": "synthetic_test"})
    r.raise_for_status()
    return r.json()["session_id"]


def checks(c):
    tpl = c.get("/exercise/line_easy").json()["reference_template"]
    out = {}
    sid = _start(c, "fail_modes_1")
    r = c.post("/attempt/submit", content=b"{not json", headers={"Content-Type": "application/json"})
    out["F1_malformed_json_422"] = r.status_code == 422 and c.get("/health").status_code == 200
    bad = sc.clean_trace(tpl)
    raw = json.dumps({"session_id": sid, "exercise_id": "line_easy", "attempt_id": "x",
                      "points": bad}).replace('"x": 0.0', '"x": NaN', 1)
    r1 = c.post("/attempt/submit", content=raw.encode(), headers={"Content-Type": "application/json"})
    neg = [dict(p) for p in bad]
    neg[3]["t"] = -1.0
    r2 = c.post("/attempt/submit", json={"session_id": sid, "exercise_id": "line_easy", "attempt_id": "y",
                                         "points": neg})
    summ = c.get(f"/session/{sid}/summary").json()
    out["F2_nan_and_negative_t_rejected_without_mutation"] = (
        r1.status_code == 422 and r2.status_code == 422 and summ["attempt_count"] == 0)
    out["F3_unknown_session_404"] = c.post("/attempt/submit", json={
        "session_id": "nope", "exercise_id": "line_easy", "attempt_id": "z", "points": bad}).status_code == 404
    out["F5_empty_points_422"] = c.post("/attempt/submit", json={
        "session_id": sid, "exercise_id": "line_easy", "attempt_id": "e", "points": []}).status_code == 422
    c.post(f"/session/{sid}/end")
    out["F4_after_end_409"] = c.post("/attempt/submit", json={
        "session_id": sid, "exercise_id": "line_easy", "attempt_id": "w", "points": bad}).status_code == 409

    # F8 duplicate submission (network retry): replayed, not processed twice
    sid8 = _start(c, "fail_modes_dup")
    body = {"session_id": sid8, "exercise_id": "line_easy", "attempt_id": "dup", "points": sc.clean_trace(tpl)}
    a1 = c.post("/attempt/submit", json=body).json()
    a2 = c.post("/attempt/submit", json=body).json()
    conflict = c.post("/attempt/submit", json=dict(body, points=sc.poor_trace(9))).status_code
    out["F8_duplicate_submission_idempotent"] = (
        a2.get("duplicate_submission") is True and a1.get("duplicate_submission") is False
        and c.get(f"/session/{sid8}/summary").json()["attempt_count"] == 1 and conflict == 409)

    def one(k):
        with httpx.Client(base_url=str(c.base_url), timeout=60) as cc:
            s = _start(cc, f"concurrent_{k:02d}")
            ex = "line_easy"
            for i in range(5):
                t = cc.get(f"/exercise/{ex}").json()["reference_template"]
                trace = sc.clean_trace(t) if k % 2 == 0 else sc.poor_trace(1000 * k + i)
                r = cc.post("/attempt/submit", json={"session_id": s, "exercise_id": ex,
                                                     "attempt_id": f"{k}_{i}", "points": trace})
                r.raise_for_status()
                ex = r.json()["next_exercise"]["exercise_id"]
            return cc.get(f"/session/{s}/summary").json()

    with ThreadPoolExecutor(max_workers=8) as pool:
        summaries = list(pool.map(one, range(8)))
    out["F6_concurrent_sessions_isolated"] = (
        all(s["attempt_count"] == 5 and len(s["score_history"]) == 5 for s in summaries)
        and len({s["session_id"] for s in summaries}) == 8
        # clean-trace sessions (even k) and scribble sessions (odd k) must keep separate histories
        and min(min(s["score_history"]) for s in summaries[0::2]) > max(max(s["score_history"]) for s in summaries[1::2]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all", choices=["all", "before", "after"])
    ap.add_argument("--db", default=os.environ.get("STORAGE_DB_PATH"))
    a = ap.parse_args()
    state_file = a.out + ".restart_state.json"
    with httpx.Client(base_url=a.base_url, timeout=60) as c:
        if a.phase in ("all",):
            res = checks(c)
        elif a.phase == "before":
            sid = _start(c, "restart_probe")
            tpl = c.get("/exercise/line_easy").json()["reference_template"]
            c.post("/attempt/submit", json={"session_id": sid, "exercise_id": "line_easy",
                                            "attempt_id": "r1", "points": sc.clean_trace(tpl)}).raise_for_status()
            json.dump({"session_id": sid}, open(state_file, "w"))
            res = {"F7_before_restart_session_created": True}
        else:
            sid = json.load(open(state_file))["session_id"]
            status = c.get(f"/session/{sid}/summary").status_code
            rows = sqlite3.connect(a.db).execute(
                "SELECT COUNT(*) FROM attempts WHERE session_id = ?", (sid,)).fetchone()[0]
            res = {"F7_in_memory_session_lost_after_restart_404": status == 404,
                   "F7_attempt_log_survives_restart": rows == 1}
    prev = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prev.update(res)
    json.dump(prev, open(a.out, "w"), indent=1)
    for k, v in res.items():
        print(f"{k:<55} {'PASS' if v else 'FAIL'}")
    sys.exit(0 if all(res.values()) else 1)
