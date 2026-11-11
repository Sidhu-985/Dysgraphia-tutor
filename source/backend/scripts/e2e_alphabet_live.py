"""Live end-to-end check of all 52 letters against a REAL running backend
(uvicorn over HTTP, not the in-process TestClient).

For every letter: GET /exercise/{id}; POST /session/start (tracing, and
free_writing for every 4th letter); submit two synthetic attempts (the guide
itself traced, then a jittered copy) following next_exercise; end the
session. Records the decision source of every attempt and checks it against
/health. All handwriting is synthetic.

Usage (from backend/):
    python scripts/e2e_alphabet_live.py OUT.json [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

LETTERS = [f"letter_upper_{c}" for c in "abcdefghijklmnopqrstuvwxyz"] + \
          [f"letter_lower_{c}" for c in "abcdefghijklmnopqrstuvwxyz"]


def call(base, method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


def trace(strokes, rng=None, scale=3.0):
    pts, t = [], 0.0
    for sid, s in enumerate(strokes):
        s = np.array(s, float)
        if rng is not None:
            s = s + rng.normal(0, 1.2, s.shape)
        for i, (x, y) in enumerate(s):
            pts.append({"x": float(x * scale), "y": float(y * scale), "t": t, "pressure": None, "stroke_id": sid,
                        "is_stroke_start": i == 0, "is_stroke_end": i == len(s) - 1})
            t += 0.02
        t += 0.3
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    base = f"http://127.0.0.1:{a.port}"
    db = tempfile.mkdtemp()
    env = dict(os.environ, STORAGE_DB_PATH=str(Path(db) / "e2e.db"))
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(a.port)],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            try:
                call(base, "GET", "/health")
                break
            except Exception:
                time.sleep(0.5)
        _, health = call(base, "GET", "/health")
        _, catalogue = call(base, "GET", "/exercises")
        rows, problems = [], []
        rng = np.random.default_rng(2026)
        letters = [e for e in catalogue if e.get("category") == "letter"]
        if sorted(e["exercise_id"] for e in letters) != sorted(LETTERS):
            problems.append("catalogue does not hold exactly the 52 letters")
        for k, eid in enumerate(LETTERS):
            _, ex = call(base, "GET", f"/exercise/{eid}")
            mode = "free_writing" if k % 4 == 3 else "tracing"
            _, start = call(base, "POST", "/session/start",
                            {"child_id": "E2E-0001", "exercise_id": eid, "data_source": "synthetic_test",
                             "practice_mode": mode})
            sid, cur = start["session_id"], start["exercise"]
            if cur["exercise_id"] != eid or cur["target"] != ex["target"] or start["practice_mode"] != mode:
                problems.append(f"{eid}: session start mismatch")
            attempts = []
            for j in range(2):
                st, res = call(base, "POST", "/attempt/submit",
                               {"session_id": sid, "exercise_id": cur["exercise_id"], "attempt_id": f"{eid}-{j}",
                                "points": trace(cur["template_strokes"], None if j == 0 else rng)})
                nxt = res["next_exercise"]
                ok_src = res["decision_source"] == ("dqn" if health["policy_mode"] == "dqn" else "rule_based_fallback")
                if st != 200 or not ok_src or res["policy_mode"] != health["policy_mode"]:
                    problems.append(f"{eid}/{j}: status {st}, source {res['decision_source']}, mode {res['policy_mode']}")
                if nxt["category"] != "letter" or nxt["letter_case"] != ex["letter_case"]:
                    problems.append(f"{eid}/{j}: next exercise left the letter track: {nxt['exercise_id']}")
                attempts.append({"exercise_id": cur["exercise_id"], "quality": round(res["quality_score"], 4),
                                 "raw_action": res["safety"]["raw_action_name"],
                                 "safe_action": res["safety"]["safe_action_name"],
                                 "safety_overridden": res["safety_overridden"],
                                 "decision_source": res["decision_source"], "policy_mode": res["policy_mode"],
                                 "intervention": res["intervention"]["type"], "next_exercise": nxt["exercise_id"]})
                cur = nxt
            call(base, "POST", f"/session/{sid}/end")
            rows.append({"exercise_id": eid, "target": ex["target"], "letter_case": ex["letter_case"],
                         "letter_zone": ex["letter_zone"], "start_mode": mode,
                         "n_strokes": len(ex["template_strokes"]), "attempts": attempts})
        out = {"server": "uvicorn app.main:app (real HTTP)", "health": {k: health[k] for k in (
            "policy_mode", "dqn_making_decisions", "behavioral_gate_enabled", "behavioral_gate_passed",
            "rl_checkpoint_path", "rl_checkpoint_sha256")},
            "letters_checked": len(rows), "attempts": sum(len(r["attempts"]) for r in rows),
            "problems": problems, "passed": not problems, "rows": rows}
        Path(a.out).write_text(json.dumps(out, indent=1))
        print(json.dumps({k: out[k] for k in ("letters_checked", "attempts", "passed", "problems")}))
        print(json.dumps(out["health"]))
        sys.exit(0 if not problems else 1)
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
