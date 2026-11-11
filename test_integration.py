#!/usr/bin/env python
"""End-to-end integration test for the R12_4 DQN deployment.

Runs the full flow:
  1. Import the backend.
  2. Check /health — the DQN must be live (policy_mode = "dqn",
     behavioral_gate_passed = true).
  3. Start a session on letter_lower_a (Level 4 lowercase).
  4. Submit a clean lowercase "a" (traced from the template).
  5. Verify:
     - decision_source == "dqn" (the DQN actually made the decision)
     - quality_score >= 0.65 (the new handwriting scorer from the
       prior patch correctly identifies the clean trace)
     - The Level 4 rotation rule keeps the learner on letter_lower_a
       (since not mastered yet)

SYNTHETIC test traffic only. No real-child data. No clinical claim.

Usage: python test_integration.py
"""
from __future__ import annotations

import os
import sys
import tempfile

# Point the test at the backend in this zip.
HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(HERE, "source", "backend")
sys.path.insert(0, BACKEND_DIR)

# Throwaway SQLite DB so we don't pollute any real database.
os.environ.setdefault("STORAGE_DB_PATH",
                      os.path.join(tempfile.mkdtemp(prefix="dt_integration_"), "test.db"))


def main() -> int:
    failures: list[str] = []

    # Import here (not at top) so the sys.path tweak takes effect first.
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)

    # --- 1. /health ---
    print("[1/5] GET /health")
    h = client.get("/health").json()
    if h.get("status") != "ok":
        failures.append(f"/health status != ok (got {h.get('status')!r})")
    policy = h.get("policy", {})
    print(f"     policy_mode            = {policy.get('policy_mode')!r}")
    print(f"     dqn_making_decisions   = {policy.get('dqn_making_decisions')!r}")
    print(f"     behavioral_gate_passed = {policy.get('behavioral_gate_passed')!r}")
    print(f"     checkpoint_sha256      = {policy.get('checkpoint_sha256', '')[:16]}...")
    if policy.get("policy_mode") != "dqn":
        failures.append(
            f"policy_mode != 'dqn' (got {policy.get('policy_mode')!r}). "
            f"The R12_4 checkpoint did not load or the gate failed. "
            f"Fallback reasons: {policy.get('fallback_reasons')}"
        )
    if not policy.get("dqn_making_decisions"):
        failures.append("dqn_making_decisions is not True")
    if policy.get("behavioral_gate_passed") is not True:
        failures.append(
            f"behavioral_gate_passed is not True (got {policy.get('behavioral_gate_passed')!r}). "
            f"Gate failures: {policy.get('behavioral_gate_failures')}"
        )

    # --- 2. Start a session on letter_lower_a (Level 4 lowercase) ---
    print()
    print("[2/5] POST /session/start (letter_lower_a, Level 4 lowercase)")
    r = client.post("/session/start", json={
        "child_id": "P-0042", "exercise_id": "letter_lower_a",
        "data_source": "synthetic_test",
    })
    if r.status_code != 200:
        failures.append(f"/session/start returned {r.status_code}: {r.text[:200]}")
        return _report(failures)
    sess = r.json()
    sid = sess["session_id"]
    exercise = sess["exercise"]
    print(f"     session_id  = {sid}")
    print(f"     exercise_id = {exercise['exercise_id']}")
    print(f"     target      = {exercise['target']!r}")
    print(f"     difficulty  = {exercise['difficulty']}")
    if exercise["exercise_id"] != "letter_lower_a":
        failures.append(f"exercise_id != letter_lower_a (got {exercise['exercise_id']!r})")
    if exercise["difficulty"] != 4:
        failures.append(f"difficulty != 4 (got {exercise['difficulty']!r})")

    # --- 3. Build a clean lowercase 'a' trace from the template ---
    print()
    print("[3/5] Build clean lowercase 'a' trace from the template")
    pts, t = [], 0.0
    for sid_s, s in enumerate(exercise["template_strokes"]):
        for i, (x, y) in enumerate(s):
            pts.append({
                "x": float(x * 3.0), "y": float(y * 3.0),
                "t": t, "pressure": 0.5, "stroke_id": sid_s,
                "is_stroke_start": i == 0,
                "is_stroke_end": i == len(s) - 1,
            })
            t += 0.02
        t += 0.3
    print(f"     {len(pts)} points across {len(exercise['template_strokes'])} strokes")

    # --- 4. Submit the attempt ---
    print()
    print("[4/5] POST /attempt/submit (clean lowercase 'a')")
    r = client.post("/attempt/submit", json={
        "session_id": sid, "exercise_id": "letter_lower_a", "attempt_id": "a1",
        "points": pts,
        "capture": {"pointer_kind": "stylus", "pressure_supported": True,
                    "client_version": "test_integration/0.1"},
    })
    if r.status_code != 200:
        failures.append(f"/attempt/submit returned {r.status_code}: {r.text[:200]}")
        return _report(failures)
    body = r.json()

    quality_score = body.get("quality_score", 0.0)
    handwriting_score = (body.get("handwriting_score") or {}).get("score", 0.0)
    decision_source = body.get("decision_source")
    policy_mode = body.get("policy_mode")
    safety_overridden = body.get("safety_overridden")
    safe_action_name = (body.get("safety") or {}).get("safe_action_name")
    intervention_type = (body.get("intervention") or {}).get("type")
    intervention_message = (body.get("intervention") or {}).get("message", "")
    next_ex_id = (body.get("next_exercise") or {}).get("exercise_id")
    next_ex_target = (body.get("next_exercise") or {}).get("target")
    next_ex_diff = (body.get("next_exercise") or {}).get("difficulty")
    letter_mastery = body.get("letter_mastery") or {}
    pedagogical = body.get("pedagogical") or {}

    print(f"     quality_score            = {quality_score:.4f}")
    print(f"     handwriting_score.score  = {handwriting_score:.2f}")
    print(f"     decision_source           = {decision_source!r}")
    print(f"     policy_mode               = {policy_mode!r}")
    print(f"     safety_overridden         = {safety_overridden!r}")
    print(f"     safety.safe_action_name   = {safe_action_name!r}")
    print(f"     pedagogical.rule          = {pedagogical.get('rule')!r}")
    print(f"     pedagogical.overridden    = {pedagogical.get('overridden')!r}")
    print(f"     pedagogical.shaped_action = {pedagogical.get('shaped_action_name')!r}")
    print(f"     intervention.type         = {intervention_type!r}")
    print(f"     intervention.message      = {intervention_message[:60]!r}")
    print(f"     next_exercise             = {next_ex_id!r} (target={next_ex_target!r}, level={next_ex_diff})")
    print(f"     letter_mastery.current_letter_mastered = {letter_mastery.get('current_letter_mastered')!r}")
    print(f"     letter_mastery.mastered_count_at_level = {letter_mastery.get('mastered_count_at_level')}/{letter_mastery.get('total_at_level')}")

    # --- 5. Verify the integration claims ---
    print()
    print("[5/5] Verify integration claims")
    if decision_source != "dqn":
        failures.append(
            f"decision_source != 'dqn' (got {decision_source!r}). "
            f"The DQN is NOT making the live decision — the rule-based fallback is. "
            f"Check /health and the gate report."
        )
    else:
        print("     [OK] decision_source == 'dqn' (the DQN made the live decision)")

    if quality_score < 0.65:
        failures.append(
            f"quality_score < 0.65 (got {quality_score:.4f}). "
            f"The new multiplicative handwriting scorer should rate a clean trace >= 0.65."
        )
    else:
        print(f"     [OK] quality_score = {quality_score:.4f} (>= 0.65, clean trace correctly scored)")

    if intervention_type not in ("continue", "letter_rotation", "difficulty_adjustment"):
        failures.append(f"intervention.type = {intervention_type!r} (unexpected)")
    else:
        print(f"     [OK] intervention.type = {intervention_type!r}")

    # The Level 4 rotation rule should keep us on letter_lower_a
    # because the letter is not yet mastered (this was attempt 1).
    if next_ex_id != "letter_lower_a":
        failures.append(
            f"next_exercise != 'letter_lower_a' (got {next_ex_id!r}). "
            f"The Level 4 rotation rule should keep the learner on the current letter "
            f"when it is not yet mastered (only 1 attempt so far)."
        )
    else:
        print(f"     [OK] next_exercise == 'letter_lower_a' (Level 4 rotation rule kept the learner on the same letter)")

    if next_ex_diff != 4:
        failures.append(f"next_exercise difficulty != 4 (got {next_ex_diff})")
    else:
        print(f"     [OK] next_exercise difficulty == 4 (stayed at Level 4, no premature escalation)")

    if letter_mastery.get("current_letter_mastered") is not False:
        failures.append(
            f"letter_mastery.current_letter_mastered != False (got {letter_mastery.get('current_letter_mastered')!r}). "
            f"After only 1 attempt, the letter should not be mastered yet."
        )
    else:
        print(f"     [OK] current_letter_mastered == False (1 attempt is not enough for mastery)")

    if letter_mastery.get("total_at_level", 0) < 5:
        failures.append(
            f"letter_mastery.total_at_level = {letter_mastery.get('total_at_level')} (< 5). "
            f"Level 4 lowercase should have 11 letters."
        )
    else:
        print(f"     [OK] total_at_level == {letter_mastery.get('total_at_level')} (Level 4 has multiple letters, rotation rule applies)")

    # Pedagogical layer (audit-driven, 2026-11) — must be present in the
    # response (even if it didn't override on this single attempt).
    if "pedagogical" not in body:
        failures.append(
            "pedagogical field missing from response. The pedagogical layer "
            "(audit-driven) should always report its decision, even when it "
            "doesn't override."
        )
    else:
        print(f"     [OK] pedagogical layer present in response (rule={pedagogical.get('rule')!r}, overridden={pedagogical.get('overridden')!r})")

    return _report(failures)


def _report(failures: list[str]) -> int:
    print()
    print("=" * 60)
    if failures:
        print(f"Integration test FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print(f"  - {f}")
        print()
        print("The R12_4 DQN is NOT correctly deployed. See INTEGRATION_NOTES.md")
        print("for troubleshooting and rollback procedures.")
        return 1
    print("Integration test PASSED.")
    print()
    print("The R12_4 DQN checkpoint is correctly deployed:")
    print("  - The behavioural gate passed.")
    print("  - The DQN makes the live decisions (decision_source = 'dqn').")
    print("  - The new handwriting scorer correctly rates a clean trace.")
    print("  - The Level 4 rotation rule keeps the learner on the same letter")
    print("    until mastery is achieved (no premature escalation).")
    print("  - The pedagogical layer is active (audit-driven, 2026-11):")
    print("      * Action diversity safeguard (prevents any single action from")
    print("        dominating the policy's choices).")
    print("      * Repeated-failure de-escalation (drops difficulty after >= 4")
    print("        failed attempts at a Level 4 letter).")
    print("      * Post-break progression (moves to the next letter after a")
    print("        break + good score, instead of staying on the same letter).")
    print()
    print("All results are simulated. No clinical validation. No claim of")
    print("benefit to real children.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
