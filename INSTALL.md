# DysgraphiaTutor — Integrated with Phase 5 R12_4 DQN checkpoint

This is the **DysgraphiaTutor project** (with the audit-driven improvements from the
prior patch — intelligent rule-based tutor, multiplicative handwriting scorer,
Level 4 letter rotation, per-letter mastery) **integrated with the Phase 5
full-training R12_4 Double DQN checkpoint**.

The checkpoint was trained in the `dgpop` child-start simulator (Phase 5 locked
protocol, seed R12_4) and **passed the tutor's strict behavioural gate**. The
backend now uses it as the live DQN policy; the intelligent rule-based tutor
remains as the automatic fallback if the gate ever fails on a future load.

**All results are simulated. No clinical validation. No claim of benefit to
real children.** See `INTEGRATION_NOTES.md` for the full eval results and the
safety net.

---

## Quick start

### 1. Backend (FastAPI)

From `source/backend/`:

```bash
# Create a venv and install deps (CPU torch per the locked protocol).
python -m venv .venv
.venv\Scripts\activate                                    # Windows
#  source .venv/bin/activate                              # macOS / Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu --timeout 600 --retries 10
pip install -r requirements.txt --timeout 600 --retries 10
pip install scikit-learn --timeout 600 --retries 10       # missing from requirements.txt
pip install gymnasium --timeout 600 --retries 10          # required by the RL bridge

# Run the backend.
uvicorn app.main:app --reload --port 8000
```

### 2. Verify the DQN is live

In a separate terminal:

```bash
curl http://127.0.0.1:8000/health | python -m json.tool
```

You should see:

```json
{
  "policy": {
    "policy_mode": "dqn",
    "dqn_making_decisions": true,
    "behavioral_gate_passed": true,
    "behavioral_gate_enabled": true,
    "fallback_reasons": [],
    "checkpoint_sha256": "411b7d07da82322d..."
  }
}
```

If `policy_mode` is `"rule_based_fallback"`, the gate failed and the
intelligent rule-based tutor is being used (still safe, but the DQN is not
live). See `INTEGRATION_NOTES.md` → "Rollback / troubleshooting".

### 3. Run the integration test

From the project root:

```bash
python test_integration.py
```

This runs the full end-to-end check: starts a session, submits a clean
lowercase `a`, and confirms `decision_source: "dqn"` and the new
handwriting scorer gives a high score. Expected output ends with
`Integration test PASSED`.

### 4. Flutter app

From `source/tablet_app/`:

```bash
flutter pub get
flutter run
```

Point the app at `http://127.0.0.1:8000` (the default). The learner-facing
UI shows child-friendly feedback only — internal policy labels
(`rule_based_fallback`, `decision_source`, etc.) are hidden by design
(audit-driven change). The supervisor-facing `PolicyStatusCard` (collapsed
by default on the dashboard) shows the audit detail.

---

## What's in this zip

| Path                                                   | What                                                  |
|--------------------------------------------------------|-------------------------------------------------------|
| `INSTALL.md`                                           | this file                                             |
| `INTEGRATION_NOTES.md`                                 | what's integrated, eval results, safety net, rollback |
| `test_integration.py`                                  | end-to-end verification script                        |
| `source/README.md`                                     | original project README                              |
| `source/docs/`                                         | original project docs                                 |
| `source/backend/`                                      | patched FastAPI backend                               |
| `source/tablet_app/`                                    | patched Flutter app                                   |
| `source/dysgraphia_rl/`                                | RL package                                            |
| `source/dysgraphia_rl/checkpoints/double_dqn.pt`       | **the new R12_4 DQN checkpoint**                     |
| `source/dysgraphia_rl/checkpoints/double_dqn.json`     | manifest (source run, SHA-256, eval results)          |
| `source/dysgraphia_rl/checkpoints/backup_pre_R12_4/`   | the old checkpoint, kept for rollback                |

### What's NOT in this zip (intentionally)

- `source/evidence/` — large evidence / screenshot folders from prior
  development; not needed to run the project.
- `source/dysgraphia_rl/results/` — Phase 1–4 training results; not
  needed for deployment (the Phase 5 R12_4 results are in
  `INTEGRATION_NOTES.md`).
- `source/dysgraphia_rl/checkpoints/seeds*`, `legacy/`,
  `deployment_candidates/` — other checkpoint variants from prior
  training; not needed (the new R12_4 + one backup is enough).
- `__pycache__/`, `.pytest_cache/`, `.venv/`, `build/`, `.dart_tool/` —
  build artifacts.
- `source/backend/data/` — the SQLite attempt-log database; created on
  first run.
- Per-child data (`data/state12_table.csv`, `data/state12_table_contexts.csv`
  from the training project) — restricted; never included.

---

## Rollback

If the DQN policy ever misbehaves on real input, or you want to revert to
the intelligent rule-based tutor:

```bash
cd source/dysgraphia_rl/checkpoints
# Option A: remove the new checkpoint — backend falls back to the rule-based tutor
mv double_dqn.pt double_dqn.R12_4.pt.bak
mv double_dqn.json double_dqn.R12_4.json.bak

# Option B: restore the pre-R12_4 checkpoint
cp backup_pre_R12_4/double_dqn_legacy_2026-11-08.pt double_dqn.pt

# Either way, restart the backend:
# (Ctrl-C the running uvicorn, then)
uvicorn app.main:app --reload --port 8000
```

After rollback, `/health` will report `policy_mode: "rule_based_fallback"`
and the intelligent rule-based tutor (the one added in the prior patch)
will make every decision. The behaviour is unchanged from before this
integration.

---

## Running the test suite (optional)

From `source/backend/`:

```bash
python -m pytest -q tests
# Expected: 687 passed, 83 skipped, 1 xfailed
```

This is the full backend suite: 575 baseline tests + 95 audit-driven tests
(handwriting scoring, intelligent tutor, Level 4 progression, integration)
+ 18 pedagogical layer tests (action diversity, de-escalation,
post-break progression). None of them modify the checkpoint; they only
test the backend code.

---

## Important caveats — read before deploying

1. **Simulated training only.** The R12_4 checkpoint was trained in the
   `dgpop` child-start simulator and evaluated in `DysgraphiaTutorEnv`.
   Neither stage used real children's handwriting data. The 17/17
   gate-pass and the +0.057 quality_gain are real numbers, but they are
   real *simulated* numbers — they tell you the policy is safe by the
   gate's definition and learns the simulated task. They do NOT tell you
   it will help real children with dysgraphia.

2. **No clinical validation.** If you deploy this in a real-child-facing
   app, you must describe the policy accurately in any user-facing
   material: "trained in simulation, passed the safety gate, not yet
   validated on real children." Do not call it "proven", "validated", or
   "clinically shown to help".

3. **The safety net is solid.** Even after deployment:
   - The behavioural gate runs at backend startup. If the checkpoint
     ever fails on a future load, the backend falls back to the
     intelligent rule-based tutor automatically.
   - The safety layer runs after every DQN decision. If the DQN picks
     an action that violates a safety rule (e.g. escalate difficulty
     right after a failure, suggest a break with low fatigue), the
     safety layer overrides it.
   - The intelligent rule-based tutor from the prior patch is always
     available as the fallback — it sees the full session context
     (score history, action log, recent exercises, current letter,
     mastery) and makes context-aware decisions.

4. **Don't weaken the gate.** If you ever want to deploy a different
   checkpoint that fails the gate, do not weaken the gate to make it
   pass. The gate is a safety filter, not a quality rubber-stamp.
