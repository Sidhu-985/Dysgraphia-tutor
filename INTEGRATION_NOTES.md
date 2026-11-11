# Integration notes — R12_4 DQN checkpoint

This document records what was integrated, the evaluation results that
justified the choice, and the safety net that protects the deployment.

**All results are simulated. No clinical validation. No claim of benefit
to real children.**

---

## What was integrated

| Item | Value |
|---|---|
| Source run | `R12_4` (Phase 5 full training, seed 4, arm R12) |
| Training protocol | `configs/phase5.yaml` v1 (locked) |
| Training simulator | `dgpop` child-start simulator (synthetic learner profiles) |
| Training budget | 150,000 steps × 16 parallel envs, reward version `sref` |
| Train / val / test split | 191 fit / 27 val / 55 test children (locked by `theta_split_seed=12345`) |
| Network architecture | MLP 12 → 128 → 64 → 10 (matches the tutor's `QNetwork`) |
| Checkpoint SHA-256 | `411b7d07da82322d98866c5d912c40d1c856d281856da4cdb05605426fb505cf` |
| Checkpoint file | `source/dysgraphia_rl/checkpoints/double_dqn.pt` |
| Manifest | `source/dysgraphia_rl/checkpoints/double_dqn.json` |
| Pre-R12_4 backup | `source/dysgraphia_rl/checkpoints/backup_pre_R12_4/double_dqn_legacy_2026-11-08.pt` |

---

## Why R12_4 was chosen

The Phase 5 full training produced 30 checkpoints (3 arms × 10 seeds).
17 passed the training-pipeline behavioural gate. Of those 17, all 17
also passed the tutor's strict behavioural gate (`gate_check.py`).

All 17 were then evaluated in the tutor's own simulator
(`DysgraphiaTutorEnv`, 200 episodes each, alongside `rule_based` and
`do_nothing` baselines on identical episode seeds). The eval results
(quality_gain and return, both averaged over 200 episodes):

| Run | quality_gain | return | note |
|---|---:|---:|---|
| **R12_4** | **+0.057** | **25.31** | best overall: positive QG and highest return (~5× rule_based) |
| R12_7 | +0.072 | 14.45 | highest QG across all 17; healthy return |
| A_8 | +0.012 | 16.43 | only A-arm checkpoint with positive QG |
| R12E_0 | -0.005 | 17.34 | approximately neutral QG; strong return |
| A_6 | -0.019 | 19.43 | negative QG; very strong return |
| R12_3 | -0.029 | 12.43 | negative QG; healthy return |
| R12E_5 | -0.030 | 13.27 | negative QG; healthy return |
| A_7 | -0.034 | 16.09 | negative QG; strong return |
| R12_9 | -0.017 | 10.33 | negative QG; healthy return |
| A_4 | -0.067 | 14.53 | negative QG; strong return |
| R12_1 | -0.068 | 12.69 | negative QG; healthy return |
| A_2 | -0.057 | 10.57 | negative QG; healthy return |
| A_3 | -0.087 | 8.62 | negative QG; modest return |
| A_9 | -0.020 | 3.83 | negative QG; small return |
| R12_8 | -0.107 | 4.22 | negative QG; small return |
| A_1 | -0.101 | 3.01 | negative QG; small return |
| A_0 | -0.146 | 0.35 | weakest; near-zero return |
| **rule_based (baseline)** | **-0.121** | **5.00** | the intelligent tutor from the prior patch |
| **do_nothing (baseline)** | **-0.241** | **-8.68** | lower bound |

**Selection rationale.** R12_4 was chosen because:
1. It has the **highest return** of any gate-passing checkpoint (25.31,
   ~5× the rule-based baseline's 5.00).
2. It has **positive quality_gain** (+0.057) — meaning it actually
   improves handwriting quality rather than only slowing decay. Only 3
   of 17 gate-passing checkpoints achieved positive QG; R12_4 is one of
   them.
3. It comes from the **R12 arm**, which was the focus of the
   pre-registered H9 hypothesis ("R12 > A on held-out children").
   Although H9 was not statistically supported at n=10 (CI straddled
   zero; MDE = 0.0217 > observed diff = 0.0101), the R12 arm's mean
   was numerically higher than A's on every test set, so picking a
   strong R12 seed is consistent with the exploratory finding.
4. It has a moderate `low_fatigue_break_rate` (0.15) — well under the
   25% G4 cap — so the safety layer should rarely need to override it
   on low-fatigue break decisions.

**Alternative candidates.** If you prefer a more conservative choice:
- **R12_7** has the highest quality_gain (+0.072) but a lower return
  (14.45). It is the best candidate if you weight handwriting quality
  improvement above reward.
- **A_8** is the only A-arm checkpoint with positive quality_gain
  (+0.012) and a strong return (16.43). It comes from the simpler
  training objective (no R12 augmentation), so it may generalise
  better to real data, though this is speculative.

To switch to one of these alternatives, copy the corresponding
`exports/childstart_<run>_best.pt` from the training project over
`source/dysgraphia_rl/checkpoints/double_dqn.pt`, update the
`double_dqn.json` manifest accordingly, and restart the backend.

---

## Phase 5 pre-registered result (the locked protocol's main finding)

The Phase 5 full training was a pre-registered protocol
(`configs/phase5.yaml` v1, run unmodified). The pre-registered
hypothesis H9 was:

> **H9: R12 > A on held-out children (T2_s12).**

Result: **NOT supported.**

| Metric | Value |
|---|---|
| `mean_A` | 0.1776 |
| `mean_R12` | 0.1877 |
| `diff (R12 − A)` | +0.0101 (in the predicted direction) |
| 95% bootstrap CI | [−0.0054, +0.0253] (straddles 0) |
| One-sided exact p | 0.1237 (well above α = 0.05) |
| Hedges' g | 0.5213 (medium effect by convention) |
| MDE | 0.0217 (larger than the observed diff of 0.0101) |

The point estimate is in the predicted direction (R12 did slightly
better than A), but with n=10 runs per arm the variance was too large
to call it a confirmed effect. The protocol was underpowered to
detect a difference as small as the one observed.

**No protocol file, reward, simulator, or test was edited to change
this outcome; the behavioural gate was not weakened.**

### Exploratory findings

- **R12E vs R12:** R12E is consistently higher than R12 by point
  estimate on every test set (+0.005 to +0.013), but none of the
  one-sided contrasts reach p < 0.05. The pre-specified E2
  non-inferiority check (R12E − R12 lower 90% bound > −0.02) passes
  on T2_s12: bootstrap 5th percentile = −0.0049, greater than the
  −0.02 margin. The R12E-over-R12 contrast grows monotonically with
  session load, suggesting R12E is more robust to load.

- **R12E vs A:** R12E > A is in the predicted direction on all five
  test sets and reaches nominal significance on two (T2_s12E
  p = 0.0446, T2_s12E_heavy_load p = 0.0262). Exploratory, not
  multiplicity-corrected.

- **All three trained arms beat all three baselines** (do_nothing,
  static_curriculum, rule) on every test set by a wide margin.

- **No dominance flags.** `dominance_flags = 0` for every run. Max
  `max_share` was 0.4055, well below the 0.6 threshold. No arm
  collapsed into a one-action policy on any of the 30 runs.

---

## End-to-end verification (run in the build environment)

After integration, the backend was started and the following
end-to-end test was run:

1. **`/health`** reports `policy_mode: "dqn"`,
   `dqn_making_decisions: true`, `behavioral_gate_passed: true`,
   `fallback_reasons: []`. ✅
2. **Session start** for `child_id: "P-0042"`,
   `exercise_id: "letter_lower_a"` returns a valid session. ✅
3. **Attempt submit** with a clean lowercase `a` (traced from the
   template) returns:
   - `quality_score: 0.9949` (the new multiplicative handwriting
     scorer from the prior patch correctly identifies the clean
     trace).
   - `handwriting_score.score: 99.49` (the learner-facing 0–100
     score, same as `quality_score * 100`).
   - `decision_source: "dqn"` — the DQN actually made the live
     decision (not the rule-based fallback). ✅
   - `policy_mode: "dqn"` — confirms the DQN is in use. ✅
   - `safety_overridden: false` — the safety layer ran but did not
     need to override the DQN's choice. ✅
   - `safety.safe_action_name: "INCREASE_DIFFICULTY"` — the DQN
     chose to escalate.
   - `intervention.type: "continue"`, message "Good effort! Let's
     keep practising this one." — the intervention controller's
     Level 4 rotation rule (from the prior patch) correctly
     intercepted the INCREASE_DIFFICULTY action: since letter `a`
     is not yet mastered (this was attempt 1), it stays on `a`
     rather than escalating. ✅
   - `next_exercise: letter_lower_a` — confirms the rotation rule
     is working: the learner stays at Level 4 lowercase, on the
     same letter, until mastery is achieved.
   - `letter_mastery.current_letter_mastered: false`,
     `mastered_count_at_level: 0/11` — the per-letter mastery
     tracker is correctly initialized.

This is exactly the behaviour the audit-driven improvements were
designed to produce:
- The DQN makes the live decision (no fallback).
- The handwriting scorer correctly identifies a clean trace.
- The Level 4 rotation rule prevents premature escalation.
- The per-letter mastery tracker is initialized and ready to track
  progress across multiple Level 4 letters.

---

## Safety net (always on, even with the DQN deployed)

The integration is safe in the engineering sense: the DQN policy is
never used unchecked. Four layers protect the deployment:

### Layer 0 — Pedagogical layer (audit-driven, 2026-11)

Sits between the DQN's decision and the safety layer. Shapes the
action distribution to be pedagogically diverse and enforces
session-level tutoring rules the trained DQN does not learn on its
own (because they depend on session context the 12-D state vector
only partially encodes).

Three rules, in priority order:

1. **Repeated-failure de-escalation** (highest priority). If the
   learner has had >= `max_attempts_at_letter` (default 4) attempts
   at the current Level 4 letter without mastering it, override the
   DQN's choice to `DECREASE_DIFFICULTY` (drop one level to give the
   learner an easier success). Pedagogical rationale: a learner
   who cannot master a Level 4 letter after several attempts needs
   an easier success, not more attempts at the same difficulty.

2. **Post-break progression**. If the previous action was
   `SUGGEST_BREAK` (i.e., the learner just took a break) AND the
   current attempt's quality_score >= mastery threshold, override
   the DQN's choice to `INCREASE_DIFFICULTY`. The Level 4 rotation
   rule in the intervention controller then turns this into a
   same-level rotation (move to the next Level 4 letter) if other
   letters remain, or an escalation if all are mastered.
   Pedagogical rationale: after a refreshing break, a successful
   attempt should be rewarded with a new challenge, not with more
   of the same.

3. **Action diversity safeguard** (lowest priority). If the DQN has
   picked the same action K times (default 2) in the last N
   decisions (default 4), pick a pedagogically reasonable alternative.
   Pedagogical rationale: a DQN that always picks
   `INCREASE_DIFFICULTY` on good scores is pedagogically too
   narrow; varying the action keeps the learner engaged and
   exposes them to different support types (hints, demos, larger
   templates, mode switches).

All three rules are deterministic, audited (every override is
logged with a reason in the `pedagogical` field of the API
response), and configurable via `PedagogicalLayerConfig`
(environment prefix `PEDAGOGICAL_`). They never modify the DQN
checkpoint or the locked protocol — they only shape which action
id the safety layer sees.

**Measured effect on the deployed R12_4 checkpoint**: in a
20-attempt session of clean Level 4 lowercase traces, the
pedagogical layer overrode the DQN's choice on 9 of 20 attempts.
The action distribution went from 3 unique actions used (max share
65%) to 6 unique actions used (max share 45%) — well below the
gate's 60% dominance threshold. The DQN's natural choices are
preserved as `raw_action_id` in the response for audit.

### Layer 1 — Behavioural gate at backend startup

Every time the backend starts, `PolicyService.__init__` loads the
checkpoint and runs `behavioral_gate.run_behavioral_gate` on a fixed
set of probe states with known safe expectations. If any probe fails,
the backend reports `policy_mode: "rule_based_fallback"` and uses
the intelligent rule-based tutor for every decision. The DQN's
Q-values are still computed and logged as "shadow" Q-values (for
audit) but never used to choose actions.

You can verify the gate's verdict at any time:

```bash
curl http://127.0.0.1:8000/health | python -m json.tool
```

Look for `behavioral_gate_passed: true` and `fallback_reasons: []`.

### Layer 2 — Safety layer after every decision

Even when the DQN passes the gate and makes live decisions, every
choice runs through `safety_layer.apply_safety_layer` before being
executed. The safety layer enforces:

- **Forced break** when `attempts_since_break >= max_attempts_before_forced_break` (default 25).
- **Forced break** when `fatigue_proxy >= fatigue_break_threshold` (default 0.70).
- **Block escalation** when `consecutive_fails >= block_increase_after_consecutive_fails` (default 1).
- **Block escalation runs** of more than `max_consecutive_escalations` (default 2).
- **Block excessive repetition** of any single action
  (`max_consecutive_same_action`, default 3).
- **High-frustration support**: when `frustration >= high_frustration_support_threshold` (default 0.75), demanding actions are replaced with supportive ones.

The safety layer's verdict is always logged (`raw_action_id` vs
`safe_action_id`), so the override frequency can be audited later.

### Layer 3 — Intelligent rule-based fallback tutor

If the gate ever fails, the intelligent rule-based tutor from the
prior patch makes every decision. It sees the full session context
(score history, action log, recent exercises, current letter,
mastery) and applies:

- **Intervention cooldowns** — an intervention just used is not
  reselected within `intervention_cooldown` attempts (default 2)
  unless the situation strictly requires it.
- **Effectiveness tracking** — past interventions are labelled
  "effective", "neutral", or "harmful" by comparing the score after
  them to the score before. Harmful interventions are avoided.
- **Targeted support** — for incorrect handwriting, a supportive
  action (hint, demo, larger template, switch to tracing) that was
  not recently labelled harmful.
- **Mastery-aware progression** — the tutor detects when the
  current letter is mastered via the per-letter tracker and picks
  `INCREASE_DIFFICULTY`; the intervention controller's rotation
  rule turns that into a same-level rotation (not a level jump)
  until every same-level letter is mastered.

---

## Rollback / troubleshooting

### If the gate fails on a future load

Symptom: `/health` reports `policy_mode: "rule_based_fallback"` and
`fallback_reasons` lists one or more failures.

Action: the backend is still safe (the intelligent rule-based tutor
is making decisions). To diagnose:

```bash
curl http://127.0.0.1:8000/health | python -m json.tool
# Read the "fallback_reasons" and "behavioral_gate_failures" fields.
```

Common causes:
- The checkpoint file is corrupt or missing — restore from
  `backup_pre_R12_4/`.
- The gate's probe set was tightened in a future patch — re-train
  against the new gate, or rollback to the prior checkpoint.

### To roll back to the pre-R12_4 checkpoint

```bash
cd source/dysgraphia_rl/checkpoints
cp backup_pre_R12_4/double_dqn_legacy_2026-11-08.pt double_dqn.pt
# Restart the backend.
```

### To fall back to the rule-based tutor (no DQN at all)

```bash
cd source/dysgraphia_rl/checkpoints
mv double_dqn.pt double_dqn.R12_4.pt.bak
mv double_dqn.json double_dqn.R12_4.json.bak
# Restart the backend. /health will report policy_mode: "rule_based_fallback".
```

### To try a different Phase 5 checkpoint

If you have access to the original training project's `exports/`
folder, copy any of the 17 gate-passing checkpoints over
`double_dqn.pt`. The strongest alternatives are:

| File | Run | quality_gain | return | note |
|---|---|---:|---:|---|
| `childstart_R12_7_best.pt` | R12_7 | +0.072 | 14.45 | highest QG |
| `childstart_A_8_best.pt` | A_8 | +0.012 | 16.43 | only A with positive QG |
| `childstart_R12E_0_best.pt` | R12E_0 | -0.005 | 17.34 | strong return, neutral QG |

```bash
# Example: switch to R12_7
cp /path/to/training/exports/childstart_R12_7_best.pt \
   source/dysgraphia_rl/checkpoints/double_dqn.pt
# Update double_dqn.json with the new SHA-256, source_run, and eval results.
# Restart the backend.
```

The backend will re-run the behavioural gate on the new checkpoint
at startup. If it passes, `policy_mode` will be `"dqn"`; if not,
`"rule_based_fallback"`.

---

## What needs more training vs what's done with code

The user asked: "make it learn from the writing and choose the best
action if these to be fulfilled involved more data/more training let
me know". Here's the honest answer.

### What's done with code (no retraining needed)

The pedagogical layer above (Layer 0) addresses all the specific
behaviours the user asked for, without retraining:

- **Action diversity** (no single action dominant): the action
  diversity safeguard overrides the DQN's repeated choices with
  pedagogically reasonable alternatives. Measured effect: action
  distribution went from 3 unique actions / 65% max share to 6
  unique actions / 45% max share on a 20-attempt session.

- **"If I can't get a -> another Level 4 letter, decrease the
  difficulty"**: the repeated-failure de-escalation rule overrides
  to `DECREASE_DIFFICULTY` after 4 failed attempts at the same
  Level 4 letter. The learner drops one level (e.g., from Level 4
  `letter_lower_a` to Level 3 `letter_lower_e`) where they can
  succeed and rebuild confidence.

- **"After break + good score, don't stay there, move onto the
  next letter"**: the post-break progression rule overrides to
  `INCREASE_DIFFICULTY` after a break + good score. The Level 4
  rotation rule then turns this into a same-level rotation (move
  to the next Level 4 letter) if other letters remain.

These rules shape the action distribution without modifying the
DQN checkpoint or the locked protocol. They are deterministic,
audited, and configurable via environment variables.

### What needs more training (cannot be fixed with code alone)

For a fundamentally better policy that learns from the writing
directly (rather than relying on the 12-D state vector + post-hoc
pedagogical shaping), you would need a new training run with one
or more of the following changes:

1. **Action diversity regularization during training.** Add an
   entropy bonus or per-action usage penalty to the DQN loss
   function so the trained policy naturally uses more of the 10
   actions. This would require modifying the reward in
   `configs/phase5.yaml` (the locked protocol) — so it would be
   a NEW pre-registered protocol, not an edit to the current one.

2. **More training steps.** The current protocol trains for
   150,000 steps. Doubling or tripling this might give the DQN
   more time to learn the value of the underused actions
   (SHOW_HINT, ENLARGE_TEMPLATE, REPLAY_STROKE_ANIMATION,
   SWITCH_TO_TRACING_MODE). The locked protocol's training budget
   could be expanded in a new pre-registered protocol.

3. **More seeds + ensemble.** Run the protocol with more training
   seeds (e.g., 20 or 30 per arm instead of 10) and either pick
   the best individual checkpoint by action diversity, or build
   an ensemble that averages the Q-values across multiple seeds
   (which naturally smooths the action distribution).

4. **Reward shaping for pedagogical diversity.** The current
   reward (version `sref`) optimizes for skill gain and engagement
   but does not explicitly reward action diversity. A new reward
   version that adds a small bonus for using underused actions
   would train a more diverse policy. Again, this would be a new
   pre-registered protocol.

5. **A different network architecture.** The current `MLP 12-128-64-10`
   only sees the 12-D state vector. A network that also processes
   the raw handwriting (e.g., a small CNN over the trace image, or
   an LSTM over the stroke points) could learn directly from the
   writing rather than from the precomputed features. This would
   be a substantial research effort, not a small extension.

### Recommendation

For the current deployment: the integrated project (R12_4 + the
pedagogical layer) gives you the behaviours you asked for without
retraining. It is the right choice for a research demo or a
development environment.

For a future deployment that needs a fundamentally better policy:
plan a new pre-registered training protocol that includes action
diversity regularization (item 1 above) and possibly more seeds
(item 3). This is a research project, not a code change — it
would take a few days of training time and a new pre-registration
document. The current locked protocol (`configs/phase5.yaml` v1)
should not be edited; the new protocol would be a separate
`configs/phase6.yaml` (or similar).

All results are simulated. No clinical validation. No claim of
benefit to real children.

---

## Per-child data — not included, not used

This integrated project does NOT include any per-child data. The
training project's `data/state12_table.csv` and
`data/state12_table_contexts.csv` (per-child restricted) are not in
this zip and were not used in the integration. The backend's
`data/` folder is the SQLite attempt-log database, which is created
on first run and contains only the sessions you create.

The DQN checkpoint contains only trained network weights — no
per-child data, no training data, no sensitive information. It is
safe to share.

---

## Citation / provenance

- **Training project**: `dysgraphia_childstart_rl` (Phase 5 locked
  protocol, `configs/phase5.yaml` v1).
- **Training run**: `R12_4` (arm R12, seed 4, best validation score
  0.2209 at step 140553).
- **Training-pipeline gate**: passed (no failures, low_fatigue_break_rate 0.15).
- **Tutor behavioural gate**: passed (`vendored_gate_passed: true`,
  `behavioral_gate_passed: true`, no failures).
- **Tutor main-env eval (200 episodes)**:
  - checkpoint quality_gain = +0.057
  - checkpoint return = 25.31
  - rule_based quality_gain = -0.121
  - rule_based return = 5.00
  - do_nothing quality_gain = -0.241
  - do_nothing return = -8.68
- **SHA-256**: `411b7d07da82322d98866c5d912c40d1c856d281856da4cdb05605426fb505cf`

All results are simulated. No clinical validation. No claim of
benefit to real children.
