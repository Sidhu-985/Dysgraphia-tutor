# Implementation summary (Oct 2026 upgrade)

The work started from commit `70c235d` (the repository as supplied). The full commit history is in `evidence/git_log.txt`.

## Backend (`backend/`)
* **Safety layer.** Seven ordered rules. New ones: fatigue ≥ 0.70 always forces a break; at most one break in a row; under high frustration, demanding actions are replaced by support; difficulty is never increased after a failed attempt; repeats under frustration stay supportive.
* **Policy service.** A behavioural gate (G1–G6) runs before the checkpoint is used live. The rule-based fallback takes over if the gate fails or the checkpoint is missing or corrupt. Invalid states fall to a safe default, and an inference exception hands the decision to the rule-based agent. `/health` reports `policy_mode`, the gate details and the fallback reasons.
* **API.**
  * Pseudonymous `child_id` validation, `data_source` provenance, and `real_child_study` refused without approval.
  * Session difficulty matches the starting exercise. There is one exercise per level 0–4, and `next_exercise` follows the delivered action.
  * Transitions are logged against the next observed attempt; the last one is stored with reward NULL at session end.
  * The reward follows reward v2/v3, matching the simulator.
  * Non-finite or degenerate features are rejected before the session is changed.
* **Features.**
  * Spacing now measures consistency. The old version scored a perfectly regular row as 0.
  * Baseline alignment now compares the trace's lower envelope with the template's. The old version scored random scribbles at about 0.9.
  * Unmeasured terms (spacing with fewer than 3 strokes, pressure on a device without it) are left out of the quality average instead of filled with neutral values.
* **Storage.** Provenance, policy-source and raw-point columns, added by non-destructive ALTERs. A `transitions` table. Singletons are created under a lock; this race was found under concurrency.
* **Robustness.** A NaN in the request body now gets a 422 instead of a dropped connection.
* **Torch-free imports.** The feature pipeline imports without PyTorch.

## RL (`dysgraphia_rl/`)
* Reward v2 (cumulative improvement, unnecessary-break penalty) and reward v3 (no practice terms on break steps). Weights are validated so that w1 is the largest. v1 reproduces the original exactly.
* Simulator: a break cap with the executed action reported, an observed first attempt, carried-forward break observations, and a difficulty-matched `latent_skill`.
* Always-break and do-nothing baselines, a seedable random agent, and a generalized reward-hacking audit.
* A 5-seed training driver, and a 7-policy × 5-run paired evaluation with 95% CIs and the acceptance checks fixed in advance.

## Tablet app (`tablet_app/`). Written, NOT compiled or run (no Flutter SDK available)
* A design system: cream, teal and lavender, with computed contrast ratios, 64 dp touch targets and large type.
* Screens: home (pseudonymous code check), exercise selection (backend exercises and policy status), practice (canvas takes most of the screen), feedback ("Practice score (shape match) … not a test result"), break (sends `break_complete` only when the break is taken), progress (backend scores only, with empty states), and completion (no claims of improvement).
* Canvas: raw pointer events (taps are kept, no pan slop), **device event timestamps**, pressure normalised or null, palm rejection, a template fitted to the canvas and scaled by `template_scale`, a hint (start dot and arrows), a demo animation, and tracing/free-writing modes. Reduced motion is respected.
* Tests: the original 4 widget tests are kept, and `test/ui_test.dart` adds canvas-capture, geometry, labelling, progress, break and API wire-format tests. **None of them has been run.**

## Not done or not possible here
* Flutter analyze, test, build and screenshots (blocked).
* Real-child data (correctly disabled). The Kaggle dataset (blocked, and not suitable).
* A learned policy that passes the acceptance gate. Two rounds were reported and no third was run.

## Final-submission pass (2026-10-29)
* All packaged results were reproduced first: tests, the Round 2 evaluation, the perturbation suite and the state comparison, each bit-identical.
* **Simulator, observation v2:**
  * the backend's quality weights and its rule for leaving out unmeasured terms;
  * the spacing placeholder for single-stroke exercises;
  * pressure available on half of episodes, with a placeholder otherwise.

  `observation_version="v1"` reproduces the earlier rounds.
* **Backend:**
  * the retry count now includes the current attempt, matching the simulator;
  * attempt submission is idempotent (the same attempt_id is replayed once; a different payload gets a 409);
  * the app reuses the attempt id on retry.
* **Round 3 retraining and evaluation.** Offline criteria met; the behavioural gate still fails, so the fallback stays live.
* **Analysis and tooling:**
  * slant investigation, with no change made;
  * per-feature state schema report;
  * `scripts/action_breakdown.py`;
  * Dart syntax-check tool;
  * live check F8 for duplicate submissions.
