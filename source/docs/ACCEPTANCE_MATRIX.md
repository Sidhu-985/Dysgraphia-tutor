# Final acceptance matrix (final-submission pass, 2026-10-29)

**PASS** means verified by the cited evidence. **FAIL** means measured and not met. **BLOCKED** means it could not be run in this environment. **NOT RUN** means it was not attempted. Nothing BLOCKED or NOT RUN is counted as PASS. Environment: `evidence/final/baseline_environment.md`. Every result was reproduced from the packaged ZIP before any change: `evidence/final/baseline_*`.

| Requirement | Status | Evidence | Remaining issue |
|---|---|---|---|
| Existing backend regression suite | **PASS** | `cd backend && python -m pytest -q -rx`: 121 collected, **119 passed, 2 xfailed (strict)**, exit 0. Run in a fresh clone (`evidence/final/final_test_results.md`) | The 2 strict xfails are the slant target and the DQN-argmax test while the checkpoint is gated out. Both are disclosed, and neither assertion was weakened |
| RL-core regression suite | **PASS** | `cd dysgraphia_rl && python -m pytest -q`: 45 collected, **45 passed**, exit 0, fresh clone. All 29 original tests are unchanged | – |
| DQN break fraction below 25% | **PASS** (offline, Round 3) | Mean executed fraction **23.3%** over 5 seeds × 150 episodes: 7,002 of 30,000 actions (`results/action_breakdown_round3.json`). Rounds 1 and 2 **FAILED** (31.3%, 26.6%) and are reported | Seed 3 alone is 25.4%. **99% of executed breaks happen at fatigue below 0.5.** The fraction is below 25%, but the breaks are mostly unneeded |
| Always-break and do-nothing baselines | **PASS** | All 7 policies × 5 runs × 150 paired episodes, 95% CIs (`dysgraphia_rl/results/evaluation_summary.json`) | – |
| Positive quality gain | **PASS** (predeclared estimator) | DQN quality gain 0.191 [0.168, 0.214]. The CI low bound is above 0 | **Mostly comes from lowering difficulty** (final difficulty 0.05 vs 2.0). Difficulty-matched skill gain over rule-based is only +0.034 [0.031, 0.037] |
| DQN reward ≥ 2× always-break | **PASS**, but the ratio is not meaningful | DQN 66.0 [61.8, 70.3] vs always-break 1.6 [0.2, 3.1], which is close to 0. Absolute paired difference **+64.4 [61.1, 67.7]** | The 41× ratio is not quoted as evidence |
| Slant correlation at least 0.80 | **FAIL** | Pooled ρ = **0.778** (n = 220). Reproduced, and the cause was investigated (`evidence/final/slant_investigation.json`, `docs/QUALITY_SCORE.md`) | Comes from the perturbation construction: a shear angle barely changes the shallow wave (0.027 normalized shape change). The proxy responds to slant: per-template ρ = 1.00, and ρ against actual shape change is 0.94 (post hoc, diagnostic only). Neither the target nor the severity scale was changed |
| All perturbation targets | **FAIL** (4 of 5) | jitter 0.828 ✓, tremor 0.863 ✓, scale 0.810 ✓, **slant 0.778 ✗**, spacing 0.899 ✓ | Slant, as above |
| State-distribution compatibility | **FAIL** (partly fixed) | Packaged: 10 of 12 flagged (reproduced). After fixing the spacing and pressure placeholders, retry timing and the quality formula: **7 of 12** (`docs/STATE_SCHEMA.md`, `evidence/final/state_distribution_after_fixes.json`, `test_sim_backend_consistency.py`) | The remaining differences come from the two synthetic populations differing and from speed having no calibration. Real data would be needed to fix them |
| Checkpoint behavioural acceptance | **FAIL** | All 5 Round 3 checkpoints fail the gate. The deployed seed 0 gives G4: 34% low-fatigue breaks, and G6: easier task for a clean writer (`evidence/final/phase_e/health.json`) | The checkpoint is **not deployed live** |
| Safe fallback | **PASS** | The real-server `/health` reports `policy_mode = rule_based_fallback` with reasons. Gate, missing/corrupt checkpoint, invalid-state and inference-failure tests pass (`test_behavioral_gate.py`, `test_live_scenarios.py`). Real-server scenarios S1–S7 all PASS (`evidence/final/phase_e/live_scenarios.json`) | – |
| Real handwriting capture | **BLOCKED** (on device) / **PASS** (backend contract) | The backend validates and stores x, y, t, nullable pressure, stroke id and capture metadata. Canvas code uses raw pointer events and device timestamps (`handwriting_canvas.dart`), and widget tests are written | The Flutter SDK is unavailable, so no touch/stylus capture was run |
| Backend-to-frontend integration | **BLOCKED** | Backend E2E over HTTP **PASS**: scenarios S1–S7 and failure modes F1–F8, covering malformed and NaN input, empty attempts, unknown and ended sessions, duplicate submissions, 8 concurrent sessions and server restart, with 0 server 5xx responses (`evidence/final/phase_e/`) | Tablet ↔ backend was not run. App-side timeout and unavailable-server handling is written (`describeError`) but not executed |
| Flutter analysis and tests | **BLOCKED** | `flutter pub get / analyze / test` NOT RUN: the SDK is missing and storage.googleapis.com is refused (re-checked 2026-10-29). Partial check: the tree-sitter syntax parse of 22/22 Dart files is clean, and all local imports resolve | Type errors, lints and test failures are possible until it runs on a machine with Flutter |
| Running-app screenshots | **BLOCKED** | None exist. **No mockups were produced or presented as screenshots** | Needs a device or emulator |
| Final package integrity | **PASS** | `dysgraphia_tutor_final_submission.zip`: contents listed, extracted to a separate directory, and both suites re-run from the extracted copy (see `evidence/final/package_check.md`) | – |
| Documentation and demo instructions | **PASS** | README, `docs/*`, `docs/DEMO.md`. The commands were executed in this pass, except the Flutter ones (BLOCKED) | Flutter steps are unverified |

## Readiness classification: **NOT READY**

Following the definitions in the completion brief, the deciding fact is that **mandatory requirements remain unverified.** The Flutter application has not been compiled, tested or run, and no screenshots exist. Real on-device handwriting capture and tablet-to-backend integration are BLOCKED. All of these are core parts of the project. They are blocked by the environment, not shown to be broken, but they are still unverified.

What is verified:
- the backend pipeline, the safety layer and the gate-plus-fallback, end to end against a real server;
- the RL evaluation protocol, with all three rounds reported;
- the predeclared offline RL criteria, met in Round 3;
- reproducibility, with fresh-clone re-runs matching bit for bit.

What fails as measured:
- the behavioural gate;
- slant sensitivity (0.778);
- simulator/backend compatibility (7 of 12 features);
- the learned policy's real contribution, which is session management through low-fatigue breaks and lowering difficulty, not better teaching.

**Path to "READY WITH DISCLOSED LIMITATIONS":** run `flutter pub get && flutter analyze && flutter test` on a machine with the Flutter SDK, fix any errors, capture the seven screens from a running app with the backend live, and run the walkthrough in `docs/DEMO.md`. The remaining FAILs above are disclosed limitations of a simulation-trained research prototype. The live system already contains them safely, because the rule-based fallback is active and the safety layer applies to every decision.
