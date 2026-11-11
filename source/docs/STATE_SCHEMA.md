# 12-D state schema: simulator vs backend (Phase C, final pass)

The canonical order is shared by `dysgraphia_rl/utils/state_utils.py::STATE_FEATURE_NAMES`, the simulator's `_build_state`, the backend's `state_builder.build_state` and `agents/baselines.py`. Existing tests check the order (`test_state_ordering_matches_canonical_spec`, `test_state_vector_has_canonical_shape_and_order`). **The schema (dimension, order and range) did not change.** Old checkpoints still load. Round 3 retrains anyway, because the meaning of three features changed in the simulator.

Comparison data:
- `evidence/final/baseline_state_distribution.json`: as packaged, reproduced exactly. **10 of 12 flagged.**
- `evidence/final/state_distribution_after_fixes.json`: after the Round 3 corrections. **7 of 12 flagged.**

Simulator states come from 100 rule-based episodes. Backend states come from 30 sessions × 12 attempts of **synthetic** traces through the live API. Neither population is real children. Script: `backend/scripts/compare_state_distributions.py`. Tests: `backend/tests/test_sim_backend_consistency.py`.

"Outside" means the share of backend values that fall outside the simulator's 1st–99th percentile range.

| # | Feature | Definition / units | Backend source | Sim source | Range | Not-measured handling | Outside, before → after | Same meaning? | Available live? |
|---|---|---|---|---|---|---|---|---|---|
| 0 | prev_score | quality proxy of the attempt just completed | `compute_quality_score` | sim quality formula | [0,1] | unmeasured terms left out | 85% → 83% | **Yes since Round 3**: same weights and the same rule for unmeasured terms. **Fixed**: the sim used 0.40/0.25/0.20/0.15 and always included spacing and pressure | yes |
| 1 | stroke_similarity | exp(−DTW/0.35), normalized frame | static features | effective skill + noise | [0,1] | – | 83% → 83% | Same meaning, different populations: sim learners are weak (mean 0.18), synthetic writers mostly clean (0.85) | yes |
| 2 | spacing_accuracy | spacing consistency, needs ≥ 3 strokes | static features | was a simulated value; now the placeholder | [0,1] | **0.5 placeholder** | 100% → **0%** | **Fixed**: the sim simulated spacing the backend can never measure on single-stroke exercises | placeholder only, since all exercises are single-stroke |
| 3 | baseline_alignment | lower-envelope agreement with the template | static features | effective skill + noise | [0,1] | – | 68% → 68% | Same meaning, different populations | yes |
| 4 | writing_speed | path length / bbox diagonal / s, ÷ 1.0 | dynamic features | 1.3 − 0.5·skill + 0.3·fatigue | [0,2] | – | 75% → 75% | **No common calibration.** The sim's formula is hand-authored, and the backend value depends on the trace's real duration. Not fixable without real timing data. **Open** | yes |
| 5 | pressure_variance | normalized pressure variance | dynamic features | simulated, or placeholder when unavailable (p = 0.5 per episode) | [0,1] | **0.3 placeholder** | 100% → **0%** | **Fixed**: the sim always had pressure, and touch devices report none | stylus only |
| 6 | retry_count | consecutive failed attempts **including the current one**, /5, capped | session `consecutive_fails` | `consecutive_fails` | [0,1] | – | 91% → **0%** | **Fixed timing**: the backend used failures counted *before* the current attempt, one step behind the sim. The means still differ (sim 0.92, backend 0.04) because of population | yes |
| 7 | fatigue_proxy | accumulated per-attempt and pause fatigue | session dynamics | profile fatigue rate | [0,1] | – | 0% → 0% | Similar heuristic | yes (a proxy) |
| 8 | engagement | heuristic: +success / −failure | session dynamics | same idea | [0,1] | – | 93% → 93% | Same rule shape, different success rates | yes (a proxy) |
| 9 | improvement_trend | mean Δ of the last 6 quality scores | `compute_improvement_trend` | same formula | [−1,1] | 0 when fewer than 3 scores | 28% → 24% | Same formula. The backend's mix of clean and scribble traces varies more | yes |
| 10 | frustration | heuristic: +failure, −success, −break | session dynamics | profile sensitivity | [0,1] | – | 81% → 81% | Same rule shape. Sim learners fail almost always (0.88), synthetic writers rarely (0.08) | yes (a proxy) |
| 11 | consistency | 1 − std of the last 5 quality scores | `compute_consistency` | same formula | [0,1] | 0.5 when fewer than 2 scores | 2% → 2% | Same | yes |

Checked and **not** found to be a cause:
- feature order;
- raw vs normalized values (every feature is normalized on both sides);
- state-update timing for fatigue, frustration and engagement (both update before the vector is built);
- action ids (shared `ACTION_NAMES`; the controller maps all 10, `test_all_ten_actions_produce_a_valid_intervention`).

## What still differs, and why it is not "fixed"

The seven features still flagged (0, 1, 3, 4, 8, 9, 10) differ because the **two synthetic populations differ**, plus one uncalibrated scale (speed). It is not an encoding bug. The simulator's learner profiles are hand-authored, and the backend traces are generated test data. Making the two match would mean inventing a learner model for which no data exists. So the mismatch is reported, not hidden. This is why the behavioural gate, the fallback and the safety layer must stay in place.
