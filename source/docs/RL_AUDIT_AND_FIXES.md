# RL audit, fixes and results

All RL numbers here are **simulated** (DysgraphiaTutorEnv, five hand-authored learner profiles). They are not evidence about real children. The protocol is `docs/EVALUATION_PROTOCOL.md`. It was written before any training. The Round 2 amendment was written after Round 1 and before Round 2 was run.

## 1. What was wrong (Phase A, reproduced on the original code)

The full evidence is in `evidence/phase_a/README.md`. In summary:

| Finding | Evidence |
|---|---|
| A policy that always suggests a break scored **49.25** per episode with quality gain **0.000**. That was 2.6× the rule-based baseline. | `repro_sim_baselines_output.json` |
| The shipped DQN chose SUGGEST_BREAK **45%** of the time and SWITCH_TO_TRACING_MODE 50%. Its quality gain was +0.04. | `repro_original_dqn_output.json` |
| The original audit flags **none** of the above, because it only looks at DECREASE_DIFFICULTY. | both files |
| "2–2.5× the strongest baseline" had no always-break or do-nothing comparison. Against always-break the ratio was 1.34×. | – |
| Live API, clean writer (quality ≈ 0.86), as the original app behaved (it never called break_complete): **65%** of raw policy actions were breaks. Otherwise the policy chose DECREASE_DIFFICULTY on 18 of 20 attempts. | `repro_live_clean_writer_*.json` |

Root causes:
1. The Δquality term telescopes: the per-step deltas over an episode add up to just q_final − q_start, while engagement and frustration are scored on every step.
2. A break step still earned the per-step terms.
3. The starting quality was a hidden constant, about 0.1 above the learner's real first attempt.
4. Break-step observations were zeroed.
5. Nothing checked the checkpoint's behaviour before deployment.

## 2. Fixes

| Area | Change | Tests |
|---|---|---|
| Reward v2 (`utils/reward_utils.py`, env) | w1·(q_t − q_start), i.e. cumulative improvement since the start, replaces the per-step Δ. New −w6·unnecessary_break term for a break below fatigue 0.5. `validate_weights` enforces that w1 is strictly the largest weight. | `tests/test_reward_v2.py` |
| Reward v3 (Round 2) | On a step where the child does not write, the reward is **only** −w6·unnecessary_break. Writing steps are the same as v2. | `test_v3_*` |
| Break cap | At most one break in a row, in both the simulator and the backend safety layer. A second request is executed as CONTINUE_NO_CHANGE, and the transition stores the **executed** action. | `test_consecutive_break_cap_*`, `test_safety_rules_v2.py` |
| Simulator honesty | The first state comes from an observed first attempt. Break steps carry the last measured features forward. v1 mode reproduces the original exactly (a test checks this). | `test_v1_mode_reproduces_original_break_hacking` |
| Baselines | Always-break and do-nothing were added permanently. RandomAgent can be seeded. | `evaluate_baselines.py` |
| Audit | Now flags non-writing excess (> 25%), single-action dominance (> 50%), no quality gain, and breaks above 50%. The original rule is kept. | `test_generalized_audit_*` |
| Evaluation | 7 policies × 5 runs × 150 paired episodes, 95% t-CIs, per-run values, the acceptance rules fixed in advance, and a difficulty-matched skill gain. | `results/evaluation_summary.json` |
| Live transitions | (s_t, a_t, r, s_{t+1}) takes r and s_{t+1} from the **next observed attempt**. At session end the transition is stored with reward NULL and outcome_observed = 0. | `S6_transition_integrity` |
| Observation v2 (Round 3) | The simulator's quality formula and its spacing/pressure placeholders match the backend, and the backend's retry count includes the current attempt. `observation_version="v1"` reproduces Rounds 1–2 exactly | `test_sim_backend_consistency.py` |
| Behavioural gate | Probes G1–G6 run before a checkpoint may decide live. If it fails, the backend uses the **rule-based fallback**, reported in `/health` and in every response. Invalid states never reach the model. If inference throws, the rule-based agent decides. | `test_behavioral_gate.py`, `test_live_scenarios.py` |
| Safety rules | Fatigue ≥ 0.70 always forces a break. High frustration replaces demanding actions with support. Difficulty is never increased after a failure. Repeats under high frustration stay supportive. | `test_safety_rules_v2.py`, S2–S4 |

## 3. Results: comparison tables (three rounds; mean over 5 runs, 95% CI)

**Round 1, reward v2** (`results/round1/`)

| Policy | Reward | Quality gain | Difficulty-matched skill gain | Exec. break % | Requested break % | Final difficulty | Fatigue-terminated |
|---|---|---|---|---|---|---|---|
| Random | −0.4 [−4.5, 3.8] | −0.127 [−0.155, −0.099] | 0.131 | 9.1% | 9.9% | 2.05 | 22% |
| Rule-based | 3.2 [1.4, 5.0] | −0.126 [−0.139, −0.112] | 0.131 | 7.0% | 7.0% | 1.09 | 0% |
| Bandit (online) | 13.4 [−0.8, 27.7] | −0.148 [−0.189, −0.107] | 0.124 | 1.0% | 1.1% | 0.80 | 73% |
| Bandit (frozen) | 14.7 [2.2, 27.2] | −0.164 [−0.222, −0.107] | 0.121 | 0.0% | 0.0% | 0.44 | 81% |
| Do-nothing | −7.3 [−8.0, −6.6] | −0.222 [−0.233, −0.211] | 0.109 | 0% | 0% | 2.00 | 81% |
| Always-break | 11.9 [8.7, 15.2] | 0.006 [−0.017, 0.028] | 0.104 | 50% | 100% | 2.00 | 0% |
| **Double DQN** | **57.9 [51.8, 64.0]** | **0.183 [0.158, 0.207]** | 0.157 | **31.3%** | 48.2% | 0.02 | 0% |

**Round 2, reward v3 with observation v1** (`results/round2/`)

| Policy | Reward | Quality gain | Difficulty-matched skill gain | Exec. break % | Requested break % | Final difficulty | Fatigue-terminated |
|---|---|---|---|---|---|---|---|
| Random | 1.6 [−2.3, 5.4] | −0.127 [−0.155, −0.099] | 0.131 | 9.1% | 9.9% | 2.05 | 22% |
| Rule-based | 5.9 [4.3, 7.4] | −0.126 [−0.139, −0.112] | 0.131 | 7.0% | 7.0% | 1.09 | 0% |
| Bandit (online) | 15.2 [0.8, 29.6] | −0.041 [−0.083, 0.001] | 0.143 | 7.7% | 8.3% | 0.35 | 10% |
| Bandit (frozen) | 12.2 [1.0, 23.4] | −0.105 [−0.191, −0.020] | 0.137 | 8.2% | 8.2% | 1.60 | 0% |
| Do-nothing | −7.3 [−8.0, −6.6] | −0.222 [−0.233, −0.211] | 0.109 | 0% | 0% | 2.00 | 81% |
| Always-break | −0.6 [−2.4, 1.2] | 0.006 [−0.017, 0.028] | 0.104 | 50% | 100% | 2.00 | 0% |
| **Double DQN** | **41.7 [24.9, 58.6]** | **0.121 [0.040, 0.202]** | 0.156 | **26.6%** | 40.6% | 0.43 | 0% |

Per-seed DQN values (Round 2, seeds 0–4):

| Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| Reward | 46.6 | 49.9 | **17.7** | 46.3 | 48.3 |
| Quality gain | 0.166 | 0.153 | **0.009** | 0.119 | 0.159 |
| Executed break fraction | 23.4% | 24.0% | **38.2%** | 22.2% | 25.0% |
| Final difficulty | 0.00 | 0.14 | 1.71 | 0.00 | 0.30 |

Seed 2 is a poor run and is included in every summary. The generalized audit flags it (non-writing excess and low mean quality).

**Round 3: reward v3 with backend-consistent observation v2. Current; the deployed checkpoint is its seed 0** (`results/evaluation_summary.json`, `results/action_breakdown_round3.json`)

Round 3 changed only the simulator's observations and quality formula so they match the backend, plus the backend's retry-count timing. Rationale: `docs/STATE_SCHEMA.md` and the Round 3 amendment in `docs/EVALUATION_PROTOCOL.md`, written before the run. The reward is unchanged.

| Policy | Reward | Quality gain | Difficulty-matched skill gain | Exec. break % | Requested break % | Final difficulty | Fatigue-terminated | Audit-flagged runs |
|---|---|---|---|---|---|---|---|---|
| Random | 2.1 [−1.0, 5.2] | −0.130 [−0.156, −0.104] | 0.132 | 9.1% | 9.9% | 2.05 | 22% | – |
| Rule-based | 5.0 [3.7, 6.3] | −0.130 [−0.150, −0.110] | 0.132 | 7.0% | 7.0% | 1.12 | 0% | all 5 |
| Bandit (online) | 17.2 [3.7, 30.7] | −0.048 [−0.084, −0.012] | 0.144 | 7.5% | 8.0% | 0.41 | 13% | 0, 2, 4 |
| Bandit (frozen) | 21.0 [8.9, 33.0] | −0.001 [−0.089, 0.087] | 0.151 | 8.0% | 8.0% | 0.36 | 0% | 2, 3 |
| Do-nothing | −8.5 [−9.5, −7.4] | −0.232 [−0.241, −0.222] | 0.109 | 0% | 0% | 2.00 | 81% | all 5 |
| Always-break | 1.6 [0.2, 3.1] | 0.019 [−0.003, 0.040] | 0.105 | 50% | 100% | 2.00 | 0% | 2, 3 |
| **Double DQN** | **66.0 [61.8, 70.3]** | **0.191 [0.168, 0.214]** | 0.166 | **23.3%** | 29.4% | 0.05 | 0% | none |

Per-seed DQN values (Round 3, seeds 0–4):

| Metric | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| Reward | 62.3 | 65.9 | 63.8 | 66.8 | 71.3 |
| Quality gain | 0.166 | 0.189 | 0.182 | 0.210 | 0.208 |
| Executed break fraction | 20.7% | 23.0% | 24.0% | **25.4%** | 23.5% |
| Requested break fraction | 27.3% | 27.9% | 32.1% | 29.2% | 30.3% |
| Final difficulty | 0.00 | 0.00 | 0.03 | 0.15 | 0.07 |

Action counts over 5 runs × 150 episodes (30,000 actions):

| Action | Proposed | Executed | Executed % |
|---|---|---|---|
| DECREASE_DIFFICULTY | 7,138 | 7,138 | 23.8% |
| INCREASE_DIFFICULTY | 0 | 0 | 0.0% |
| SHOW_HINT | 6,279 | 6,279 | 20.9% |
| ENLARGE_TEMPLATE | 1,778 | 1,778 | 5.9% |
| REPLAY_STROKE_ANIMATION | 184 | 184 | 0.6% |
| SWITCH_TO_TRACING_MODE | 4,064 | 4,064 | 13.5% |
| SWITCH_TO_FREE_WRITING_MODE | 0 | 0 | 0.0% |
| SUGGEST_BREAK | 8,808 | 7,002 | 23.3% |
| REPEAT_SAME_LETTER | 414 | 414 | 1.4% |
| CONTINUE_NO_CHANGE | 1,335 | 3,141 | 10.5% |

The consecutive-break cap replaced 1,806 proposed breaks with CONTINUE_NO_CHANGE.

**98.96%** of executed breaks still happen at fatigue below 0.5, with a median fatigue of 0.06 and a median frustration of 0.20.

In the simulator nothing sits between "accepted" and "executed" except the break cap, so the two are the same there. Live, proposed, accepted and executed are recorded per attempt as `raw_action_id`, `safe_action_id` and the intervention action.

### Acceptance (predeclared criteria)

| Criterion | Round 1 | Round 2 | Round 3 (current) |
|---|---|---|---|
| 1. Reward vs always-break | PASS (4.86×) | PASS (CI rule) | PASS, but the ratio (41×) is **not meaningful**, because always-break is close to 0 (1.6 [0.2, 3.1]). Absolute paired difference: **+64.4 [61.1, 67.7]** |
| 2. Quality-gain CI low bound > 0 | PASS (0.158) | PASS (0.040) | PASS (0.168), **though it mostly reflects a lowered difficulty** (final difficulty 0.05) |
| 3. Executed break fraction < 25% (mean over runs) | **FAIL (31.3%)** | **FAIL (26.6%)** | **PASS (23.3%)**. Seed 3 alone is 25.4%, which is reported |
| 4. Audit flags DQN in no run | PASS | **FAIL** (run 2) | PASS. Run 3 carries the informational flag `non_writing_excess` but is not suspected, because its quality gain is positive |
| Backend behavioural gate, all 5 seeds | **FAIL** | **FAIL** | **FAIL**: G4 low-fatigue break rate 34–70%; G6 easier task for a clean writer (4 of 5 seeds); G1/G2 breaks (seeds 2, 3) |

**Outcome:** Round 3 meets the four predeclared **offline** criteria for the first time. It still **fails the behavioural gate**: in backend-shaped, low-fatigue states it asks for a break 34% of the time (seed 0) and makes the task easier for a clean writer. Under the protocol the fallback therefore stays live. `/health` reports `policy_mode = "rule_based_fallback"` and gives the reasons (`evidence/final/phase_e/health.json`). No Round 4 was run.

## 4. What does a do-nothing policy score, and what does that say about the DQN?

Round 3 (current) numbers, paired over 5 runs:
- **Do-nothing** (always CONTINUE_NO_CHANGE) scores **−8.5** reward with a quality gain of **−0.232**. 81% of its sessions end early at the fatigue cutoff, and the difficulty stays at 2.00.
- The DQN beats it by **+74.5** [70.8, 78.2] reward, **+0.42** [0.41, 0.44] quality gain and **+0.057** [0.054, 0.059] difficulty-matched skill gain.

Most of that comes from two shortcuts, not from better teaching:
* **Making the task easier.** The DQN ends at a mean difficulty of 0.05, against 2.00. In this simulator, moving from level 2 to level 0 adds about +0.16 to the quality proxy by itself. That is most of the DQN's +0.19 raw quality gain.
* **Keeping sessions alive with breaks.** 99% of its executed breaks happen at fatigue below 0.5. They remove fatigue-cutoff endings (0% against 81%) and so give the learner more practice attempts. More attempts is what raises the simulator's skill measure.

The difficulty-matched skill gain cannot be raised by lowering difficulty. On it the DQN beats rule-based by only **+0.034** [0.031, 0.037] and the frozen bandit by **+0.014** [0.004, 0.025]. In this simulator the learned policy's real advantage is session management. There is no evidence that it chooses better interventions. Its live behaviour shows the same pattern: with the gate disabled it chose DECREASE_DIFFICULTY on 6 of 6 attempts for a clean writer scoring about 0.98 (`evidence/final/phase_e/live_scenarios_gate_disabled_round3_DQN.json`).

Round 2 values, for the record: do-nothing −7.3; DQN − do-nothing +49.0 reward and +0.047 skill gain.

## 5. Sim-to-real mismatch (Section 6 check)

The full per-feature table is in `docs/STATE_SCHEMA.md`. The packaged report flagged **10 of 12** features, and that was reproduced. Three of them were real encoding or timing inconsistencies and are now fixed:
- spacing placeholder;
- pressure availability placeholder;
- retry-count timing.

The simulator's quality formula now also matches the backend. **7 of 12** remain flagged (`evidence/final/state_distribution_after_fixes.json`). The cause is that the two synthetic populations differ: simulated learners mostly fail, and synthetic backend writers mostly succeed. Speed also has no common calibration. That is not an encoding error, and it cannot be fixed without real interaction data. It is one reason the gate, the fallback and the safety layer remain required.

## 6. Mapping actions to the controller, and making each action visible

Each of the 10 actions maps to an intervention, `interventions/intervention_controller.py`, and each one changes what the child sees:
- difficulty actions change the exercise (there is now one exercise per level 0–4);
- the hint shows a start dot and arrows;
- replay animates the shape;
- enlarge rescales the guide;
- the mode actions show or hide the guide;
- the break action opens the break screen, which sends `break_complete` only if the break was taken;
- repeat and continue keep the current exercise.

The tablet side of this is **written but not compiled** (no Flutter SDK; see the acceptance matrix).

## 7. Double DQN internals (checked, unchanged)

- Online and target networks are 12→128→64→10.
- The target is the Double-DQN target: argmax from the online network, value from the target network.
- Other settings: γ = 0.95, target sync every 200 steps, PER with α = 0.6 and β annealed from 0.4 to 1.0.
- `done` is `terminated` only. A time-limit truncation still bootstraps.

These were verified by the original test suite, which still passes unchanged: 29 of 29.
