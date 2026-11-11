# Predeclared RL evaluation protocol (reward version 2)

Written and committed **before** any reward-v2 DQN was trained or evaluated.
The script that implements it is `dysgraphia_rl/scripts/evaluate_baselines.py`.
Results are reported for every seed. No seed, episode or run may be dropped or
re-run selectively. If a criterion fails, the result is reported as a failure.

## What is being evaluated

All numbers come from the **simulator** (`DysgraphiaTutorEnv`). They are
synthetic, offline results. They are not evidence about real children.

| Setting | Value |
|---|---|
| Environment | `DysgraphiaTutorEnv(reward_version="v2", max_steps=40, max_consecutive_breaks=1, break_fatigue_threshold=0.5)` |
| Reward weights | `RewardWeights.v2()`: w1=3.0 (largest), w2=1.0, w3=1.0, w4=1.5, w5=0.5, w6=1.5 |
| DQN training | `scripts/train_dqn.py` defaults except `--episodes 800`. Training seeds k ∈ {0, 1, 2, 3, 4}. Training episode k+ep uses env seed `k + ep`. |
| Evaluation episodes | 150 per (policy, run k). Env seeds `100000 + 1000·k + ep`, ep = 0…149. These are disjoint from every training seed. |
| Paired design | Every policy in run k is evaluated on exactly the same 150 env seeds. |
| DQN action selection | Greedy (ε = 0). The policy is frozen during evaluation. |
| Contextual bandit (frozen) | Pretrained for 800 episodes on env seeds `50000 + 1000·k + ep`, then frozen. |
| Contextual bandit (online) | Starts from scratch on the evaluation episodes and updates online. |
| Random | `RandomAgent(seed=k)`. |
| Policies | Random, Rule-based, Contextual Bandit (online), Contextual Bandit (frozen), Do-nothing (always CONTINUE_NO_CHANGE), Always-break (always requests SUGGEST_BREAK), Double DQN. |

## Per-run metrics (one value per policy per run k)

* **Mean episode reward**: undiscounted sum of reward-v2 per episode, averaged over 150 episodes.
* **Quality gain**: `score_history[-1] − score_history[0]` per episode, averaged. This is the final observed quality proxy minus the learner's starting level. It is the same definition used in the Phase A audit.
* **Difficulty-matched skill gain**: `latent_skill(end) − latent_skill(start)` at the episode's starting difficulty, with no fatigue, frustration or noise. This is a secondary metric. It is reported because the quality proxy can rise just by lowering the difficulty.
* **Mean final difficulty**: shows whether a quality gain came from making the task easier.
* **Executed break fraction**: share of executed actions that were SUGGEST_BREAK. **Requested break fraction** is also reported, and so is the share of steps where the break cap replaced a requested break.
* **Action distribution**: executed actions, all 10.
* **Termination**: share of episodes that ended because fatigue went above 0.95 (terminated). The rest were truncated at max_steps.
* **Reward-hacking audit**: `audit_action_distribution(..., quality_gain=mean quality gain)`. The flags are reported.

## Aggregation across runs

For each policy and metric, the 5 per-run values are summarised as a mean and a
95% confidence interval: mean ± t₀.₉₇₅,₄ · s/√5, where t₀.₉₇₅,₄ = 2.776. Per-run values are always printed next to the summary.

## Acceptance criteria (fixed before seeing results)

1. **Reward vs always-break.** If the always-break mean reward is > 0, the DQN mean reward must be ≥ 2× that value. If it is ≤ 0, a ratio means nothing. In that case the DQN's 95% CI lower bound must be above the always-break 95% CI upper bound.
2. **Positive quality gain.** The lower bound of the DQN's 95% CI for mean quality gain must be > 0.
3. **Break fraction.** The DQN's executed break fraction, averaged over runs, must be < 0.25.
4. **No reward hacking.** The generalized audit must not flag the DQN in any run.
5. **Contribution over do-nothing** is reported but has no pass/fail threshold. It covers the paired difference (DQN − Do-nothing) in reward, quality gain and difficulty-matched skill gain, with a 95% CI over runs.

Failing any of criteria 1 to 4 means the deployed checkpoint does not meet the
RL acceptance gate. The final report then says so.

### Clarification (added before any v2 results were generated)

"Frozen" contextual bandit means fixed weights **and** greedy action selection (ε = 0). That matches the greedy DQN. The online bandit keeps its ε = 0.1 exploration because it learns during evaluation.

### Which checkpoint is deployed (fixed before training)

The backend's `dysgraphia_rl/checkpoints/double_dqn.pt` is a copy of the **seed-0** checkpoint, `checkpoints/seeds/double_dqn_seed0.pt`. Seed 0 was chosen before any v2 training run, so the best seed cannot be picked afterwards. The deployed checkpoint must also pass the backend behavioural gate (`backend/app/rl_integration/behavioral_gate.py`). If it fails, the backend serves the rule-based fallback and reports this in `/health`. The checkpoint is not swapped for another seed.

## Round 1 result and Round 2 amendment (written after Round 1 and BEFORE Round 2 was run)

**Round 1** used reward v2 and is reported in full in `dysgraphia_rl/results/round1/`. It **failed** criterion 3: the executed break fraction was 31.3%, against a limit of 25%. Criteria 1, 2 and 4 passed: 4.86× the always-break reward, quality-gain CI low bound 0.158, and no audit flag. Diagnosis (`results/round1/break_diagnostics.json`): 99.5% of the DQN's executed breaks happened at fatigue below 0.5, with a median fatigue of 0.03. All five Round 1 checkpoints also **fail** the backend behavioural gate. They request SUGGEST_BREAK in 44–77% of low-fatigue probe states. So under the deployed gate, the backend would serve the rule-based fallback.

Cause: in reward v2 a break step still earns the per-step engagement, consistency and carried-forward quality-improvement terms. The only cost is the unnecessary-break penalty (at most 1.5). A break every third step keeps simulated fatigue near 0 at a small price.

**Round 2 change (reward v3):** on a step where the child does not write (an executed break), the reward is **only** `−w6 · unnecessary_break`. The quality, completion, engagement, frustration and consistency terms are credited only on steps where the child actually practises. A break therefore has to pay for itself through better later practice, for example by avoiding fatigue degradation and early termination. It is not rewarded for the calm state it produces while nobody writes. Writing steps are scored exactly as in v2. The weights are unchanged and w1 is still the largest.

Everything else is identical to Round 1: seeds, episodes, evaluation seeds, metrics, CI rule and acceptance criteria. Round 2 is evaluated with the v3 environment. The always-break and do-nothing baselines are re-evaluated under v3 as well, because rewards are only comparable within one reward version. **The deployed checkpoint becomes the Round 2 seed-0 checkpoint, whatever its result.** Both rounds are reported, and Round 1 is not hidden.

## Round 3 amendment (written after the Phase C state-mismatch investigation and BEFORE Round 3 was run)

Round 2 failed criterion 3 (executed break fraction 26.6%) and criterion 4 (audit flag on run 2). Root-cause diagnostics are in `evidence/final/break_root_cause_diagnostics.json` (Round 2 DQN, evaluation seeds):
- Two simulated profiles succeed on only **0.2%** of writing attempts (spacing_struggler and slow_improver).
- A break lowers frustration by 0.14 on average. Writing cannot do that for those learners, so breaks are chosen at low fatigue as frustration relief.
- spacing_struggler fails partly because the simulator scores spacing in its quality formula. The deployed backend **cannot** measure spacing on its single-stroke exercises and leaves it out.

**Round 3 changes only how the simulator observes and scores attempts, so they match the deployed backend.** Each change is a correction found by the per-feature comparison (`docs/STATE_SCHEMA.md`). None is a reward change:

1. **Quality formula.** The simulator uses the backend's `compute_quality_score` weights (0.45 similarity, 0.20 spacing, 0.20 baseline, 0.15 pressure stability) and the same rule for leaving out unmeasured terms. Previously the weights were 0.40 / 0.25 / 0.20 / 0.15 and every term was always included.
2. **Spacing.** Every backend exercise is single-stroke, so spacing is not observable. The simulator now emits the backend's placeholder 0.5 and leaves spacing out of quality (`spacing_observable=False`).
3. **Pressure.** Each episode has pressure available with probability 0.5, an explicit assumption for a mix of stylus and finger devices. When it is not available, the simulator emits the backend's placeholder 0.3 and leaves pressure out of quality.
4. **Retry timing (backend).** The backend state used retries counted **before** the current attempt, while the simulator counted them **including** it. The backend now uses consecutive failures including the current attempt, which matches the simulator.

Everything else is identical to Round 2: reward v3, the weights, seeds, episodes, evaluation seeds, metrics, CI rule, acceptance criteria and the behavioural gate. `observation_version="v1"` reproduces Round 1 and Round 2 exactly. **The deployed checkpoint becomes the Round 3 seed-0 checkpoint whatever its result, and it goes live only if it passes the gate.** All three rounds are reported. No Round 4 will be run to chase the thresholds.
