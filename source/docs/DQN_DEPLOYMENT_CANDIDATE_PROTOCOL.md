# DQN deployment-candidate protocol

This protocol was declared **before** any candidate was trained, on 2026-11-08, on branch `alphabet-dqn-upgrade`.

## Why it exists

All 18 existing checkpoints fail the unchanged behavioural gate (`evidence/alphabet_dqn/gate_existing_checkpoints.json`). The deployed `double_dqn.pt` is byte-identical to `seeds/double_dqn_seed0.pt` (Round 3, seed 0). It fails in two ways:

- **G6:** it chooses DECREASE_DIFFICULTY for the fresh good writer, both the plain and the deployed form.
- **G4:** it requests a break in 34 % of low-fatigue states, against a limit of 25 %.

## Diagnosis (from the code, before training)

### Cause 1: DECREASE_DIFFICULTY is rewarded

The reward's quality term is `q_t − q_start` on the raw quality proxy. In the simulator the raw quality includes `−0.35·difficulty/4`, a tracing bonus of +0.08, and fatigue and frustration penalties (`_simulate_trial`). Lowering the difficulty therefore raises q, and the reward books that as "improvement".

This is reward exploitation. It matches G6: a fresh, clean writer is made easier because easier means higher measured quality.

### Cause 2: breaks pay off at low fatigue

A break step earns only `−w6·unnecessary_break` (reward v3). It also cuts frustration by `0.6·recovery_rate`, which raises the reward of later steps.

For learners whose practice steps score low, a break is cheap and buys frustration relief. The earlier diagnosis (`evidence/final/break_root_cause_diagnostics.json`) found two profiles that succeed on only 0.2 % of attempts.

## Correction tried: a new reward signal, opt-in

The environment gets a new option, `improvement_signal="latent"`.

- **How it works:** the quality term becomes the change in `latent_skill()`. That is the existing difficulty-matched, noise-free skill measure, previously used only for evaluation reporting.
- **Why it should help:** it grows only with practice, so making the task easier is not rewarded and a break adds no learning.
- **Historical results are unaffected:** the default stays `"observed"`, so every historical checkpoint and result is still reproducible.
- **Nothing else changes:** network, optimiser, replay, observation (Round 3, v2), break cap and reward weights are all the same.

## Candidate ladder (fixed in advance; each rung runs only if no seed of the previous rung qualifies)

| Rung | Change against the Round-3 defaults | Seeds |
|---|---|---|
| C1 | `improvement_signal="latent"`, 800 episodes | 0–4 |
| C2 | C1 with 1,600 episodes | 0–4 |
| C3 | C1 with `w6_unnecessary_break = 2.5` (w1 = 3.0 stays the largest weight) | 0–4 |

No further rungs. If no candidate qualifies, the rule-based fallback stays live and the blocker is reported.

## Qualification (all required)

1. **The existing behavioural gate:** unchanged code, unchanged thresholds, every probe passes.
2. **Supplementary held-out check (stricter, added here, never used to relax the gate):**
   - the same low-fatigue generator with 5 unseen seeds (2001–2005) must give a break rate below 0.25 in each;
   - G1–G3 must also hold under 50 small jitters of each probe state, at most ±0.03 per feature, seed 77.
   - This guards against a checkpoint that passes only the gate's fixed probes.
3. **Offline evaluation:**
   - setup: Round-3 environment, 150 episodes on evaluation seeds 100,000–100,149;
   - difficulty-matched skill gain must be greater than 0;
   - share of executed breaks must be below 0.25.

## Selection among qualifying candidates

1. Lowest rung first.
2. Then the highest offline difficulty-matched skill gain.
3. Then the lowest seed.

The selected file is copied to `dysgraphia_rl/checkpoints/deployment/double_dqn_deployment.pt`. No historical checkpoint is touched.

## Limitation acknowledged in advance

The gate's probe states are known before training. The supplementary check (2) and the offline evaluation (3) reduce, but cannot remove, the risk of selecting for the probes.

Everything here is simulated. It is not evidence about real children.
