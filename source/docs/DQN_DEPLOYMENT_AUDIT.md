# Double DQN deployment audit (2026-11-08)

> **Result: no checkpoint passes the existing behavioural gate. The DQN is NOT deployed.**
>
> - **What is running:** `/health` reports `policy_mode = "rule_based_fallback"` and `dqn_making_decisions = false`. That is true: the rule-based policy makes every decision, and the safety layer stays on.
> - **What was tried:** 18 existing checkpoints and 15 new ones from a protocol declared before training. All 33 fail.
> - **Kept unchanged:** the gate and its 25 % threshold.
> - **Not produced:** `checkpoints/deployment/double_dqn_deployment.pt`. Brief §10 forbids presenting an invalid one.

## 1. Verified (code and checkpoints)

| Item | Finding |
|---|---|
| Double DQN | **Correct.** `train_step` picks the next action with the online net and evaluates it with the target net: y = r + γ·Q_target(s′, argmax Q_online(s′, ·))·(1 − done). γ = 0.95, target sync every 200 steps, gradient clip 10, Adam, lr 1e-3 |
| Prioritised replay | **Proportional PER.** Priority = (\|TD\| + 1e-3)^α with α = 0.6. Importance weights with β annealed from 0.4 to 1.0. New transitions get the current maximum priority |
| Network | `QNetwork`: Linear 12→128, ReLU, 128→64, ReLU, 64→10. Online and target networks are identical |
| State / actions | 12 features (`utils/state_utils.STATE_FEATURE_NAMES`); 10 actions, in a fixed order checked by a test |
| Preprocessing | The backend builds the state with the same `build_state_vector`. Invalid states (wrong shape, NaN/inf, outside the canonical range) never reach the network; a test checks this |
| Checkpoint path | Resolved against the repository root, then `dysgraphia_rl/`. Path and SHA-256 are reported in `/health` |
| CPU/GPU | `map_location` = the agent device (`RL_DEVICE`, default auto); all tests ran on CPU |
| Deterministic inference | ε = 0 in deployment (`RL_DEPLOYMENT_EPSILON`); greedy argmax; a test checks determinism |
| Deployed file | `checkpoints/double_dqn.pt` (`7b17f6f3…0c43`) is **byte-identical to `checkpoints/seeds/double_dqn_seed0.pt`** (Round 3, seed 0). It is the intended Round-3 deployment candidate, not a stale copy, and it fails the gate |

## 2. All existing checkpoints against the unchanged gate (`evidence/alphabet_dqn/gate_existing_checkpoints.json`)

18 checkpoints were tested: the deployed file, legacy v1, and Rounds 1, 2 and 3 (seeds 0–4 each). **All 18 fail.**

- **Low-fatigue break rate (G4):** 34 %–77 % across the 18 checkpoints, against a limit of 25 %.
- **Most common extra failures:** G6 (DECREASE_DIFFICULTY for a fresh good writer) and G1/G2 (a break for a fresh or low-fatigue writer).
- **Deployed checkpoint:** fails G6 in both forms, and G4 at 34 %.

## 3. Diagnosis

**1. DECREASE_DIFFICULTY is rewarded.** The reward's quality term is `q_t − q_start` on the raw quality proxy. In the simulator, raw quality contains `−0.35 · difficulty / 4`, so making the task easier raises the "improvement" even when no learning happened. This is reward exploitation, and G6 is its symptom.

**2. Breaks pay off at low fatigue.** A break cuts frustration by 0.6 × recovery rate. Each later step is charged −1.5 × frustration, and with γ = 0.95 that relief outweighs the immediate penalty for an unneeded break. The simulator's affect dynamics make the break a frustration-management action, not a fatigue action. G4 is the symptom.

**3. The gate states are partly off the training distribution.** Half of the gate population uses the backend's placeholder values (spacing 0.5, pressure variance 0.3), which training only partly covers (`docs/STATE_SCHEMA.md`: 7 of 12 features still differ).

## 4. Correction attempted (protocol declared before training: `docs/DQN_DEPLOYMENT_CANDIDATE_PROTOCOL.md`, commit `ab8c3a0`)

**The change.** A new opt-in environment option, `improvement_signal="latent"`, credits the change in the difficulty-matched latent skill. It cannot be raised by making the task easier.

- **Defaults unchanged:** all historical checkpoints and results stay reproducible. Seed 0 retrained with defaults gives the same gate behaviour as the shipped checkpoint.
- **Everything else unchanged:** the existing `train()` function, network, PER, observation and break cap. Rung C3 also raises w6 from 1.5 to 2.5; w1 = 3.0 is still the largest weight.

**Results** (`evidence/alphabet_dqn/candidate_ladder_results.json`; checkpoints in `dysgraphia_rl/checkpoints/deployment_candidates/`). The offline columns come from 150 episodes on evaluation seeds 100,000–100,149.

| Rung | Seed | Low-fatigue break rate (limit 0.25) | Failed probes | Offline skill gain | Executed breaks |
|---|---|---|---|---|---|
| C1 latent, 800 ep | 0 | 0.513 | G1, G2, G4 | 0.152 | 0.258 |
| | 1 | **0.203** | G6 | 0.155 | 0.183 |
| | 2 | 0.540 | G1, G4, G6 | 0.153 | 0.286 |
| | 3 | 0.510 | G1, G2, G4, G6 | 0.157 | 0.231 |
| | 4 | 0.403 | G4, G6 | 0.155 | 0.255 |
| C2 latent, 1,600 ep | 0 | 0.653 | G1, G4 | 0.156 | 0.249 |
| | 1 | 0.397 | G4, G6 | 0.161 | 0.207 |
| | 2 | 0.670 | G1, G4 | 0.146 | 0.349 |
| | 3 | 0.510 | G4, G6 | 0.156 | 0.264 |
| | 4 | 0.400 | G1, G4 | 0.152 | 0.262 |
| C3 latent, w6 = 2.5 | 0 | 0.380 | G1, G4, G6 | 0.159 | 0.195 |
| | 1 | 0.430 | G4, G6 | 0.161 | 0.222 |
| | 2 | 0.343 | G4, G6 | 0.164 | 0.201 |
| | 3 | **0.253** | G4, G6 | 0.164 | 0.183 |
| | 4 | 0.483 | G4 | 0.159 | 0.170 |

**What changed.** No candidate passes all gate conditions, and none passes the supplementary held-out check either.

- C1 seed 1 passes G4 but still fails G6.
- C3 seed 3 is closest on G4 (0.253) but fails G6.
- C3 seed 4 is the only candidate with correct actions on all five probe states (G1–G3 and G6), but it requests breaks in 48 % of low-fatigue states.
- Offline, every candidate learns (skill gain 0.15–0.16). The latent signal reduced, but did not remove, the preference for making the task easier.
- The break preference persisted in every rung.

**The ladder stopped at C3, as declared.** Nothing was tuned against the gate after the results were seen.

## 5. Remaining blocker and what would be needed

The blocker is the simulator's affect dynamics together with the per-step frustration penalty. Together they make a break (frustration relief) valuable at low fatigue. Fixing that means changing the learner model or the reward definition, for example:

- frustration relief from a break that depends on fatigue;
- a frustration term that counts only practice steps;
- more training coverage of the backend's placeholder states.

Each of these is a change to the research simulator. It would need a new pre-declared protocol, and the project owner to approve it. It was not done here.

## 6. Deployment safety changes made (all tested: `backend/tests/test_dqn_deployment.py`)

- **The gate cannot be bypassed.** It always runs on a loaded checkpoint, and `policy_mode = "dqn"` requires a passed gate. `RL_BEHAVIORAL_GATE_ENABLED=false` is now only reported (`behavioral_gate_config_requested`).
- **`/health` reports:**
  - checkpoint loaded, path, SHA-256 and optional metadata;
  - architecture and dimensions, device and deployment ε;
  - gate enabled and gate result;
  - gate failures, policy mode, `dqn_making_decisions` and decision maker;
  - `safety_layer_active`.
- **Every attempt response reports** `policy_mode`, `decision_source` (`dqn`, `rule_based_fallback`, `safe_default_invalid_state` or `rule_based_after_inference_error`) and `safety_overridden`.
- **The tablet** shows a DQN badge only when the backend reports both `policy_mode == "dqn"` and `dqn_making_decisions`. It also shows the real decision source on each feedback card.

Everything here is simulated. Nothing in it is evidence about real children.
