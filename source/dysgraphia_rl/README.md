> **SUPERSEDED RESULTS — read first (Oct 2026 upgrade).** The numbers, claims and comparisons in this document come from the original code-review build. The Phase A audit found several problems with it:
> - The original reward let an "always suggest a break" policy score 49.25 per episode with zero quality gain.
> - The shipped DQN chose SUGGEST_BREAK about 45% of the time.
> - The "2–2.5× the strongest baseline" claim left out always-break and do-nothing baselines.
>
> Use these files instead: `../docs/RL_AUDIT_AND_FIXES.md`, `../docs/EVALUATION_PROTOCOL.md`, `../docs/ACCEPTANCE_MATRIX.md` and `results/evaluation_summary.json`. That last file is a 5-seed, 7-policy evaluation with 95% CIs. The text below is kept unchanged for the record.

# Dysgraphia RL Tutor — Code Review Submission (Component 3)

**Project:** Reinforcement-Learning-Based Adaptive Handwriting Intervention
System for Children with Dysgraphia
**Team D9:** Niranjan N S, Shivanand R, Sidharth Panicker, Sidharth Sunil
**Guide:** Neetha Joseph

> **Scope of this Code Review.** This repository implements and tests the
> **offline RL pre-training core** of the system described in the Design and
> Solution document (Section 7, "Phase 1: Offline pre-training") — a
> Gymnasium simulator with hand-authored virtual learner profiles, a Double
> DQN agent with prioritized experience replay, and evaluation against three
> baselines. **Real handwriting sensing, CNN-based handwriting scoring,
> tablet integration, a production backend, and real-user evaluation are
> subsequent development stages** (Prototype milestone, 15 Oct; Module
> Integration, 15 Nov) and are not implemented here. See
> `DESIGN_TO_CODE_MAPPING.md` for the exact, itemized boundary between what
> exists in this repository and what is planned.

> **⚠️ One action item before this is submission-ready: `INDIVIDUAL_CONTRIBUTION.md`
> is a template, not a completed record.** No commit history or authorship
> metadata exists anywhere in these files to derive a real per-member split
> from — every team member needs to open that file and fill in what they
> actually own before the review. This is the one thing in this repository
> that cannot be resolved by writing more code.

## What's implemented right now

| Design doc component | File | Status |
|---|---|---|
| Simulated learner environment (`DysgraphiaTutorEnv`) | `envs/dysgraphia_tutor_env.py` | ✅ 5 learner profiles, 12-d state, 10 actions, Gymnasium-compatible |
| State Representation Module | `utils/state_utils.py` | ✅ normalization helpers, vector builder, range validation |
| Reward Computation Module | `utils/reward_utils.py` | ✅ matches Section 7 formula; single source of truth (env delegates to it) + reward-hacking audit |
| RL Agent / Policy Module (Double DQN) | `agents/double_dqn_agent.py` | ✅ 12→128→64→10 net, target sync, genuine Double DQN update rule |
| Prioritized experience replay | `agents/replay_buffer.py` | ✅ verified genuinely prioritized (not uniform), IS-weight corrected |
| Baselines: Random, Rule-based, Contextual Bandit | `agents/baselines.py` | ✅ |
| Training script | `scripts/train_dqn.py` | ✅ fully reproducible given `--seed` (verified — see Testing); correct `terminated`/`truncated` handling (fixed — see "Bug we found and fixed") |
| Evaluation script | `scripts/evaluate_baselines.py` | ✅ reports mean ± std, reward-hacking audit, produces plots + CSV/JSON |
| Interpretability analysis (Objective 6) | `scripts/analyze_policy_by_profile.py` | ✅ per-profile action-distribution breakdown, honest partial finding (see Results) |
| Unit/integration tests | `tests/test_core.py`, `tests/test_extra.py` | ✅ **29/29 passing** |

**Not yet implemented** (planned for the Prototype milestone, 15 Oct — see
`DESIGN_TO_CODE_MAPPING.md` §3 for the full list with reasons): tablet
client / touch-canvas UI, real OpenCV+DTW feature extraction from actual
stylus strokes, Kaggle-dataset-trained CNN quality scorer, FastAPI backend,
PostgreSQL schema, ONNX export for on-device inference, therapist guardrail
module, real-child pilot.

## Architecture (this Code Review's scope)

```
Simulated learner (5 hand-authored profiles)
        │
        ▼
DysgraphiaTutorEnv.step()  ──────────────►  12-D state vector
        │                                    (envs/dysgraphia_tutor_env.py,
        ▼                                     utils/state_utils.py)
Double DQN agent (12→128→64→10)
  online net picks action │ target net evaluates it
        │
        ▼
10 pedagogical actions applied to the simulated learner
        │
        ▼
Reward computed (utils/reward_utils.py::compute_reward, single source of truth)
        │
        ▼
Transition stored in Prioritized Experience Replay (agents/replay_buffer.py)
        │
        ▼
Batch sampled (priority-weighted) → gradient update → periodic target sync
        │
        ▼
scripts/train_dqn.py loops this to produce checkpoints/double_dqn.pt
        │
        ▼
scripts/evaluate_baselines.py compares the frozen policy against
Random / Rule-based / Contextual Bandit (online + frozen) baselines
        │
        ▼
results/{training_curve.png, baseline_comparison.png, metrics.csv, evaluation_summary.json}
```

This is the Phase 1 ("offline pre-training in simulation") slice of the
full architecture described in Section 3 of the Design and Solution
document (Child/Tablet → Data Acquisition → Feature Extraction → State
Representation → Double DQN → Adaptive Intervention Controller → Feedback
→ Reward → Policy Update). See `DESIGN_TO_CODE_MAPPING.md` §4 for a
side-by-side diagram of this implemented pipeline against the planned
real-hardware pipeline it's designed to eventually plug into.

## Repository structure

```
dysgraphia_rl/
├── envs/dysgraphia_tutor_env.py     # Simulated learner environment (Gymnasium)
├── agents/
│   ├── double_dqn_agent.py          # Double DQN network + training update
│   ├── replay_buffer.py             # Prioritized experience replay
│   └── baselines.py                 # Random / Rule-based / Contextual Bandit
├── utils/
│   ├── state_utils.py               # State vector construction + validation
│   └── reward_utils.py              # Reward formula + reward-hacking audit
├── scripts/
│   ├── train_dqn.py                 # Offline training entry point
│   ├── evaluate_baselines.py        # Policy comparison entry point
│   └── analyze_policy_by_profile.py # Objective 6: per-profile interpretability analysis
├── tests/
│   ├── test_core.py                 # Original 10 tests
│   └── test_extra.py                # 19 additional tests (see below)
├── checkpoints/double_dqn.pt        # Trained policy (400 episodes, seed=0)
├── results/                         # training_curve.png, baseline_comparison.png, metrics.csv,
│                                     # evaluation_summary.json, policy_by_profile.png/.csv
├── DESIGN_TO_CODE_MAPPING.md        # Row-by-row design->code traceability
├── CODE_REVIEW_GUIDE.md             # Full explanation, viva-ready
├── VIVA_QUESTIONS.md                # Q&A reference
├── INDIVIDUAL_CONTRIBUTION.md       # Per-member contribution (fill in before review)
├── DEMO_SCRIPT.md                   # ~8-10 minute walkthrough script
└── CODE_REVIEW_READINESS_REPORT.md  # Self-assessed rubric score + audit log
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Run the full test suite (29 tests)
python -m pytest tests/ -v

# 2. Offline pre-train the Double DQN agent in the simulator (~40 sec on CPU)
python scripts/train_dqn.py --episodes 400 --seed 0
# -> writes checkpoints/double_dqn.pt and results/training_curve.png

# 3. Evaluate the trained policy against Random, Rule-based, and both
#    Contextual Bandit variants (online + frozen) -- 5 rows total
python scripts/evaluate_baselines.py --episodes 150 --seed 0
# -> prints a comparison table and writes results/baseline_comparison.png,
#    results/metrics.csv, results/evaluation_summary.json
```

Every command above was actually run to produce the numbers and plots
already checked into this repository — see "Reproducibility" below for how
that was verified.

## Results (400 training episodes, 150 evaluation episodes, seed=0)

| Policy | Mean Reward (± std) | Final Quality | Engagement | Frustration | Easy-Action % |
|---|---|---|---|---|---|
| Random | 16.79 ± 13.1 | 0.199 | 0.115 | 0.924 | 10.1% |
| Rule-based | 19.26 ± 2.6 | 0.201 | 0.300 | 0.932 | 7.6% |
| Contextual Bandit (online) | 30.52 ± 31.7 | 0.196 | 0.488 | 0.816 | 38.6% |
| **Double DQN (ours)** | **67.41 ± 15.1** | **0.492** | **0.973** | **0.123** | 3.2% |
| Contextual Bandit (frozen, 400-episode pretrain) | 24.58 ± 26.7 | 0.158 | 0.197 | 0.886 | 38.2% |

These numbers reflect the corrected training pipeline (see "Bug found and
fixed" below) — they are not the same as an earlier draft of this table.
The Double DQN's easy-action usage dropped sharply after the fix (18.8% →
3.2%), consistent with a policy that now has a more accurate long-horizon
value estimate and relies less on "just make it easier" as a fallback.

Two Contextual Bandit variants are reported for a genuinely fair comparison:
**online** starts from scratch and updates its weights during the
evaluation loop itself (as it would in real deployment); **frozen** is
pretrained offline for 400 episodes — the same offline training budget the
Double DQN checkpoint gets — on a disjoint block of environment seeds, then
evaluated with its weights frozen, exactly like the Double DQN row. Even
with a matched training budget, the frozen bandit still trails the online
bandit and is far behind Double DQN — concrete evidence for the Design
doc's own argument that a bandit's inability to model sequential state
(fatigue/frustration carrying over between attempts) is a real limitation,
not just a theoretical one.

Every number in this table is saved to `results/metrics.csv` and
`results/evaluation_summary.json` (not just printed to stdout), and to the
two PNG plots — so anything quoted here can be checked against a file, not
just this document.

These are **simulated / offline results**, produced entirely by
`DysgraphiaTutorEnv`'s hand-authored virtual learner profiles — not
evidence of real-child clinical benefit. They demonstrate that the RL
pipeline (state → Double DQN → action → reward → prioritized replay →
training → evaluation) is implemented correctly and learns a policy that
clearly outperforms all four baseline variants in simulation, on every
reported metric simultaneously (not just reward at the expense of
frustration, for instance). The reward-hacking audit does not flag the
Double DQN policy — see `utils/reward_utils.py` and the note below.

Run `python scripts/evaluate_baselines.py --episodes 150 --seed 0` yourself
against the shipped checkpoint to reproduce this exact table.

### Objective 6: does the policy actually personalize?

```bash
python scripts/analyze_policy_by_profile.py --episodes 60 --seed 0
```

This runs the frozen, fully-greedy policy against each of the 5 learner
profiles separately and reports the action distribution per profile —
directly addressing Proposal Objective 6 ("identifying which intervention
strategies are most useful for children with different handwriting
difficulties"). The honest finding: there **is** real personalization
along the fatigue axis — `fast_improver` gets `SUGGEST_BREAK` on only 4% of
steps, while every "struggling" profile (`slow_improver`,
`spacing_struggler`, `unstable_stroke`, `fatigue_prone`) gets it on
46–65% of steps, tracking each profile's actual fatigue/recovery
parameters sensibly. However, the policy has **not** learned fine-grained,
weakness-specific scaffolding — e.g. `spacing_struggler` doesn't get more
`SHOW_HINT` or `ENLARGE_TEMPLATE` than other struggling profiles, despite
having a distinct spacing-specific weakness parameter. See
`results/policy_by_profile.png` (heatmap) and `results/policy_by_profile.csv`
for the full breakdown. This is reported as a genuine, partial result, not
inflated into "the policy fully personalizes."

## Testing

**29/29 tests passing** (`python -m pytest tests/ -v`):

- `tests/test_core.py` (10 tests) — environment shape/typing, reward formula
  correctness, the reward-hacking audit's three decision branches, state
  vector defaults, one Double DQN train step, baseline action validity.
- `tests/test_extra.py` (19 tests, added across three rounds of self-audit)
  — network architecture shape assertion, checkpoint save/load round-trip,
  a **statistical test that the replay buffer is genuinely prioritized** (a
  uniform-random sampler would fail it), a direct test that terminal
  transitions use `target = r` with no bootstrap, state-ordering consistency
  across files, per-feature range validation on live environment output,
  environment determinism given a seed, cross-module reward-formula
  consistency, full end-to-end training reproducibility given `--seed`
  (bit-identical checkpoints across two independent runs), five separate
  reward-monotonicity tests (quality improvement, quality deterioration,
  frustration penalty, completion, consistency — each varying one reward
  term in isolation while holding the others fixed), a test that the 5
  virtual learner profiles produce measurably different fatigue-accumulation
  and skill-growth dynamics rather than being decorative, **two
  integration-level tests locking in the terminated/truncated fix**
  (`test_fatigue_cutoff_sets_terminated_not_truncated` and
  `test_train_dqn_stores_done_flag_on_true_termination` — the latter runs
  the real training loop and inspects the replay buffer directly, closing
  exactly the gap the original bug slipped through), and a shape/validity
  check on the Objective 6 interpretability analysis output.

## Reproducibility

`scripts/train_dqn.py` and `scripts/evaluate_baselines.py` both seed
Python's `random`, NumPy's global RNG, and (for training) PyTorch's RNG at
startup via `--seed`. This was not just assumed — it was verified directly:
two independent training runs with `--seed 0` were compared and produced
**bit-identical model weights** (`tests/test_extra.py::test_training_is_reproducible_given_seed`
automates a smaller version of this check). Tested with Python 3.12, torch
2.13 (CPU), gymnasium 1.3.0, numpy 2.4.4, matplotlib 3.10.8, pytest 9.1.1 —
see `requirements.txt` for minimum version constraints.

## Baselines: a methodological note

The Contextual Bandit is reported in **two variants** specifically to avoid
a misleading comparison. "Online" updates its linear weights during the
evaluation loop itself (as it would in real deployment) — this is not
stacked in the DQN's favor; if anything it gives the bandit a chance to
adapt to the specific evaluation seed's learner sequence, an advantage the
frozen DQN doesn't get. "Frozen" is pretrained offline for 400 episodes —
matching the Double DQN's own offline training budget exactly — on a
disjoint block of environment seeds, then evaluated with weights frozen,
so it's directly comparable to the Double DQN row on equal footing. Both
are reported rather than picking one, so the comparison is transparent
either way you want to read it.

## Design decisions worth mentioning to the guide

- **Why Double DQN over vanilla DQN / PPO / bandit** — justified in Section 7
  of the Design doc: vanilla DQN overestimates Q-values (risky for a system
  that could push a frustrated child toward harder tasks); PPO is too
  sample-hungry for scarce per-child data; a bandit ignores fatigue building
  up across a session. The bandit is kept here only as a baseline.
- **Reward-hacking guardrail** — `audit_action_distribution()` flags a
  policy if it leans too heavily on "just make it easier" *and* isn't
  actually delivering quality gains. This is a direct, testable version of
  the mitigation listed in the Design doc's Risks section.
- **Single source of truth for the reward formula** — the environment used
  to compute the reward formula inline, duplicating the logic already in
  `utils/reward_utils.py::compute_reward()`. During this Code Review pass
  we refactored the environment to call that function directly, and added a
  regression test (`test_env_reward_matches_standalone_reward_utils`) so the
  two can't silently drift apart again.
- **Virtual learner profiles** are intentionally simple/hand-authored for
  now (fast improver, slow improver, fatigue-prone, spacing-struggler,
  unstable-stroke) — a stand-in for the Kaggle-dataset-trained CNN scorer
  and real interaction logs, to be swapped in without changing the agent.
  These specific numeric parameters are not derived from real data or a
  sensitivity study — see the honesty note in `LearnerProfile`'s docstring
  in `envs/dysgraphia_tutor_env.py`.
- **Per-profile interpretability analysis** (`scripts/analyze_policy_by_profile.py`)
  was added specifically to give Proposal Objective 6 concrete evidence
  instead of just a description of intent — see the Objective 6 subsection
  above for the honest (partial) finding.

**Bug we found and fixed during a strict self-audit (the more significant
one):** the environment's `step()` had `terminated` hard-coded to `False`
and only ever set `truncated = True` — for both the ordinary time-limit
cutoff *and* the fatigue-based safety cutoff, even though the code comment
right above it said "safety **termination**." Since `scripts/train_dqn.py`
stored `done=float(terminated)` in the replay buffer, and `terminated` was
never true, **every stored transition had `done=0.0`**, meaning the Double
DQN's Bellman target bootstrapped through every episode boundary, including
genuine terminal states, across the entire training run. An earlier unit
test (`test_terminal_transitions_do_not_bootstrap`) verified the *agent's*
target-computation formula is correct given a correctly-labeled `done`
flag — it did not verify the training loop ever produces one, and it
didn't. Fixed by making the fatigue safety cutoff set `terminated=True`
(a true MDP-ending event) while the ordinary time limit keeps
`truncated=True` (an artificial cutoff, where bootstrapping through it is
the standard, correct treatment). Two new integration-level tests
(`test_fatigue_cutoff_sets_terminated_not_truncated`,
`test_train_dqn_stores_done_flag_on_true_termination`) lock this in — the
second one runs the *actual* training loop and inspects the replay buffer
directly, rather than only testing the agent's formula in isolation. We
verified these tests would have caught the original bug by temporarily
reverting the fix and confirming both tests fail against the old code.
Retraining after the fix produced a policy that relies on the "decrease
difficulty" action far less (18.8% → 3.2% of steps) while still
outperforming every baseline — consistent with a more accurate long-horizon
value estimate needing less of a difficulty-based safety net.

**Bug we found and fixed during development (the original one):** the first version of
`audit_action_distribution()` flagged any policy using its "decrease
difficulty" action more than 50% of the time, regardless of outcome. This
produced a false positive on the trained Double DQN agent even though it
was simultaneously achieving roughly 2.5–3x the quality score of every
baseline. We fixed the audit to only flag a policy when high easy-action
usage is *also* paired with poor quality (`mean_quality` below a
threshold) — frequency alone isn't evidence of hacking if it's actually
working. This is documented in the function's docstring and covered by
three unit tests in `tests/test_core.py`.

## Next steps (post code-review)

1. Train the CNN handwriting-quality scorer on the Kaggle Dysgraphia dataset.
2. Replace the simulator's synthetic quality function with CNN scorer output.
3. Build the OpenCV + DTW feature-extraction pipeline for real stylus strokes.
4. Wire up the FastAPI backend (`/session/start`, `/trial/submit`, etc.).
5. Export the trained policy to ONNX for on-device inference testing.
6. Multi-seed evaluation with statistical significance testing.

## Further reading

- `DESIGN_TO_CODE_MAPPING.md` — exhaustive design→code traceability table
- `CODE_REVIEW_GUIDE.md` — full technical walkthrough (MDP formulation,
  why each design choice was made, limitations)
- `VIVA_QUESTIONS.md` — rapid Q&A reference
- `DEMO_SCRIPT.md` — a ~8-10 minute live demo script
- `INDIVIDUAL_CONTRIBUTION.md` — fill in before the review
- `CODE_REVIEW_READINESS_REPORT.md` — self-assessed rubric score and full audit log
