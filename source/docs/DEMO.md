# Reproducible demo script

All commands are run from the repository root. Python 3.10–3.13.

## 1. Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r backend/requirements.txt          # includes dysgraphia_rl's dependencies
```

## 2. Run every test suite

```bash
(cd dysgraphia_rl && python -m pytest -q)
(cd backend && python -m pytest -q)
(cd tablet_app && flutter pub get && flutter analyze && flutter test)   # needs the Flutter SDK (NOT run in this environment)
python tablet_app/tool/dart_syntax_check.py tablet_app                # syntax-only fallback check (pip install tree-sitter tree-sitter-language-pack)
```

## 3. Reproduce the Phase A defect on the original code

`reward_version="v1"` together with `max_consecutive_breaks=None` reproduces the original simulator exactly. A test checks this.

```bash
python evidence/phase_a/repro_sim_baselines.py dysgraphia_rl v1   # always-break = 49.25, gain 0.0
python evidence/phase_a/repro_sim_baselines.py dysgraphia_rl v2   # same fixed policies under the current defaults (reward v3, observation v2)
```

## 4. Retrain and evaluate. This follows the predeclared protocol in docs/EVALUATION_PROTOCOL.md.

```bash
cd dysgraphia_rl
python scripts/train_all_seeds.py --episodes 800              # Round 3 settings: 5 seeds; copies seed 0 to checkpoints/double_dqn.pt
python scripts/evaluate_baselines.py                          # 7 policies x 5 runs x 150 episodes, 95% CIs
python scripts/action_breakdown.py --out results/action_breakdown_round3.json   # proposed vs executed counts
# reproduce earlier rounds: --reward-version v2|v3 --observation-version v1 --checkpoints "checkpoints/seeds_round1/double_dqn_seed{k}.pt"
cd ..
```

Outputs: `dysgraphia_rl/results/evaluation_summary.json` (aggregate results, acceptance checks and per-run values), `metrics.csv` and `baseline_comparison.png`.

## 5. Quality-proxy perturbation suite (synthetic)

```bash
(cd backend && python scripts/run_perturbation_suite.py ../evidence/phase_c/perturbation_report.json)
```

## 5b. Simulator vs backend state comparison

```bash
(cd backend && python scripts/compare_state_distributions.py ../evidence/final/state_distribution_after_fixes.json)
```

## 6. Live backend and scenario checks

`backend/scripts/run_phase_e.sh <python> <evidence_dir> [port]` does all of this against a real uvicorn server and includes a restart. The individual steps are:

```bash
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
curl -s localhost:8000/health | python -m json.tool          # shows policy_mode and behavioural-gate result
python scripts/live_scenarios.py --base-url http://127.0.0.1:8000 --out ../evidence/phase_e/live_scenarios.json
```

The scenarios send synthetic traces with `data_source="synthetic_test"`:
- a fresh good writer gets no break;
- fatigue above 0.7 always gets a break;
- high frustration gets support;
- failing never raises the difficulty;
- separate sessions stay separate;
- each transition is linked to the next observed attempt;
- non-pseudonymous IDs and real-child study sessions are refused.

## 7. Tablet app

```bash
cd tablet_app
flutter run --dart-define=BACKEND_URL=http://<host-ip>:8000
```

Walk-through:
1. Home: enter a code such as `P-0042` (a name like "Jane Doe" is rejected) and press **Start Session**.
2. Pick a shape. The canvas fills most of the screen.
3. Trace the shape, then press **Submit**. The feedback card shows the tutor's message and a "Practice score (shape match)", which is labelled as an estimate.
4. Press the action button. A hint shows the start dot and arrows, a demo animates the shape, enlarge redraws the guide larger, and a break opens the break screen. **I'm ready** sends `break_complete`; **Skip break** sends nothing.
5. **Progress** charts only the scores the backend recorded. **Finish** ends the session and shows the attempts and breaks.

## 8. Alphabet A–Z and policy status (Nov 2026 upgrade)

Backend only, without the tablet:

```bash
cd backend
python scripts/e2e_alphabet_live.py ../evidence/alphabet_dqn/e2e_alphabet_live.json   # real uvicorn, all 52 letters
python scripts/render_alphabet_sheet.py /tmp/alphabet.png                              # all 52 guides on one sheet
curl -s localhost:8000/health | python -m json.tool   # policy_mode, dqn_making_decisions, gate result, checkpoint SHA-256
```

On the tablet:
1. On the dashboard, open **Learn & Write the Alphabet** and switch between **Lowercase a–z** and **Uppercase A–Z**.
   - Each tab has 26 tiles in A–Z order.
   - The chips (Straight lines, Round letters, Tall letters, Tail letters, Humps and bumps) narrow the grid.
2. Tap a letter, for example **g**.
   - The page shows the big letter and "Tail letter: goes below the baseline", plus the guide on four-line paper with numbered start dots and arrows.
   - **Show me how** draws the strokes in order.
3. Press **a / A** to switch to the same letter in the other case. **Previous** and **Next** walk A–Z, or the chosen group.
4. Start the session:
   - **Trace capital G** starts a real backend session in tracing mode.
   - **Write it on my own** starts one with the guide hidden.
5. After **Submit**, the feedback card ends with "For the adult: next step chosen by ...". This is the real decision source the backend reported for that attempt.
6. The policy card at the top of the dashboard shows the policy the backend is actually running. Expand it to see the gate result, the reasons and the checkpoint hash.

## What this demo does not show

- No real child data is used.
- No clinical effect is shown.
- The RL results come from a simulator of five hand-authored learner profiles.
