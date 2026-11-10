# Dysgraphia Tutor — RL-Based Adaptive Handwriting Intervention System

**Project:** Reinforcement-Learning-Based Adaptive Handwriting Intervention
System for Children with Dysgraphia

This repository contains the full pipeline:

```
Tablet handwriting input (Flutter)
        │  REST (JSON)
        ▼
FastAPI backend (backend/)
        │
        ▼
Preprocessing  →  Static + Dynamic feature extraction  →  12-D state
        │
        ▼
Double DQN policy (dysgraphia_rl/)  →  Safety layer  →  Intervention controller
        │
        ▼
JSON response  →  Tablet renders feedback / next exercise
```

Three parts, each independently developed and independently testable:

| Part | Path | What it is |
|---|---|---|
| Offline RL pre-training core | [`dysgraphia_rl/`](dysgraphia_rl/README.md) | Gymnasium simulator, Double DQN agent + prioritized replay, baselines, training/eval scripts. **Changed in the Oct 2026 upgrade** (reward v2/v3, break cap, new baselines, generalized audit, 5-seed evaluation). See below. Its own README is kept for the record and is marked superseded. |
| Backend (this build's main contribution) | [`backend/`](backend/) | FastAPI app: raw-stroke preprocessing, static (DTW/shape) + dynamic (kinematics) feature extraction, `StateBuilder`, RL policy integration, safety layer, intervention controller, REST API. |
| Tablet client (this build's main contribution) | [`tablet_app/`](tablet_app/) | Flutter app: handwriting canvas (x/y/t/pressure/stroke-id capture), exercise display, session controls, feedback rendering. **No ML/RL/feature-extraction logic** — it only captures input and renders the backend's response. |

## Alphabet A–Z and DQN deployment upgrade (Nov 2026)

| Area | Status |
|---|---|
| **52 letters** (A–Z, a–z) | **Done (backend).** Each letter has its own exercise id, multi-stroke guide, four-line writing lines (with a descender line), instruction and session path. A difficulty step stays within shapes, words, capitals or small letters. All 52 pass the font-match test and a live run against the real server (`docs/ALPHABET_UPGRADE.md`) |
| **Tablet alphabet dashboard** | **Written, not compiled.** The "Learn & Write the Alphabet" section has lowercase and uppercase tabs with 26 tiles each, practice groups, a letter page (stroke order, numbers, arrows, "show me how", trace or write on my own, a small ↔ capital switch, previous/next) and a supervisor policy card. The Flutter SDK could not be installed here, so `flutter analyze` and `flutter test` were **not run**; only a syntax check passed |
| **DQN deployment** | **NOT deployed.** None of the 18 existing checkpoints and none of the 15 newly trained candidates (from a protocol declared before training) passes the unchanged behavioural gate. The rule-based fallback makes the decisions, `/health` says so, and the safety layer is always on (`docs/DQN_DEPLOYMENT_AUDIT.md`) |
| **Gate reporting** | **Done.** The gate cannot be bypassed. `/health` reports the checkpoint SHA-256, architecture and gate result. Every attempt reports its real decision source |

## Upgrade status (Oct 2026). Read this before quoting any number

The audit, the fixes and the results are documented in:

| Document | What it covers |
|---|---|
| [`docs/ACCEPTANCE_MATRIX.md`](docs/ACCEPTANCE_MATRIX.md) | Every requirement as PASS / FAIL / BLOCKED, with evidence, plus the readiness classification |
| [`docs/RL_AUDIT_AND_FIXES.md`](docs/RL_AUDIT_AND_FIXES.md) | The reproduced defects, the fixes, the 7-policy × 5-seed comparison for both rounds, and what do-nothing scores |
| [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) | The evaluation protocol, written before training, including the Round 2 amendment |
| [`docs/QUALITY_SCORE.md`](docs/QUALITY_SCORE.md) | What the quality proxy is, the perturbation and discrimination results, and the fixes |
| [`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) | Provenance labels, pseudonymous IDs, real-child collection (disabled), retention, and the Kaggle assessment |
| [`docs/DEMO.md`](docs/DEMO.md) | Reproducible commands and a demo walk-through |
| [`docs/STATE_SCHEMA.md`](docs/STATE_SCHEMA.md) | Simulator vs backend, feature by feature |
| [`docs/TASK_LEDGER.md`](docs/TASK_LEDGER.md) | Task ledger for the final pass |
| [`evidence/`](evidence/) | The raw outputs behind every number |

In short (final-submission pass, 2026-10-29). Full detail is in [`docs/ACCEPTANCE_MATRIX.md`](docs/ACCEPTANCE_MATRIX.md); the readiness classification is **NOT READY**, because the Flutter app could not be compiled, tested or screenshotted in this environment.
* The original reward could be gamed: always-break scored 49.25 per episode with zero quality gain. This is fixed and tested.
* Three predeclared retraining rounds are all reported. Round 3 fixed the simulator/backend observation inconsistencies and is the first to meet the offline criteria: executed breaks 23.3%, quality gain 0.191 [0.168, 0.214], no audit flag. Even so, 99% of its breaks come at low fatigue, its quality gain mostly reflects lowered difficulty, and it **fails the behavioural gate**. **The live backend therefore runs the transparent rule-based fallback.** `/health` reports `policy_mode` and the reasons, and the safety layer applies in every case.
* Quality-proxy perturbation targets: 4 of 5 met. Slant is ρ 0.778 (FAIL, investigated in `docs/QUALITY_SCORE.md`).
* Simulator vs backend state: 3 encoding inconsistencies fixed, 7 of 12 features still differ (`docs/STATE_SCHEMA.md`).
* Tests (fresh clone): backend 119 passed + 2 disclosed strict xfails; `dysgraphia_rl` 45 passed.
* Tablet app: redesigned, and parses cleanly at the syntax level. It is **not compiled or run**, because the Flutter SDK host is blocked. No screenshots exist.

## Honesty notes (please read before quoting any number from this repo)

- The 12-D state, Double DQN, reward formula, and baseline comparison
  numbers in `dysgraphia_rl/README.md` come from a **simulated** environment
  with 5 hand-authored virtual learner profiles — **not real children**, and
  not presented as such anywhere in that module.
- The backend's `quality_score` (`backend/app/state/state_builder.py::compute_quality_score`)
  is an **engineering proxy** computed from measured static features
  (stroke similarity, spacing, baseline alignment, pressure stability) — it
  is explicitly **not** the CNN-based, dataset-trained handwriting-quality
  scorer described as future work. Nothing in this repo claims clinical
  validity or diagnostic accuracy.
- `fatigue_proxy`, `engagement`, and `frustration` are heuristic proxies
  built from measurable session behaviour (retries, pauses, quality
  deltas) — there is no sensor that measures a child's affect directly.
  See `backend/app/config.py::SessionDynamicsConfig` for the exact rules.
- The exercise reference templates in `backend/app/exercises.py` (line,
  wave, loop, figure-eight, zigzag: one per difficulty level 0-4) are **synthetic placeholder shapes** used to exercise the
  DTW similarity pipeline end-to-end. They are not derived from the Kaggle
  dysgraphia dataset and are not clinical handwriting content. See
  "Dataset setup" below for how a real template/letter set plugs in.
- All feature-extraction tests validate **engineering properties**
  (repeatability, noise/scale/translation/resampling robustness, numerical
  stability, sensitivity to intended changes) — none claim clinical
  accuracy, because there is no clinical ground truth available in this
  environment to validate against.

## What was actually run and verified in the ORIGINAL build (historical; see "Upgrade status" for current results)

- **`backend/`**: dependencies installed into a real virtualenv, the FastAPI
  app imported and served in-process, a full `/session/start` →
  `/attempt/submit` round trip executed with synthetic stroke data, and the
  **full pytest suite (72 tests) run and passing**, covering preprocessing,
  static features, dynamic features, the state builder, the safety layer,
  the intervention controller, the RL policy integration, the REST API
  (via `TestClient`), and two end-to-end pipeline tests.
- **`dysgraphia_rl/`**: its own pytest suite (29 tests) run and passing
  after two Windows-portability fixes (see table above); its shipped
  checkpoint loads and produces valid, deterministic greedy actions.
- **`tablet_app/`**: written as complete Flutter/Dart source, but **not
  built, run, or compiled** — there is no Flutter SDK in the environment
  this was built in. It has not been visually verified. See "Known
  limitations" below. A `flutter analyze` / `flutter test` pass by you
  after `flutter pub get` is the first thing to do with it.

## Repository layout

```
dysgraphia_tutor/
├── dysgraphia_rl/            # Offline RL core (own README/tests/checkpoint/results)
├── backend/                  # FastAPI backend (this build)
│   ├── app/
│   │   ├── config.py                     # ALL tunable parameters, in one place
│   │   ├── preprocessing/stroke_preprocessing.py
│   │   ├── features/static_features.py   # "what" was written (DTW/shape/geometry)
│   │   ├── features/dynamic_features.py  # "how" it was written (kinematics/timing)
│   │   ├── state/state_builder.py        # features + session -> 12-D state
│   │   ├── state/session_context.py      # per-session rolling state
│   │   ├── rl_integration/policy_service.py  # the ONLY file touching the RL agent
│   │   ├── dysgraphia_rl_bridge.py       # wires into dysgraphia_rl/ without modifying it
│   │   ├── safety/safety_layer.py        # policy -> safety-checked action
│   │   ├── interventions/intervention_controller.py  # action -> tablet instruction
│   │   ├── exercises.py                  # exercise/reference-template registry
│   │   ├── storage.py                    # session store + durable attempt/reward log
│   │   ├── schemas.py                    # all REST request/response models
│   │   ├── api/routes.py                 # the REST surface
│   │   └── main.py                       # FastAPI app entry point
│   ├── tests/                # 72 tests (unit + integration + API + e2e)
│   ├── requirements.txt
│   └── pytest.ini
├── tablet_app/                # Flutter client (this build)
│   ├── lib/
│   │   ├── models/            # StrokePoint, Exercise, AttemptResult (wire-format mirrors)
│   │   ├── services/api_service.dart      # the ONLY file that calls the backend
│   │   ├── widgets/            # HandwritingCanvas, ExerciseDisplay, SessionControls, FeedbackOverlay
│   │   └── screens/            # SessionScreen, WritingScreen
│   ├── test/widget_test.dart  # written, not executed (no Flutter SDK here)
│   └── pubspec.yaml
├── README.md                  # this file
└── .gitignore
```

## Prerequisites

- Python 3.10–3.12
- Flutter SDK 3.13+ (Dart 3+) — only needed to run/build the tablet app
- No GPU required (CPU-only PyTorch is sufficient for both training and inference at this scale)

## Installation

Nothing is vendored in this ZIP — install everything yourself:

```bash
# Backend (includes dysgraphia_rl's own dependencies, since the backend
# imports it directly — see backend/app/dysgraphia_rl_bridge.py)
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

```bash
# Tablet app
cd tablet_app
flutter pub get
```

If you want to run `dysgraphia_rl/`'s own scripts/tests standalone (its
own README documents this in more detail), its `requirements.txt` is a
subset of the backend's, so the same venv works:

```bash
cd dysgraphia_rl
pip install -r requirements.txt   # already satisfied if you installed backend/requirements.txt
```

## Backend setup & running

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/docs` for interactive Swagger docs, or
`GET http://127.0.0.1:8000/health` to confirm the RL checkpoint loaded.

Key endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/session/start` | Start a session for a child, get the first exercise |
| POST | `/attempt/submit` | Submit one handwriting attempt (the main pipeline call) |
| POST | `/session/{id}/break_complete` | Tell the backend a suggested break was taken |
| GET  | `/session/{id}/summary` | Current session state (fatigue/engagement/history/etc) |
| POST | `/session/{id}/end` | End a session |
| GET  | `/exercises`, `/exercise/{id}` | List/inspect exercises |
| GET  | `/health` | Liveness + whether the RL checkpoint loaded |

Attempt/session state is process-local (in memory); every submitted
attempt's full feature/state/action/reward record is durably logged to a
local SQLite file at `backend/data/dysgraphia_tutor.db` (created on first
run) for later analysis — see `backend/app/storage.py`.

## Flutter (tablet) setup & running

```bash
cd tablet_app
flutter pub get
flutter run --dart-define=BACKEND_URL=http://127.0.0.1:8000
```

Backend URL notes (see `lib/main.dart`):
- Android emulator → host machine: `http://10.0.2.2:8000`
- iOS simulator / desktop / web on the same machine: `http://127.0.0.1:8000`
- Physical tablet on the same network: your host machine's LAN IP

**This app was written but not compiled/run/visually verified in this
build** (no Flutter SDK available in the environment this was produced
in). Run `flutter analyze` and `flutter test` first; fix whatever surfaces
before trusting it against a real device.

## Dataset setup

No dataset is bundled. Two independent kinds of data feed this system,
and they are **not interchangeable**:

1. **Image-only handwriting datasets** (e.g. a Kaggle dysgraphia dataset)
   contain shape/appearance information only — no pressure, no timing, no
   stroke order. Useful for training a CNN quality scorer (see "Next
   steps"), but cannot supply dynamic (kinematic) features.
2. **Real tablet telemetry** (x, y, t, pressure, stroke id) is what
   `backend/app/features/dynamic_features.py` actually needs, and only a
   real or realistically-mocked tablet session can supply it (see
   `backend/tests/test_e2e_pipeline.py::_mock_tablet_attempt` for what
   "realistic mock" means here).

To wire in a real exercise/template set: edit `backend/app/exercises.py`
(`EXERCISES` dict) — nothing else needs to change, since `StateBuilder`,
the RL agent, and the UI never see exercise definitions directly.

## Model setup

The Double DQN checkpoint used by the backend is
`dysgraphia_rl/checkpoints/double_dqn.pt`, already trained and shipped
(see `dysgraphia_rl/README.md` for how/why). `backend/app/config.py::RLConfig`
points at it by relative path; override `RL_CHECKPOINT_PATH` to point at a
different one.

## How to train

```bash
cd dysgraphia_rl
python scripts/train_dqn.py --episodes 400 --seed 0
# -> writes checkpoints/double_dqn.pt and results/training_curve.png
python scripts/evaluate_baselines.py --episodes 150 --seed 0
```

See `dysgraphia_rl/README.md` for full detail (results table, baselines,
reproducibility, the two real bugs found and fixed during that module's
own development).

## How the complete pipeline works (attempt lifecycle)

1. Tablet captures raw strokes (`HandwritingCanvas`) and calls
   `POST /attempt/submit` with a flat, time-ordered list of points.
2. `api/deps.py::points_to_strokes` regroups points by `stroke_id`.
3. `preprocessing/stroke_preprocessing.py` cleans each stroke (outlier
   clipping, timestamp de-duplication).
4. `features/static_features.py` computes DTW-based shape similarity to
   the exercise's reference template, spacing, baseline alignment, etc.
5. `features/dynamic_features.py` computes speed, smoothness, pauses,
   pressure stats, retry count, etc, independently of (4).
6. `state/state_builder.py` combines (4) + (5) + the session's rolling
   history (fatigue/frustration/engagement/quality history) into the
   canonical 12-D state vector, via `dysgraphia_rl`'s own
   `build_state_vector` (single source of truth, shared with the
   simulator/training pipeline).
7. `rl_integration/policy_service.py` runs the trained Double DQN's
   online network greedily (`epsilon=0` by default) to pick one of 10
   actions.
8. `safety/safety_layer.py` checks the proposed action against session
   history and safety limits, possibly overriding it.
9. `interventions/intervention_controller.py` turns the final action into
   a concrete instruction (difficulty delta, hint, break, etc).
10. Every stage's output, plus the reward and its individual components,
    is logged to SQLite (`storage.py`) and returned to the tablet as one
    JSON response.

## How to modify/tune the static feature extractor

Edit `backend/app/config.py::StaticFeatureConfig` (DTW band width,
distance→similarity scale, expected spacing/tolerance, baseline-fit
fraction) or `backend/app/features/static_features.py` directly. Nothing
else needs to change — the dynamic extractor, StateBuilder, RL agent, and
UI don't know how static features are computed, only their names/ranges.

## How to modify/tune the dynamic feature extractor

Edit `backend/app/config.py::DynamicFeatureConfig` (pause threshold,
minimum reliable sampling rate, expected speed baseline) or
`backend/app/features/dynamic_features.py` directly. Same independence
guarantee as above.

## How to modify/tune RL hyperparameters

`dysgraphia_rl/agents/double_dqn_agent.py::DoubleDQNAgent.__init__` takes
`lr`, `gamma`, `buffer_capacity`, `target_sync_every` as constructor
arguments; `scripts/train_dqn.py --help` exposes the training-time knobs
(episodes, epsilon schedule, warmup steps, seed). Reward weights are
tunable at the backend layer without retraining via
`backend/app/config.py::RewardConfigOverride` (falls back to
`dysgraphia_rl`'s own defaults when left `None`), and independently at the
simulator/training layer via `dysgraphia_rl`'s own `RewardWeights`.

## How to tune the safety layer

`backend/app/config.py::SafetyConfig` — escalation limits, repetition
limits, fatigue/frustration thresholds, forced-break attempt cap. All
overrides are also available as environment variables (`SAFETY_*`, see
each config class's `model_config`).

## Testing

```bash
cd backend
source .venv/bin/activate
pytest -q            # 119 passed + 2 strict xfail (disclosed in docs/ACCEPTANCE_MATRIX.md)

cd ../dysgraphia_rl
pytest -q            # 45 passed (29 original + 16 reward-v2/v3, break-cap and audit tests)
```

```bash
cd tablet_app
flutter test          # written, not run in this build -- run it yourself
```

Backend test files, by concern:

| File | Covers |
|---|---|
| `test_preprocessing.py` | resampling/scale/translation robustness, outlier clipping |
| `test_static_features.py` | repeatability, noise/scale/translation/resampling robustness, numerical stability, sensitivity to intended shape changes |
| `test_dynamic_features.py` | speed/pause/pressure/retry correctness, missing-pressure handling, low-sample-rate acceleration guard |
| `test_state_builder.py` | canonical 12-D ordering, history accumulation, fatigue/frustration dynamics |
| `test_safety.py` | escalation blocking, repetition blocking, forced breaks, override transparency |
| `test_intervention_controller.py` | every action maps to a valid, bounded intervention |
| `test_rl_integration.py` | checkpoint loads, deterministic greedy action, argmax consistency |
| `test_api.py` | full REST surface via `TestClient`, error handling |
| `test_e2e_pipeline.py` | raw mocked stylus input → ... → intervention, single- and multi-attempt |
| `test_safety_rules_v2.py` | fatigue ≥ 0.7 forced break, break cap, high-frustration support, no escalation after failure |
| `test_behavioral_gate.py` | gate rejects the original checkpoint, always-break and NaN policies; missing or corrupt checkpoint → fallback |
| `test_live_scenarios.py` | scenarios S1–S7, inference failure, invalid state, NaN body, concurrency singleton race |
| `test_quality_perturbation.py` | Spearman ρ ≥ 0.8 per perturbation type (slant is a disclosed strict xfail) |

## Known limitations

- The Flutter app is source-complete but **unbuilt/untested** in this
  environment — no visual verification, no `flutter analyze`/`flutter
  test` run. Treat it as a first-compile candidate, not a verified UI.
- No real child interaction data exists; the RL policy was trained purely
  in `dysgraphia_rl`'s simulator (5 hand-authored virtual profiles), and
  the backend's `quality_score` is a measured-feature proxy, not a
  clinically validated score.
- No CNN handwriting-quality scorer is trained or integrated (Kaggle
  dataset training is listed as future work in `dysgraphia_rl/README.md`
  and not duplicated here).
- Session/attempt state is in-memory + local SQLite — fine for a research
  prototype, not for multi-instance production deployment.
- The exercise/reference-template set is 3 synthetic placeholder shapes,
  not real letter/word content.
- Reward-hacking auditing (`dysgraphia_rl.utils.reward_utils.audit_action_distribution`)
  is implemented and tested but not yet wired into a backend endpoint —
  the raw ingredients (`safe_action_id` per attempt, in SQLite) are there;
  running the audit against real logs is a follow-up script, not
  implemented here.

## Next steps (beyond this build)

1. Train the CNN handwriting-quality scorer on a real dataset (e.g.
   Kaggle's dysgraphia dataset) and substitute it for
   `compute_quality_score` at its single call site in
   `backend/app/state/state_builder.py::build_state`.
2. Replace the synthetic exercise templates with a real letter/word set.
3. Build/run/visually verify the Flutter app against the running backend.
4. Collect real (consented, IRB-appropriate) child interaction data and
   compare the simulator-trained policy's behaviour against it.
5. Export the trained policy to ONNX for on-device inference, if offline
   operation becomes a requirement.
#   D y s g r a p h i a - t u t o r  
 