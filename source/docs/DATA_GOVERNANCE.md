# Data provenance and governance

## 1. What kinds of data exist in this repository

| Kind | Where it comes from | Label on records | May be called "real-child data"? |
|---|---|---|---|
| **Real app capture** | Pen events from the Flutter canvas (`tablet_app/lib/widgets/handwriting_canvas.dart`) sent to `/attempt/submit` | `data_source` = `demo` (a developer or adult tester using the app) or `real_child_study` (blocked, see §3) | Only `real_child_study`, and only after approval. **None exists in this repository.** |
| **Synthetic test traffic** | Generated traces: `backend/scripts/live_scenarios.py`, `backend/app/synthetic/handwriting_perturbations.py`, `backend/tests/conftest.py` | `data_source = synthetic_test`. Perturbation outputs are labelled "SYNTHETIC" in every JSON report. | No |
| **Simulated learners** | `dysgraphia_rl/envs/dysgraphia_tutor_env.py`. Five hand-authored profiles. Not fitted to any data. | Every results file has `"note": "SIMULATED / OFFLINE ..."` | No |
| **Public dataset** | The Kaggle dysgraphia image dataset named in the design document. **Not downloaded and not used anywhere in this code** (see §5). | `public_dataset_replay` is reserved for future replay of public data | No |
| **Model outputs** | DQN checkpoints, evaluation tables, quality-proxy scores | Saved under `dysgraphia_rl/checkpoints/`, `dysgraphia_rl/results/`, `evidence/` | No. These are model outputs, not observations. |

Every attempt row and every transition row in the SQLite log stores `data_source`.
Test runs write to a temporary database (`backend/tests/conftest.py`), so test
traffic never mixes with app data.

## 2. What is captured for one attempt

Per point: `x`, `y` (canvas logical pixels), `t` (seconds since the attempt's
first pointer event, taken from the **device event timestamp**), `pressure`
(normalized 0–1, or **`null` when the device reports no pressure range**;
it is never invented), `stroke_id`.

Per attempt: exercise id, attempt id, an optional `capture` block (pointer
kind, whether pressure is supported, canvas size, retry index, completion
flag, client version), the extracted features, the state vector, the
intervention actually delivered (`safe_action_id`) and the policy's raw
proposal, the policy source (`dqn`, `rule_based_fallback`, ...), and the
reward terms. The raw points are kept (`STORAGE_STORE_RAW_POINTS`, on by
default) so features can be recomputed and audited later.

Per transition: `(state, action, reward, next_state, done)`. `next_state`
and `reward` always come from the **next observed attempt**. If a session
ends before that attempt, the transition is stored with `reward = NULL`
and `outcome_observed = 0`.

## 3. Real-child data collection is disabled

`POST /session/start` with `data_source = "real_child_study"` returns
**HTTP 403**, unless **both** of these are set:

* `STUDY_REAL_CHILD_COLLECTION_APPROVED=true`
* `STUDY_ETHICS_APPROVAL_REFERENCE=<the approval reference>`

Set these only after institutional ethics approval, parental consent and
child assent, and safeguarding arrangements are in place. `/health`
reports `real_child_collection_enabled`. This is covered by a test
(`tests/test_live_scenarios.py::test_s7_data_governance`).

## 4. Pseudonymous identifiers and minimisation

* `child_id` must match `^[A-Za-z0-9][A-Za-z0-9_-]{2,39}$` (e.g. `P-0042`). Names with spaces and e-mail addresses are rejected with HTTP 422. Only the person running the study should hold the key that maps codes to children, and it must be kept outside this system.
* No name, age, date of birth, school, diagnosis, photo or audio is collected. The app asks only for the pseudonymous code.
* **Retention:** 180 days by default (`STUDY_RETENTION_DAYS`). Deletion is an operational task: delete rows by `session_id` or `child_id` from `attempts` and `transitions`. No automatic purge job is implemented. That is a known gap.
* **Access control:** the API has **no authentication** in this prototype. It is for a local demo network only and must not be exposed to the internet. Adding authentication is required before any study.

## 5. Kaggle dysgraphia dataset: honest assessment

* **Not used.** This session could not reach kaggle.com, and the dataset also needs Kaggle credentials. Nothing in this repository was trained or validated on it.
* **Modality mismatch.** The dataset is described as labelled **images** of handwriting (dysgraphic vs non-dysgraphic). The tutor's features are computed from **online pen trajectories** (x, y, t, pressure). Static images have no timing, speed, pause or pressure data. So the dataset cannot validate `writing_speed`, pauses, `movement_smoothness` or `pressure_variance`, and it cannot be replayed through `/attempt/submit`.
* **Label mismatch.** The labels are group/diagnostic labels, not per-sample quality scores. Using a classifier's "non-dysgraphic" probability as a quality reward would mix up "looks like a non-dysgraphic writer's sample" with "this attempt is better than the last one". A model trained that way also must not be described as diagnosing dysgraphia.
* **Domain shift.** Scanned paper handwriting differs from finger or stylus tracing on a tablet in tool, surface, scale and task.
* **What it could reasonably support** (future work): a separate image-based legibility estimate for the static part only, validated against human ratings before use, with all results reported under `public_dataset` provenance.

## 6. What may and may not be claimed

The system does **not** diagnose dysgraphia, does **not** deliver a clinically validated treatment, and has **not** been shown to improve any child's handwriting. Every RL result is from a simulator of hand-authored learner profiles. `quality_score` is an engineering proxy (see `docs/QUALITY_SCORE.md`).
