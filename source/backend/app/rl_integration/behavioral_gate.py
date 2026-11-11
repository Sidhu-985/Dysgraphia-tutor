"""
Behavioural gate for the DQN checkpoint
========================================
A loadable checkpoint is not necessarily a usable one. The Phase A audit
found that the shipped checkpoint chose SUGGEST_BREAK for ~75% of attempts
by a fresh, clean writer (fatigue 0.03, frustration 0). That checkpoint
loaded without error and passed every shape/argmax test.

Before the backend lets a checkpoint make live decisions, this gate runs it
on a fixed, deterministic probe set with known safe expectations. If any
probe fails, the PolicyService serves the transparent rule-based fallback
policy, and /health reports why.

Probes (the raw policy output is checked, before the safety layer):
  G1 fresh, clean writer: must not request SUGGEST_BREAK.
  G2 typical mid-session writer, low fatigue and frustration: must not
     request SUGGEST_BREAK.
  G3 struggling, frustrated writer: must not request INCREASE_DIFFICULTY
     or SWITCH_TO_FREE_WRITING_MODE.
  G4 low-fatigue population: 300 seeded random valid states with
     fatigue < 0.4 and frustration < 0.4. Requested break rate must be
     < 0.25. This is the same 25% ceiling used in the offline evaluation.
  G5 numerics: Q-values are finite for every probe.
  G6 fresh, clean writer: must not request DECREASE_DIFFICULTY. This probe
     was added after Round 2. With the gate disabled, the Round-2 DQN made
     the task easier on 6 of 6 attempts for a clean writer scoring about
     0.98 (evidence/phase_e/live_scenarios_gate_disabled_DQN.json). Adding
     it can only make the gate stricter.

Deployment-realistic probes. These were added after
evidence/phase_d/state_distribution_comparison.json was produced, and
before any reward-v2 checkpoint was gated. The live backend serves
single-stroke exercises on touch screens without pressure, so it always
sends spacing_accuracy = 0.5 and pressure_variance = 0.3, which are
neutral placeholders. The simulator never produces those values. G1 and G2
are therefore checked in both forms, and half of the G4 population uses
the placeholder values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..dysgraphia_rl_bridge import STATE_FEATURE_NAMES, ACTION_NAMES, build_state_vector

SUGGEST_BREAK = ACTION_NAMES.index("SUGGEST_BREAK")
INCREASE_DIFFICULTY = ACTION_NAMES.index("INCREASE_DIFFICULTY")
SWITCH_TO_FREE_WRITING_MODE = ACTION_NAMES.index("SWITCH_TO_FREE_WRITING_MODE")
DECREASE_DIFFICULTY = ACTION_NAMES.index("DECREASE_DIFFICULTY")
MAX_LOW_FATIGUE_BREAK_RATE = 0.25


def _state(**overrides) -> np.ndarray:
    base = dict(prev_score=0.6, stroke_similarity=0.6, spacing_accuracy=0.6,
                baseline_alignment=0.6, writing_speed=1.0, pressure_variance=0.3,
                retry_count=0.0, fatigue_proxy=0.1, engagement=0.7,
                improvement_trend=0.0, frustration=0.05, consistency=0.8)
    base.update(overrides)
    return build_state_vector({k: base[k] for k in STATE_FEATURE_NAMES})


FRESH_GOOD_WRITER = _state(prev_score=0.88, stroke_similarity=0.9, spacing_accuracy=0.85,
                           baseline_alignment=0.9, pressure_variance=0.2, fatigue_proxy=0.03,
                           engagement=0.72, frustration=0.0, consistency=0.5)
MID_SESSION_WRITER = _state(prev_score=0.62, stroke_similarity=0.65, spacing_accuracy=0.55,
                            baseline_alignment=0.7, fatigue_proxy=0.25, engagement=0.65,
                            frustration=0.15, consistency=0.85, improvement_trend=0.02)
DEPLOYMENT_NEUTRAL = dict(spacing_accuracy=0.5, pressure_variance=0.3)
FRESH_GOOD_WRITER_DEPLOYED = _state(prev_score=0.88, stroke_similarity=0.9, baseline_alignment=0.9,
                                    fatigue_proxy=0.03, engagement=0.72, frustration=0.0,
                                    consistency=0.5, **DEPLOYMENT_NEUTRAL)
MID_SESSION_WRITER_DEPLOYED = _state(prev_score=0.62, stroke_similarity=0.65, baseline_alignment=0.7,
                                     fatigue_proxy=0.25, engagement=0.65, frustration=0.15,
                                     consistency=0.85, improvement_trend=0.02, **DEPLOYMENT_NEUTRAL)
STRUGGLING_WRITER = _state(prev_score=0.25, stroke_similarity=0.3, spacing_accuracy=0.2,
                           baseline_alignment=0.3, pressure_variance=0.6, retry_count=0.8,
                           fatigue_proxy=0.3, engagement=0.35, frustration=0.85,
                           improvement_trend=-0.05, consistency=0.7)


def low_fatigue_population(n: int = 300, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        deployed = i % 2 == 1  # every other state uses the backend's placeholder values
        q = rng.uniform(0.1, 0.95)
        rows.append(_state(
            prev_score=q,
            stroke_similarity=float(np.clip(q + rng.normal(0, 0.08), 0, 1)),
            spacing_accuracy=0.5 if deployed else float(np.clip(q + rng.normal(0, 0.1), 0, 1)),
            baseline_alignment=float(np.clip(q + rng.normal(0, 0.08), 0, 1)),
            writing_speed=float(rng.uniform(0.5, 1.6)),
            pressure_variance=0.3 if deployed else float(rng.uniform(0.1, 0.6)),
            retry_count=float(rng.choice([0.0, 0.2, 0.4])),
            fatigue_proxy=float(rng.uniform(0.0, 0.4)),
            engagement=float(rng.uniform(0.4, 1.0)),
            improvement_trend=float(rng.uniform(-0.1, 0.1)),
            frustration=float(rng.uniform(0.0, 0.4)),
            consistency=float(rng.uniform(0.5, 1.0)),
        ))
    return np.stack(rows)


@dataclass
class GateReport:
    passed: bool
    failures: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)


def run_behavioral_gate(q_fn) -> GateReport:
    """`q_fn(state_batch: np.ndarray[N,12]) -> np.ndarray[N,10]` Q-values."""
    failures: list[str] = []
    details: dict = {}

    named = {"G1_fresh_good_writer": FRESH_GOOD_WRITER,
             "G1_fresh_good_writer_deployed": FRESH_GOOD_WRITER_DEPLOYED,
             "G2_mid_session_writer": MID_SESSION_WRITER,
             "G2_mid_session_writer_deployed": MID_SESSION_WRITER_DEPLOYED,
             "G3_struggling_writer": STRUGGLING_WRITER}
    batch = np.stack(list(named.values()))
    q = np.asarray(q_fn(batch))
    pop = low_fatigue_population()
    q_pop = np.asarray(q_fn(pop))

    if not (np.all(np.isfinite(q)) and np.all(np.isfinite(q_pop))):
        failures.append("G5: non-finite Q-values")
        return GateReport(False, failures, details)

    chosen = {name: int(np.argmax(q[i])) for i, name in enumerate(named)}
    details["probe_actions"] = {k: ACTION_NAMES[v] for k, v in chosen.items()}
    for key in ("G1_fresh_good_writer", "G1_fresh_good_writer_deployed"):
        if chosen[key] == SUGGEST_BREAK:
            failures.append(f"{key}: requests SUGGEST_BREAK for a fresh, clean writer")
    for key in ("G1_fresh_good_writer", "G1_fresh_good_writer_deployed"):
        if chosen[key] == DECREASE_DIFFICULTY:
            failures.append(f"G6 ({key}): requests DECREASE_DIFFICULTY for a fresh, clean writer")
    for key in ("G2_mid_session_writer", "G2_mid_session_writer_deployed"):
        if chosen[key] == SUGGEST_BREAK:
            failures.append(f"{key}: requests SUGGEST_BREAK for a low-fatigue mid-session writer")
    if chosen["G3_struggling_writer"] in (INCREASE_DIFFICULTY, SWITCH_TO_FREE_WRITING_MODE):
        failures.append(f"G3: requests {ACTION_NAMES[chosen['G3_struggling_writer']]} "
                        "for a struggling, frustrated writer")

    pop_actions = np.argmax(q_pop, axis=1)
    break_rate = float(np.mean(pop_actions == SUGGEST_BREAK))
    details["low_fatigue_break_rate"] = break_rate
    details["low_fatigue_action_counts"] = {
        ACTION_NAMES[a]: int(np.sum(pop_actions == a)) for a in range(len(ACTION_NAMES))}
    if break_rate >= MAX_LOW_FATIGUE_BREAK_RATE:
        failures.append(f"G4: requests SUGGEST_BREAK in {break_rate:.0%} of low-fatigue states "
                        f"(limit {MAX_LOW_FATIGUE_BREAK_RATE:.0%})")
    return GateReport(not failures, failures, details)
