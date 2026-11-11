"""
DysgraphiaTutorEnv
==================
A Gymnasium-compatible simulated environment for training and evaluating the
RL-based adaptive handwriting intervention agent described in the project's
Design and Solution document (Section 6 - Dataset Description, Section 7 -
Algorithms/Methodology).

This is the *offline pre-training simulator* (Phase 1 of the training regime).
It does NOT talk to a real tablet or a real child. Instead it models a small
population of "virtual learner profiles" with different handwriting-difficulty
dynamics, so the Double DQN agent can be trained safely before any bounded
online fine-tuning on real interaction data.

State vector s_t in R^12 -- CANONICAL ORDERING. This exact order/naming is
shared verbatim with utils/state_utils.py:STATE_FEATURE_NAMES and
agents/baselines.py:IDX_* constants. Do not reorder without updating all
three locations and tests/test_extra.py::test_state_ordering_matches_canonical_spec.
    0. prev_score            - previous trial's handwriting-quality score [0,1]
    1. stroke_similarity     - DTW-based shape similarity to template [0,1]
    2. spacing_accuracy      - letter/word spacing consistency [0,1]
    3. baseline_alignment    - how well strokes sit on the writing line [0,1]
    4. writing_speed         - normalized speed (1.0 = age-appropriate) [0,2]
    5. pressure_variance     - stylus pressure variability [0,1]
    6. retry_count           - retries on current trial, normalized [0,1]
    7. fatigue_proxy         - accumulated fatigue, derived from inter-stroke
                               pause duration and session length [0,1]
    8. engagement            - engagement proxy [0,1]
    9. improvement_trend     - short-term rolling improvement [-1,1]
    10. frustration          - frustration proxy [0,1]
    11. consistency          - cross-attempt consistency [0,1]

Action space (10 discrete pedagogical actions, matches Section 4/6/7):
    0. DECREASE_DIFFICULTY
    1. INCREASE_DIFFICULTY
    2. SHOW_HINT
    3. ENLARGE_TEMPLATE
    4. REPLAY_STROKE_ANIMATION
    5. SWITCH_TO_TRACING_MODE
    6. SWITCH_TO_FREE_WRITING_MODE
    7. SUGGEST_BREAK
    8. REPEAT_SAME_LETTER
    9. CONTINUE_NO_CHANGE

Reward (reward_version="v2", the default since the Phase A audit):
    R_t = w1*(q_t - q_start) + w2*completion + w3*engagement
          - w4*frustration + w5*consistency - w6*unnecessary_break
    with w1 the largest weight. q_t is the most recent OBSERVED quality
    proxy (carried forward unchanged on a break step, because no writing
    happens during a break) and q_start is the learner's starting level.
    `unnecessary_break` penalizes a break taken while the fatigue proxy is
    below `break_fatigue_threshold`. reward_version="v1" reproduces the
    original per-step-delta formula exactly (used only to reproduce the
    Phase A defect; see docs/RL_AUDIT_AND_FIXES.md).

Reward version "v3" (the default since Round 2): same as v2 on writing
    steps. On an executed break step the reward is ONLY
    -w6*unnecessary_break, because no practice happened, so the practice
    terms (quality, completion, engagement, frustration, consistency) are
    not credited. In Round 1, v2 let the DQN take a break roughly every
    third step at fatigue ~0.03 (docs/EVALUATION_PROTOCOL.md).

Break safety rule (both versions unless max_consecutive_breaks=None):
    at most `max_consecutive_breaks` (default 1) SUGGEST_BREAK actions may
    execute back to back. A further requested break is executed as
    CONTINUE_NO_CHANGE; `info["requested_action"]`, `info["executed_action"]`
    and `info["break_capped"]` record this so the training loop stores the
    action that actually ran. The live backend safety layer enforces the
    same rule (backend/app/safety/safety_layer.py).
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import Optional

# Single source of truth for the reward formula (Design and Solution Section 7).
# The environment does NOT re-implement the formula -- it calls the same
# function used by the standalone reward-hacking audit and by
# tests/test_core.py::test_reward_formula_matches_spec, so the two can never
# silently drift apart.
from utils.reward_utils import (
    RewardWeights, compute_reward, unnecessary_break_penalty, validate_weights,
)


# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------
DECREASE_DIFFICULTY = 0
INCREASE_DIFFICULTY = 1
SHOW_HINT = 2
ENLARGE_TEMPLATE = 3
REPLAY_STROKE_ANIMATION = 4
SWITCH_TO_TRACING_MODE = 5
SWITCH_TO_FREE_WRITING_MODE = 6
SUGGEST_BREAK = 7
REPEAT_SAME_LETTER = 8
CONTINUE_NO_CHANGE = 9

N_ACTIONS = 10
STATE_DIM = 12

ACTION_NAMES = [
    "DECREASE_DIFFICULTY", "INCREASE_DIFFICULTY", "SHOW_HINT",
    "ENLARGE_TEMPLATE", "REPLAY_STROKE_ANIMATION", "SWITCH_TO_TRACING_MODE",
    "SWITCH_TO_FREE_WRITING_MODE", "SUGGEST_BREAK", "REPEAT_SAME_LETTER",
    "CONTINUE_NO_CHANGE",
]


@dataclass
class LearnerProfile:
    """Parameters describing a virtual dysgraphic learner's dynamics.

    These are deliberately simple, hand-authored rule-based dynamics (as
    described in Section 6 of the Design and Solution doc: "a simulated
    handwriting-practice environment ... to model different learner
    profiles"). They stand in for real children until real interaction
    logs / the Kaggle-derived CNN scorer are integrated.

    Honesty note on where these specific numbers come from: they are NOT
    derived from real dysgraphic-child data, a literature-sourced
    parameter table, or a sensitivity/ablation study -- they were chosen by
    hand to produce plausible, clearly-differentiated relative behavior
    across profiles (e.g. "fatigue_prone accumulates fatigue roughly 3x
    faster than fast_improver"), which is what `PROFILES` below and
    tests/test_extra.py::test_learner_profiles_produce_measurably_different_dynamics
    verify. Treat the RELATIVE differences between profiles as the
    meaningful signal, not the absolute values. A rigorous next step before
    any claim of real-world validity would be a sensitivity analysis over
    these constants and/or fitting them to actual interaction logs once
    available -- currently listed as future work, not done here.
    """
    name: str
    base_skill: float            # underlying competence, 0-1
    learning_rate: float         # how fast skill improves with practice
    fatigue_rate: float          # how fast fatigue accumulates
    frustration_sensitivity: float  # how quickly frustration builds on failure
    spacing_weakness: float      # extra difficulty specific to spacing
    stroke_instability: float    # extra noise in stroke similarity
    recovery_rate: float         # how fast fatigue/frustration recover on break


PROFILES = [
    LearnerProfile("fast_improver", base_skill=0.55, learning_rate=0.045,
                   fatigue_rate=0.02, frustration_sensitivity=0.05,
                   spacing_weakness=0.05, stroke_instability=0.05,
                   recovery_rate=0.25),
    LearnerProfile("slow_improver", base_skill=0.35, learning_rate=0.015,
                   fatigue_rate=0.03, frustration_sensitivity=0.08,
                   spacing_weakness=0.10, stroke_instability=0.10,
                   recovery_rate=0.20),
    LearnerProfile("fatigue_prone", base_skill=0.45, learning_rate=0.025,
                   fatigue_rate=0.06, frustration_sensitivity=0.10,
                   spacing_weakness=0.08, stroke_instability=0.08,
                   recovery_rate=0.35),
    LearnerProfile("spacing_struggler", base_skill=0.40, learning_rate=0.02,
                   fatigue_rate=0.025, frustration_sensitivity=0.07,
                   spacing_weakness=0.30, stroke_instability=0.06,
                   recovery_rate=0.22),
    LearnerProfile("unstable_stroke", base_skill=0.40, learning_rate=0.02,
                   fatigue_rate=0.03, frustration_sensitivity=0.09,
                   spacing_weakness=0.08, stroke_instability=0.28,
                   recovery_rate=0.22),
]


# Backend-consistent quality weights and "not measured" placeholders
# (observation_version "v2"). These must equal
# backend/app/config.py::QualityScoreConfig and the backend's placeholder
# values. backend/tests/test_sim_backend_consistency.py checks this.
QUALITY_W_SIMILARITY = 0.45
QUALITY_W_SPACING = 0.20
QUALITY_W_BASELINE = 0.20
QUALITY_W_PRESSURE = 0.15
SPACING_PLACEHOLDER = 0.5            # spacing not measurable (< 3 strokes)
PRESSURE_VARIANCE_PLACEHOLDER = 0.3  # device reports no pressure


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class DysgraphiaTutorEnv(gym.Env):
    """Gymnasium environment simulating a dysgraphic child's handwriting
    practice session for offline RL pre-training.

    One `step()` = one writing attempt (trial) followed by the tutor's
    chosen intervention, mirroring Stage 1-6 of the System Workflow.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 40,
        difficulty_levels: int = 5,
        profile: Optional[LearnerProfile] = None,
        seed: Optional[int] = None,
        reward_weights: Optional[dict] = None,
        reward_version: str = "v3",
        max_consecutive_breaks: Optional[int] = 1,
        break_fatigue_threshold: float = 0.5,
        observation_version: Optional[str] = None,
        spacing_observable: bool = False,
        pressure_available_prob: float = 0.5,
        improvement_signal: str = "observed",
        profile_sampling: str = "catalog",
        synthetic_profile_count: int = 1000,
    ):
        super().__init__()
        # improvement_signal (deployment-candidate training, 2026-11-08,
        # docs/DQN_DEPLOYMENT_AUDIT.md). Default "observed" keeps every
        # historical result and checkpoint reproducible.
        #   "observed": the quality term is q_t - q_start on the RAW quality
        #               proxy. That proxy rises when the exercise is made
        #               easier (difficulty penalty) or switched to tracing,
        #               so lowering difficulty is rewarded as "improvement".
        #   "latent":   the quality term is the change in latent_skill(),
        #               the difficulty-matched, noise-free skill at the
        #               episode's starting difficulty. It grows only with
        #               practice, so it cannot be raised by making the task
        #               easier, and a break step adds no practice.
        if improvement_signal not in ("observed", "latent"):
            raise ValueError(f"improvement_signal must be 'observed' or 'latent', got {improvement_signal!r}")
        if improvement_signal == "latent" and reward_version == "v1":
            raise ValueError("improvement_signal='latent' requires reward_version v2 or v3")
        self.improvement_signal = improvement_signal
        self._latent_start = 0.0
        # observation_version (Round 3, docs/STATE_SCHEMA.md):
        #   "v1": original observation/quality semantics, kept so Rounds 1-2
        #         and the Phase A reproduction stay bit-exact. Quality uses
        #         0.40/0.25/0.20/0.15 and spacing and pressure are always measured.
        #   "v2": matches the deployed backend. Quality uses the backend's
        #         compute_quality_score weights and its rule for leaving out
        #         unmeasured terms. Spacing is unobservable on single-stroke
        #         exercises (placeholder 0.5). Pressure is available with
        #         probability `pressure_available_prob` per episode
        #         (placeholder 0.3 otherwise).
        if observation_version is None:
            observation_version = "v1" if reward_version == "v1" else "v2"
        if observation_version not in ("v1", "v2"):
            raise ValueError(f"observation_version must be 'v1' or 'v2', got {observation_version!r}")
        self.observation_version = observation_version
        self.spacing_observable = spacing_observable
        self.pressure_available_prob = pressure_available_prob
        self.pressure_available = True
        if reward_version not in ("v1", "v2", "v3"):
            raise ValueError(f"reward_version must be 'v1', 'v2' or 'v3', got {reward_version!r}")
        self.reward_version = reward_version
        self.max_consecutive_breaks = max_consecutive_breaks
        self.break_fatigue_threshold = break_fatigue_threshold
        self.max_steps = max_steps
        self.difficulty_levels = difficulty_levels
        self._fixed_profile = profile
        if profile_sampling not in ("catalog", "synthetic_population"):
            raise ValueError("profile_sampling must be 'catalog' or 'synthetic_population'")
        if synthetic_profile_count < 1:
            raise ValueError("synthetic_profile_count must be >= 1")
        self.profile_sampling = profile_sampling
        self.synthetic_profile_count = int(synthetic_profile_count)
        self.rng = np.random.default_rng(seed)
        self._synthetic_profiles = self._build_synthetic_population(
            self.synthetic_profile_count, seed
        ) if profile_sampling == "synthetic_population" and profile is None else []

        # Reward weights, matches Section 7: w1 highest to avoid reward hacking.
        # Uses the same RewardWeights dataclass as utils/reward_utils.py so
        # there is exactly one place these numbers are defined.
        if reward_version == "v1":
            self.reward_weights = (
                RewardWeights(**reward_weights) if reward_weights else RewardWeights()
            )
        else:
            self.reward_weights = RewardWeights.v2(**(reward_weights or {}))
        validate_weights(self.reward_weights)

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(STATE_DIM,), dtype=np.float32
        )

        # internal (hidden from agent) learner state
        self.profile: LearnerProfile = None
        self.difficulty = 0
        self.fatigue = 0.0
        self.frustration = 0.0
        self.engagement = 0.0
        self.consecutive_fails = 0
        self.step_count = 0
        self.score_history = []
        self.mode = "tracing"  # or "free_writing"
        self.consecutive_breaks = 0
        self.start_difficulty = 0
        self._last_trial = None

    @staticmethod
    def _build_synthetic_population(count: int, seed: Optional[int]) -> list[LearnerProfile]:
        """Create a reproducible, broad synthetic parameter population.

        Bounds are the min/max of the five documented hand-authored profiles.
        This increases simulator coverage; it does NOT make the parameters
        empirically calibrated or representative of children.
        """
        rng = np.random.default_rng(seed)
        fields = (
            ("base_skill", 0.35, 0.55),
            ("learning_rate", 0.015, 0.045),
            ("fatigue_rate", 0.020, 0.060),
            ("frustration_sensitivity", 0.050, 0.100),
            ("spacing_weakness", 0.050, 0.300),
            ("stroke_instability", 0.050, 0.280),
            ("recovery_rate", 0.200, 0.350),
        )
        # Latin-hypercube sampling spreads profiles across each parameter's
        # range instead of clustering all samples around a few random draws.
        unit = np.empty((count, len(fields)), dtype=np.float64)
        for j in range(len(fields)):
            unit[:, j] = (rng.permutation(count) + rng.random(count)) / count
        population = []
        for i in range(count):
            values = {}
            for j, (name, low, high) in enumerate(fields):
                values[name] = float(low + unit[i, j] * (high - low))
            population.append(LearnerProfile(name=f"synthetic_{i:04d}", **values))
        return population

    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self._fixed_profile is not None:
            self.profile = self._fixed_profile
        elif self.profile_sampling == "synthetic_population":
            self.profile = self._synthetic_profiles[int(self.rng.integers(0, len(self._synthetic_profiles)))]
        else:
            self.profile = PROFILES[self.rng.integers(0, len(PROFILES))]
        if self.observation_version == "v2":
            self.pressure_available = bool(self.rng.random() < self.pressure_available_prob)
        else:
            self.pressure_available = True
        self.difficulty = self.difficulty_levels // 2
        self.fatigue = 0.05
        self.frustration = 0.05
        self.engagement = 0.7
        self.consecutive_fails = 0
        self.step_count = 0
        self.mode = "tracing"
        self.consecutive_breaks = 0
        self.start_difficulty = self.difficulty

        if self.reward_version == "v1":
            # Original behaviour (kept only for exact Phase A reproduction):
            # the episode "starts" from the hidden profile constant
            # base_skill, which is NOT an observed measurement and sits
            # ~0.1-0.15 above the quality of the learner's first actual
            # attempt -- biasing every "quality gain" negative.
            self.score_history = [self.profile.base_skill]
            self._last_trial = dict(quality=self.profile.base_skill,
                                    stroke_sim=self.profile.base_skill,
                                    spacing=self.profile.base_skill,
                                    alignment=self.profile.base_skill,
                                    speed=1.0, pressure_var=0.3)
            self._last_retries = 0.0
            obs = self._build_state(prev_quality=self.profile.base_skill,
                                     stroke_sim=self.profile.base_skill,
                                     spacing=self.profile.base_skill,
                                     alignment=self.profile.base_skill,
                                     speed=1.0, pressure_var=0.3, retries=0.0)
        else:
            # v2: like the live backend, the first state comes from an
            # OBSERVED first attempt (an initial assessment trial at the
            # starting difficulty), so q_start and every later q_t are
            # measurements on the same scale.
            self.score_history = []
            trial = self._simulate_trial()
            retries = self._apply_trial_dynamics(trial)
            self.score_history.append(trial["quality"])
            self._last_trial = dict(trial)
            self._last_retries = retries
            self._latent_start = self.latent_skill()
            obs = self._build_state(prev_quality=trial["quality"], stroke_sim=trial["stroke_sim"],
                                     spacing=trial["spacing"], alignment=trial["alignment"],
                                     speed=trial["speed"], pressure_var=trial["pressure_var"],
                                     retries=retries)
        info = {"profile": self.profile.name, "start_quality": self.score_history[0],
                "pressure_available": self.pressure_available,
                "observation_version": self.observation_version}
        return obs, info

    def _apply_trial_dynamics(self, trial: dict) -> float:
        """Natural fatigue/frustration/engagement response to one writing
        attempt. Returns the normalized retry count after the attempt."""
        p = self.profile
        success = trial["quality"] >= 0.55
        self.consecutive_fails = 0 if success else self.consecutive_fails + 1
        retries = min(self.consecutive_fails, 5) / 5.0
        self.fatigue = _clip01(self.fatigue + p.fatigue_rate)
        if not success:
            self.frustration = _clip01(
                self.frustration + p.frustration_sensitivity * (1 + 0.3 * self.consecutive_fails)
            )
            self.engagement = _clip01(self.engagement - 0.04)
        else:
            self.frustration = _clip01(self.frustration - 0.03)
            self.engagement = _clip01(self.engagement + 0.02)
        return retries

    # ------------------------------------------------------------------
    def _simulate_trial(self):
        """Simulate the child's next writing attempt given current internal
        state and difficulty. Returns raw handwriting-quality features."""
        p = self.profile
        difficulty_penalty = (self.difficulty / (self.difficulty_levels - 1)) * 0.35
        fatigue_penalty = self.fatigue * 0.4
        frustration_penalty = self.frustration * 0.2
        mode_bonus = 0.08 if self.mode == "tracing" else 0.0

        effective_skill = _clip01(
            p.base_skill + self._practice_bonus()
            - difficulty_penalty - fatigue_penalty - frustration_penalty
            + mode_bonus
        )

        noise = self.rng.normal(0, 0.05 + p.stroke_instability)
        stroke_sim = _clip01(effective_skill + noise)
        spacing = _clip01(effective_skill - p.spacing_weakness + self.rng.normal(0, 0.05))
        alignment = _clip01(effective_skill + self.rng.normal(0, 0.05))
        speed = float(np.clip(1.3 - effective_skill * 0.5 + self.fatigue * 0.3
                               + self.rng.normal(0, 0.08), 0.3, 2.0))
        pressure_var = _clip01(0.5 - effective_skill * 0.3 + self.fatigue * 0.2
                                + self.rng.normal(0, 0.05))

        if self.observation_version == "v1":
            quality = _clip01(0.4 * stroke_sim + 0.25 * spacing + 0.2 * alignment
                               + 0.15 * (1.0 - pressure_var))
        else:
            # Same weights and the same rule for unmeasured terms as
            # backend/app/state/state_builder.py::compute_quality_score
            # (tests/test_sim_backend_consistency.py keeps them in sync).
            terms = [(QUALITY_W_SIMILARITY, stroke_sim), (QUALITY_W_BASELINE, alignment)]
            if self.spacing_observable:
                terms.append((QUALITY_W_SPACING, spacing))
            else:
                spacing = SPACING_PLACEHOLDER
            if self.pressure_available:
                terms.append((QUALITY_W_PRESSURE, 1.0 - pressure_var))
            else:
                pressure_var = PRESSURE_VARIANCE_PLACEHOLDER
            quality = _clip01(sum(w * v for w, v in terms) / sum(w for w, _ in terms))
        return dict(quality=quality, stroke_sim=stroke_sim, spacing=spacing,
                    alignment=alignment, speed=speed, pressure_var=pressure_var)

    def latent_skill(self, difficulty: Optional[int] = None) -> float:
        """Noise-free expected quality at a FIXED difficulty (default: the
        episode's starting difficulty) with zero fatigue/frustration in
        tracing mode. Used only for evaluation reporting as a
        difficulty-matched learning measure: unlike the raw quality proxy,
        it cannot be raised simply by lowering the exercise difficulty."""
        d = self.start_difficulty if difficulty is None else difficulty
        difficulty_penalty = (d / (self.difficulty_levels - 1)) * 0.35
        return _clip01(self.profile.base_skill + self._practice_bonus()
                       - difficulty_penalty + 0.08)

    def _practice_bonus(self):
        """Skill grows slowly with number of successful attempts (learning)."""
        n = len(self.score_history)
        recent = np.mean(self.score_history[-5:]) if n else self.profile.base_skill
        return self.profile.learning_rate * min(n, 30) * (0.5 + recent * 0.5) * 0.3

    # ------------------------------------------------------------------
    def step(self, action: int):
        assert self.action_space.contains(action)
        p = self.profile
        completion = 1.0
        break_taken = False
        requested_action = int(action)
        break_capped = False
        if (action == SUGGEST_BREAK and self.max_consecutive_breaks is not None
                and self.consecutive_breaks >= self.max_consecutive_breaks):
            action = CONTINUE_NO_CHANGE
            break_capped = True
        fatigue_before = self.fatigue

        # --- apply the chosen pedagogical action -------------------------------
        if action == DECREASE_DIFFICULTY:
            self.difficulty = max(0, self.difficulty - 1)
        elif action == INCREASE_DIFFICULTY:
            self.difficulty = min(self.difficulty_levels - 1, self.difficulty + 1)
        elif action == SHOW_HINT:
            self.frustration = _clip01(self.frustration - 0.10)
            self.engagement = _clip01(self.engagement + 0.03)
        elif action == ENLARGE_TEMPLATE:
            self.frustration = _clip01(self.frustration - 0.05)
        elif action == REPLAY_STROKE_ANIMATION:
            self.frustration = _clip01(self.frustration - 0.07)
            self.engagement = _clip01(self.engagement + 0.02)
        elif action == SWITCH_TO_TRACING_MODE:
            self.mode = "tracing"
            self.frustration = _clip01(self.frustration - 0.05)
        elif action == SWITCH_TO_FREE_WRITING_MODE:
            self.mode = "free_writing"
            self.engagement = _clip01(self.engagement + 0.05)
        elif action == SUGGEST_BREAK:
            self.fatigue = _clip01(self.fatigue - p.recovery_rate)
            self.frustration = _clip01(self.frustration - p.recovery_rate * 0.6)
            self.engagement = _clip01(self.engagement + 0.05)
            completion = 0.0  # no writing trial happens this step
            break_taken = True
        elif action == REPEAT_SAME_LETTER:
            pass  # no direct effect, just repetition
        elif action == CONTINUE_NO_CHANGE:
            pass

        # --- simulate the (possibly skipped) writing trial ----------------------
        if break_taken:
            self.consecutive_breaks += 1
            if self.reward_version == "v1":
                # Original behaviour, kept only so the Phase A defect can be
                # reproduced exactly: zeroed stroke features on a break step.
                trial = dict(quality=self.score_history[-1], stroke_sim=0.0,
                             spacing=0.0, alignment=0.0, speed=1.0, pressure_var=0.3)
                retries = 0
            else:
                # No writing happens during a break, so nothing new is
                # measured: carry the last OBSERVED attempt's features
                # forward instead of fabricating zeros (the live backend
                # never produces an all-zero stroke-feature state).
                trial = dict(self._last_trial)
                retries = self._last_retries
        else:
            self.consecutive_breaks = 0
            trial = self._simulate_trial()
            retries = self._apply_trial_dynamics(trial)

            self.score_history.append(trial["quality"])
            self._last_trial = dict(trial)
            self._last_retries = retries

        prev_quality = self.score_history[-2] if len(self.score_history) > 1 else self.score_history[-1]
        delta_quality = trial["quality"] - prev_quality if not break_taken else 0.0
        consistency = 1.0 - float(np.std(self.score_history[-5:])) if len(self.score_history) >= 2 else 0.5
        consistency = _clip01(consistency)
        improvement_trend = float(np.clip(
            np.mean(np.diff(self.score_history[-6:])) if len(self.score_history) >= 3 else 0.0,
            -1.0, 1.0
        ))

        # Delegate to the canonical reward formula (utils/reward_utils.py)
        # instead of re-deriving it here -- see the Design-to-Code Mapping
        # doc for why this matters (avoids two formulas silently diverging).
        if self.reward_version == "v1":
            quality_signal = delta_quality
            unnecessary_break = 0.0
        else:
            if self.improvement_signal == "latent":
                quality_signal = self.latent_skill() - self._latent_start
            else:
                quality_signal = self.score_history[-1] - self.score_history[0]
            unnecessary_break = unnecessary_break_penalty(
                break_taken, fatigue_before, self.break_fatigue_threshold)
        if self.reward_version == "v3" and break_taken:
            # v3: no practice happened on this step, so no practice terms
            # are credited. Only the unnecessary-break penalty applies
            # (docs/EVALUATION_PROTOCOL.md, Round 2 amendment).
            reward = -self.reward_weights.w6_unnecessary_break * unnecessary_break
        else:
            reward = compute_reward(
                delta_quality=quality_signal,
                completion=completion,
                engagement=self.engagement,
                frustration=self.frustration,
                consistency=consistency,
                weights=self.reward_weights,
                unnecessary_break=unnecessary_break,
            )

        self.step_count += 1
        # `terminated` (a true, MDP-internal ending) vs `truncated` (an
        # artificial external cutoff) are NOT interchangeable for the
        # Bellman target: `done=terminated` should zero the bootstrap term
        # (there genuinely is no valid next state), while a time-limit
        # truncation should still bootstrap (the episode could, in
        # principle, have continued). The ordinary max_steps cutoff is a
        # time limit -> truncated. The fatigue safety cutoff is a real,
        # MDP-internal ending condition (the session is being forced to
        # stop because it is no longer safe to continue) -> terminated.
        # See tests/test_extra.py::test_fatigue_cutoff_sets_terminated_not_truncated
        # and ::test_train_dqn_stores_done_flag_on_true_termination for the
        # tests that lock this distinction in, and CODE_REVIEW_READINESS_REPORT.md
        # for the bug this fixes (previously both cases set `truncated`, and
        # the training loop stored `done=float(terminated)`, which was
        # therefore always 0.0 -- the Bellman target bootstrapped through
        # every episode boundary, including genuine terminal states).
        terminated = self.fatigue > 0.95
        truncated = (self.step_count >= self.max_steps) and not terminated

        obs = self._build_state(
            prev_quality=trial["quality"], stroke_sim=trial["stroke_sim"],
            spacing=trial["spacing"], alignment=trial["alignment"],
            speed=trial["speed"], pressure_var=trial["pressure_var"], retries=retries,
            improvement_trend=improvement_trend, consistency=consistency,
        )

        info = {
            "quality": trial["quality"],
            "delta_quality": delta_quality,
            "fatigue": self.fatigue,
            "frustration": self.frustration,
            "engagement": self.engagement,
            "difficulty": self.difficulty,
            "action_name": ACTION_NAMES[action],
            "profile": p.name,
            "break_taken": break_taken,
            "requested_action": requested_action,
            "executed_action": int(action),
            "break_capped": break_capped,
            "quality_improvement": self.score_history[-1] - self.score_history[0],
            "unnecessary_break": unnecessary_break,
            "fatigue_before": fatigue_before,
            "reward_version": self.reward_version,
        }
        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    def _build_state(self, prev_quality, stroke_sim, spacing, alignment, speed,
                      pressure_var, retries, improvement_trend=0.0, consistency=0.5):
        state = np.array([
            prev_quality,
            stroke_sim,
            spacing,
            alignment,
            speed,
            pressure_var,
            retries,
            self.fatigue,
            self.engagement,
            improvement_trend,
            self.frustration,
            consistency,
        ], dtype=np.float32)
        return state

    def render(self):
        print(f"step={self.step_count} profile={self.profile.name} "
              f"difficulty={self.difficulty} fatigue={self.fatigue:.2f} "
              f"frustration={self.frustration:.2f} engagement={self.engagement:.2f} "
              f"last_quality={self.score_history[-1]:.2f}")
