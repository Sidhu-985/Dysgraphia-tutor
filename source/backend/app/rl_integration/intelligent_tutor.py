"""
Intelligent rule-based tutor
==============================

A drop-in replacement for the brittle :class:`agents.baselines.RuleBasedAgent`
used as the rule-based fallback when the gated Double DQN checkpoint is
not allowed to make live decisions. Unlike the legacy agent, which only
saw the 12-D state vector and applied a handful of fixed thresholds, this
tutor sees the FULL session context:

  * current handwriting score (``state[IDX_PREV_SCORE]``)
  * recent score history (``session.score_history``) -- improvement /
    deterioration trends, consistency
  * current letter / exercise difficulty (``session.difficulty_level``,
    ``session.current_exercise_id``)
  * learner fatigue, frustration, engagement (state vector components)
  * writing consistency (state vector component)
  * number of failed attempts in a row (``session.consecutive_fails``)
  * previous interventions (``session.action_log``) and which were
    effective (inferred from score changes after each intervention)
  * current learning mode (``session.mode``)
  * recent exercise history (``session.recent_exercises``)

It applies the tutoring behaviours required by the audit:

  A. Incorrect handwriting -- never praise a malformed letter as correct;
     give an encouraging, age-appropriate hint or demonstration; do NOT
     increase difficulty.
  B. Repeated mistakes -- identify the weakness; pick a targeted
     intervention; do NOT repeat the same ineffective intervention; if a
     hint did not help, try a stroke animation or an enlarged template.
  C. Improving handwriting -- positive feedback; do not preemptively
     reduce difficulty; gradually encourage independent writing (free
     writing mode) once the learner has multiple consecutive successes.
  D. Fatigue or frustration -- if either is elevated, prefer a short
     break (the safety layer still caps repeated breaks); do NOT
     recommend a break solely because of one poor score.
  E. Consistent high performance -- progress appropriately; do not give
     unnecessary hints; let the learner practise different letters at
     the same difficulty (the rotation rule in the intervention
     controller handles the actual letter switch).

The tutor NEVER bypasses the safety layer. It returns an action id; the
safety layer still runs after it and may override.

Implementation discipline:
  * No randomness -- deterministic for identical input (the audit
    requires this).
  * Intervention cooldowns and repetition prevention: an intervention is
    not re-selected within the cooldown window unless the situation
    strictly requires it (e.g. a forced break).
  * Effectiveness tracking: each past intervention is labelled
    "effective" / "neutral" / "harmful" by comparing the score AFTER it
    to the score BEFORE it. The tutor avoids interventions labelled
    harmful in the last few attempts.
  * Missing state values are tolerated (e.g. when the state vector has
    NaN-adjacent proxy values); the tutor falls back to neutral defaults
    rather than crashing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import MasteryConfig, SessionDynamicsConfig, settings
from ..dysgraphia_rl_bridge import ACTION_NAMES, N_ACTIONS
from ..state.session_context import SessionContext


# RL state vector indices (kept in sync with dysgraphia_rl.utils.state_utils).
IDX_PREV_SCORE = 0
IDX_STROKE_SIMILARITY = 1
IDX_SPACING = 2
IDX_BASELINE = 3
IDX_WRITING_SPEED = 4
IDX_PRESSURE_VARIANCE = 5
IDX_RETRY_COUNT = 6
IDX_FATIGUE = 7
IDX_ENGAGEMENT = 8
IDX_IMPROVEMENT_TREND = 9
IDX_FRUSTRATION = 10
IDX_CONSISTENCY = 11

# Action ids (must match dysgraphia_rl.envs.dysgraphia_tutor_env).
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

SUPPORTIVE_ACTIONS = (
    DECREASE_DIFFICULTY, SHOW_HINT, ENLARGE_TEMPLATE, REPLAY_STROKE_ANIMATION,
    SWITCH_TO_TRACING_MODE, SUGGEST_BREAK,
)
ESCALATING_ACTIONS = (INCREASE_DIFFICULTY,)


@dataclass
class TutorConfig:
    """Tunable thresholds for the intelligent tutor. Kept simple and
    documented so a clinician / educator can adjust the tutor's
    strictness without editing code."""
    # Score below which an attempt is "incorrect" (very poor shape).
    incorrect_score_threshold: float = 0.35
    # Score above which an attempt is "good".
    good_score_threshold: float = 0.65
    # Number of consecutive fails that triggers a targeted intervention.
    repeated_mistake_threshold: int = 2
    # Number of consecutive successes that triggers progression / free
    # writing encouragement.
    consistent_success_threshold: int = 3
    # Cooldown: do not repeat an intervention within this many attempts
    # UNLESS the situation strictly requires it (e.g. still failing).
    intervention_cooldown: int = 2
    # Window for "recent" interventions to track effectiveness.
    effectiveness_window: int = 4
    # Score improvement needed to label a previous intervention as
    # "effective" (vs neutral / harmful).
    effectiveness_improvement: float = 0.05
    # Fatigue / frustration levels at which a break is preferred.
    fatigue_break_threshold: float = 0.65
    frustration_support_threshold: float = 0.65
    # Score threshold for the "frustration-justified break" decision: a
    # break is suggested ONLY when fatigue / frustration is high AND
    # the recent score history is poor. A single poor score does not
    # justify a break on its own.
    break_score_floor: float = 0.45


@dataclass
class TutorDecision:
    action_id: int
    action_name: str
    reason: str
    effectiveness_labels: dict[int, str] = field(default_factory=dict)


def _safe_state(state: np.ndarray, idx: int, default: float = 0.0) -> float:
    """Read state[idx] tolerating wrong shape / NaN. Never raises."""
    if state is None:
        return default
    try:
        v = float(state[idx])
        if not np.isfinite(v):
            return default
        return v
    except (IndexError, TypeError, ValueError):
        return default


def _label_intervention_effectiveness(session: SessionContext,
                                       window: int,
                                       improvement: float) -> dict[int, str]:
    """For each intervention in the recent action log, label it
    "effective", "neutral" or "harmful" based on the score change
    after it. Returns a dict of action_id -> label (only the most
    recent label per action is kept)."""
    if not session.action_log or len(session.score_history) < 2:
        return {}
    n = len(session.action_log)
    labels: dict[int, str] = {}
    history = session.score_history
    # action_log is 1-indexed relative to score_history: action_log[k]
    # was decided at attempt k (0-indexed), the next attempt's score is
    # score_history[k+1] if it exists.
    for k in range(max(0, n - window), n):
        action_id = session.action_log[k]
        # The score BEFORE the action was the k-th score in history.
        before = history[k] if 0 <= k < len(history) else None
        # The score AFTER the action is the (k+1)-th score.
        after = history[k + 1] if (k + 1) < len(history) else None
        if before is None or after is None:
            labels[action_id] = "neutral"
            continue
        delta = float(after) - float(before)
        if delta >= improvement:
            labels[action_id] = "effective"
        elif delta <= -improvement:
            labels[action_id] = "harmful"
        else:
            labels[action_id] = "neutral"
    return labels


def _in_cooldown(session: SessionContext, action_id: int,
                 cooldown: int) -> bool:
    """Whether the action was selected within the last `cooldown`
    attempts. Returns True if so."""
    if cooldown <= 0 or not session.action_log:
        return False
    recent = session.action_log[-cooldown:]
    return action_id in recent


def _recent_scores(session: SessionContext, n: int) -> list[float]:
    if not session.score_history:
        return []
    return list(session.score_history[-n:])


class IntelligentRuleBasedTutor:
    """A genuinely adaptive rule-based tutor that replaces the brittle
    legacy ``RuleBasedAgent`` for the live backend's fallback policy.

    The tutor is stateless apart from the configuration it holds; all
    session state is read from the :class:`SessionContext` it is given
    per decision, so it is safe to share one instance across sessions.
    """
    name = "intelligent_rule_based"

    def __init__(self, cfg: Optional[TutorConfig] = None,
                 mastery_cfg: Optional[MasteryConfig] = None,
                 session_cfg: Optional[SessionDynamicsConfig] = None):
        self.cfg = cfg or TutorConfig()
        self.mastery_cfg = mastery_cfg or settings.mastery
        self.session_cfg = session_cfg or settings.session_dynamics

    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, session: Optional[SessionContext] = None) -> int:
        """Pick the next action. When ``session`` is None (e.g. the
        simulator calls this with the legacy signature) the tutor
        degrades to the legacy :class:`RuleBasedAgent` behaviour so the
        RL training tests keep passing."""
        if session is None:
            return self._legacy_act(state)
        decision = self.decide(state, session)
        return decision.action_id

    # ------------------------------------------------------------------
    def decide(self, state: np.ndarray, session: SessionContext) -> TutorDecision:
        """Make a context-aware tutoring decision. The decision flow is:

        1. Safety-critical overrides:
             - Very high fatigue -> SUGGEST_BREAK
             - Very high frustration + demanding action -> SHOW_HINT
        2. Incorrect handwriting -> targeted support (hint, demo, larger
           template) and never escalate.
        3. Repeated mistakes -> pick a different supportive action than
           the ones that did not help.
        4. Improving -> CONTINUE_NO_CHANGE (let the learner keep going)
           or SWITCH_TO_FREE_WRITING after a long success run.
        5. Consistent mastery -> INCREASE_DIFFICULTY (the intervention
           controller's rotation rule will rotate to another same-level
           letter, NOT jump to a higher level, until all are mastered).
        6. Default -> CONTINUE_NO_CHANGE.
        """
        cfg = self.cfg

        prev_score = _safe_state(state, IDX_PREV_SCORE, default=0.5)
        fatigue = _safe_state(state, IDX_FATIGUE, default=0.0)
        frustration = _safe_state(state, IDX_FRUSTRATION, default=0.0)
        engagement = _safe_state(state, IDX_ENGAGEMENT, default=0.7)
        improvement = _safe_state(state, IDX_IMPROVEMENT_TREND, default=0.0)
        consistency = _safe_state(state, IDX_CONSISTENCY, default=0.5)
        stroke_sim = _safe_state(state, IDX_STROKE_SIMILARITY, default=0.5)
        retry_count = _safe_state(state, IDX_RETRY_COUNT, default=0.0)

        recent = _recent_scores(session, 5)
        consecutive_fails = session.consecutive_fails
        recent_fail_count = sum(1 for s in recent if s < self.session_cfg.quality_success_threshold)
        recent_success_count = sum(1 for s in recent if s >= cfg.good_score_threshold)
        effectiveness = _label_intervention_effectiveness(
            session, cfg.effectiveness_window, cfg.effectiveness_improvement
        )

        # ----- 1. Safety-critical overrides -------------------------------
        # Forced break: high fatigue OR (high frustration AND poor recent
        # scores). A break is NOT suggested solely because of one poor
        # score -- the recent score history must also be poor.
        if fatigue >= cfg.fatigue_break_threshold:
            return TutorDecision(SUGGEST_BREAK, ACTION_NAMES[SUGGEST_BREAK],
                                  f"fatigue {fatigue:.2f} >= {cfg.fatigue_break_threshold:.2f}",
                                  effectiveness)
        if (frustration >= cfg.frustration_support_threshold
                and recent and max(recent) < cfg.break_score_floor):
            return TutorDecision(SUGGEST_BREAK, ACTION_NAMES[SUGGEST_BREAK],
                                  f"frustration {frustration:.2f} high and recent best "
                                  f"{max(recent):.2f} below {cfg.break_score_floor:.2f}",
                                  effectiveness)

        # ----- 2. Incorrect handwriting ----------------------------------
        if prev_score < cfg.incorrect_score_threshold:
            # The handwriting is malformed. Pick a supportive action
            # that was NOT already tried and labelled harmful in the
            # recent window, and is not in cooldown.
            candidates = [SHOW_HINT, REPLAY_STROKE_ANIMATION, ENLARGE_TEMPLATE,
                          SWITCH_TO_TRACING_MODE]
            for cand in candidates:
                if effectiveness.get(cand) == "harmful":
                    continue
                if _in_cooldown(session, cand, cfg.intervention_cooldown):
                    continue
                return TutorDecision(cand, ACTION_NAMES[cand],
                                      f"incorrect handwriting (score {prev_score:.2f}); "
                                      f"supportive action {ACTION_NAMES[cand]}",
                                      effectiveness)
            # All candidates are in cooldown or harmful -- fall through
            # to a non-escalating CONTINUE rather than praising or
            # escalating.
            return TutorDecision(CONTINUE_NO_CHANGE, ACTION_NAMES[CONTINUE_NO_CHANGE],
                                  f"incorrect handwriting but all supportive actions "
                                  f"in cooldown; staying on the same letter",
                                  effectiveness)

        # ----- 3. Repeated mistakes --------------------------------------
        if consecutive_fails >= cfg.repeated_mistake_threshold or recent_fail_count >= 3:
            # Pick a different supportive action than the most recent
            # one (which clearly did not help). Prefer switching to
            # tracing mode (concrete support) or a demo.
            last_action = session.action_log[-1] if session.action_log else None
            ordered = [SHOW_HINT, REPLAY_STROKE_ANIMATION, ENLARGE_TEMPLATE,
                       SWITCH_TO_TRACING_MODE, DECREASE_DIFFICULTY]
            for cand in ordered:
                if cand == last_action:
                    continue
                if effectiveness.get(cand) == "harmful":
                    continue
                if _in_cooldown(session, cand, cfg.intervention_cooldown):
                    # Allow cooldown override when the situation is bad
                    # enough (recent_fail_count >= 3): we need to do
                    # SOMETHING, even if it was recently tried.
                    if recent_fail_count < 3:
                        continue
                return TutorDecision(cand, ACTION_NAMES[cand],
                                      f"repeated mistakes (consecutive_fails "
                                      f"{consecutive_fails}); switching support to "
                                      f"{ACTION_NAMES[cand]}",
                                      effectiveness)
            return TutorDecision(CONTINUE_NO_CHANGE, ACTION_NAMES[CONTINUE_NO_CHANGE],
                                  "repeated mistakes but no new supportive action available",
                                  effectiveness)

        # ----- 4. Improving handwriting ----------------------------------
        if improvement > 0.10 and prev_score >= cfg.good_score_threshold:
            # The learner is improving and now scoring well. Encourage
            # independent writing once they have a longer success run.
            if recent_success_count >= cfg.consistent_success_threshold \
                    and session.mode == "tracing":
                return TutorDecision(SWITCH_TO_FREE_WRITING_MODE,
                                      ACTION_NAMES[SWITCH_TO_FREE_WRITING_MODE],
                                      f"consistent improvement; encouraging free writing",
                                      effectiveness)
            return TutorDecision(CONTINUE_NO_CHANGE, ACTION_NAMES[CONTINUE_NO_CHANGE],
                                  f"improving steadily; let the learner keep going",
                                  effectiveness)

        # ----- 5. Consistent mastery -------------------------------------
        # The learner has mastered the current letter when EITHER:
        #   (a) the per-letter mastery tracker has marked it mastered
        #       (after `mastery_consecutive_successes_required`
        #       consecutive successes -- configurable in MasteryConfig),
        #   OR
        #   (b) the recent_success_count meets the tutor's own
        #       consistent_success_threshold AND consistency is high.
        # In both cases the tutor picks INCREASE_DIFFICULTY; the
        # intervention controller's rotation rule then turns that into
        # a same-level rotation (NOT a level jump) until every same-
        # level letter is mastered.
        from ..mastery.mastery_tracker import is_letter_mastered
        current_letter_mastered_via_tracker = bool(
            session.current_exercise_id
            and is_letter_mastered(session, session.current_exercise_id, self.mastery_cfg)
        )
        if (current_letter_mastered_via_tracker
                or (recent_success_count >= cfg.consistent_success_threshold
                    and prev_score >= cfg.good_score_threshold
                    and consistency >= 0.7)):
            return TutorDecision(INCREASE_DIFFICULTY, ACTION_NAMES[INCREASE_DIFFICULTY],
                                  f"consistent mastery (current letter mastered or "
                                  f"{recent_success_count} recent successes); progressing",
                                  effectiveness)

        # ----- 6. Default ------------------------------------------------
        # Mild progress / mild struggle -> let the learner keep going.
        # Avoid escalating difficulty when the learner is mid-range.
        if prev_score >= cfg.good_score_threshold:
            return TutorDecision(CONTINUE_NO_CHANGE, ACTION_NAMES[CONTINUE_NO_CHANGE],
                                  f"good score; continuing",
                                  effectiveness)
        # Mid-range score with high frustration -> support.
        if frustration >= cfg.frustration_support_threshold:
            for cand in (SHOW_HINT, ENLARGE_TEMPLATE, REPLAY_STROKE_ANIMATION):
                if _in_cooldown(session, cand, cfg.intervention_cooldown):
                    continue
                return TutorDecision(cand, ACTION_NAMES[cand],
                                      f"high frustration {frustration:.2f}; supportive action",
                                      effectiveness)
        return TutorDecision(CONTINUE_NO_CHANGE, ACTION_NAMES[CONTINUE_NO_CHANGE],
                              f"neutral continuation",
                              effectiveness)

    # ------------------------------------------------------------------
    def _legacy_act(self, state: np.ndarray) -> int:
        """Fallback behaviour that matches the legacy ``RuleBasedAgent``
        when no session context is supplied. Used so the simulator /
        training tests that call ``agent.act(state)`` keep working."""
        fatigue = _safe_state(state, IDX_FATIGUE, default=0.0)
        retries = _safe_state(state, IDX_RETRY_COUNT, default=0.0)
        prev_score = _safe_state(state, IDX_PREV_SCORE, default=0.5)
        if fatigue >= 0.75:
            return SUGGEST_BREAK
        if retries >= 0.6:
            return SHOW_HINT
        if prev_score >= 0.75:
            return INCREASE_DIFFICULTY
        if prev_score <= 0.3:
            return DECREASE_DIFFICULTY
        return CONTINUE_NO_CHANGE


# Module-level singleton, mirroring the policy_service's
# get_policy_service() pattern.
_intelligent_tutor: Optional[IntelligentRuleBasedTutor] = None


def get_intelligent_tutor() -> IntelligentRuleBasedTutor:
    global _intelligent_tutor
    if _intelligent_tutor is None:
        _intelligent_tutor = IntelligentRuleBasedTutor()
    return _intelligent_tutor
