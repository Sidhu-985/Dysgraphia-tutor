"""
Pedagogical layer
==================

Sits between the RL policy's choice and the safety layer. Shapes the
action distribution to be pedagogically diverse and to enforce
session-level tutoring rules that the trained DQN does not learn on
its own (because they depend on session context the DQN's 12-D state
vector only partially encodes).

This layer runs in BOTH modes (DQN live AND rule-based fallback). It
always runs after the policy and before the safety layer. It never
bypasses the safety layer — its decisions are still subject to safety
override.

Motivation (audit-driven, 2026-11)
-----------------------------------

The deployed R12_4 DQN checkpoint (Phase 5 full training, locked
protocol) has a concentrated action distribution: on a 20-attempt
session of clean Level 4 lowercase traces, the DQN picked
INCREASE_DIFFICULTY 65% of the time, SUGGEST_BREAK 20%, and
SWITCH_TO_FREE_WRITING_MODE 15% — only 3 of 10 actions were ever
used. The DQN learned to escalate on good scores, which is what its
training reward shaped, but it is pedagogically too narrow.

The pedagogical layer introduces three session-level rules that
broaden the action distribution and add the specific tutoring
behaviours the audit requires:

  1. **Action diversity safeguard** — if the policy has picked the
     same action K times in the last N decisions (default K=2, N=4),
     the layer replaces the policy's choice with a different
     pedagogically appropriate action. This prevents the DQN from
     over-picking INCREASE_DIFFICULTY on every clean attempt.

  2. **Repeated-failure de-escalation** — if the learner has had
     ``max_attempts_at_letter`` (default 4) attempts at the current
     Level 4 letter without mastering it, the layer overrides the
     policy's choice to DECREASE_DIFFICULTY (drop one level to give
     the learner an easier letter to build confidence). The
     pedagogical rationale: a learner who cannot master a Level 4
     letter after several attempts needs an easier success, not
     more attempts at the same difficulty.

  3. **Post-break progression** — if the previous action was
     SUGGEST_BREAK (i.e., the learner just took a break) AND the
     current attempt's quality_score is at or above the mastery
     threshold, the layer overrides the policy's choice to
     INCREASE_DIFFICULTY. The Level 4 rotation rule in the
     intervention controller then turns this into a same-level
     rotation (move to the next Level 4 letter) if other letters
     remain, or an escalation if all are mastered. The
     pedagogical rationale: after a refreshing break, a successful
     attempt should be rewarded with a new challenge, not with
     more of the same.

These rules are deterministic (no randomness), audited (every
override is logged with a reason), and configurable via
:class:`app.config.PedagogicalLayerConfig`. They never modify the
DQN's checkpoint, the locked protocol, or the safety layer's
thresholds.

The layer's output is a :class:`PedagogicalDecision` containing the
shaped action id, the original (raw) action id, the override reason
(or None if no override), and the rule that fired (or None).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..config import PedagogicalLayerConfig, settings
from ..dysgraphia_rl_bridge import ACTION_NAMES
from ..state.session_context import SessionContext


# Action id constants (must match the canonical order in dysgraphia_rl).
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

# Pedagogically reasonable actions when the learner is doing well.
# Used by the action-diversity safeguard to pick a *different* action
# than the policy's repeated one. Ordered by pedagogical preference:
# the higher in the list, the more appropriate as a "vary the action"
# choice when the learner is succeeding.
PROGRESSIVE_ACTIONS = (
    SHOW_HINT,
    REPLAY_STROKE_ANIMATION,
    ENLARGE_TEMPLATE,
    SWITCH_TO_TRACING_MODE,
    SWITCH_TO_FREE_WRITING_MODE,
)
SUPPORTIVE_ACTIONS = (
    SHOW_HINT,
    REPLAY_STROKE_ANIMATION,
    ENLARGE_TEMPLATE,
    SWITCH_TO_TRACING_MODE,
    DECREASE_DIFFICULTY,
)


@dataclass
class PedagogicalDecision:
    """Result of the pedagogical layer.

    ``raw_action_id`` is what the policy (DQN or rule-based fallback)
    proposed. ``shaped_action_id`` is what the pedagogical layer
    forwards to the safety layer. ``overridden`` is True iff they
    differ. ``reason`` is a short human-readable string explaining
    which rule fired (or None if no override).
    """
    raw_action_id: int
    shaped_action_id: int
    overridden: bool
    reason: Optional[str] = None
    rule: Optional[str] = None  # one of: "diversity", "deescalation", "post_break"

    @property
    def raw_action_name(self) -> str:
        return ACTION_NAMES[self.raw_action_id]

    @property
    def shaped_action_name(self) -> str:
        return ACTION_NAMES[self.shaped_action_id]


def _recent_action_distribution(session: SessionContext, window: int) -> dict[int, int]:
    """Count of each action id in the last `window` actions of the
    session's action_log. Returns a dict of action_id -> count."""
    if not session.action_log:
        return {}
    recent = session.action_log[-window:]
    counts: dict[int, int] = {}
    for a in recent:
        counts[a] = counts.get(a, 0) + 1
    return counts


def _pick_diverse_alternative(repeated_action: int, recent_counts: dict[int, int],
                              feature_dict: dict) -> int:
    """Pick a pedagogically reasonable action different from
    ``repeated_action`` and from the most-recently-used actions.

    Heuristic:
      - If the learner is doing well (quality_score >= mastery threshold),
        pick a progressive action (hint / demo / larger template / mode
        switch) that hasn't been used recently.
      - If the learner is struggling (quality_score < success threshold),
        pick a supportive action (hint / demo / easier / tracing mode)
        that hasn't been used recently.
      - Fallback: CONTINUE_NO_CHANGE if no better option exists.
    """
    quality = float(feature_dict.get("prev_score", 0.5))
    success_threshold = settings.session_dynamics.quality_success_threshold
    mastery_threshold = settings.mastery.mastery_score_threshold

    # Build the set of recently-used actions to avoid.
    recent_actions = set(recent_counts.keys())

    if quality >= mastery_threshold:
        # Learner is doing well: vary with a progressive action.
        for cand in PROGRESSIVE_ACTIONS:
            if cand != repeated_action and cand not in recent_actions:
                return cand
        # All progressive actions have been used recently; pick one
        # different from the repeated one anyway.
        for cand in PROGRESSIVE_ACTIONS:
            if cand != repeated_action:
                return cand
    elif quality < success_threshold:
        # Learner is struggling: pick a supportive action.
        for cand in SUPPORTIVE_ACTIONS:
            if cand != repeated_action and cand not in recent_actions:
                return cand
        for cand in SUPPORTIVE_ACTIONS:
            if cand != repeated_action:
                return cand
    else:
        # Mid-range: just pick something different.
        all_actions = list(PROGRESSIVE_ACTIONS) + [CONTINUE_NO_CHANGE] + list(SUPPORTIVE_ACTIONS)
        for cand in all_actions:
            if cand != repeated_action and cand not in recent_actions:
                return cand
        for cand in all_actions:
            if cand != repeated_action:
                return cand

    return CONTINUE_NO_CHANGE if repeated_action != CONTINUE_NO_CHANGE else SHOW_HINT


def apply_pedagogical_layer(
    proposed_action_id: int,
    session: SessionContext,
    feature_dict: dict,
    cfg: Optional[PedagogicalLayerConfig] = None,
) -> PedagogicalDecision:
    """Apply the three pedagogical rules in priority order:

    1. Repeated-failure de-escalation (highest priority — a stuck
       learner needs an easier success, regardless of what the DQN
       wants to do).
    2. Post-break progression (high priority — fresh start after a
       break + success should move forward, not stay).
    3. Action diversity safeguard (lowest priority — only kicks in
       if the policy has been picking the same action repeatedly).

    The output of this function is forwarded to the safety layer,
    which may further override based on safety rules (forced break,
    escalation block, etc.).
    """
    cfg = cfg or settings.pedagogical_layer
    action = int(proposed_action_id)
    reason: Optional[str] = None
    rule: Optional[str] = None

    # Master switch: when the layer is fully disabled, it's a
    # pass-through (no overrides).
    if not cfg.enabled:
        return PedagogicalDecision(
            raw_action_id=proposed_action_id,
            shaped_action_id=proposed_action_id,
            overridden=False,
        )

    # ----- 1. Repeated-failure de-escalation ------------------------
    # If the learner has had >= max_attempts_at_letter attempts at the
    # current letter without mastering it, override to
    # DECREASE_DIFFICULTY (drop one level).
    if cfg.enable_deescalation_after_repeated_failure:
        from ..mastery.mastery_tracker import is_letter_mastered
        current_eid = session.current_exercise_id
        attempts_at_current = session.retries_on_current_exercise
        # retries_on_current_exercise is reset to 0 on success, so it
        # counts the *current* run of failed attempts at this letter.
        if (current_eid
                and attempts_at_current >= cfg.max_attempts_at_letter
                and not is_letter_mastered(session, current_eid)
                and action != DECREASE_DIFFICULTY):
            action = DECREASE_DIFFICULTY
            reason = (
                f"repeated-failure de-escalation: learner has had "
                f"{attempts_at_current} failed attempts at {current_eid} "
                f"without mastering it (>= {cfg.max_attempts_at_letter}); "
                f"overriding to DECREASE_DIFFICULTY"
            )
            rule = "deescalation"

    # ----- 2. Post-break progression -------------------------------
    # If the previous action was SUGGEST_BREAK and the current
    # attempt scored >= mastery threshold, override to
    # INCREASE_DIFFICULTY (which the Level 4 rotation rule turns
    # into a same-level rotation if other letters remain, or an
    # escalation if all are mastered).
    if (cfg.enable_post_break_progression
            and reason is None  # don't override the de-escalation
            and session.last_action == SUGGEST_BREAK
            and action != INCREASE_DIFFICULTY):
        quality = float(feature_dict.get("prev_score", 0.5))
        if quality >= settings.mastery.mastery_score_threshold:
            action = INCREASE_DIFFICULTY
            reason = (
                f"post-break progression: previous action was SUGGEST_BREAK "
                f"and current quality {quality:.3f} >= mastery threshold "
                f"{settings.mastery.mastery_score_threshold:.2f}; "
                f"overriding to INCREASE_DIFFICULTY (Level 4 rotation rule "
                f"will turn this into a same-level rotation if other "
                f"letters remain)"
            )
            rule = "post_break"

    # ----- 3. Action diversity safeguard ---------------------------
    # If the policy has picked the same action K times in the last N
    # decisions (default K=2 of N=4), pick a pedagogically reasonable
    # alternative.
    if (cfg.enable_diversity_safeguard
            and reason is None  # don't override the higher-priority rules
            and session.action_log):
        recent_counts = _recent_action_distribution(
            session, cfg.diversity_window
        )
        # If the proposed action is the same as the most-recent action
        # AND it has been picked >= K times in the last N decisions,
        # pick a different action.
        most_recent = session.action_log[-1] if session.action_log else None
        if (most_recent is not None
                and action == most_recent
                and recent_counts.get(action, 0) >= cfg.diversity_repeat_threshold):
            # Only override if we're not in the middle of a high-priority
            # sequence (e.g., don't override SUGGEST_BREAK when fatigue is
            # high — the safety layer will handle that).
            fatigue = float(feature_dict.get("fatigue_proxy", 0.0))
            frustration = float(feature_dict.get("frustration", 0.0))
            if (action != SUGGEST_BREAK or fatigue < 0.5) and frustration < 0.7:
                new_action = _pick_diverse_alternative(
                    action, recent_counts, feature_dict
                )
                if new_action != action:
                    action = new_action
                    reason = (
                        f"action diversity safeguard: policy picked "
                        f"{ACTION_NAMES[proposed_action_id]} which has been "
                        f"used {recent_counts[proposed_action_id]} times in "
                        f"the last {cfg.diversity_window} decisions (>= "
                        f"{cfg.diversity_repeat_threshold}); varying to "
                        f"{ACTION_NAMES[new_action]}"
                    )
                    rule = "diversity"

    overridden = action != proposed_action_id
    return PedagogicalDecision(
        raw_action_id=proposed_action_id,
        shaped_action_id=action,
        overridden=overridden,
        reason=reason,
        rule=rule,
    )
