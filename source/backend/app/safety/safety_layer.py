"""
Safety layer
=============
Sits strictly between the RL policy and the intervention controller / UI.
The RL agent NEVER reaches the child directly -- every action it proposes
passes through here first. This module has no notion of Q-values or
training; it only knows the 10 discrete action ids, the session's recent
history, and the configured safety limits (`config.SafetyConfig`).

Responsibilities (project spec Section 10):
  - prevent inappropriate escalation (repeated difficulty increases while
    the child is already struggling/frustrated)
  - prevent excessive repetition (the same action fired over and over)
  - cap session intensity (force a break after too many attempts, or when
    the fatigue proxy is high, regardless of what the policy chose)
  - provide a safe fallback action whenever it overrides the policy

The policy's own choice is always logged (`raw_action_id`) alongside the
possibly-overridden one (`safe_action_id`) so override frequency can be
audited later -- a safety layer that silently overrides is much harder to
evaluate than one that reports its own intervention rate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SafetyConfig, settings
from ..dysgraphia_rl_bridge import ACTION_NAMES
from ..state.session_context import SessionContext

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

ESCALATING_ACTIONS = (INCREASE_DIFFICULTY,)
SUPPORTIVE_FALLBACK = SHOW_HINT


@dataclass
class SafetyDecision:
    raw_action_id: int
    raw_action_name: str
    safe_action_id: int
    safe_action_name: str
    overridden: bool
    reasons: list[str]


DEMANDING_ACTIONS = (
    INCREASE_DIFFICULTY, SWITCH_TO_FREE_WRITING_MODE, REPEAT_SAME_LETTER, CONTINUE_NO_CHANGE,
)
SUPPORTIVE_ACTIONS = (
    DECREASE_DIFFICULTY, SHOW_HINT, ENLARGE_TEMPLATE, REPLAY_STROKE_ANIMATION,
    SWITCH_TO_TRACING_MODE, SUGGEST_BREAK,
)


def _supportive_alternative(avoid: int | None, last_action: int | None) -> int:
    """A supportive action different from `avoid` and from the previous
    action, in a fixed priority order (deterministic, auditable)."""
    for candidate in (SHOW_HINT, DECREASE_DIFFICULTY, ENLARGE_TEMPLATE, REPLAY_STROKE_ANIMATION):
        if candidate != avoid and candidate != last_action:
            return candidate
    return SHOW_HINT


def apply_safety_layer(
    proposed_action_id: int,
    session: SessionContext,
    feature_dict: dict,
    cfg: SafetyConfig | None = None,
) -> SafetyDecision:
    """Rules, in priority order (the first matching rule decides):

    1. attempts-since-break cap          -> SUGGEST_BREAK (forced)
    2. fatigue_proxy >= threshold        -> SUGGEST_BREAK (forced)
    3. consecutive-break cap             -> requested break replaced
    4. high frustration                  -> demanding action replaced by a
                                            supportive one (hint, or an
                                            easier task if a hint was just given)
    5. failed last attempt(s)            -> INCREASE_DIFFICULTY blocked
    6. consecutive-escalation cap        -> INCREASE_DIFFICULTY blocked
    7. same-action repetition cap        -> different action
    Rules 3-5 were added after the Phase A audit (docs/RL_AUDIT_AND_FIXES.md).
    """
    cfg = cfg or settings.safety
    reasons: list[str] = []
    action = proposed_action_id

    fatigue = feature_dict.get("fatigue_proxy", 0.0)
    frustration = feature_dict.get("frustration", 0.0)
    high_frustration = frustration >= cfg.high_frustration_support_threshold

    # 1. Hard session-intensity cap: too many attempts since the last
    #    break. Counts the attempt currently being decided on (+1) so that
    #    `max_attempts_before_forced_break=N` means "force a break on the
    #    Nth attempt", not the (N+1)th.
    if (session.attempts_since_break + 1) >= cfg.max_attempts_before_forced_break:
        action = SUGGEST_BREAK
        reasons.append(
            f"attempts_since_break={session.attempts_since_break} >= "
            f"max_attempts_before_forced_break={cfg.max_attempts_before_forced_break}"
        )

    # 2. Fatigue-triggered forced break overrides everything else.
    elif fatigue >= cfg.fatigue_break_threshold:
        action = SUGGEST_BREAK
        reasons.append(f"fatigue_proxy={fatigue:.3f} >= fatigue_break_threshold={cfg.fatigue_break_threshold}")

    # 3. Consecutive-break cap (policy-requested breaks only; forced breaks
    #    above always win).
    elif action == SUGGEST_BREAK and session.consecutive_breaks >= cfg.max_consecutive_breaks:
        action = _supportive_alternative(SUGGEST_BREAK, session.last_action) if high_frustration \
            else CONTINUE_NO_CHANGE
        reasons.append(
            f"consecutive_breaks={session.consecutive_breaks} >= max_consecutive_breaks="
            f"{cfg.max_consecutive_breaks}; blocked SUGGEST_BREAK"
        )

    # 4. High frustration: never keep (or raise) the demand.
    elif high_frustration and action in DEMANDING_ACTIONS:
        action = SHOW_HINT if session.last_action != SHOW_HINT else DECREASE_DIFFICULTY
        reasons.append(
            f"frustration={frustration:.3f} >= high_frustration_support_threshold="
            f"{cfg.high_frustration_support_threshold}; replaced demanding action "
            f"{ACTION_NAMES[proposed_action_id]} with supportive {ACTION_NAMES[action]}"
        )

    # 5. Never escalate difficulty right after failure(s).
    elif action == INCREASE_DIFFICULTY and session.consecutive_fails >= cfg.block_increase_after_consecutive_fails:
        action = CONTINUE_NO_CHANGE
        reasons.append(
            f"consecutive_fails={session.consecutive_fails} >= block_increase_after_consecutive_fails="
            f"{cfg.block_increase_after_consecutive_fails}; blocked INCREASE_DIFFICULTY"
        )

    # 6. Block repeated escalation runs even without high frustration yet --
    #    a guardrail against the policy ratcheting difficulty up too fast.
    elif action in ESCALATING_ACTIONS and session.consecutive_escalation_count >= cfg.max_consecutive_escalations:
        action = CONTINUE_NO_CHANGE
        reasons.append(
            f"consecutive_escalation_count={session.consecutive_escalation_count} >= "
            f"max_consecutive_escalations={cfg.max_consecutive_escalations}"
        )

    # 7. Block excessive repetition of any single action.
    elif (
        action == session.last_action
        and session.consecutive_same_action_count >= cfg.max_consecutive_same_action
    ):
        if high_frustration:
            action = _supportive_alternative(action, session.last_action)
        else:
            action = CONTINUE_NO_CHANGE if action != CONTINUE_NO_CHANGE else SHOW_HINT
        reasons.append(
            f"action {ACTION_NAMES[proposed_action_id]} repeated "
            f"{session.consecutive_same_action_count} times consecutively >= "
            f"max_consecutive_same_action={cfg.max_consecutive_same_action}"
        )

    overridden = action != proposed_action_id

    # --- update session counters based on the FINAL (possibly overridden) action ---
    if action == session.last_action:
        session.consecutive_same_action_count += 1
    else:
        session.consecutive_same_action_count = 1
    if action in ESCALATING_ACTIONS:
        session.consecutive_escalation_count += 1
    else:
        session.consecutive_escalation_count = 0
    if action == SUGGEST_BREAK:
        session.attempts_since_break = 0
        session.consecutive_breaks += 1
    else:
        session.attempts_since_break += 1
        session.consecutive_breaks = 0
    session.last_action = action

    return SafetyDecision(
        raw_action_id=proposed_action_id,
        raw_action_name=ACTION_NAMES[proposed_action_id],
        safe_action_id=action,
        safe_action_name=ACTION_NAMES[action],
        overridden=overridden,
        reasons=reasons,
    )
