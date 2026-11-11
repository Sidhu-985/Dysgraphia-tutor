"""
Intervention controller
=========================
Converts a (safety-checked) RL action id into a concrete instruction the
tablet UI can render, mutating only the session's *presentational* state
(difficulty level, template scale, mode) -- never the state vector or
reward, which are already finalized by the time this runs. The UI itself
contains zero RL/ML logic; it only ever receives the JSON this module
produces.

The INCREASE_DIFFICULTY action now also applies the Level 4 letter
rotation rule (audit-driven): when the learner has not yet mastered the
current letter, the action does NOT escalate the difficulty; instead the
learner keeps practising the same letter. When the learner has mastered
the current letter AND other eligible same-level letters remain, the
action rotates to a different same-level letter at the SAME difficulty
level (no escalation). Only when every same-level letter is mastered
does INCREASE_DIFFICULTY finally escalate the difficulty by one level.
DECREASE_DIFFICULTY keeps its original behaviour.

Kept as a pure mapping (action id -> Intervention) so it can be extended
or restyled without touching the safety layer, the RL agent, or the
StateBuilder.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from ..config import InterventionConfig, MasteryConfig, settings
from ..state.session_context import SessionContext
from ..exercises import exercise_for_difficulty, levels_in, track_of, CATALOGUE
from ..mastery.mastery_tracker import (
    all_same_level_mastered,
    is_letter_mastered,
    select_next_exercise_with_rotation,
    record_attempt,
    eligible_same_level_exercises,
)

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


@dataclass
class Intervention:
    action_id: int
    action_name: str
    type: str
    message: str
    params: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


def _record_recent_exercise(session: SessionContext, exercise_id: str) -> None:
    """Track recently-practised exercise ids (overall and same-level) so
    the rotation rule avoids bouncing between just two letters."""
    session.recent_exercises.append(exercise_id)
    if len(session.recent_exercises) > 16:
        session.recent_exercises = session.recent_exercises[-16:]
    session.recent_same_level_exercises.append(exercise_id)
    if len(session.recent_same_level_exercises) > 16:
        session.recent_same_level_exercises = session.recent_same_level_exercises[-16:]


def _rotation_message(letter: str, level: int) -> str:
    return f"Great work! Now let's try another letter at the same level: {letter}."


def build_intervention(
    action_id: int,
    action_name: str,
    session: SessionContext,
    cfg: InterventionConfig | None = None,
    mastery_cfg: MasteryConfig | None = None,
) -> Intervention:
    cfg = cfg or settings.intervention
    mastery_cfg = mastery_cfg or settings.mastery

    if action_id in (DECREASE_DIFFICULTY, INCREASE_DIFFICULTY):
        step = -cfg.difficulty_step if action_id == DECREASE_DIFFICULTY else cfg.difficulty_step
        before = session.difficulty_level
        category = track_of(session.current_exercise_id)
        available = levels_in(category)

        if action_id == INCREASE_DIFFICULTY:
            # ----------------------------------------------------------------
            # Level 4 rotation rule (audit-driven). The action's INTENT is
            # "increase difficulty", but the rotation rule may override it.
            # The rule ONLY applies when the current level has MULTIPLE
            # eligible same-level exercises in the same track (e.g.
            # lowercase letters at Level 4: a, b, d, f, g, h, m, n, p, q,
            # r). For tracks with one exercise per level (shapes, words),
            # the rotation rule is a no-op and INCREASE_DIFFICULTY behaves
            # exactly as before (escalate by one level).
            #
            # When the rotation rule applies:
            #   1. If the current letter is NOT yet mastered, ignore the
            #      escalation and stay on the same letter. The learner
            #      needs more practice at this level first.
            #   2. If the current letter IS mastered AND other eligible
            #      same-level letters remain, rotate to one of them at
            #      the SAME difficulty level (do NOT escalate).
            #   3. Only when EVERY eligible same-level letter is mastered
            #      does the action finally escalate the difficulty by one
            #      level (the original behaviour, plus the safety layer's
            #      consecutive-fails check above).
            # ----------------------------------------------------------------
            current_id = session.current_exercise_id or ""
            same_level_alternatives = eligible_same_level_exercises(current_id) \
                if current_id else []
            rotation_applies = bool(same_level_alternatives)

            if rotation_applies:
                current_mastered = is_letter_mastered(session, current_id, mastery_cfg) \
                    if current_id else False

                if not current_mastered:
                    # Case 1: not mastered -- keep practising this letter.
                    next_exercise_id = current_id or exercise_for_difficulty(
                        session.difficulty_level, category).exercise_id
                    if not next_exercise_id:
                        next_exercise_id = session.current_exercise_id or ""
                    _record_recent_exercise(session, next_exercise_id)
                    return Intervention(
                        action_id, action_name, "continue",
                        "Good effort! Let's keep practising this one.",
                        {"difficulty_level": session.difficulty_level,
                         "delta": 0,
                         "requested_delta": step,
                         "next_exercise_id": next_exercise_id,
                         "rotation_reason": "current_letter_not_mastered"},
                    )

                # Current letter mastered: pick the next exercise via the
                # rotation rule. The rule returns either "rotate_same_level"
                # (rotate to another same-level letter) or "all_mastered" /
                # "no_alternatives" (escalate, since nothing else remains at
                # this level).
                next_id, reason = select_next_exercise_with_rotation(
                    session, current_id, current_letter_mastered=True, cfg=mastery_cfg
                )
                if reason == "rotate_same_level":
                    # Case 2: rotate to a different same-level letter. Do NOT
                    # change difficulty_level.
                    rotated = CATALOGUE.get(next_id)
                    if rotated is not None:
                        _record_recent_exercise(session, rotated.exercise_id)
                        return Intervention(
                            action_id, action_name, "letter_rotation",
                            _rotation_message(rotated.target or rotated.exercise_id,
                                              session.difficulty_level),
                            {"difficulty_level": session.difficulty_level,
                             "delta": 0,
                             "requested_delta": step,
                             "next_exercise_id": rotated.exercise_id,
                             "rotation_reason": reason,
                             "previous_exercise_id": current_id},
                        )
                    # If rotation failed for any reason, fall through to
                    # the normal escalation path below.

            # No rotation applies (single exercise per level) OR all
            # same-level letters are mastered -- escalate the difficulty
            # normally. This is the original behaviour.
            target = int(np.clip(before + step, cfg.difficulty_min, cfg.difficulty_max))
            candidates = [lv for lv in available if target <= lv <= cfg.difficulty_max]
            new_level = min(candidates) if candidates else before
            session.difficulty_level = int(new_level)
            changed = session.difficulty_level != before
            next_exercise_id = (exercise_for_difficulty(session.difficulty_level, category).exercise_id
                                if changed else (session.current_exercise_id
                                                 or exercise_for_difficulty(session.difficulty_level, category).exercise_id))
            _record_recent_exercise(session, next_exercise_id)
            if rotation_applies:
                msg = ("Great work on every letter at this level! Let's try a harder one."
                       if changed else "This is already the hardest one. Let's keep practising it.")
                rotation_reason = "all_mastered_escalate" if changed else "no_further_levels"
            else:
                msg = ("Great work! Let's try something a bit harder." if changed
                       else "This is already the hardest one. Let's keep practising it.")
                rotation_reason = "single_exercise_per_level"
            return Intervention(action_id, action_name, "difficulty_adjustment", msg,
                                 {"difficulty_level": session.difficulty_level,
                                  "delta": session.difficulty_level - before,
                                  "requested_delta": step,
                                  "next_exercise_id": next_exercise_id,
                                  "rotation_reason": rotation_reason})

        # DECREASE_DIFFICULTY: original behaviour (no rotation rule).
        target = int(np.clip(before + step, cfg.difficulty_min, cfg.difficulty_max))
        candidates = [lv for lv in available if cfg.difficulty_min <= lv <= target]
        new_level = max(candidates) if candidates else before
        session.difficulty_level = int(new_level)
        changed = session.difficulty_level != before
        next_exercise_id = (exercise_for_difficulty(session.difficulty_level, category).exercise_id
                            if changed else (session.current_exercise_id
                                             or exercise_for_difficulty(session.difficulty_level, category).exercise_id))
        _record_recent_exercise(session, next_exercise_id)
        if action_id == DECREASE_DIFFICULTY:
            msg = ("Let's try something a little easier." if changed
                   else "This is already the easiest one. Let's keep practising it.")
        else:
            msg = ("Great work! Let's try something a bit harder." if changed
                   else "This is already the hardest one. Let's keep practising it.")
        return Intervention(action_id, action_name, "difficulty_adjustment", msg,
                             {"difficulty_level": session.difficulty_level,
                              "delta": session.difficulty_level - before,
                              "requested_delta": step,
                              "next_exercise_id": next_exercise_id})

    if action_id == SHOW_HINT:
        return Intervention(action_id, action_name, "hint",
                             "Start at the green dot and follow the arrows.",
                             {"display_seconds": cfg.hint_display_seconds,
                              "highlight_start_point": True, "show_direction_arrows": True})

    if action_id == ENLARGE_TEMPLATE:
        session.template_scale = float(np.clip(
            session.template_scale * cfg.template_enlarge_factor,
            cfg.template_scale_min, cfg.template_scale_max,
        ))
        return Intervention(action_id, action_name, "template_size",
                             "Making the shape a bit bigger.",
                             {"template_scale": session.template_scale})

    if action_id == REPLAY_STROKE_ANIMATION:
        return Intervention(action_id, action_name, "stroke_animation",
                             "Watch how this one is drawn.",
                             {"replay": True, "animation_seconds": 3})

    if action_id == SWITCH_TO_TRACING_MODE:
        session.mode = "tracing"
        return Intervention(action_id, action_name, "mode_change",
                             "Let's trace it together first.",
                             {"mode": "tracing"})

    if action_id == SWITCH_TO_FREE_WRITING_MODE:
        session.mode = "free_writing"
        return Intervention(action_id, action_name, "mode_change",
                             "Now try writing it on your own.",
                             {"mode": "free_writing"})

    if action_id == SUGGEST_BREAK:
        return Intervention(action_id, action_name, "break",
                             "Great effort! Let's take a short break.",
                             {"duration_seconds": cfg.break_duration_seconds})

    if action_id == REPEAT_SAME_LETTER:
        _record_recent_exercise(session, session.current_exercise_id or "")
        return Intervention(action_id, action_name, "repeat_exercise",
                             "Let's try that one again.",
                             {"exercise_id": session.current_exercise_id})

    # CONTINUE_NO_CHANGE (and safe fallback default)
    if session.current_exercise_id:
        _record_recent_exercise(session, session.current_exercise_id)
    return Intervention(action_id, action_name, "continue",
                         "Keep going, you're doing great!", {})
