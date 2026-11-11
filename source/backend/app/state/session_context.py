"""
SessionContext: the per-session rolling state needed to turn a single
attempt's static/dynamic features into the 12-d RL state (fatigue,
engagement, frustration, and quality history don't exist within a single
attempt -- they accumulate across a session). Pure data + no dependency on
the UI, the RL agent, or the feature extractors themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionContext:
    session_id: str
    child_id: str
    attempt_count: int = 0
    score_history: list[float] = field(default_factory=list)
    fatigue: float = 0.05
    frustration: float = 0.05
    engagement: float = 0.7
    consecutive_fails: int = 0
    current_exercise_id: Optional[str] = None
    retries_on_current_exercise: int = 0
    action_log: list[int] = field(default_factory=list)
    difficulty_level: int = 2
    ended: bool = False
    attempts_since_break: int = 0
    last_action: Optional[int] = None
    consecutive_same_action_count: int = 0
    consecutive_escalation_count: int = 0
    template_scale: float = 1.0
    mode: str = "tracing"
    consecutive_breaks: int = 0
    break_count: int = 0
    data_source: str = "demo"
    # Pending RL transition: the state and (safe) action decided at the
    # previous attempt, waiting for the NEXT observed attempt to supply
    # its reward and next_state. Never filled with invented outcomes.
    pending_state: Optional[list] = None
    pending_action: Optional[int] = None
    pending_raw_action: Optional[int] = None
    pending_fatigue_at_decision: Optional[float] = None
    pending_attempt_index: Optional[int] = None
    # Idempotency for client retries (Phase G): attempt_id -> (payload
    # hash, response). Kept in memory only and NOT included in to_dict().
    processed_attempts: dict = field(default_factory=dict)
    # Per-letter mastery state (audit-driven addition for Level 4
    # rotation): exercise_id -> LetterMastery. Mutated by
    # app.mastery.mastery_tracker.record_attempt after every attempt.
    # Owned by the session so it is naturally per-session (one learner's
    # mastery of "a" does not leak into another learner's session).
    letter_mastery: dict = field(default_factory=dict)
    # Most recently practised exercise ids (audit-driven). Used by the
    # rotation rule to avoid bouncing between just two letters.
    recent_exercises: list[str] = field(default_factory=list)
    # Most recently practised exercise ids WITHIN the current same-level
    # band. Used by select_next_exercise_with_rotation to skip the last
    # N letters when picking the next one.
    recent_same_level_exercises: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "child_id": self.child_id,
            "attempt_count": self.attempt_count,
            "score_history": self.score_history,
            "fatigue": self.fatigue,
            "frustration": self.frustration,
            "engagement": self.engagement,
            "consecutive_fails": self.consecutive_fails,
            "current_exercise_id": self.current_exercise_id,
            "retries_on_current_exercise": self.retries_on_current_exercise,
            "action_log": self.action_log,
            "difficulty_level": self.difficulty_level,
            "ended": self.ended,
            "attempts_since_break": self.attempts_since_break,
            "last_action": self.last_action,
            "consecutive_same_action_count": self.consecutive_same_action_count,
            "consecutive_escalation_count": self.consecutive_escalation_count,
            "template_scale": self.template_scale,
            "mode": self.mode,
            "consecutive_breaks": self.consecutive_breaks,
            "break_count": self.break_count,
            "data_source": self.data_source,
            "pending_state": self.pending_state,
            "pending_action": self.pending_action,
            "pending_raw_action": self.pending_raw_action,
            "pending_fatigue_at_decision": self.pending_fatigue_at_decision,
            "pending_attempt_index": self.pending_attempt_index,
            "letter_mastery": {k: v.to_dict() if hasattr(v, "to_dict") else v
                                for k, v in self.letter_mastery.items()},
            "recent_exercises": list(self.recent_exercises),
            "recent_same_level_exercises": list(self.recent_same_level_exercises),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SessionContext":
        # Backwards-compatible deserialization: tolerate older dicts that
        # predate letter_mastery / recent_exercises (added by the audit-
        # driven Level 4 rotation patch). LetterMastery objects are
        # reconstructed lazily on first access via the mastery tracker.
        letter_mastery_raw = d.get("letter_mastery") or {}
        # Defer LetterMastery reconstruction: keep raw dicts here and let
        # the mastery tracker rehydrate them on first use.
        return cls(
            session_id=d["session_id"],
            child_id=d["child_id"],
            attempt_count=d.get("attempt_count", 0),
            score_history=list(d.get("score_history", [])),
            fatigue=d.get("fatigue", 0.05),
            frustration=d.get("frustration", 0.05),
            engagement=d.get("engagement", 0.7),
            consecutive_fails=d.get("consecutive_fails", 0),
            current_exercise_id=d.get("current_exercise_id"),
            retries_on_current_exercise=d.get("retries_on_current_exercise", 0),
            action_log=list(d.get("action_log", [])),
            difficulty_level=d.get("difficulty_level", 2),
            ended=d.get("ended", False),
            attempts_since_break=d.get("attempts_since_break", 0),
            last_action=d.get("last_action"),
            consecutive_same_action_count=d.get("consecutive_same_action_count", 0),
            consecutive_escalation_count=d.get("consecutive_escalation_count", 0),
            template_scale=d.get("template_scale", 1.0),
            mode=d.get("mode", "tracing"),
            consecutive_breaks=d.get("consecutive_breaks", 0),
            break_count=d.get("break_count", 0),
            data_source=d.get("data_source", "demo"),
            pending_state=d.get("pending_state"),
            pending_action=d.get("pending_action"),
            pending_raw_action=d.get("pending_raw_action"),
            pending_fatigue_at_decision=d.get("pending_fatigue_at_decision"),
            pending_attempt_index=d.get("pending_attempt_index"),
            letter_mastery=letter_mastery_raw,
            recent_exercises=list(d.get("recent_exercises", [])),
            recent_same_level_exercises=list(d.get("recent_same_level_exercises", [])),
        )
