"""Pydantic request/response models for the FastAPI backend. The tablet
app (Flutter) only ever exchanges these JSON shapes -- it never sees a
feature dict, a state vector, or a Q-value directly."""

from __future__ import annotations

import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Where a record came from. Kept on every session/attempt/transition so
# real app data, synthetic test traffic, public-dataset replays and model
# outputs are never mixed up. "real_child_study" is refused unless an
# ethics approval is configured (config.StudyConfig).
DataSource = Literal["demo", "synthetic_test", "public_dataset_replay", "real_child_study"]


class StrokePointIn(BaseModel):
    x: float
    y: float
    t: float = Field(..., ge=0.0, description="Seconds since the attempt started (device event time)")
    pressure: Optional[float] = Field(
        None, ge=0.0,
        description="Normalized stylus pressure, or null when the device does not report pressure. "
                    "Never fabricated by the client.")
    stroke_id: int = Field(..., ge=0)
    is_stroke_start: bool = False
    is_stroke_end: bool = False

    @field_validator("x", "y", "t", "pressure")
    @classmethod
    def _finite(cls, v):
        if v is not None and not math.isfinite(v):
            raise ValueError("coordinates, timestamps and pressure must be finite numbers")
        return v


class CaptureMetadata(BaseModel):
    """Optional client-side capture facts, logged for data-provenance
    checks. Nothing here feeds the state vector."""
    pointer_kind: Optional[Literal["stylus", "touch", "mouse", "unknown"]] = None
    pressure_supported: Optional[bool] = None
    canvas_width: Optional[float] = None
    canvas_height: Optional[float] = None
    retry_index: Optional[int] = Field(None, ge=0)
    completed: Optional[bool] = None
    client_version: Optional[str] = None


class AttemptSubmitRequest(BaseModel):
    session_id: str
    exercise_id: str
    attempt_id: str
    points: list[StrokePointIn]
    capture: Optional[CaptureMetadata] = None


class SessionStartRequest(BaseModel):
    child_id: str = Field(..., description="Pseudonymous learner code, never a real name")
    exercise_id: Optional[str] = None
    data_source: DataSource = "demo"
    # Optional starting practice mode (2026-11-08, backward compatible; the
    # default stays "tracing"). "free_writing" lets a child who chose
    # "write it myself" start without the guide; the policy may still switch
    # modes later through its own actions.
    practice_mode: Optional[Literal["tracing", "free_writing"]] = None


class ExerciseOut(BaseModel):
    exercise_id: str
    prompt: str
    difficulty: int
    reference_template: list[list[float]]
    # Added for child-facing instructions (backward compatible: optional, and
    # older clients ignore them). category: "shape" | "letter" | "word";
    # target: the exact shape name, character or word; template_strokes: the
    # guide as separate pen strokes (reference_template is their concatenation).
    category: Optional[str] = None
    target: Optional[str] = None
    template_strokes: Optional[list[list[list[float]]]] = None
    # letters/words: {"ascender", "x_height", "baseline", "descender"} in template units
    writing_lines: Optional[dict[str, float]] = None
    # letters only (2026-11-08, backward compatible): "upper" | "lower";
    # "capital" | "ascender" | "x_height" | "descender"; child-facing sentence
    letter_case: Optional[str] = None
    letter_zone: Optional[str] = None
    instruction: Optional[str] = None


class SessionStartResponse(BaseModel):
    session_id: str
    child_id: str
    exercise: ExerciseOut
    data_source: str = "demo"
    policy_mode: Optional[str] = None
    practice_mode: str = "tracing"


class SafetyOut(BaseModel):
    raw_action_id: int
    raw_action_name: str
    safe_action_id: int
    safe_action_name: str
    overridden: bool
    reasons: list[str]


class InterventionOut(BaseModel):
    action_id: int
    action_name: str
    type: str
    message: str
    params: dict


class AttemptSubmitResponse(BaseModel):
    session_id: str
    attempt_id: str
    exercise_id: str
    quality_score: float
    static_features: dict
    dynamic_features: dict
    state_vector: list[float]
    state_feature_names: list[str]
    safety: SafetyOut
    intervention: InterventionOut
    # Attempt-level reward (reward version 2 terms evaluated at THIS
    # attempt). "delta_quality_term" keeps its historical key name but
    # holds w1 * (q_t - q_first): cumulative improvement relative to the
    # session's first attempt.
    reward: float
    reward_components: dict
    next_exercise: ExerciseOut
    session_summary: dict
    # RL transition completed by this attempt: the reward credited to the
    # PREVIOUS decision (includes the unnecessary-break penalty). Null on
    # the first attempt of a session.
    transition: Optional[dict] = None
    policy: Optional[dict] = None
    presentation: Optional[dict] = None
    # Who actually made this decision (2026-11-08, backward compatible):
    # policy_mode is the service mode ("dqn" | "rule_based_fallback");
    # decision_source is the source of THIS decision before the safety
    # layer ("dqn", "rule_based_fallback", "safe_default_invalid_state",
    # "rule_based_after_inference_error"); safety_overridden says whether
    # the independent safety layer then replaced it.
    policy_mode: Optional[str] = None
    decision_source: Optional[str] = None
    safety_overridden: Optional[bool] = None
    # True when this response replays an earlier one: the client resent an
    # attempt_id with identical points (e.g. a network retry). The attempt
    # was NOT processed a second time.
    duplicate_submission: bool = False
    # Handwriting score breakdown (audit-driven, 2026-11). Contains the
    # learner-facing 0-100 score and the per-component breakdown (shape
    # similarity, stroke count, aspect ratio, curvature, direction,
    # baseline alignment, optional pressure stability) used internally
    # by the mastery tracker and exposed for an optional therapist /
    # debug view. The learner only ever sees the rounded "score" field.
    handwriting_score: Optional[dict] = None
    # Per-letter mastery snapshot (audit-driven, 2026-11). Tells the
    # Flutter app which letters at the current difficulty level have been
    # mastered and how many remain, so the UI can show progress without
    # exposing internal policy labels.
    letter_mastery: Optional[dict] = None
    # Pedagogical layer decision (audit-driven, 2026-11). Reports what
    # the pedagogical layer did to the policy's choice before the
    # safety layer ran. Contains the raw policy action id, the shaped
    # action id (after pedagogical rules), whether it was overridden,
    # the rule that fired ("diversity", "deescalation", "post_break",
    # or None), and a human-readable reason. Audit-only; not shown to
    # the learner.
    pedagogical: Optional[dict] = None


class SessionSummaryResponse(BaseModel):
    session_id: str
    child_id: str
    attempt_count: int
    score_history: list[float]
    fatigue: float
    frustration: float
    engagement: float
    difficulty_level: int
    ended: bool
    break_count: int = 0
    data_source: str = "demo"
    current_exercise_id: Optional[str] = None
    # Audit-driven (2026-11): per-letter mastery snapshot, exposed so a
    # supervisor / progress view can show how many letters at each
    # level the learner has mastered. Internal policy labels (decision
    # source, etc.) are NOT included here -- the AttemptSubmitResponse
    # keeps those for developer/audit use, but the summary stays
    # learner-facing.
    letter_mastery: Optional[dict] = None
    recent_exercises: Optional[list[str]] = None
