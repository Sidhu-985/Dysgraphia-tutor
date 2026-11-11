"""
API routes -- the FastAPI surface the Flutter tablet app talks to over
REST. This file wires the pipeline stages together in one place
(preprocessing -> features -> state -> policy -> safety -> intervention ->
reward/transition logging) but implements none of them itself; every stage
is a call into an independently testable module.

Transition integrity (docs/RL_AUDIT_AND_FIXES.md, section "Live
transitions"): the decision made at attempt t is credited with the reward
and next_state observed at attempt t+1. The pending (state, action) pair is
held on the session until then. When a session ends before another attempt,
the last decision is logged with reward NULL and outcome_observed=0. No
outcome is ever invented.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from fastapi import APIRouter, HTTPException

from ..config import settings
from ..exercises import CATALOGUE, DEFAULT_EXERCISE_ID, get_exercise
from ..features.static_features import extract_static_features
from ..features.dynamic_features import extract_dynamic_features
from ..state.state_builder import build_state
from ..rl_integration.policy_service import get_policy_service
from ..safety.safety_layer import apply_safety_layer, SUGGEST_BREAK
from ..interventions.intervention_controller import build_intervention
from ..dysgraphia_rl_bridge import RewardWeights, compute_reward, unnecessary_break_penalty
from ..mastery.mastery_tracker import (
    is_letter_mastered, level_mastered_count, record_attempt,
    all_same_level_mastered, eligible_same_level_exercises,
)
from ..storage import get_session_store, get_attempt_log_store
from .deps import points_to_strokes, require_session
from ..schemas import (
    SessionStartRequest, SessionStartResponse, ExerciseOut,
    AttemptSubmitRequest, AttemptSubmitResponse, SafetyOut, InterventionOut,
    SessionSummaryResponse,
)

router = APIRouter()


def _exercise_out(exercise_id: str) -> ExerciseOut:
    ex = get_exercise(exercise_id)
    return ExerciseOut(exercise_id=ex.exercise_id, prompt=ex.prompt,
                        difficulty=ex.difficulty, reference_template=ex.reference_template,
                        category=ex.category, target=ex.target, template_strokes=ex.strokes,
                        writing_lines=ex.writing_lines, letter_case=ex.letter_case,
                        letter_zone=ex.letter_zone, instruction=ex.instruction)


def _reward_weights() -> RewardWeights:
    """Reward version 2 weights (same defaults as the simulator), with
    optional REWARD_* environment overrides."""
    override = settings.reward
    defaults = RewardWeights.v2()
    names = ("w1_quality", "w2_completion", "w3_engagement", "w4_frustration",
             "w5_consistency", "w6_unnecessary_break")
    return RewardWeights(**{
        n: getattr(override, n) if getattr(override, n) is not None else getattr(defaults, n)
        for n in names
    })


def _validate_learner_id(child_id: str) -> None:
    if not re.fullmatch(settings.study.learner_id_pattern, child_id or ""):
        raise HTTPException(
            status_code=422,
            detail="child_id must be a pseudonymous code (3-40 letters, digits, '_' or '-', "
                   "e.g. 'P-0042'); do not use a child's name or e-mail address.")


def _policy_info(decision=None) -> dict:
    status = get_policy_service().status()
    info = {"mode": status["policy_mode"], "dqn_making_decisions": status["dqn_making_decisions"],
            "fallback_reasons": status["fallback_reasons"]}
    if decision is not None:
        info["source"] = decision.source
        info["notes"] = decision.notes
    return info


@router.get("/health")
def health():
    policy = get_policy_service()
    status = policy.status()
    return {"status": "ok",
            "rl_checkpoint_loaded": policy.checkpoint_loaded,
            "rl_checkpoint_path": policy.checkpoint_path,
            "rl_checkpoint_sha256": status["checkpoint_sha256"],
            "policy_mode": status["policy_mode"],
            "dqn_making_decisions": status["dqn_making_decisions"],
            "behavioral_gate_enabled": status["behavioral_gate_enabled"],
            "behavioral_gate_passed": status["behavioral_gate_passed"],
            "policy": status,
            "real_child_collection_enabled": bool(
                settings.study.real_child_collection_approved and settings.study.ethics_approval_reference)}


@router.get("/exercises", response_model=list[ExerciseOut])
def list_exercises():
    return [_exercise_out(eid) for eid in CATALOGUE]


@router.get("/exercise/{exercise_id}", response_model=ExerciseOut)
def get_exercise_route(exercise_id: str):
    try:
        return _exercise_out(exercise_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown exercise_id: {exercise_id}")


@router.post("/session/start", response_model=SessionStartResponse)
def start_session(req: SessionStartRequest):
    _validate_learner_id(req.child_id)
    if req.data_source == "real_child_study" and not (
            settings.study.real_child_collection_approved and settings.study.ethics_approval_reference):
        raise HTTPException(
            status_code=403,
            detail="Real-child data collection is disabled: it requires institutional ethics approval, "
                   "consent and safeguards (set STUDY_REAL_CHILD_COLLECTION_APPROVED and "
                   "STUDY_ETHICS_APPROVAL_REFERENCE only once these are in place).")
    exercise_id = req.exercise_id or DEFAULT_EXERCISE_ID
    try:
        exercise = _exercise_out(exercise_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown exercise_id: {exercise_id}")
    store = get_session_store()
    session = store.create(child_id=req.child_id)
    session.current_exercise_id = exercise.exercise_id
    # Keep the policy-facing difficulty level consistent with what the
    # child is actually shown (previously always 2 regardless of exercise).
    session.difficulty_level = exercise.difficulty
    session.data_source = req.data_source
    if req.practice_mode is not None:
        session.mode = req.practice_mode
    store.save(session)
    return SessionStartResponse(session_id=session.session_id, child_id=session.child_id,
                                exercise=exercise, data_source=session.data_source,
                                policy_mode=get_policy_service().mode, practice_mode=session.mode)


def _close_pending_transition(session) -> None:
    if session.pending_state is None:
        return
    get_attempt_log_store().log_transition({
        "session_id": session.session_id, "child_id": session.child_id,
        "data_source": session.data_source,
        "decision_attempt_index": session.pending_attempt_index, "outcome_attempt_index": None,
        "state": session.pending_state, "action": session.pending_action,
        "raw_action": session.pending_raw_action, "reward": None, "reward_components": None,
        "next_state": None, "done": True, "outcome_observed": False,
    })
    session.pending_state = None
    session.pending_action = None
    session.pending_raw_action = None
    session.pending_fatigue_at_decision = None
    session.pending_attempt_index = None


@router.post("/session/{session_id}/end")
def end_session(session_id: str):
    store = get_session_store()
    session = require_session(store, session_id)
    _close_pending_transition(session)
    session.ended = True
    store.save(session)
    return {"session_id": session_id, "ended": True}


@router.get("/session/{session_id}/summary", response_model=SessionSummaryResponse)
def session_summary(session_id: str):
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    # Audit-driven (2026-11): include the per-letter mastery snapshot
    # so a progress view can show how many letters at each level the
    # learner has mastered. Internal policy labels are NOT exposed
    # here; this dict only contains the mastery info itself.
    letter_mastery_dict = {
        eid: (v.to_dict() if hasattr(v, "to_dict") else v)
        for eid, v in session.letter_mastery.items()
    }
    return SessionSummaryResponse(
        session_id=session.session_id, child_id=session.child_id,
        attempt_count=session.attempt_count, score_history=session.score_history,
        fatigue=session.fatigue, frustration=session.frustration, engagement=session.engagement,
        difficulty_level=session.difficulty_level, ended=session.ended,
        break_count=session.break_count, data_source=session.data_source,
        current_exercise_id=session.current_exercise_id,
        letter_mastery=letter_mastery_dict,
        recent_exercises=list(session.recent_exercises),
    )


@router.get("/session/{session_id}/transitions")
def session_transitions(session_id: str):
    """Logged RL transitions for one session (research/audit use)."""
    store = get_session_store()
    if store.get(session_id) is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return get_attempt_log_store().get_session_transitions(session_id)


@router.post("/session/{session_id}/break_complete")
def break_complete(session_id: str):
    """The tablet calls this once the child finishes a break, so the
    session's fatigue/frustration recovery is applied without running the
    full feature-extraction pipeline (no stroke data for a break)."""
    from ..state.state_builder import update_session_dynamics
    from ..features.dynamic_features import DynamicFeatures

    store = get_session_store()
    session = require_session(store, session_id)
    empty_dynamic = DynamicFeatures(
        writing_speed=1.0, velocity_variation=0.0, acceleration_rms=None,
        acceleration_reliable=False, movement_smoothness=0.5, pause_count=0,
        pause_total_s=0.0, pause_mean_s=0.0, stroke_duration_s=0.0,
        inter_stroke_interval_mean_s=0.0, pressure_mean=0.5,
        pressure_variance_normalized=0.3, pressure_available=False,
        retry_count_normalized=0.0, temporal_consistency=0.5,
    )
    update_session_dynamics(session, quality=0.0, dynamic=empty_dynamic, break_taken=True)
    session.attempts_since_break = 0
    session.break_count += 1
    store.save(session)
    return {"session_id": session_id, "fatigue": session.fatigue,
            "frustration": session.frustration, "engagement": session.engagement,
            "break_count": session.break_count}


_STATE_FEEDING_STATIC = ("stroke_similarity", "spacing_accuracy", "baseline_alignment")
_STATE_FEEDING_DYNAMIC = ("writing_speed", "pressure_variance_normalized", "retry_count_normalized",
                          "pause_total_s")


def _require_finite_features(static_feats, dynamic_feats) -> None:
    """Reject the attempt BEFORE any session state is mutated if a feature
    that feeds the state vector / quality proxy is not finite (e.g. a
    degenerate trace). A NaN must never enter score_history."""
    import math
    bad = [n for n in _STATE_FEEDING_STATIC if not math.isfinite(getattr(static_feats, n))]
    bad += [n for n in _STATE_FEEDING_DYNAMIC if not math.isfinite(getattr(dynamic_feats, n))]
    if bad:
        raise HTTPException(status_code=422,
                            detail=f"could not extract valid features from this attempt ({', '.join(bad)}); "
                                   "please try again")


@router.post("/attempt/submit", response_model=AttemptSubmitResponse)
def submit_attempt(req: AttemptSubmitRequest):
    store = get_session_store()
    session = require_session(store, req.session_id)

    try:
        exercise = get_exercise(req.exercise_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown exercise_id: {req.exercise_id}")

    if not req.points:
        raise HTTPException(status_code=422, detail="attempt has no stroke points")

    # Idempotent resubmission (Phase G). The same attempt_id with the same
    # points replays the stored response without processing the attempt
    # again, so a network retry cannot count an attempt twice or add a second
    # transition. The same attempt_id with DIFFERENT points is refused.
    payload_hash = hashlib.sha256(json.dumps(
        [p.model_dump() for p in req.points] + [req.exercise_id], sort_keys=True).encode()).hexdigest()
    previous = session.processed_attempts.get(req.attempt_id)
    if previous is not None:
        if previous[0] != payload_hash:
            raise HTTPException(status_code=409, detail=f"attempt_id {req.attempt_id} was already used for "
                                                        "a different attempt")
        return AttemptSubmitResponse(**{**previous[1], "duplicate_submission": True})

    strokes = points_to_strokes(req.points)

    if req.exercise_id != session.current_exercise_id:
        # The child (or the UI's exercise picker) switched exercise.
        session.retries_on_current_exercise = 0
        session.current_exercise_id = req.exercise_id
        session.difficulty_level = exercise.difficulty
    retries_before_this_attempt = session.retries_on_current_exercise

    static_feats = extract_static_features(strokes, reference_template=exercise.reference_array())
    dynamic_feats = extract_dynamic_features(strokes, retries_so_far=retries_before_this_attempt)
    _require_finite_features(static_feats, dynamic_feats)

    # Build state using the EXERCISE-AWARE multiplicative scorer (audit-
    # driven). The scorer combines DTW shape similarity with structural
    # checks (stroke count, aspect ratio, curvature, direction) so a
    # malformed shape cannot earn a moderately high score based on
    # unrelated features. Falls back to the legacy reference-free proxy
    # only if a scorer bug raises (which is caught inside build_state).
    result = build_state(
        static_feats, dynamic_feats, session, break_taken=False,
        strokes=strokes,
        reference_template=exercise.reference_array(),
        reference_strokes=exercise.strokes,
    )
    success = result.quality_score >= settings.session_dynamics.quality_success_threshold
    session.retries_on_current_exercise = 0 if success else session.retries_on_current_exercise + 1

    # Record per-letter mastery for the Level 4 rotation rule (audit-
    # driven). record_attempt is called EXACTLY ONCE per attempt, after
    # the quality score is finalized, so a repeated API submission
    # (handled by the idempotent path above) cannot advance mastery
    # twice.
    record_attempt(session, req.exercise_id, result.quality_score)

    # --- attempt-level reward terms (reward version 2) ---
    weights = _reward_weights()
    quality_improvement = result.quality_score - session.score_history[0]
    reward_components = {
        "delta_quality_term": weights.w1_quality * quality_improvement,
        "completion_term": weights.w2_completion * 1.0,
        "engagement_term": weights.w3_engagement * result.feature_dict["engagement"],
        "frustration_term": -weights.w4_frustration * result.feature_dict["frustration"],
        "consistency_term": weights.w5_consistency * result.feature_dict["consistency"],
    }
    reward = compute_reward(
        delta_quality=quality_improvement, completion=1.0,
        engagement=result.feature_dict["engagement"],
        frustration=result.feature_dict["frustration"],
        consistency=result.feature_dict["consistency"],
        weights=weights,
    )

    # --- complete the PREVIOUS decision's transition with this outcome ---
    log_store = get_attempt_log_store()
    transition_out = None
    if session.pending_state is not None:
        penalty = unnecessary_break_penalty(
            session.pending_action == SUGGEST_BREAK, session.pending_fatigue_at_decision or 0.0,
            settings.reward.break_fatigue_threshold)
        t_components = dict(reward_components)
        t_components["unnecessary_break_term"] = -weights.w6_unnecessary_break * penalty
        t_reward = reward - weights.w6_unnecessary_break * penalty
        log_store.log_transition({
            "session_id": session.session_id, "child_id": session.child_id,
            "data_source": session.data_source,
            "decision_attempt_index": session.pending_attempt_index,
            "outcome_attempt_index": session.attempt_count,
            "state": session.pending_state, "action": session.pending_action,
            "raw_action": session.pending_raw_action, "reward": t_reward,
            "reward_components": t_components, "next_state": result.state_vector.tolist(),
            "done": False, "outcome_observed": True,
        })
        transition_out = {"credited_action_id": session.pending_action,
                          "decision_attempt_index": session.pending_attempt_index,
                          "outcome_attempt_index": session.attempt_count,
                          "reward": t_reward, "reward_components": t_components}

    # --- decide the next intervention ---
    policy = get_policy_service()
    # Pass the session so the intelligent rule-based tutor (used when
    # the DQN is gated out) can see the full session context: score
    # history, fatigue, frustration, previous interventions, current
    # letter / level. The behavioural gate is NOT bypassed; the
    # decision source still reports "rule_based_fallback" when the
    # checkpoint is gated out.
    decision = policy.select_action(result.state_vector, session=session)

    # --- pedagogical layer (audit-driven, 2026-11) ---
    # Sits between the policy decision and the safety layer. Shapes
    # the action distribution to be pedagogically diverse and enforces
    # session-level tutoring rules the trained DQN does not learn on
    # its own:
    #   1. Repeated-failure de-escalation (>= N failed attempts at a
    #      Level 4 letter -> DECREASE_DIFFICULTY).
    #   2. Post-break progression (after a break + good score ->
    #      INCREASE_DIFFICULTY, which the Level 4 rotation rule turns
    #      into a same-level rotation if other letters remain).
    #   3. Action diversity safeguard (same action K of last N ->
    #      pick a pedagogically reasonable alternative).
    # The layer never modifies the DQN checkpoint or the locked
    # protocol; it only shapes which action id the safety layer sees.
    from ..rl_integration.pedagogical_layer import apply_pedagogical_layer
    pedagogical_decision = apply_pedagogical_layer(
        decision.action_id, session, result.feature_dict
    )
    shaped_action_id = pedagogical_decision.shaped_action_id

    safety_decision = apply_safety_layer(shaped_action_id, session, result.feature_dict)
    session.action_log.append(safety_decision.safe_action_id)

    intervention = build_intervention(
        safety_decision.safe_action_id, safety_decision.safe_action_name, session
    )
    next_exercise_id = intervention.params.get("next_exercise_id", session.current_exercise_id)
    if next_exercise_id != session.current_exercise_id:
        session.current_exercise_id = next_exercise_id
        session.retries_on_current_exercise = 0

    session.pending_state = result.state_vector.tolist()
    session.pending_action = safety_decision.safe_action_id
    session.pending_raw_action = safety_decision.raw_action_id
    session.pending_fatigue_at_decision = float(result.feature_dict["fatigue_proxy"])
    session.pending_attempt_index = session.attempt_count
    store.save(session)

    log_store.log_attempt({
        "session_id": session.session_id, "child_id": session.child_id,
        "exercise_id": req.exercise_id, "attempt_index": session.attempt_count,
        "created_at": time.time(),
        "static_features": static_feats.to_dict(), "dynamic_features": dynamic_feats.to_dict(),
        "state_vector": result.state_vector.tolist(), "quality_score": result.quality_score,
        "raw_action_id": safety_decision.raw_action_id, "safe_action_id": safety_decision.safe_action_id,
        "safety_overridden": safety_decision.overridden, "safety_reasons": safety_decision.reasons,
        "reward": reward, "reward_components": reward_components,
        "data_source": session.data_source, "policy_source": decision.source,
        "pressure_available": dynamic_feats.pressure_available,
        "n_points": len(req.points), "n_strokes": len(strokes),
        "raw_points": [p.model_dump() for p in req.points] if settings.storage.store_raw_points else None,
    })

    # Per-letter mastery snapshot for the UI (audit-driven): lets the
    # Flutter app show the learner which letters they have mastered and
    # how many remain at the current level. Thelearner-facing UI MUST
    # NOT show internal policy labels (rule_based_fallback etc.) -- it
    # only ever shows this mastery info and the child-friendly message.
    next_ex = get_exercise(session.current_exercise_id)
    track = next_ex.category if next_ex.category != "letter" else f"letter_{next_ex.letter_case}"
    mastered_count, total_at_level = level_mastered_count(session, session.difficulty_level, track)
    eligible = eligible_same_level_exercises(session.current_exercise_id)
    letter_mastery_snapshot = {
        "current_exercise_id": session.current_exercise_id,
        "current_letter": next_ex.target if next_ex else None,
        "current_difficulty_level": session.difficulty_level,
        "current_letter_mastered": is_letter_mastered(session, session.current_exercise_id),
        "mastered_count_at_level": mastered_count,
        "total_at_level": total_at_level,
        "remaining_at_level": max(0, total_at_level - mastered_count),
        "next_exercise_id": session.current_exercise_id,
        "next_letter": next_ex.target if next_ex else None,
        "all_at_level_mastered": mastered_count >= total_at_level and total_at_level > 0,
    }

    # Handwriting score breakdown (audit-driven). The "score" field is
    # the learner-facing 0-100 score; the "breakdown" is for an optional
    # therapist / debug view and is NOT shown to the learner.
    handwriting_breakdown = result.handwriting_score.to_dict() if result.handwriting_score else None

    # Pedagogical layer decision (audit-only). Reports what the
    # pedagogical layer did to the policy's choice before the safety
    # layer ran. The learner-facing UI does not show this; it's for
    # the supervisor / audit view.
    pedagogical_info = {
        "raw_action_id": pedagogical_decision.raw_action_id,
        "raw_action_name": pedagogical_decision.raw_action_name,
        "shaped_action_id": pedagogical_decision.shaped_action_id,
        "shaped_action_name": pedagogical_decision.shaped_action_name,
        "overridden": pedagogical_decision.overridden,
        "rule": pedagogical_decision.rule,
        "reason": pedagogical_decision.reason,
    }

    response = AttemptSubmitResponse(
        session_id=session.session_id, attempt_id=req.attempt_id, exercise_id=req.exercise_id,
        quality_score=result.quality_score,
        static_features=static_feats.to_dict(), dynamic_features=dynamic_feats.to_dict(),
        state_vector=result.state_vector.tolist(),
        state_feature_names=list(result.feature_dict.keys()),
        safety=SafetyOut(**vars(safety_decision)),
        intervention=InterventionOut(**intervention.to_dict()),
        reward=reward, reward_components=reward_components,
        next_exercise=_exercise_out(session.current_exercise_id),
        session_summary=session.to_dict(),
        transition=transition_out,
        policy=_policy_info(decision),
        policy_mode=policy.mode, decision_source=decision.source,
        safety_overridden=safety_decision.overridden,
        presentation={"difficulty_level": session.difficulty_level,
                      "template_scale": session.template_scale, "mode": session.mode},
        handwriting_score=handwriting_breakdown,
        letter_mastery=letter_mastery_snapshot,
        pedagogical=pedagogical_info,
    )
    session.processed_attempts[req.attempt_id] = (payload_hash, response.model_dump())
    if len(session.processed_attempts) > 200:  # bound memory: keep the most recent 200
        session.processed_attempts.pop(next(iter(session.processed_attempts)))
    store.save(session)
    return response
