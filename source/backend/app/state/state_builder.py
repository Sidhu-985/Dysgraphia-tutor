"""
StateBuilder
=============
Converts validated static + dynamic features, plus session context, into
the project's canonical 12-dimensional RL state vector. This is the single
seam between the "real pipeline" (tablet -> preprocessing -> features) and
the RL agent -- it is independent of both the UI (which never sees a
feature or a state vector) and the RL agent implementation (which only
ever consumes the finished 12-d vector via `dysgraphia_rl_bridge`).

State dimension reference (must match dysgraphia_rl.utils.state_utils
.STATE_FEATURE_NAMES exactly -- enforced by test_state_builder.py):
    0. prev_score          - composite quality proxy of the just-completed attempt
    1. stroke_similarity   - DTW shape similarity to reference [0,1]
    2. spacing_accuracy    - inter-stroke spacing consistency [0,1]
    3. baseline_alignment  - how well strokes sit on the writing line [0,1]
    4. writing_speed       - normalized speed, 1.0 = age-appropriate [0,2]
    5. pressure_variance   - normalized stylus pressure variability [0,1]
    6. retry_count         - retries on current exercise, normalized [0,1]
    7. fatigue_proxy       - accumulated session fatigue heuristic [0,1]
    8. engagement          - engagement heuristic [0,1]
    9. improvement_trend   - rolling quality trend [-1,1]
    10. frustration        - frustration heuristic [0,1]
    11. consistency         - cross-attempt consistency [0,1]

`fatigue_proxy` / `engagement` / `frustration` are heuristic proxies built
from measurable session behaviour (retries, pauses, quality deltas) --
NOT sensor-measured psychological states. This mirrors (and is tunable
independently of) the simulator's internal dynamics in
`dysgraphia_rl/envs/dysgraphia_tutor_env.py`, but is a separate
implementation because a real session has real timestamps/pauses to key
off of, which the simulator fabricates instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import QualityScoreConfig, SessionDynamicsConfig, settings
from ..features.static_features import StaticFeatures
from ..features.dynamic_features import DynamicFeatures
from ..dysgraphia_rl_bridge import (
    STATE_FEATURE_NAMES,
    build_state_vector,
    compute_improvement_trend,
    compute_consistency,
    normalize_retries,
    validate_state_vector,
)
from ..scoring.handwriting_quality import HandwritingScore, compute_handwriting_score
from ..preprocessing.stroke_preprocessing import Stroke
from .session_context import SessionContext


@dataclass
class StateBuildResult:
    state_vector: np.ndarray
    feature_dict: dict
    quality_score: float
    break_taken: bool
    handwriting_score: Optional[HandwritingScore] = None


def compute_quality_score(
    static: StaticFeatures,
    dynamic: DynamicFeatures,
    cfg: QualityScoreConfig | None = None,
) -> float:
    """Composite handwriting-quality PROXY computed purely from measured
    static features. This is explicitly NOT a clinical accuracy score and
    NOT the CNN-based scorer described as future work in the project
    README -- it exists so the pipeline has *some* honest, reproducible
    notion of 'quality' to drive prev_score/history until that model is
    trained on the Kaggle dataset and substituted in (a one-function swap
    at the call site in state/state_builder.py, no other module changes).

    Retained as a backwards-compatible, reference-free proxy for callers
    that do not have an exercise reference (e.g. some unit tests). The
    exercise-aware, multiplicative :func:`compute_handwriting_score` is
    preferred whenever a reference template is available, because it
    cannot be fooled into a moderately high score by a coincidentally
    good bounding box or stroke count.
    """
    cfg = cfg or settings.quality_score
    # Only terms that were actually measured enter the average (Phase C
    # fix). Spacing consistency needs >= 3 strokes. Pressure stability
    # needs a device that reports pressure. Previously their neutral
    # placeholders (0.5 and 0.7) were averaged in as if they were
    # measurements, adding about 0.2 to every score.
    terms = [(cfg.w_stroke_similarity, static.stroke_similarity),
             (cfg.w_baseline_alignment, static.baseline_alignment)]
    if static.n_strokes >= 3:
        terms.append((cfg.w_spacing_accuracy, static.spacing_accuracy))
    if dynamic.pressure_available:
        terms.append((cfg.w_pressure_stability, 1.0 - dynamic.pressure_variance_normalized))
    total_weight = sum(w for w, _ in terms)
    if total_weight <= 0:
        return 0.5
    return float(np.clip(sum(w * v for w, v in terms) / total_weight, 0.0, 1.0))


def update_session_dynamics(
    session: SessionContext,
    quality: float,
    dynamic: DynamicFeatures,
    break_taken: bool = False,
    cfg: SessionDynamicsConfig | None = None,
) -> None:
    """Mutates `session` in place: fatigue/frustration/engagement/
    consecutive_fails, given the just-completed attempt's quality proxy
    and dynamic signal (pauses). Kept separate from `build_state` so it
    can be unit-tested (and retuned) independently of vector assembly.
    """
    cfg = cfg or settings.session_dynamics

    if break_taken:
        session.fatigue = float(np.clip(session.fatigue - cfg.fatigue_recovery_on_break, 0.0, 1.0))
        session.frustration = float(np.clip(
            session.frustration - cfg.fatigue_recovery_on_break * 0.6, 0.0, 1.0
        ))
        session.engagement = float(np.clip(session.engagement + cfg.engagement_gain_on_success, 0.0, 1.0))
        return

    success = quality >= cfg.quality_success_threshold
    session.consecutive_fails = 0 if success else session.consecutive_fails + 1

    fatigue_delta = cfg.fatigue_per_attempt + cfg.fatigue_per_pause_second * dynamic.pause_total_s
    session.fatigue = float(np.clip(session.fatigue + fatigue_delta, 0.0, 1.0))

    if success:
        session.frustration = float(np.clip(
            session.frustration - cfg.frustration_recovery_on_success, 0.0, 1.0
        ))
        session.engagement = float(np.clip(session.engagement + cfg.engagement_gain_on_success, 0.0, 1.0))
    else:
        frustration_delta = cfg.frustration_per_failed_attempt * (
            1.0 + cfg.frustration_retry_multiplier * session.consecutive_fails
        )
        session.frustration = float(np.clip(session.frustration + frustration_delta, 0.0, 1.0))
        session.engagement = float(np.clip(session.engagement - cfg.engagement_decay_on_failure, 0.0, 1.0))


def build_state(
    static: StaticFeatures,
    dynamic: DynamicFeatures,
    session: SessionContext,
    break_taken: bool = False,
    quality_cfg: QualityScoreConfig | None = None,
    dynamics_cfg: SessionDynamicsConfig | None = None,
    strokes: Optional[list[Stroke]] = None,
    reference_template: Optional[np.ndarray] = None,
    reference_strokes: Optional[list] = None,
) -> StateBuildResult:
    """Full pipeline: features + session -> quality proxy -> updated
    session dynamics -> updated score history -> assembled 12-d vector.

    Mutates `session.score_history` / `.attempt_count` by appending this
    attempt's result -- callers should persist the returned/updated
    `session` object (see storage.SessionStore) so the next call sees an
    up-to-date history.

    When ``reference_template`` (and optionally ``reference_strokes`` and
    ``strokes``) are provided, the exercise-aware multiplicative
    :func:`compute_handwriting_score` is used in place of the legacy
    reference-free proxy. The multiplicative scorer is required so a
    malformed shape (e.g. a zigzag instead of a lowercase ``a``) cannot
    earn a moderately high score based on coincidentally good
    non-shape features (see docs/HANDWRITING_QUALITY.md).
    """
    dynamics_cfg = dynamics_cfg or settings.session_dynamics

    handwriting_score: Optional[HandwritingScore] = None
    if not break_taken and reference_template is not None and strokes is not None:
        try:
            handwriting_score = compute_handwriting_score(
                strokes=strokes,
                reference_template=reference_template,
                reference_strokes=reference_strokes,
                static=static,
                dynamic=dynamic,
            )
            quality = handwriting_score.normalized
        except Exception:  # never crash the request on a scorer bug
            quality = compute_quality_score(static, dynamic, quality_cfg)
    elif not break_taken:
        quality = compute_quality_score(static, dynamic, quality_cfg)
    else:
        quality = session.score_history[-1] if session.score_history else 0.5

    update_session_dynamics(session, quality, dynamic, break_taken=break_taken, cfg=dynamics_cfg)

    if not break_taken:
        session.score_history.append(quality)
        session.attempt_count += 1

    improvement_trend = compute_improvement_trend(
        session.score_history, window=dynamics_cfg.improvement_trend_window
    )
    consistency = compute_consistency(
        session.score_history, window=dynamics_cfg.consistency_window
    )

    feature_dict = {
        "prev_score": quality,
        "stroke_similarity": static.stroke_similarity,
        "spacing_accuracy": static.spacing_accuracy,
        "baseline_alignment": static.baseline_alignment,
        "writing_speed": dynamic.writing_speed,
        "pressure_variance": dynamic.pressure_variance_normalized,
        # Phase C (Round 3) timing fix: count consecutive failures INCLUDING
        # the attempt just completed (session.consecutive_fails, already
        # updated above), the same way the simulator does. Previously this
        # was the count from BEFORE the attempt, one step behind the
        # simulator. A break does not change it.
        "retry_count": normalize_retries(session.consecutive_fails),
        "fatigue_proxy": session.fatigue,
        "engagement": session.engagement,
        "improvement_trend": improvement_trend,
        "frustration": session.frustration,
        "consistency": consistency,
    }

    vector = build_state_vector(feature_dict)
    validate_state_vector(vector, strict=False)  # non-fatal sanity check; logged by caller if non-empty

    assert list(feature_dict.keys()) == STATE_FEATURE_NAMES, (
        "feature_dict key order must exactly match STATE_FEATURE_NAMES "
        "(canonical ordering shared with the RL agent)"
    )

    return StateBuildResult(
        state_vector=vector,
        feature_dict=feature_dict,
        quality_score=quality,
        break_taken=break_taken,
        handwriting_score=handwriting_score,
    )
