"""
Central configuration for the backend pipeline.
=================================================
Every tunable parameter used by preprocessing, static/dynamic feature
extraction, the state builder, the safety layer, and the RL integration
lives here (Section 12 of the project spec: "all important parameters must
be configurable rather than hard-coded"). Nothing below is clinical ground
truth -- these are engineering defaults, documented with *why* they were
chosen, and meant to be tuned against real tablet data as it becomes
available.

Values can be overridden via environment variables (see `Settings.Config`)
or by editing this file directly -- no code outside this module should
need to change to retune the system.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class PreprocessingConfig(BaseSettings):
    # Target number of points strokes are resampled to before shape
    # comparison (DTW). Equal-arc-length resampling makes comparisons
    # robust to the tablet's raw sampling rate.
    resample_points: int = 64
    # Savitzky-Golay smoothing window (must be odd) and polynomial order,
    # applied to x/y before velocity/curvature derivatives are taken --
    # raw stylus coordinates are noisy enough that unsmoothed derivatives
    # are dominated by sensor jitter rather than real hand motion.
    smoothing_window: int = 7
    smoothing_polyorder: int = 2
    # Points whose instantaneous jump exceeds this many MADs (median
    # absolute deviations) from the stroke's median step size are treated
    # as outliers and clipped to the local neighbourhood average, rather
    # than dropped (dropping breaks timestamp-based dynamic features).
    outlier_mad_threshold: float = 6.0
    min_points_per_stroke: int = 2

    model_config = SettingsConfigDict(env_prefix="PREP_")


class StaticFeatureConfig(BaseSettings):
    # Sakoe-Chiba DTW band width as a fraction of series length. A full
    # (unconstrained) DTW is O(n*m) and can match distant-in-time points,
    # which is undesirable for handwriting where local temporal order
    # should mostly be preserved; a band keeps DTW close to the diagonal
    # while still tolerating speed differences.
    dtw_band_fraction: float = 0.25
    # DTW distance -> similarity mapping: similarity = exp(-distance / scale).
    # `scale` is expressed as a fraction of the shape's own bounding-box
    # diagonal (post scale-normalization the diagonal is 1.0 by
    # construction -- see preprocessing.normalize_shape), so this constant
    # is dimensionless and does not need retuning per exercise size.
    dtw_distance_to_similarity_scale: float = 0.35
    # Expected inter-stroke gap for multi-stroke exercises (e.g. words with
    # separate letters), in normalized shape units (same frame as DTW,
    # bounding-box diagonal = 1.0). Used only as the *reference* spacing;
    # actual spacing is compared against this and against the exercise's
    # own reference template spacing when available.
    expected_spacing: float = 0.15
    spacing_tolerance: float = 0.08
    # Spacing consistency (Phase C fix): a gap that differs from the
    # attempt's median gap by this fraction of the median scores 0.
    # expected_spacing/spacing_tolerance above are no longer used by
    # _spacing_accuracy. They are kept so existing STATICFEAT_* overrides
    # do not fail.
    spacing_relative_tolerance: float = 0.5
    # Baseline alignment: how many points from the bottom of each stroke's
    # bounding box are used to fit the "resting line" the child was
    # supposed to write on.
    baseline_fit_fraction: float = 0.2  # no longer used (kept so existing overrides do not fail)
    # Baseline alignment (Phase C fix): lower envelope of trace vs template,
    # compared over this many x-bins in the normalized frame. A mean
    # envelope difference equal to `baseline_envelope_scale` (normalized
    # units, bounding-box diagonal = 1) maps to exp(-1) ~ 0.37. Both values
    # were fixed in advance and are not tuned against data.
    baseline_bins: int = 8
    baseline_envelope_scale: float = 0.1

    model_config = SettingsConfigDict(env_prefix="STATICFEAT_")


class DynamicFeatureConfig(BaseSettings):
    # A gap between consecutive points larger than this (seconds) is
    # counted as a pause rather than normal writing motion.
    pause_threshold_s: float = 0.35
    # Acceleration is only computed (rather than left as "unreliable") when
    # the stroke's median sampling interval is at or below this value --
    # coarse sampling makes second-derivative estimates dominated by noise.
    min_sampling_rate_hz: float = 30.0
    # Expected writing speed baseline, in normalized "character widths per
    # second", used to convert raw path-length/time into the state's
    # writing_speed feature (1.0 == age-appropriate speed per the design
    # doc's normalize_speed()).
    expected_speed_chars_per_sec: float = 1.0
    # Pressure variance is only meaningful when the device reports
    # pressure; if pressure is entirely absent, the feature is marked
    # unavailable and a neutral default is used instead of a fabricated
    # value.
    pressure_available_min_fraction: float = 0.5

    model_config = SettingsConfigDict(env_prefix="DYNFEAT_")


class SessionDynamicsConfig(BaseSettings):
    """Heuristics turning raw dynamic signals into the session-level
    fatigue/engagement/frustration proxies consumed by the state vector.

    These are proxies for affect/fatigue, not measured psychological
    constructs -- there is no sensor that measures a child's frustration
    directly. They are explicitly engineering approximations, tunable here,
    and should be revisited once real interaction logs are available.
    """
    fatigue_per_attempt: float = 0.03
    fatigue_per_pause_second: float = 0.01
    fatigue_recovery_on_break: float = 0.3
    frustration_per_failed_attempt: float = 0.08
    frustration_recovery_on_success: float = 0.03
    frustration_retry_multiplier: float = 0.3
    engagement_decay_on_failure: float = 0.04
    engagement_gain_on_success: float = 0.02
    consistency_window: int = 5
    improvement_trend_window: int = 6
    quality_success_threshold: float = 0.55

    model_config = SettingsConfigDict(env_prefix="SESSDYN_")


class QualityScoreConfig(BaseSettings):
    """Weights for the composite handwriting-quality proxy computed purely
    from measured static features, used as `prev_score` / history until a
    dataset-trained CNN scorer (see project README "Next steps") replaces
    it. NOT a clinical score -- documented as an engineering proxy only.
    """
    w_stroke_similarity: float = 0.45
    w_spacing_accuracy: float = 0.20
    w_baseline_alignment: float = 0.20
    w_pressure_stability: float = 0.15

    model_config = SettingsConfigDict(env_prefix="QUALITY_")


class HandwritingScoreConfig(BaseSettings):
    """Configuration for the exercise-aware handwriting quality scorer
    (``app.scoring.handwriting_quality``). Every component lives in
    [0, 1]; the final score multiplies the dominant DTW shape similarity
    by a structural factor (stroke count, aspect ratio, curvature,
    direction) and a smaller auxiliary factor (baseline alignment,
    optional pressure stability). The multiplicative composition makes
    a single weak component pull the score down, so a malformed shape
    (e.g. a zigzag drawn instead of a lowercase ``a``) cannot earn a
    moderately high score merely because its bounding box, stroke count
    or baseline alignment coincidentally match the reference.

    None of these values are clinical ground truth -- they are
    engineering defaults, documented with the reasoning behind each
    choice, and tunable here without touching the scorer.
    """
    # Stroke count: 1 extra/missing stroke is tolerated (a child may
    # draw the dot of an ``i`` as a separate stroke, or merge the bowl
    # and stem of an ``a`` into one continuous stroke). Each further
    # deviation costs roughly 1/3 of the stroke-count score.
    stroke_count_tolerance: int = 1
    # Number of bins in the turning-angle histogram used by the
    # curvature-similarity component. 16 is enough to distinguish a
    # smooth ``a`` (dominant small-angle turns) from a zigzag
    # (alternating +/- ~90 degree turns) without overfitting to noise.
    curvature_bins: int = 16
    # Number of equal-arc-length segments used by the direction-
    # consistency component. 8 is the smallest count that still captures
    # the gross direction sequence of a single letter.
    direction_segments: int = 8
    # Exponent applied to the auxiliary factor in the final composition.
    # 0.5 means a moderately off baseline is half as damaging as it would
    # be at exponent 1.0; a perfect baseline cannot rescue a wrong shape
    # because the shape and structural factors are still multiplicative
    # caps. Lower this to make neatness matter more; raise it to make
    # the score depend almost entirely on shape and structure.
    auxiliary_weight: float = 0.5
    # Diagnostic floor: when the DTW shape similarity is below this
    # value, a note is added to the breakdown (the score is not forced
    # to any particular value -- the geometric-mean composition already
    # makes a poor shape produce a poor score).
    shape_floor: float = 0.45

    model_config = SettingsConfigDict(env_prefix="HWSCORE_")


class MasteryConfig(BaseSettings):
    """Per-letter mastery and Level 4 rotation configuration.

    Level 4 is treated as a difficulty level containing MULTIPLE
    exercises (multiple letters), not as a single letter. The mastery
    tracker records, per exercise id, the recent scores and consecutive
    successes; the exercise selector uses this to rotate among
    eligible same-level letters and only escalate the difficulty once
    all letters at the current level have been mastered.

    Tunable so a clinician / educator can adjust the strictness without
    editing code. All values are documented with the reasoning behind
    the defaults.
    """
    # Quality score (in [0, 1]) at or above which an attempt counts as a
    # success for mastery purposes. 0.65 is comfortably above the legacy
    # 0.55 success threshold but below the "excellent" range so that a
    # single slightly imperfect attempt is not rejected.
    mastery_score_threshold: float = 0.65
    # Number of consecutive successes required to mark an exercise as
    # mastered. 2 keeps the learner from being advanced after a single
    # lucky attempt without making the criterion so strict that the
    # learner gets stuck indefinitely.
    mastery_consecutive_successes_required: int = 2
    # Maximum attempts at one exercise before the system rotates to
    # another eligible same-level letter (if any) even if mastery has
    # not been achieved. Prevents indefinite frustration on one letter.
    # Set to 0 to disable.
    max_attempts_before_rotation: int = 6
    # When rotating, avoid the N most recently practised exercises at
    # the same level so the learner does not bounce between just two
    # letters.
    rotation_avoid_recent: int = 2
    # When the learner has mastered every eligible same-level letter,
    # the next INCREASE_DIFFICULTY action escalates the difficulty by
    # one level. The escalation still respects the safety layer and the
    # track (case is preserved).
    escalate_after_all_mastered: bool = True

    model_config = SettingsConfigDict(env_prefix="MASTERY_")


class RewardConfigOverride(BaseSettings):
    """Optional override of the RL reward weights (defaults match
    dysgraphia_rl.utils.reward_utils.RewardWeights and Section 7 of the
    design doc). Left as None fields use the RL package's own defaults.
    """
    w1_quality: float | None = None
    w2_completion: float | None = None
    w3_engagement: float | None = None
    w4_frustration: float | None = None
    w5_consistency: float | None = None
    w6_unnecessary_break: float | None = None
    # A break taken while fatigue_proxy is below this value is
    # "unnecessary" and penalized (reward version 2; same default as the
    # simulator's DysgraphiaTutorEnv.break_fatigue_threshold).
    break_fatigue_threshold: float = 0.5

    model_config = SettingsConfigDict(env_prefix="REWARD_")


class SafetyConfig(BaseSettings):
    # Maximum number of consecutive DIFFICULTY-escalating actions the
    # safety layer allows before forcing a non-escalating fallback.
    max_consecutive_escalations: int = 2
    # Maximum number of times the *same* action may be chosen in a row
    # before it is treated as excessive repetition and overridden.
    max_consecutive_same_action: int = 3
    # If fatigue_proxy crosses this value, the safety layer forces a break
    # recommendation regardless of what the policy chose.
    # Lowered from 0.85 to 0.70 after the Phase A audit: the scenario
    # requirement is that a learner whose fatigue proxy exceeds 0.7 is
    # always offered a break, whatever the policy proposes.
    fatigue_break_threshold: float = 0.70
    # If frustration crosses this value, escalating difficulty is always
    # blocked (the safety layer substitutes a supportive action instead).
    frustration_escalation_block_threshold: float = 0.75
    # Hard session-length cap (number of attempts) after which a break is
    # forced regardless of the policy's own action, independent of fatigue
    # estimation -- a pure session-intensity guardrail.
    max_attempts_before_forced_break: int = 25
    # --- rules added after the Phase A audit (break over-selection) ---
    # At most this many breaks may be delivered back to back (mirrors the
    # simulator's max_consecutive_breaks). A further requested break is
    # replaced with CONTINUE_NO_CHANGE unless fatigue is above
    # fatigue_break_threshold (then the forced-break rule wins).
    max_consecutive_breaks: int = 1
    # At or above this frustration level, actions that keep the same or
    # higher demand (INCREASE_DIFFICULTY, SWITCH_TO_FREE_WRITING_MODE,
    # REPEAT_SAME_LETTER, CONTINUE_NO_CHANGE) are replaced with SHOW_HINT
    # (or DECREASE_DIFFICULTY if a hint was already the previous action).
    high_frustration_support_threshold: float = 0.75
    # INCREASE_DIFFICULTY is blocked whenever the learner has failed at
    # least this many attempts in a row (default 1: never escalate right
    # after a failed attempt).
    block_increase_after_consecutive_fails: int = 1

    model_config = SettingsConfigDict(env_prefix="SAFETY_")


class RLConfig(BaseSettings):
    # Relative to the repository root (backend/'s parent directory).
    checkpoint_path: str = "dysgraphia_rl/checkpoints/double_dqn.pt"
    # Relative to backend/ itself (see dysgraphia_rl_bridge.py).
    dysgraphia_rl_root: str = "../dysgraphia_rl"
    device: str | None = None
    # Epsilon for the deployed policy. 0.0 = fully greedy (recommended for
    # a deployed tutor -- exploration noise is a training-time concept and
    # should not be sprung on a real child's session by default).
    deployment_epsilon: float = 0.0
    # Behavioural gate: before the DQN checkpoint is used for live
    # decisions, it is run on a fixed set of probe states with known safe
    # expectations (rl_integration/behavioral_gate.py). If any probe fails,
    # or the checkpoint is missing/unloadable, the backend serves the
    # transparent rule-based fallback policy instead, and /health reports
    # policy_mode="rule_based_fallback" with the reasons.
    behavioral_gate_enabled: bool = True

    model_config = SettingsConfigDict(env_prefix="RL_")


class PedagogicalLayerConfig(BaseSettings):
    """Configuration for the pedagogical layer
    (``app.rl_integration.pedagogical_layer``).

    The pedagogical layer sits between the RL policy's choice and the
    safety layer. It shapes the action distribution to be
    pedagogically diverse and enforces session-level tutoring rules
    that the trained DQN does not learn on its own (because they
    depend on session context the 12-D state vector only partially
    encodes).

    Three rules, in priority order:
      1. Repeated-failure de-escalation (highest priority).
      2. Post-break progression.
      3. Action diversity safeguard (lowest priority).

    All rules are deterministic, audited (every override is logged
    with a reason), and configurable here. They never modify the DQN
    checkpoint or the locked protocol.
    """
    # Master switch for the whole layer. When False, the layer is a
    # pass-through (no overrides).
    enabled: bool = True

    # Rule 1: Repeated-failure de-escalation.
    # If the learner has had >= max_attempts_at_letter attempts at
    # the current Level 4 letter without mastering it, override the
    # policy's choice to DECREASE_DIFFICULTY (drop one level).
    enable_deescalation_after_repeated_failure: bool = True
    max_attempts_at_letter: int = 4

    # Rule 2: Post-break progression.
    # If the previous action was SUGGEST_BREAK and the current
    # attempt's quality_score >= mastery threshold, override the
    # policy's choice to INCREASE_DIFFICULTY (which the Level 4
    # rotation rule turns into a same-level rotation if other
    # letters remain).
    enable_post_break_progression: bool = True

    # Rule 3: Action diversity safeguard.
    # If the policy has picked the same action K times in the last N
    # decisions, pick a pedagogically reasonable alternative.
    enable_diversity_safeguard: bool = True
    # Window of recent decisions to consider (N).
    diversity_window: int = 4
    # Number of times the same action must appear in that window to
    # trigger the override (K).
    diversity_repeat_threshold: int = 2

    model_config = SettingsConfigDict(env_prefix="PEDAGOGICAL_")


class InterventionConfig(BaseSettings):
    """Magnitudes used by the intervention controller when it turns an RL
    action id into a concrete tablet instruction. Purely presentational/
    pedagogical knobs -- do not affect the state vector or reward."""
    difficulty_step: int = 1
    difficulty_min: int = 0
    difficulty_max: int = 4
    template_enlarge_factor: float = 1.25
    template_shrink_factor: float = 0.9
    template_scale_min: float = 0.5
    template_scale_max: float = 2.0
    break_duration_seconds: int = 60
    hint_display_seconds: int = 8

    model_config = SettingsConfigDict(env_prefix="INTERVENTION_")


class StorageConfig(BaseSettings):
    # SQLite file backing session/attempt history. Kept local/file-based
    # for this research prototype -- swappable for a real DB later without
    # touching any other module (storage.py is the only place that knows
    # about the backing store).
    db_path: str = "./data/dysgraphia_tutor.db"
    # Keep the raw captured points (x, y, t, pressure, stroke_id) of each
    # attempt so feature extraction can be re-run and audited later. Raw
    # handwriting is linked only to a pseudonymous learner code.
    store_raw_points: bool = True

    model_config = SettingsConfigDict(env_prefix="STORAGE_")


class StudyConfig(BaseSettings):
    """Data-governance switches. Collection of data from real children
    stays DISABLED unless an institutional ethics approval reference is
    configured AND this flag is explicitly turned on. Until then the
    backend refuses attempts labelled data_source="real_child_study"."""
    real_child_collection_approved: bool = False
    ethics_approval_reference: str | None = None
    # Pseudonymous learner identifiers only: short opaque codes such as
    # "P-0042" or "demo_child_1". Names, e-mail addresses and spaces are
    # rejected by the pattern below.
    learner_id_pattern: str = r"^[A-Za-z0-9][A-Za-z0-9_-]{2,39}$"
    # Documented retention period for attempt logs (enforcement is an
    # operational task; see docs/DATA_GOVERNANCE.md).
    retention_days: int = 180

    model_config = SettingsConfigDict(env_prefix="STUDY_")


class Settings(BaseSettings):
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    static_features: StaticFeatureConfig = Field(default_factory=StaticFeatureConfig)
    dynamic_features: DynamicFeatureConfig = Field(default_factory=DynamicFeatureConfig)
    session_dynamics: SessionDynamicsConfig = Field(default_factory=SessionDynamicsConfig)
    quality_score: QualityScoreConfig = Field(default_factory=QualityScoreConfig)
    handwriting_score: HandwritingScoreConfig = Field(default_factory=HandwritingScoreConfig)
    mastery: MasteryConfig = Field(default_factory=MasteryConfig)
    reward: RewardConfigOverride = Field(default_factory=RewardConfigOverride)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    pedagogical_layer: PedagogicalLayerConfig = Field(default_factory=PedagogicalLayerConfig)
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    study: StudyConfig = Field(default_factory=StudyConfig)


settings = Settings()
