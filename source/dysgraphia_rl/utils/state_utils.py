"""
State Representation Module
============================
Utilities for normalizing raw handwriting features into the fixed-length
state vector consumed by the RL agent (Design and Solution Section 3/4).

In the simulator, DysgraphiaTutorEnv already emits a normalized 12-d vector
directly. This module exists as the seam where the REAL feature-extraction
pipeline (OpenCV + DTW on real tablet strokes) will plug in later: it takes
raw, unnormalized feature dicts (as they will arrive from a real tablet /
the Feature Extraction Module) and converts them into the same 12-d format,
so the same trained policy can be reused for both simulation and real data.
"""

from __future__ import annotations
import numpy as np

STATE_FEATURE_NAMES = [
    "prev_score", "stroke_similarity", "spacing_accuracy", "baseline_alignment",
    "writing_speed", "pressure_variance", "retry_count", "fatigue_proxy",
    "engagement", "improvement_trend", "frustration", "consistency",
]

STATE_DIM = len(STATE_FEATURE_NAMES)


def normalize_speed(raw_speed_chars_per_sec: float, expected_speed: float = 1.0) -> float:
    """Normalize raw writing speed against an age/task-expected baseline.
    Returned value: 1.0 == expected speed, <1 slower, >1 faster."""
    if expected_speed <= 0:
        return 1.0
    return float(np.clip(raw_speed_chars_per_sec / expected_speed, 0.0, 2.0))


def normalize_pressure_variance(raw_variance: float, max_expected_variance: float = 50.0) -> float:
    """Map raw stylus-pressure variance (device units) into [0, 1]."""
    return float(np.clip(raw_variance / max_expected_variance, 0.0, 1.0))


def normalize_retries(retry_count: int, cap: int = 5) -> float:
    return float(np.clip(retry_count / cap, 0.0, 1.0))


def compute_improvement_trend(score_history: list[float], window: int = 6) -> float:
    """Short-term rolling trend: average step-to-step delta over the last
    `window` handwriting-quality scores, clipped to [-1, 1]."""
    if len(score_history) < 3:
        return 0.0
    recent = score_history[-window:]
    diffs = np.diff(recent)
    return float(np.clip(np.mean(diffs), -1.0, 1.0))


def compute_consistency(score_history: list[float], window: int = 5) -> float:
    """1 - std of recent scores => higher means more consistent performance."""
    if len(score_history) < 2:
        return 0.5
    recent = score_history[-window:]
    return float(np.clip(1.0 - np.std(recent), 0.0, 1.0))


def build_state_vector(features: dict) -> np.ndarray:
    """Assemble the 12-d state vector from a raw feature dict.

    Expected keys in `features` (all already normalized to roughly [0,1] or
    the documented range -- use the helper functions above upstream if they
    are not):
        prev_score, stroke_similarity, spacing_accuracy, baseline_alignment,
        writing_speed, pressure_variance, retry_count, fatigue_proxy,
        engagement, improvement_trend, frustration, consistency
    Missing keys default to a neutral value (0.5, or 0.0 for trend-type features).
    """
    neutral_defaults = {
        "prev_score": 0.5, "stroke_similarity": 0.5, "spacing_accuracy": 0.5,
        "baseline_alignment": 0.5, "writing_speed": 1.0, "pressure_variance": 0.3,
        "retry_count": 0.0, "fatigue_proxy": 0.0, "engagement": 0.5,
        "improvement_trend": 0.0, "frustration": 0.0, "consistency": 0.5,
    }
    vec = np.array(
        [features.get(name, neutral_defaults[name]) for name in STATE_FEATURE_NAMES],
        dtype=np.float32,
    )
    return vec


def describe_state(vec: np.ndarray) -> dict:
    """Inverse of build_state_vector, for logging/debugging/reports."""
    return {name: float(v) for name, v in zip(STATE_FEATURE_NAMES, vec)}


# Per-feature expected ranges, used only for validation/sanity-checking (not
# for clipping during normal operation -- DysgraphiaTutorEnv already clips
# internally). improvement_trend is the only signed feature; writing_speed
# is allowed up to 2.0 (see normalize_speed); everything else is [0, 1].
STATE_FEATURE_RANGES = {
    "prev_score": (0.0, 1.0), "stroke_similarity": (0.0, 1.0),
    "spacing_accuracy": (0.0, 1.0), "baseline_alignment": (0.0, 1.0),
    "writing_speed": (0.0, 2.0), "pressure_variance": (0.0, 1.0),
    "retry_count": (0.0, 1.0), "fatigue_proxy": (0.0, 1.0),
    "engagement": (0.0, 1.0), "improvement_trend": (-1.0, 1.0),
    "frustration": (0.0, 1.0), "consistency": (0.0, 1.0),
}


def validate_state_vector(vec: np.ndarray, strict: bool = True) -> list[str]:
    """Sanity-check a state vector against the canonical shape and the
    documented per-feature ranges from Section 3 of the Design and Solution
    doc. Returns a list of human-readable problems (empty list = valid).

    `strict=True` raises AssertionError on the first problem (useful as a
    one-line guard at integration points, e.g. before feeding a real-tablet
    feature vector to the trained policy); `strict=False` just returns the
    full list of problems, e.g. for a diagnostic/report use.
    """
    problems = []
    if vec.shape != (STATE_DIM,):
        problems.append(f"expected shape ({STATE_DIM},), got {vec.shape}")
    else:
        for name, (lo, hi) in zip(STATE_FEATURE_NAMES, STATE_FEATURE_RANGES.values()):
            value = float(vec[STATE_FEATURE_NAMES.index(name)])
            if not (lo - 1e-6 <= value <= hi + 1e-6):
                problems.append(f"{name}={value:.4f} outside expected range [{lo}, {hi}]")

    if strict and problems:
        raise AssertionError("Invalid state vector: " + "; ".join(problems))
    return problems
