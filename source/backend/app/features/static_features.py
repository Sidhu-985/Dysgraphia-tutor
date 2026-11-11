"""
Static feature extraction ("what the child wrote").
======================================================
Fully independent of the dynamic feature extractor, the RL agent, and the
UI -- this module only ever sees geometric (x, y) stroke data (already
cleaned by `preprocessing.stroke_preprocessing`) plus an optional reference
template, and returns a `StaticFeatures` dataclass. It can be edited,
retuned, or replaced without touching anything else in the pipeline.

Feature list (matches project spec Section 3 "Static features"):
  - stroke_similarity   : DTW-based shape similarity to the reference template
  - spacing_accuracy     : consistency of inter-stroke spacing vs. expected
  - baseline_alignment   : how well strokes sit on the intended writing line
  - geometry              : width, height, aspect_ratio (descriptive, logged
                            but not part of the 12-d RL state directly)
  - intra_attempt_consistency : how similar per-stroke shapes are to each
                            other within the same attempt (distinct from the
                            *cross-attempt* consistency the StateBuilder
                            computes from session history)
  - curvature_smoothness : how smooth vs. jagged the trajectory's direction
                            changes are (auxiliary, feeds the quality proxy)

All thresholds/parameters are configurable via
`backend.app.config.StaticFeatureConfig` -- nothing here is an unexplained
magic number; see that module's docstrings for the reasoning behind each
default.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from ..config import StaticFeatureConfig, settings
from ..preprocessing.stroke_preprocessing import (
    Stroke, clean_strokes, resample_by_arc_length, smooth_series,
    normalize_shape, concatenate_strokes,
)


@dataclass
class StaticFeatures:
    stroke_similarity: float
    spacing_accuracy: float
    baseline_alignment: float
    width: float
    height: float
    aspect_ratio: float
    intra_attempt_consistency: float
    curvature_smoothness: float
    dtw_distance_raw: float
    n_strokes: int

    def to_dict(self) -> dict:
        return asdict(self)


def _resample_and_smooth(xs: np.ndarray, ys: np.ndarray, cfg: StaticFeatureConfig,
                          n_points: int) -> tuple[np.ndarray, np.ndarray]:
    rx, ry = resample_by_arc_length(xs, ys, n_points)
    rx = smooth_series(rx)
    ry = smooth_series(ry)
    return rx, ry


def _banded_dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray, band_fraction: float) -> float:
    """Sakoe-Chiba-banded DTW over 2D point sequences (rows = [x, y]).
    Returns the raw (average, path-length-normalized) DTW distance in
    whatever units `seq_a`/`seq_b` are expressed in -- callers are expected
    to pass already scale/translation-normalized coordinates.

    A band restricts alignment to |i - j| <= band_width, which keeps DTW
    from matching temporally-distant points (a real risk with unconstrained
    DTW on repetitive strokes like loops) while still tolerating genuine
    speed differences between the child's stroke and the template.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    band_width = max(1, int(round(band_fraction * max(n, m))))
    INF = np.inf
    cost = np.full((n + 1, m + 1), INF, dtype=np.float64)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - band_width)
        j_hi = min(m, i + band_width)
        for j in range(j_lo, j_hi + 1):
            d = float(np.hypot(seq_a[i - 1, 0] - seq_b[j - 1, 0],
                                seq_a[i - 1, 1] - seq_b[j - 1, 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    path_len = max(n, m)
    return float(cost[n, m] / path_len)


def _default_reference_template(n_points: int) -> np.ndarray:
    """A neutral straight-line reference used only when the exercise did
    not supply a real template (e.g. free-writing mode, or an exercise
    definition missing a template) -- lets stroke_similarity degrade
    gracefully to "how line-like/controlled is the stroke" rather than
    crashing or silently returning a meaningless constant."""
    t = np.linspace(-0.5, 0.5, n_points)
    return np.stack([t, np.zeros_like(t)], axis=1)


def _lower_envelope(seq: np.ndarray, n_bins: int) -> np.ndarray:
    """Per-x-bin lowest point (largest y: screen y grows downward) of a
    normalized (N, 2) shape, over the shape's own x-range. Empty bins are
    filled from the nearest non-empty bin."""
    xs, ys = seq[:, 0], seq[:, 1]
    lo, hi = float(xs.min()), float(xs.max())
    if hi - lo < 1e-6:
        return np.full(n_bins, float(ys.max()))
    idx = np.clip(((xs - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    env = np.full(n_bins, np.nan)
    for b in range(n_bins):
        if np.any(idx == b):
            env[b] = float(ys[idx == b].max())
    filled = np.where(~np.isnan(env))[0]
    for b in range(n_bins):
        if np.isnan(env[b]):
            env[b] = env[filled[np.argmin(np.abs(filled - b))]]
    return env


def _baseline_alignment(child_seq: np.ndarray, ref_seq: np.ndarray, cfg: StaticFeatureConfig) -> float:
    """How closely the trace's lower envelope (where it "sits") follows
    the template's lower envelope. Both shapes are in the same
    translation/scale-normalized frame used for DTW. Score =
    exp(-mean|envelope difference| / baseline_envelope_scale), in [0, 1].
    1.0 means the writing rests where the model shape rests.

    Phase C fix: the original version measured the spread of the lowest
    20% of y-values. Those points are near the bottom by construction, so
    random scribbles scored about 0.9 and the feature did not respond to
    alignment at all (evidence/phase_c/discrimination_before.json).
    """
    if len(child_seq) < 3 or len(ref_seq) < 3:
        return 0.5
    n = max(2, int(cfg.baseline_bins))
    diff = np.abs(_lower_envelope(child_seq, n) - _lower_envelope(ref_seq, n))
    return float(np.exp(-float(np.mean(diff)) / max(cfg.baseline_envelope_scale, 1e-6)))


def _spacing_accuracy(strokes: list[Stroke], cfg: StaticFeatureConfig) -> float:
    """Spacing CONSISTENCY between successive strokes (the state feature is
    defined as "letter/word spacing consistency").

    Each centroid-to-centroid gap is compared with the attempt's own median
    gap. Per-gap score = 1 - |gap - median| / (spacing_relative_tolerance *
    median), clipped to [0, 1]. The result is the mean per-gap score. Fewer
    than three strokes give fewer than two gaps, so regularity cannot be
    measured and the neutral 0.5 is returned.

    Phase C fix: the original version compared each gap with a fixed
    constant (expected_spacing = 0.15 of the attempt's bounding-box
    diagonal). A perfectly regular row of four strokes therefore scored
    0.0. The score depended on how many strokes there were, not on how
    even the spacing was (evidence/phase_c/).
    """
    if len(strokes) < 3:
        return 0.5
    centroids = np.array([
        (float(np.mean([p.x for p in s.points])), float(np.mean([p.y for p in s.points])))
        for s in strokes
    ])
    gaps = np.hypot(*np.diff(centroids, axis=0).T)
    median_gap = float(np.median(gaps))
    if median_gap < 1e-9:
        return 0.0
    tolerance = max(cfg.spacing_relative_tolerance, 1e-6) * median_gap
    per_gap_score = np.clip(1.0 - np.abs(gaps - median_gap) / tolerance, 0.0, 1.0)
    return float(np.mean(per_gap_score))


def _curvature_smoothness(rx: np.ndarray, ry: np.ndarray) -> float:
    """Turning-angle-based smoothness: mean absolute change in heading
    angle between consecutive segments, mapped to [0, 1] with 1.0 == a
    perfectly straight/smoothly-curving path and 0.0 == extremely jagged
    (large, frequent direction reversals -- a hallmark of poor motor
    control in dysgraphia literature, used here only as a descriptive
    signal, not a diagnostic one)."""
    if len(rx) < 4:
        return 0.5
    dx, dy = np.diff(rx), np.diff(ry)
    angles = np.arctan2(dy, dx)
    dtheta = np.diff(angles)
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]
    mean_abs_turn = float(np.mean(np.abs(dtheta)))
    # Normalize: pi radians of average turn per step is "maximally jagged".
    return float(np.clip(1.0 - mean_abs_turn / np.pi, 0.0, 1.0))


def extract_static_features(
    strokes: list[Stroke],
    reference_template: Optional[np.ndarray] = None,
    cfg: Optional[StaticFeatureConfig] = None,
) -> StaticFeatures:
    """
    Parameters
    ----------
    strokes: raw (already outlier-clipped, e.g. via clean_strokes) strokes
        for one handwriting attempt.
    reference_template: optional (N, 2) array of the exercise's reference
        shape in arbitrary units -- will be independently resampled and
        scale/translation-normalized, so it does not need to match the
        child's coordinate system or point count.
    """
    cfg = cfg or settings.static_features
    strokes = clean_strokes(strokes)
    strokes = [s for s in strokes if len(s.points) >= 1]
    if not strokes:
        return StaticFeatures(0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.5, 0.5, float("inf"), 0)

    all_xs, all_ys = concatenate_strokes(strokes)
    width = float(all_xs.max() - all_xs.min()) if len(all_xs) else 0.0
    height = float(all_ys.max() - all_ys.min()) if len(all_ys) else 0.0
    aspect_ratio = float(width / height) if height > 1e-9 else float("inf") if width > 1e-9 else 1.0

    n_points = settings.preprocessing.resample_points
    rx, ry = _resample_and_smooth(all_xs, all_ys, cfg, n_points)
    norm_x, norm_y = normalize_shape(rx, ry)
    child_seq = np.stack([norm_x, norm_y], axis=1)

    if reference_template is not None and len(reference_template) >= 2:
        ref_x, ref_y = resample_by_arc_length(
            reference_template[:, 0], reference_template[:, 1], n_points
        )
        ref_norm_x, ref_norm_y = normalize_shape(ref_x, ref_y)
        ref_seq = np.stack([ref_norm_x, ref_norm_y], axis=1)
    else:
        ref_seq = _default_reference_template(n_points)

    dtw_dist = _banded_dtw_distance(child_seq, ref_seq, cfg.dtw_band_fraction)
    stroke_similarity = float(np.exp(-dtw_dist / cfg.dtw_distance_to_similarity_scale)) \
        if np.isfinite(dtw_dist) else 0.0

    baseline_alignment = _baseline_alignment(child_seq, ref_seq, cfg)
    spacing_accuracy = _spacing_accuracy(strokes, cfg)
    curvature_smoothness = _curvature_smoothness(rx, ry)

    # Intra-attempt consistency: how similar each individual stroke's shape
    # is to the whole attempt's average shape (only meaningful with >=2
    # strokes -- e.g. repeated letters in a word).
    if len(strokes) >= 2:
        per_stroke_scores = []
        for s in strokes:
            sx = np.array([p.x for p in s.points])
            sy = np.array([p.y for p in s.points])
            if len(sx) < 2:
                continue
            srx, sry = _resample_and_smooth(sx, sy, cfg, n_points)
            snx, sny = normalize_shape(srx, sry)
            s_seq = np.stack([snx, sny], axis=1)
            d = _banded_dtw_distance(s_seq, child_seq, cfg.dtw_band_fraction)
            per_stroke_scores.append(np.exp(-d / cfg.dtw_distance_to_similarity_scale) if np.isfinite(d) else 0.0)
        intra_attempt_consistency = float(1.0 - np.std(per_stroke_scores)) if per_stroke_scores else 0.5
        intra_attempt_consistency = float(np.clip(intra_attempt_consistency, 0.0, 1.0))
    else:
        intra_attempt_consistency = 0.5

    return StaticFeatures(
        stroke_similarity=stroke_similarity,
        spacing_accuracy=spacing_accuracy,
        baseline_alignment=baseline_alignment,
        width=width,
        height=height,
        aspect_ratio=aspect_ratio,
        intra_attempt_consistency=intra_attempt_consistency,
        curvature_smoothness=curvature_smoothness,
        dtw_distance_raw=dtw_dist,
        n_strokes=len(strokes),
    )
