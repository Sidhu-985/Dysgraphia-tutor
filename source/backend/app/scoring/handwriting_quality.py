"""
Exercise-specific handwriting quality scoring
==============================================

A robust scorer that fixes the core defect reported in the audit:
a learner could draw a completely incorrect zigzag in place of lowercase
``a`` and still receive a score around 40-50%, because the legacy composite
``compute_quality_score`` averages the DTW shape similarity with two
non-shape features (``baseline_alignment`` and ``spacing_accuracy``) plus a
pressure-stability term. Those non-shape features respond to bounding-box
placement and stroke regularity, not to whether the trace actually looks
like the requested letter. A zigzag and a lowercase ``a`` have roughly
similar bounding boxes, so the non-shape terms contributed ~0.20 of "free"
score and lifted a malformed attempt above the 0.55 success threshold.

This module replaces that behaviour with an exercise-aware scorer that
combines a *dominant* shape signal with structural / topological checks,
using a multiplicative (geometric-mean) combination so a single weak
component pulls the final score down rather than being averaged away.

Components (each in [0, 1], higher is better):

  - ``shape_similarity``  : DTW-based shape similarity to the reference
                            template (re-uses the static feature extractor's
                            existing computation -- only the *weighting*
                            of that signal changes here).
  - ``stroke_count_match``: 1.0 when the number of strokes is within
                            ``stroke_count_tolerance`` of the template's
                            stroke count, falling off linearly outside that
                            window. 0 when no strokes were captured.
  - ``aspect_ratio_match``: 1 - |child_ratio - ref_ratio| / ref_ratio, in
                            [0, 1]. Letters such as ``i`` are tall and
                            narrow; ``w`` is wide and short. A zigzag
                            instead of an ``a`` tends to have a very
                            different aspect ratio and is caught here.
  - ``curvature_similarity``: similarity between the turning-angle
                            histograms of the child's trace and the
                            reference template. Curvature is shift and
                            scale invariant, so a correct letter drawn at
                            a different size or position still scores
                            well, while a zigzag (alternating +90/-90
                            degree turns) scores very differently from a
                            smooth ``a``.
  - ``direction_consistency``: cosine-similarity between the dominant
                            direction vectors of corresponding segments
                            of the child and reference traces. A
                            scribble's direction vector is random; a
                            correct letter's is consistent with the
                            template's.
  - ``baseline_alignment``: re-uses the static feature extractor's
                            baseline alignment score (how well the trace
                            sits on the intended writing line). Only
                            contributes to the auxiliary factor.
  - ``pressure_stability``: only when reliable pressure data is
                            available. When absent, the term is dropped
                            and the remaining weights are renormalized --
                            it never silently contributes a "perfect"
                            0.5 placeholder.

Final score (in [0, 1]):

    final = shape_similarity * structural_factor * auxiliary_factor

  - ``structural_factor`` is the geometric mean of ``stroke_count_match``,
    ``aspect_ratio_match``, ``curvature_similarity`` and
    ``direction_consistency``. Each one is a *necessary* condition for the
    trace to be the requested letter, so any single failure pulls the
    score down multiplicatively.
  - ``auxiliary_factor`` is the geometric mean of ``baseline_alignment``
    (always) and ``pressure_stability`` (only when reliable). It rewards
    neatness without overpowering the shape signal: a correct shape with
    a slightly off baseline still scores well; a wrong shape with a
    perfect baseline still scores poorly because the shape term
    dominates.

All thresholds are configurable via
:class:`app.config.HandwritingScoreConfig` and exposed transparently in
the returned :class:`HandwritingScoreBreakdown` for diagnostic /
therapist use.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..config import HandwritingScoreConfig, settings
from ..features.static_features import StaticFeatures
from ..features.dynamic_features import DynamicFeatures
from ..preprocessing.stroke_preprocessing import (
    Stroke, clean_strokes, resample_by_arc_length, smooth_series,
    normalize_shape, concatenate_strokes,
)


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------
@dataclass
class HandwritingScoreBreakdown:
    """Transparent component breakdown for one attempt.

    Every component is in [0, 1]; ``final_score`` is the combined
    handwriting-quality score in [0, 1]. Reported to the UI for an
    optional therapist / debug view (the learner only sees the rounded
    percentage), and used by the mastery tracker.
    """
    shape_similarity: float
    stroke_count_match: float
    aspect_ratio_match: float
    curvature_similarity: float
    direction_consistency: float
    baseline_alignment: float
    pressure_stability: Optional[float]
    structural_factor: float
    auxiliary_factor: float
    final_score: float
    expected_strokes: int
    actual_strokes: int
    pressure_available: bool
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "shape_similarity": round(self.shape_similarity, 4),
            "stroke_count_match": round(self.stroke_count_match, 4),
            "aspect_ratio_match": round(self.aspect_ratio_match, 4),
            "curvature_similarity": round(self.curvature_similarity, 4),
            "direction_consistency": round(self.direction_consistency, 4),
            "baseline_alignment": round(self.baseline_alignment, 4),
            "pressure_stability": (None if self.pressure_stability is None
                                   else round(self.pressure_stability, 4)),
            "structural_factor": round(self.structural_factor, 4),
            "auxiliary_factor": round(self.auxiliary_factor, 4),
            "final_score": round(self.final_score, 4),
            "expected_strokes": self.expected_strokes,
            "actual_strokes": self.actual_strokes,
            "pressure_available": self.pressure_available,
            "notes": list(self.notes),
        }


@dataclass
class HandwritingScore:
    """Result of one scoring call. ``score`` is in [0, 100] for the
    learner-facing UI; ``normalized`` is the same value in [0, 1] used
    internally by the state builder and mastery tracker."""
    score: float            # 0-100, learner-facing
    normalized: float       # 0-1, internal
    breakdown: HandwritingScoreBreakdown

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 2),
            "normalized": round(self.normalized, 4),
            "breakdown": self.breakdown.to_dict(),
        }


# ---------------------------------------------------------------------------
# Exercise metadata helpers
# ---------------------------------------------------------------------------
# Letters whose body is a closed bowl (loop). The scorer uses this set as
# a conservative hint: only letters whose primary body is unambiguously a
# closed bowl are listed. Open letters (c, m, n, s, v, w, x, z, G, S) and
# triangle/wedge letters (A, M, N, V, W) are NOT in the set.
_CLOSED_LOOP_LETTERS = frozenset("abdegopqBDOPQR")


def is_closed_loop_letter(target: str) -> bool:
    """Whether a letter's main body is a closed bowl/loop."""
    return target in _CLOSED_LOOP_LETTERS


def expected_stroke_count(reference_strokes: list) -> int:
    """The number of pen strokes the reference template defines.

    The reference template stored on an :class:`Exercise` is the flattened
    concatenation; the original per-stroke structure lives in
    ``exercise.strokes``. Callers pass the per-stroke list directly.
    Falls back to 1 stroke if no per-stroke info is available.
    """
    if not reference_strokes:
        return 1
    n = 0
    for s in reference_strokes:
        if s and len(s) >= 2:
            n += 1
    return max(1, n)


# ---------------------------------------------------------------------------
# Per-component scorers
# ---------------------------------------------------------------------------
def _geometric_mean(values: list[float]) -> float:
    """Geometric mean, treating an empty list as 1.0 (neutral) and
    clipping each input to [0, 1]. A single 0 forces the result to 0
    (this is the desired multiplicative-cap behaviour)."""
    if not values:
        return 1.0
    vals = [max(0.0, min(1.0, float(v))) for v in values]
    if any(v <= 0.0 for v in vals):
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def _stroke_count_score(actual: int, expected: int, tolerance: int) -> float:
    """1.0 when ``actual`` is within ``tolerance`` of ``expected``;
    linearly falls off to 0 as the gap grows. A zero-stroke attempt is
    always 0."""
    if actual <= 0:
        return 0.0
    if expected <= 0:
        return 1.0 if actual >= 1 else 0.0
    gap = abs(actual - expected)
    if gap <= tolerance:
        return 1.0
    excess = gap - tolerance
    # Linearly decay so each extra missing/extra stroke costs ~1/3 of the
    # score until it reaches 0 at 3 strokes beyond tolerance.
    return float(max(0.0, 1.0 - excess / 3.0))


def _aspect_ratio_score(child_w: float, child_h: float,
                        ref_w: float, ref_h: float,
                        max_ratio: float = 1e3) -> float:
    """Aspect-ratio match in [0, 1]. Compares w/h ratios (guarded against
    division by zero). Returns 1.0 when the ratios are equal, falling off
    linearly with the relative difference."""
    def safe_ratio(w: float, h: float) -> float:
        if h < 1e-6:
            return max_ratio if w > 1e-6 else 1.0
        return float(np.clip(w / h, 0.0, max_ratio))

    cr, rr = safe_ratio(child_w, child_h), safe_ratio(ref_w, ref_h)
    if rr < 1e-6:
        return 0.5
    rel = abs(cr - rr) / max(rr, 1e-6)
    return float(np.clip(1.0 - rel, 0.0, 1.0))


def _turning_angle_sequence(xs: np.ndarray, ys: np.ndarray,
                            n_bins: int = 16) -> np.ndarray:
    """Histogram of turning angles (radians, wrapped to [-pi, pi]) along a
    2-D path, normalized to sum to 1. Robust to resampling rate because it
    is a histogram over angles, not a sequence comparison."""
    if len(xs) < 4:
        return np.zeros(n_bins)
    dx, dy = np.diff(xs), np.diff(ys)
    angles = np.arctan2(dy, dx)
    dtheta = np.diff(angles)
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi  # wrap to [-pi, pi]
    hist, _ = np.histogram(dtheta, bins=n_bins, range=(-np.pi, np.pi))
    total = float(hist.sum())
    if total <= 0:
        return np.zeros(n_bins)
    return hist.astype(np.float64) / total


def _curvature_similarity(child_seq: np.ndarray, ref_seq: np.ndarray,
                           n_bins: int = 16) -> float:
    """Bhattacharyya-style coefficient between the turning-angle histograms
    of two normalized shapes. Returns 1.0 when the histograms are
    identical, falling off to 0 when they are disjoint. Scale and
    translation invariant by construction."""
    if len(child_seq) < 4 or len(ref_seq) < 4:
        return 0.5
    h_c = _turning_angle_sequence(child_seq[:, 0], child_seq[:, 1], n_bins)
    h_r = _turning_angle_sequence(ref_seq[:, 0], ref_seq[:, 1], n_bins)
    # Bhattacharyya coefficient: sum(sqrt(p_i * q_i)) in [0, 1].
    bc = float(np.sum(np.sqrt(np.clip(h_c * h_r, 0.0, None))))
    return float(np.clip(bc, 0.0, 1.0))


def _dominant_directions(seq: np.ndarray, n_segments: int = 8) -> np.ndarray:
    """Unit direction vectors of `n_segments` equally arc-length-spaced
    segments along a 2-D path. Returns shape (n_segments, 2). Used for a
    coarse direction-sequence comparison that is invariant to point
    density along the trace."""
    if len(seq) < 2:
        return np.zeros((n_segments, 2))
    seg_lengths = np.hypot(np.diff(seq[:, 0]), np.diff(seq[:, 1]))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total < 1e-9:
        return np.zeros((n_segments, 2))
    targets = np.linspace(0.0, total, n_segments + 1)
    dirs = []
    for i in range(n_segments):
        lo, hi = targets[i], targets[i + 1]
        if hi - lo < 1e-9:
            dirs.append((0.0, 0.0))
            continue
        # Average direction over [lo, hi]: integrate the unit tangent.
        x_lo = np.interp(lo, cumulative, seq[:, 0])
        y_lo = np.interp(lo, cumulative, seq[:, 1])
        x_hi = np.interp(hi, cumulative, seq[:, 0])
        y_hi = np.interp(hi, cumulative, seq[:, 1])
        dx, dy = x_hi - x_lo, y_hi - y_lo
        n = float(np.hypot(dx, dy))
        dirs.append((dx / n, dy / n) if n > 1e-9 else (0.0, 0.0))
    return np.array(dirs, dtype=np.float64)


def _direction_consistency(child_seq: np.ndarray, ref_seq: np.ndarray,
                            n_segments: int = 8) -> float:
    """Mean cosine similarity between corresponding dominant direction
    vectors of the child and reference shapes. Returns 1.0 when the two
    traces follow the same direction at every segment, 0.0 when their
    directions are orthogonal/opposite at every segment. A scribble's
    directions are random, so its mean cosine similarity is near 0."""
    d_c = _dominant_directions(child_seq, n_segments)
    d_r = _dominant_directions(ref_seq, n_segments)
    if d_c.shape != d_r.shape or np.all(d_c == 0.0) or np.all(d_r == 0.0):
        return 0.5
    dots = np.sum(d_c * d_r, axis=1)
    # Cosine similarity per segment, mapped from [-1, 1] to [0, 1].
    cos = (dots + 1.0) / 2.0
    return float(np.clip(np.mean(cos), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def compute_handwriting_score(
    strokes: list[Stroke],
    reference_template: Optional[np.ndarray],
    reference_strokes: Optional[list] = None,
    static: Optional[StaticFeatures] = None,
    dynamic: Optional[DynamicFeatures] = None,
    cfg: Optional[HandwritingScoreConfig] = None,
) -> HandwritingScore:
    """Score one handwriting attempt against an exercise's reference.

    Parameters
    ----------
    strokes : list[Stroke]
        Raw (already outlier-clipped) stylus strokes for one attempt.
    reference_template : np.ndarray (N, 2) or None
        Flattened reference template. When ``None``, the scorer degrades
        to a "controlled line" reference (the static feature extractor's
        default) and the structural checks that need a per-stroke
        reference are skipped.
    reference_strokes : list[list[[x, y]]] or None
        Per-stroke reference (e.g. ``exercise.strokes``). Used to determine
        the expected stroke count. Falls back to 1 if not supplied.
    static : StaticFeatures or None
        Pre-computed static features. The DTW shape similarity from this
        is used as a starting point, but is RECOMPUTED symmetrically
        (both child and reference smoothed) so letters with sharp
        transitions (m, n, r, ...) are not penalized by the asymmetry in
        the legacy ``extract_static_features`` (which smooths the child
        but not the reference).
    dynamic : DynamicFeatures or None
        Pre-computed dynamic features. Used to read pressure availability
        and the pressure-stability term.
    cfg : HandwritingScoreConfig or None
        Scoring configuration. Falls back to ``settings.handwriting_score``.
    """
    cfg = cfg or settings.handwriting_score
    notes: list[str] = []
    # NOTE: we do NOT call clean_strokes on the input strokes here. The
    # legacy clean_strokes applies MAD-based outlier clipping that is
    # appropriate for smooth shape traces but mistreats letters with
    # sharp vertical transitions (m, n, r -- the stem bottoms look like
    # "outliers" because they are large single-step jumps in y). For
    # the shape-similarity and structural checks computed here we want
    # the raw trace; the legacy extract_static_features still applies
    # its own cleaning when we use it for the baseline_alignment value.
    raw_strokes = [s for s in strokes if len(s.points) >= 1]
    actual_strokes = len(raw_strokes)

    # ----- pre-computed features (re-use when supplied) -----
    if static is None:
        from ..features.static_features import extract_static_features
        static = extract_static_features(strokes, reference_template=reference_template)
    if dynamic is None:
        from ..features.dynamic_features import extract_dynamic_features
        dynamic = extract_dynamic_features(strokes)

    baseline_alignment = float(np.clip(static.baseline_alignment, 0.0, 1.0))

    # ----- SYMMETRIC shape similarity (DTW with both sides smoothed) -----
    # The legacy extract_static_features smooths the child but not the
    # reference, which works fine for the 5 smooth shape exercises but
    # unfairly penalizes letters with sharp transitions (m, n, r, etc.).
    # We recompute the DTW here with BOTH sides smoothed so the shape
    # comparison is symmetric. We also use the RAW (un-cleaned) strokes
    # because clean_strokes' MAD-based outlier clipping mistreats sharp
    # stem-bottom transitions in letters.
    n_points = settings.preprocessing.resample_points
    if raw_strokes:
        all_xs, all_ys = concatenate_strokes(raw_strokes)
        rx, ry = resample_by_arc_length(all_xs, all_ys, n_points)
        rx = smooth_series(rx)
        ry = smooth_series(ry)
        child_nx, child_ny = normalize_shape(rx, ry)
        child_seq = np.stack([child_nx, child_ny], axis=1)
    else:
        child_seq = np.zeros((n_points, 2))

    if reference_template is not None and len(reference_template) >= 2:
        ref_arr = np.asarray(reference_template, dtype=np.float64)
        ref_x, ref_y = resample_by_arc_length(ref_arr[:, 0], ref_arr[:, 1], n_points)
        # Apply the SAME smoothing to the reference so the comparison is
        # symmetric.
        ref_x = smooth_series(ref_x)
        ref_y = smooth_series(ref_y)
        ref_nx, ref_ny = normalize_shape(ref_x, ref_y)
        ref_seq = np.stack([ref_nx, ref_ny], axis=1)

        # Symmetric DTW (both sides smoothed, raw strokes used).
        from ..features.static_features import _banded_dtw_distance
        scfg = settings.static_features
        dtw_dist = _banded_dtw_distance(child_seq, ref_seq, scfg.dtw_band_fraction)
        shape_similarity = float(np.exp(-dtw_dist / scfg.dtw_distance_to_similarity_scale)) \
            if np.isfinite(dtw_dist) else 0.0
    else:
        # No reference: fall back to the legacy (asymmetric) value, which
        # for a no-reference shape is fine.
        shape_similarity = float(np.clip(static.stroke_similarity, 0.0, 1.0))
        ref_seq = np.zeros((n_points, 2))

    # ----- stroke-count check -----
    expected = expected_stroke_count(reference_strokes) if reference_strokes else 1
    stroke_count_match = _stroke_count_score(actual_strokes, expected,
                                              cfg.stroke_count_tolerance)

    # ----- aspect-ratio check -----
    if raw_strokes:
        all_xs, all_ys = concatenate_strokes(raw_strokes)
        child_w = float(all_xs.max() - all_xs.min()) if len(all_xs) else 0.0
        child_h = float(all_ys.max() - all_ys.min()) if len(all_ys) else 0.0
    else:
        child_w, child_h = 0.0, 0.0
    if reference_template is not None and len(reference_template) >= 2:
        ref_arr = np.asarray(reference_template, dtype=np.float64)
        ref_w = float(ref_arr[:, 0].max() - ref_arr[:, 0].min())
        ref_h = float(ref_arr[:, 1].max() - ref_arr[:, 1].min())
    else:
        # Without a reference, neutral: do not penalize on aspect ratio.
        ref_w, ref_h = max(child_w, 1e-6), max(child_h, 1e-6)
    aspect_ratio_match = _aspect_ratio_score(child_w, child_h, ref_w, ref_h)

    # ----- curvature & direction checks -----
    if reference_template is not None and len(reference_template) >= 2:
        curvature_similarity = _curvature_similarity(child_seq, ref_seq,
                                                      n_bins=cfg.curvature_bins)
        direction_consistency = _direction_consistency(child_seq, ref_seq,
                                                         n_segments=cfg.direction_segments)
    else:
        # No reference: degrade the structural checks to neutral so we
        # do not silently fail an attempt that simply has no template.
        curvature_similarity = 0.5
        direction_consistency = 0.5

    # ----- pressure stability (only when reliable) -----
    pressure_stability: Optional[float] = None
    pressure_available = bool(dynamic.pressure_available)
    if pressure_available:
        pressure_stability = float(np.clip(1.0 - dynamic.pressure_variance_normalized,
                                           0.0, 1.0))

    # ----- combine -----
    # Structural factor: a multiplicative cap. Any single weak structural
    # signal pulls the final score down, so a zigzag (poor curvature
    # similarity AND poor direction consistency) cannot be rescued by a
    # coincidentally good aspect-ratio match.
    structural_components = [
        stroke_count_match,
        aspect_ratio_match,
        float(np.clip(curvature_similarity, 0.0, 1.0)),
        float(np.clip(direction_consistency, 0.0, 1.0)),
    ]
    structural_factor = _geometric_mean(structural_components)

    # Auxiliary factor: rewards neatness without overpowering shape.
    # The structural factor is weighted more heavily than the auxiliary
    # factor because auxiliary features (baseline alignment, pressure)
    # can be coincidentally good for a wrong shape (a zigzag can sit on
    # the baseline perfectly). To keep the auxiliary factor from tanking
    # the score for letters whose baseline-alignment metric is poorly
    # defined (m, n, r -- the legacy envelope metric was tuned for
    # smooth shapes), we use a soft floor on the auxiliary contribution.
    aux_components = [baseline_alignment]
    if pressure_stability is not None:
        aux_components.append(pressure_stability)
    raw_aux = _geometric_mean(aux_components)
    # Soft floor: a perfect shape should not be tanked by a poorly-
    # defined auxiliary metric. The auxiliary factor can ADD up to
    # (1 - auxiliary_weight) of bonus, but cannot SUBTRACT more than
    # auxiliary_weight from the shape*structural product.
    soft_aux = float(np.clip(raw_aux, 0.5, 1.0))
    auxiliary_factor = soft_aux
    aux_pow = cfg.auxiliary_weight
    aux_term = float(np.clip(auxiliary_factor, 0.0, 1.0)) ** aux_pow

    final = float(np.clip(shape_similarity * structural_factor * aux_term, 0.0, 1.0))

    # Diagnostic notes
    if actual_strokes == 0:
        notes.append("no strokes captured")
    if expected and abs(actual_strokes - expected) > cfg.stroke_count_tolerance:
        notes.append(f"stroke count {actual_strokes} differs from expected {expected}")
    if shape_similarity < cfg.shape_floor:
        notes.append(f"shape_similarity {shape_similarity:.2f} below floor {cfg.shape_floor:.2f}")
    if not pressure_available:
        notes.append("pressure data unavailable; pressure term omitted")

    breakdown = HandwritingScoreBreakdown(
        shape_similarity=shape_similarity,
        stroke_count_match=stroke_count_match,
        aspect_ratio_match=aspect_ratio_match,
        curvature_similarity=float(curvature_similarity),
        direction_consistency=float(direction_consistency),
        baseline_alignment=baseline_alignment,
        pressure_stability=pressure_stability,
        structural_factor=structural_factor,
        auxiliary_factor=auxiliary_factor,
        final_score=final,
        expected_strokes=expected,
        actual_strokes=actual_strokes,
        pressure_available=pressure_available,
        notes=notes,
    )
    return HandwritingScore(
        score=float(np.clip(final * 100.0, 0.0, 100.0)),
        normalized=final,
        breakdown=breakdown,
    )
