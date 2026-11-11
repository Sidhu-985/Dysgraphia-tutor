"""
Stroke preprocessing
=====================
Shared, low-level cleaning applied to raw tablet strokes before either the
static or dynamic feature extractors see them. This module knows nothing
about "handwriting quality" -- it only turns noisy, irregularly-sampled
(x, y, t, pressure) point sequences into cleaner sequences that downstream
extractors can trust:

  * outlier clipping (robust to occasional sensor glitches / jumps)
  * smoothing (Savitzky-Golay) to stabilize derivative-based features
  * equal-arc-length resampling (removes sampling-rate as a confound for
    shape comparison)
  * scale/translation normalization (removes absolute position/size as a
    confound for shape comparison)

Kept independent of the static/dynamic feature modules so it can be tuned
or replaced (e.g. a different smoothing filter) without touching either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

from ..config import PreprocessingConfig, settings


@dataclass
class Point:
    x: float
    y: float
    t: float                       # seconds, monotonically increasing within a stroke
    pressure: Optional[float] = None


@dataclass
class Stroke:
    stroke_id: int
    points: list[Point] = field(default_factory=list)

    def as_arrays(self):
        xs = np.array([p.x for p in self.points], dtype=np.float64)
        ys = np.array([p.y for p in self.points], dtype=np.float64)
        ts = np.array([p.t for p in self.points], dtype=np.float64)
        pressures = np.array(
            [p.pressure if p.pressure is not None else np.nan for p in self.points],
            dtype=np.float64,
        )
        return xs, ys, ts, pressures


def _clip_outliers_1d(values: np.ndarray, mad_threshold: float) -> np.ndarray:
    """Robust outlier clipping via median-absolute-deviation on the
    point-to-point step size. A single bad sample is replaced by the local
    neighbour average rather than dropped, which would otherwise break
    timestamp alignment for the dynamic feature extractor."""
    if len(values) < 3:
        return values
    diffs = np.abs(np.diff(values))
    median_diff = np.median(diffs)
    mad = np.median(np.abs(diffs - median_diff)) + 1e-9
    out = values.copy()
    for i in range(1, len(out) - 1):
        if abs(diffs[i - 1]) > median_diff + mad_threshold * mad and \
           abs(diffs[min(i, len(diffs) - 1)]) > median_diff + mad_threshold * mad:
            out[i] = 0.5 * (out[i - 1] + out[i + 1])
    return out


def clean_stroke(stroke: Stroke, cfg: Optional[PreprocessingConfig] = None) -> Stroke:
    """Outlier-clip + drop degenerate (duplicate-timestamp) points. Does NOT
    resample or smooth -- those are separate steps callers opt into,
    because dynamic features need original timing while static features
    need resampled/smoothed geometry."""
    cfg = cfg or settings.preprocessing
    if len(stroke.points) < cfg.min_points_per_stroke:
        return stroke

    xs, ys, ts, pressures = stroke.as_arrays()

    # Drop non-monotonic / duplicate timestamps (keep first occurrence).
    keep = np.ones(len(ts), dtype=bool)
    last_t = -np.inf
    for i, t in enumerate(ts):
        if t <= last_t:
            keep[i] = False
        else:
            last_t = t
    if keep.sum() < 2:
        return stroke

    xs, ys, ts, pressures = xs[keep], ys[keep], ts[keep], pressures[keep]
    xs = _clip_outliers_1d(xs, cfg.outlier_mad_threshold)
    ys = _clip_outliers_1d(ys, cfg.outlier_mad_threshold)

    cleaned_points = [
        Point(float(xs[i]), float(ys[i]), float(ts[i]),
              None if np.isnan(pressures[i]) else float(pressures[i]))
        for i in range(len(xs))
    ]
    return Stroke(stroke_id=stroke.stroke_id, points=cleaned_points)


def clean_strokes(strokes: list[Stroke], cfg: Optional[PreprocessingConfig] = None) -> list[Stroke]:
    return [clean_stroke(s, cfg) for s in strokes if len(s.points) >= 1]


def smooth_series(values: np.ndarray, cfg: Optional[PreprocessingConfig] = None) -> np.ndarray:
    """Savitzky-Golay smoothing, falling back to the raw series when there
    are too few points for the configured window (short strokes / a single
    dot are common and must not crash the pipeline)."""
    cfg = cfg or settings.preprocessing
    n = len(values)
    window = cfg.smoothing_window
    if window % 2 == 0:
        window += 1
    if n < window or window <= cfg.smoothing_polyorder:
        return values.astype(np.float64)
    return savgol_filter(values, window_length=window, polyorder=cfg.smoothing_polyorder)


def resample_by_arc_length(xs: np.ndarray, ys: np.ndarray, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Resample a 2D path to `n_points` equally spaced along arc length.
    This is what makes shape comparison (DTW similarity) independent of
    the tablet's raw sampling rate and of how fast the child drew --
    without it, a slowly-drawn stroke would simply have more points than a
    fast one and bias any point-wise or DTW comparison.
    """
    if len(xs) < 2:
        return np.repeat(xs, n_points), np.repeat(ys, n_points)

    seg_lengths = np.hypot(np.diff(xs), np.diff(ys))
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]

    if total_length < 1e-9:
        return np.repeat(xs[0], n_points), np.repeat(ys[0], n_points)

    targets = np.linspace(0.0, total_length, n_points)
    resampled_x = np.interp(targets, cumulative, xs)
    resampled_y = np.interp(targets, cumulative, ys)
    return resampled_x, resampled_y


def normalize_shape(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to centroid and scale by bounding-box diagonal so shape
    comparisons (DTW) are invariant to where on the tablet the child wrote
    and how large they wrote it. If the shape is a single point (zero
    diagonal), returns zeros to avoid division by zero."""
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    x0, y0 = xs - cx, ys - cy
    diag = float(np.hypot(x0.max() - x0.min(), y0.max() - y0.min())) if len(x0) else 0.0
    if diag < 1e-9:
        return np.zeros_like(x0), np.zeros_like(y0)
    return x0 / diag, y0 / diag


def concatenate_strokes(strokes: list[Stroke]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate all strokes' points into one (x, y) path, in stroke
    order, for whole-attempt shape comparison. Multi-stroke gaps are kept
    implicit in point order (no interpolation across strokes)."""
    xs, ys = [], []
    for s in strokes:
        for p in s.points:
            xs.append(p.x)
            ys.append(p.y)
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)
