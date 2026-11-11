"""
Deterministic synthetic handwriting perturbations (SYNTHETIC DATA ONLY)
========================================================================
Generates clean synthetic pen trajectories for the exercise templates and
applies controlled distortions at a given severity in [0, 1]. Used only
to test whether the quality proxy (state.state_builder.compute_quality_score)
gets worse as the distortion gets stronger. That is a perturbation-sensitivity
test. Nothing generated here is, or may be labelled as, real-child
handwriting. Every record built from it carries data_source="synthetic_test".

Perturbation types. Magnitudes are fractions of the specimen's
bounding-box diagonal D, so they do not depend on scale:
  jitter       i.i.d. Gaussian noise on x and y, sigma = 0.06*s*D.
  tremor       sinusoidal displacement perpendicular to the direction of
               travel, amplitude 0.05*s*D, frequency 9 Hz (inside the
               4-12 Hz range usually cited for tremor), random phase.
  scale        progressive size distortion: the size factor changes
               linearly along the trajectory from 1 to 1 - 0.6*s, about
               the specimen centroid (inconsistent letter size).
  slant        horizontal shear x' = x + tan(35deg * s) * (y_c - y).
  spacing      multi-stroke specimen (a row of evenly spaced strokes).
               Each inter-stroke gap is changed by a factor
               1 + 0.8*s*u, u ~ U(-1, 1) (irregular spacing).
The random draws come from numpy.random.default_rng(seed), so a given
(type, severity, seed) always produces the same trajectory.
"""

from __future__ import annotations

import numpy as np

from ..preprocessing.stroke_preprocessing import Point, Stroke

SAMPLE_RATE_HZ = 120.0
PERTURBATION_TYPES = ("jitter", "tremor", "scale", "slant", "spacing")


def _resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, xy[:, 0]), np.interp(t, s, xy[:, 1])], axis=1)


def clean_specimen(template: np.ndarray, n_points: int = 120) -> np.ndarray:
    """Dense, evenly sampled version of a template polyline (N, 2)."""
    return _resample_polyline(np.asarray(template, dtype=float), n_points)


def spacing_specimen(n_strokes: int = 4, gap: float = 30.0, height: float = 40.0,
                     points_per_stroke: int = 30) -> list[np.ndarray]:
    """A row of `n_strokes` short vertical strokes, evenly spaced (synthetic)."""
    strokes = []
    for i in range(n_strokes):
        x = 10.0 + i * gap
        ys = np.linspace(30.0, 30.0 + height, points_per_stroke)
        strokes.append(np.stack([np.full_like(ys, x), ys], axis=1))
    return strokes


def _diag(xy: np.ndarray) -> float:
    return float(np.hypot(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))) or 1.0


def perturb_single(xy: np.ndarray, kind: str, severity: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, dtype=float).copy()
    D = _diag(xy)
    s = float(np.clip(severity, 0.0, 1.0))
    if kind == "jitter":
        return xy + rng.normal(0.0, 0.06 * s * D, size=xy.shape)
    if kind == "tremor":
        n = len(xy)
        t = np.arange(n) / SAMPLE_RATE_HZ
        d = np.gradient(xy, axis=0)
        norm = np.hypot(d[:, 0], d[:, 1])[:, None] + 1e-9
        perp = np.stack([-d[:, 1], d[:, 0]], axis=1) / norm
        phase = rng.uniform(0, 2 * np.pi)
        amp = 0.05 * s * D * np.sin(2 * np.pi * 9.0 * t + phase)
        return xy + perp * amp[:, None]
    if kind == "scale":
        c = xy.mean(axis=0)
        factor = np.linspace(1.0, 1.0 - 0.6 * s, len(xy))[:, None]
        return c + (xy - c) * factor
    if kind == "slant":
        yc = xy[:, 1].mean()
        out = xy.copy()
        out[:, 0] = xy[:, 0] + np.tan(np.deg2rad(35.0 * s)) * (yc - xy[:, 1])
        return out
    raise ValueError(f"unknown single-stroke perturbation {kind!r}")


def perturb_spacing(strokes: list[np.ndarray], severity: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    s = float(np.clip(severity, 0.0, 1.0))
    xs = [float(st[:, 0].mean()) for st in strokes]
    gaps = np.diff(xs)
    new_gaps = gaps * (1.0 + 0.8 * s * rng.uniform(-1.0, 1.0, size=len(gaps)))
    new_x = np.concatenate([[xs[0]], xs[0] + np.cumsum(new_gaps)])
    return [st + np.array([nx - ox, 0.0]) for st, nx, ox in zip(strokes, new_x, xs)]


def to_strokes(paths: list[np.ndarray], pressure: float | None = None,
               inter_stroke_gap_s: float = 0.25) -> list[Stroke]:
    """Wrap (N, 2) paths as timestamped strokes sampled at 120 Hz.
    `pressure=None` means "device reports no pressure". No value is invented."""
    strokes, t0 = [], 0.0
    for sid, xy in enumerate(paths):
        pts = [Point(x=float(x), y=float(y), t=t0 + i / SAMPLE_RATE_HZ, pressure=pressure)
               for i, (x, y) in enumerate(xy)]
        strokes.append(Stroke(stroke_id=sid, points=pts))
        t0 = pts[-1].t + inter_stroke_gap_s
    return strokes
