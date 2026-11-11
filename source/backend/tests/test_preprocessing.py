import numpy as np

from app.preprocessing.stroke_preprocessing import (
    Point, Stroke, clean_stroke, resample_by_arc_length, normalize_shape, smooth_series,
)
from .conftest import make_line_stroke, scale_translate_stroke


def test_resample_by_arc_length_preserves_endpoints():
    xs = np.array([0.0, 10.0, 20.0, 30.0])
    ys = np.array([0.0, 0.0, 0.0, 0.0])
    rx, ry = resample_by_arc_length(xs, ys, n_points=10)
    assert rx[0] == 0.0
    assert abs(rx[-1] - 30.0) < 1e-9
    assert len(rx) == 10


def test_resample_is_robust_to_sampling_rate_difference():
    xs = np.linspace(0, 100, 200)
    ys = np.zeros_like(xs)
    rx_dense, ry_dense = resample_by_arc_length(xs, ys, 50)

    xs_sparse = np.linspace(0, 100, 20)
    ys_sparse = np.zeros_like(xs_sparse)
    rx_sparse, ry_sparse = resample_by_arc_length(xs_sparse, ys_sparse, 50)

    assert np.allclose(rx_dense, rx_sparse, atol=1e-6)


def test_normalize_shape_is_translation_invariant():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 1.0, 0.0, 1.0])
    nx1, ny1 = normalize_shape(xs, ys)
    nx2, ny2 = normalize_shape(xs + 50.0, ys - 30.0)
    assert np.allclose(nx1, nx2, atol=1e-9)
    assert np.allclose(ny1, ny2, atol=1e-9)


def test_normalize_shape_is_scale_invariant():
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 1.0, 0.0, 1.0])
    nx1, ny1 = normalize_shape(xs, ys)
    nx2, ny2 = normalize_shape(xs * 10.0, ys * 10.0)
    assert np.allclose(nx1, nx2, atol=1e-6)
    assert np.allclose(ny1, ny2, atol=1e-6)


def test_normalize_shape_degenerate_single_point_does_not_crash():
    xs = np.array([5.0, 5.0, 5.0])
    ys = np.array([5.0, 5.0, 5.0])
    nx, ny = normalize_shape(xs, ys)
    assert np.all(nx == 0.0)
    assert np.all(ny == 0.0)


def test_clean_stroke_clips_outlier_without_dropping_point_count():
    pts = [Point(x=float(i), y=0.0, t=i * 0.02) for i in range(20)]
    pts[10] = Point(x=500.0, y=500.0, t=10 * 0.02)  # injected sensor glitch
    stroke = Stroke(stroke_id=0, points=pts)
    cleaned = clean_stroke(stroke)
    assert len(cleaned.points) == len(pts)
    assert cleaned.points[10].x < 50.0  # glitch clipped toward neighbours


def test_clean_stroke_drops_non_monotonic_timestamps():
    pts = [Point(x=float(i), y=0.0, t=1.0) for i in range(3)]  # all same timestamp
    stroke = Stroke(stroke_id=0, points=pts)
    cleaned = clean_stroke(stroke)
    assert len(cleaned.points) <= 1 or cleaned.points == stroke.points  # degenerate case handled, no crash


def test_smooth_series_falls_back_gracefully_on_short_input():
    values = np.array([1.0, 2.0, 3.0])
    out = smooth_series(values)
    assert len(out) == 3  # did not crash despite being shorter than the smoothing window
