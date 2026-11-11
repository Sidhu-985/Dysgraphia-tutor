"""
Feature validation for the static extractor (project spec Section 4):
repeatability, noise robustness, resampling robustness, scale/translation
robustness, sensitivity to intended changes, numerical stability. No
clinical-accuracy claims are made or tested -- there is no ground truth
for that here.
"""

import numpy as np
import pytest

from app.features.static_features import extract_static_features
from app.preprocessing.stroke_preprocessing import Stroke
from .conftest import make_line_stroke, make_loop_stroke, scale_translate_stroke, resample_stroke_timestamps


def test_repeatability_identical_input_gives_identical_output():
    stroke = make_line_stroke()
    f1 = extract_static_features([stroke])
    f2 = extract_static_features([stroke])
    assert f1.to_dict() == f2.to_dict()


def test_stroke_similarity_is_high_for_shape_matching_its_own_reference():
    stroke = make_line_stroke()
    ref = np.array([[p.x, p.y] for p in stroke.points])
    feats = extract_static_features([stroke], reference_template=ref)
    assert feats.stroke_similarity > 0.9


def test_stroke_similarity_is_lower_for_a_dissimilar_shape():
    line = make_line_stroke()
    loop = make_loop_stroke()
    ref = np.array([[p.x, p.y] for p in line.points])

    line_feats = extract_static_features([line], reference_template=ref)
    loop_feats = extract_static_features([loop], reference_template=ref)

    assert line_feats.stroke_similarity > loop_feats.stroke_similarity


def test_scale_invariance_of_stroke_similarity():
    stroke = make_line_stroke()
    ref = np.array([[p.x, p.y] for p in stroke.points])

    scaled = scale_translate_stroke(stroke, scale=3.0)
    f_original = extract_static_features([stroke], reference_template=ref)
    f_scaled = extract_static_features([scaled], reference_template=ref)

    assert abs(f_original.stroke_similarity - f_scaled.stroke_similarity) < 0.05


def test_translation_invariance_of_stroke_similarity():
    stroke = make_line_stroke()
    ref = np.array([[p.x, p.y] for p in stroke.points])

    translated = scale_translate_stroke(stroke, scale=1.0, dx=500.0, dy=-300.0)
    f_original = extract_static_features([stroke], reference_template=ref)
    f_translated = extract_static_features([translated], reference_template=ref)

    assert abs(f_original.stroke_similarity - f_translated.stroke_similarity) < 0.02


def test_resampling_rate_robustness_of_stroke_similarity():
    stroke = make_line_stroke(n=100)
    ref = np.array([[p.x, p.y] for p in stroke.points])

    sparse = resample_stroke_timestamps(stroke, factor=0.2)
    f_dense = extract_static_features([stroke], reference_template=ref)
    f_sparse = extract_static_features([sparse], reference_template=ref)

    assert abs(f_dense.stroke_similarity - f_sparse.stroke_similarity) < 0.05


def test_noise_robustness_of_stroke_similarity():
    clean = make_line_stroke(noise=0.0)
    noisy = make_line_stroke(noise=1.5, seed=42)
    ref = np.array([[p.x, p.y] for p in clean.points])

    f_clean = extract_static_features([clean], reference_template=ref)
    f_noisy = extract_static_features([noisy], reference_template=ref)

    # Similarity should degrade gracefully, not catastrophically, under
    # small positional noise relative to the shape's own scale (100 units).
    assert f_noisy.stroke_similarity > 0.5
    assert f_clean.stroke_similarity >= f_noisy.stroke_similarity


def test_baseline_alignment_is_high_for_a_flat_line():
    stroke = make_line_stroke(y=50.0)
    feats = extract_static_features([stroke])
    assert feats.baseline_alignment > 0.95


def test_baseline_alignment_degrades_for_a_sloped_line():
    n = 40
    from app.preprocessing.stroke_preprocessing import Point
    pts = [Point(x=float(i), y=float(i) * 2.0, t=i * 0.02) for i in range(n)]
    stroke = Stroke(stroke_id=0, points=pts)
    flat = make_line_stroke(n=n)

    sloped_feats = extract_static_features([stroke])
    flat_feats = extract_static_features([flat])
    assert sloped_feats.baseline_alignment < flat_feats.baseline_alignment


def test_spacing_accuracy_neutral_for_single_stroke():
    stroke = make_line_stroke()
    feats = extract_static_features([stroke])
    assert feats.spacing_accuracy == 0.5


def test_empty_strokes_do_not_crash_and_return_neutral_defaults():
    feats = extract_static_features([])
    assert feats.n_strokes == 0
    assert 0.0 <= feats.stroke_similarity <= 1.0


def test_numerical_stability_all_outputs_finite():
    stroke = make_line_stroke()
    feats = extract_static_features([stroke])
    for k, v in feats.to_dict().items():
        if isinstance(v, float):
            assert not np.isnan(v), f"{k} is NaN"


def test_sensitivity_to_intended_change_more_curvy_shape_lowers_smoothness():
    straight = make_line_stroke()
    jagged_points = []
    for i, p in enumerate(straight.points):
        y = p.y + (10 if i % 2 == 0 else -10)
        jagged_points.append(type(p)(x=p.x, y=y, t=p.t, pressure=p.pressure))
    jagged = Stroke(stroke_id=0, points=jagged_points)

    f_straight = extract_static_features([straight])
    f_jagged = extract_static_features([jagged])
    assert f_jagged.curvature_smoothness < f_straight.curvature_smoothness
