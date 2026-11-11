"""
Handwriting-quality scoring tests (audit-driven, 2026-11)
=========================================================

These tests verify the new exercise-aware multiplicative scorer
(``app.scoring.handwriting_quality.compute_handwriting_score``)
correctly distinguishes correct handwriting from malformed
handwriting across multiple letters and failure modes.

The legacy reference-free ``compute_quality_score`` is tested
separately (test_state_builder.py) and continues to pass.

These tests use SYNTHETIC traces generated deterministically from
the project's own letter templates; they make no clinical-accuracy
claims.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.exercises import CATALOGUE, LETTER_EXERCISES, get_exercise
from app.preprocessing.stroke_preprocessing import Point, Stroke
from app.scoring.handwriting_quality import (
    compute_handwriting_score,
    expected_stroke_count,
    is_closed_loop_letter,
)


# ---------------------------------------------------------------------------
# Synthetic trace builders (deterministic)
# ---------------------------------------------------------------------------
def _trace_template(strokes, scale=3.0, dt=0.02, lift=0.3, noise=0.0, seed=0):
    """Trace the per-stroke template, returning a list[Stroke]. Each
    stroke is one pen-down; the loop adds a pen-lift between strokes
    so the stroke count matches the template's stroke count."""
    rng = np.random.default_rng(seed) if noise else None
    out = []
    t = 0.0
    for sid, s in enumerate(strokes):
        pts = []
        for i, (x, y) in enumerate(s):
            nx = float(rng.normal(0, noise)) if noise else 0.0
            ny = float(rng.normal(0, noise)) if noise else 0.0
            pts.append(Point(x=float(x * scale + nx), y=float(y * scale + ny),
                             t=t, pressure=0.5))
            t += dt
        out.append(Stroke(stroke_id=sid, points=pts))
        t += lift
    return out


def _flatten_trace(template, scale=3.0, dt=0.02):
    """One continuous stroke through the flattened template points
    (no pen lifts)."""
    pts = []
    t = 0.0
    for x, y in template:
        pts.append(Point(x=float(x * scale), y=float(y * scale), t=t, pressure=0.5))
        t += dt
    return [Stroke(stroke_id=0, points=pts)]


def _zigzag(width=100.0, height=80.0, n=60, dt=0.02):
    """A zigzag covering the same bounding-box area as a letter."""
    pts = []
    for i in range(n):
        x = (i / (n - 1)) * width
        y = 50 + (height / 2) * (1 if i % 2 == 0 else -1)
        pts.append(Point(x=float(x), y=float(y), t=i * dt, pressure=0.5))
    return [Stroke(stroke_id=0, points=pts)]


def _scribble(seed=0, n=80, dt=0.02):
    """A random-walk scribble unrelated to any letter."""
    rng = np.random.default_rng(seed)
    pts = []
    x, y = 50.0, 50.0
    for i in range(n):
        x += float(rng.normal(0, 4.0))
        y += float(rng.normal(0, 4.0))
        pts.append(Point(x=x, y=y, t=i * dt, pressure=0.5))
    return [Stroke(stroke_id=0, points=pts)]


def _vertical_line(n=40, dt=0.02):
    """A straight vertical line."""
    pts = []
    for i in range(n):
        x = 50.0
        y = 10 + (i / (n - 1)) * 60
        pts.append(Point(x=x, y=y, t=i * dt, pressure=0.5))
    return [Stroke(stroke_id=0, points=pts)]


def _horizontal_line(n=40, dt=0.02):
    pts = []
    for i in range(n):
        x = 10 + (i / (n - 1)) * 80
        y = 50.0
        pts.append(Point(x=x, y=y, t=i * dt, pressure=0.5))
    return [Stroke(stroke_id=0, points=pts)]


# ---------------------------------------------------------------------------
# Per-letter correctness tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("char", list("abcolOALmnpxrR"))
def test_correct_letter_scores_high(char):
    eid = f"letter_lower_{char}" if char.islower() else f"letter_upper_{char.lower()}"
    ex = get_exercise(eid)
    trace = _trace_template(ex.strokes)
    r = compute_handwriting_score(trace, ex.reference_array(), ex.strokes)
    assert r.normalized >= 0.65, (
        f"clean {char!r} should score at least 0.65; got {r.normalized:.3f} "
        f"(shape={r.breakdown.shape_similarity:.3f}, "
        f"structural={r.breakdown.structural_factor:.3f})"
    )


@pytest.mark.parametrize("char", ["a", "b", "o", "L", "A", "O"])
def test_letter_score_is_bounded_in_0_100(char):
    eid = f"letter_lower_{char}" if char.islower() else f"letter_upper_{char.lower()}"
    ex = get_exercise(eid)
    r = compute_handwriting_score(_trace_template(ex.strokes), ex.reference_array(), ex.strokes)
    assert 0.0 <= r.score <= 100.0
    assert 0.0 <= r.normalized <= 1.0


def test_correct_a_scores_high():
    ex = get_exercise("letter_lower_a")
    trace = _trace_template(ex.strokes)
    r = compute_handwriting_score(trace, ex.reference_array(), ex.strokes)
    assert r.score >= 80.0, f"clean lowercase a should score >= 80; got {r.score:.1f}"


def test_zigzag_instead_of_a_scores_very_low():
    """The core defect the audit requires fixed: a zigzag instead of a
    lowercase ``a`` must NOT receive a moderately high score."""
    ex = get_exercise("letter_lower_a")
    zigzag = _zigzag(width=100, height=80, n=60)
    r = compute_handwriting_score(zigzag, ex.reference_array(), ex.strokes)
    assert r.score <= 35.0, (
        f"zigzag instead of 'a' must score <= 35 (was getting ~50% before the fix); "
        f"got {r.score:.1f}"
    )
    # And the breakdown should explain why: weak shape (DTW) and weak
    # structural (curvature + direction) signals together drive the
    # score down. The structural factor itself may be moderate (a
    # zigzag has good aspect ratio and stroke count) but the shape
    # similarity is poor.
    assert r.breakdown.shape_similarity < 0.55
    assert r.breakdown.curvature_similarity < 0.5
    assert r.breakdown.direction_consistency < 0.55


def test_scribble_instead_of_a_scores_very_low():
    ex = get_exercise("letter_lower_a")
    for seed in range(5):
        r = compute_handwriting_score(_scribble(seed=seed), ex.reference_array(), ex.strokes)
        assert r.score <= 35.0, f"scribble (seed={seed}) must score <= 35; got {r.score:.1f}"


def test_incomplete_a_scores_low():
    """Just the bowl, no stem -- an incomplete letter should not pass."""
    ex = get_exercise("letter_lower_a")
    # Take only the first stroke (the bowl) of the template
    incomplete = _trace_template(ex.strokes[:1])
    r = compute_handwriting_score(incomplete, ex.reference_array(), ex.strokes)
    assert r.score <= 60.0, f"incomplete a should score <= 60; got {r.score:.1f}"


def test_distorted_a_scores_lower_than_clean():
    ex = get_exercise("letter_lower_a")
    clean = _trace_template(ex.strokes, noise=0.0)
    distorted = _trace_template(ex.strokes, noise=4.0, seed=7)
    rc = compute_handwriting_score(clean, ex.reference_array(), ex.strokes)
    rd = compute_handwriting_score(distorted, ex.reference_array(), ex.strokes)
    assert rc.score > rd.score, "distorted a should score lower than clean a"


# ---------------------------------------------------------------------------
# Scale, translation, and speed invariance
# ---------------------------------------------------------------------------
def test_correct_a_at_larger_scale_remains_high():
    ex = get_exercise("letter_lower_a")
    base = _trace_template(ex.strokes, scale=3.0)
    bigger = _trace_template(ex.strokes, scale=6.0)
    rb = compute_handwriting_score(bigger, ex.reference_array(), ex.strokes)
    r0 = compute_handwriting_score(base, ex.reference_array(), ex.strokes)
    # Should be within a few percentage points (DTW + curvature are
    # scale-invariant; only resampling discretization introduces drift).
    assert abs(rb.normalized - r0.normalized) < 0.10, (
        f"scale invariance broken: base={r0.normalized:.3f} bigger={rb.normalized:.3f}"
    )
    assert rb.score >= 80.0


def test_correct_a_at_smaller_scale_remains_high():
    ex = get_exercise("letter_lower_a")
    smaller = _trace_template(ex.strokes, scale=1.5)
    r = compute_handwriting_score(smaller, ex.reference_array(), ex.strokes)
    assert r.score >= 80.0


def test_correct_a_translated_remains_high():
    ex = get_exercise("letter_lower_a")
    base = _trace_template(ex.strokes, scale=3.0)
    # Translate by a large offset
    translated = [
        Stroke(stroke_id=s.stroke_id,
               points=[Point(x=p.x + 500.0, y=p.y - 300.0, t=p.t, pressure=p.pressure)
                       for p in s.points])
        for s in base
    ]
    r = compute_handwriting_score(translated, ex.reference_array(), ex.strokes)
    assert r.score >= 80.0


def test_correct_a_at_different_speed_remains_high():
    """Slower timestamps must not unfairly penalize the score (the
    scorer is geometry-based and should be largely time-invariant)."""
    ex = get_exercise("letter_lower_a")
    fast = _trace_template(ex.strokes, scale=3.0, dt=0.01)
    slow = _trace_template(ex.strokes, scale=3.0, dt=0.06)
    rf = compute_handwriting_score(fast, ex.reference_array(), ex.strokes)
    rs = compute_handwriting_score(slow, ex.reference_array(), ex.strokes)
    assert abs(rf.normalized - rs.normalized) < 0.10, (
        f"speed invariance broken: fast={rf.normalized:.3f} slow={rs.normalized:.3f}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_identical_input_gives_identical_score():
    ex = get_exercise("letter_lower_a")
    trace = _trace_template(ex.strokes)
    r1 = compute_handwriting_score(trace, ex.reference_array(), ex.strokes)
    r2 = compute_handwriting_score(trace, ex.reference_array(), ex.strokes)
    assert r1.score == r2.score
    assert r1.normalized == r2.normalized
    assert r1.breakdown.to_dict() == r2.breakdown.to_dict()


# ---------------------------------------------------------------------------
# Component breakdown transparency
# ---------------------------------------------------------------------------
def test_breakdown_contains_all_components():
    ex = get_exercise("letter_lower_a")
    trace = _trace_template(ex.strokes)
    r = compute_handwriting_score(trace, ex.reference_array(), ex.strokes)
    d = r.breakdown.to_dict()
    for k in ("shape_similarity", "stroke_count_match", "aspect_ratio_match",
              "curvature_similarity", "direction_consistency",
              "baseline_alignment", "structural_factor", "auxiliary_factor",
              "final_score", "expected_strokes", "actual_strokes",
              "pressure_available"):
        assert k in d, f"breakdown missing {k}"


def test_pressure_stability_only_when_available():
    ex = get_exercise("letter_lower_a")
    # No pressure
    trace_no_p = [
        Stroke(stroke_id=s.stroke_id,
               points=[Point(x=p.x, y=p.y, t=p.t, pressure=None) for p in s.points])
        for s in _trace_template(ex.strokes)
    ]
    r1 = compute_handwriting_score(trace_no_p, ex.reference_array(), ex.strokes)
    assert r1.breakdown.pressure_available is False
    assert r1.breakdown.pressure_stability is None
    # With pressure
    trace_p = _trace_template(ex.strokes)
    r2 = compute_handwriting_score(trace_p, ex.reference_array(), ex.strokes)
    assert r2.breakdown.pressure_available is True
    assert r2.breakdown.pressure_stability is not None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_strokes_score_zero():
    ex = get_exercise("letter_lower_a")
    r = compute_handwriting_score([], ex.reference_array(), ex.strokes)
    assert r.score <= 5.0
    assert r.normalized <= 0.05


def test_wrong_letter_scores_lower_than_correct_letter():
    """A vertical line drawn where 'a' is requested must score lower
    than a correct 'a'."""
    ex = get_exercise("letter_lower_a")
    correct = _trace_template(ex.strokes)
    rc = compute_handwriting_score(correct, ex.reference_array(), ex.strokes)
    rv = compute_handwriting_score(_vertical_line(), ex.reference_array(), ex.strokes)
    assert rc.score > rv.score + 30, (
        f"correct 'a' ({rc.score:.1f}) should beat vertical line ({rv.score:.1f}) by > 30"
    )


def test_old_vs_new_score_table_for_zigzag():
    """Reproduce the audit's headline defect and confirm the new scorer
    fixes it. Documents the OLD vs NEW score for the same deterministic
    zigzag-so-the-audit-can-see-the-improvement."""
    from app.features.static_features import extract_static_features
    from app.features.dynamic_features import extract_dynamic_features
    from app.state.state_builder import compute_quality_score

    ex = get_exercise("letter_lower_a")
    zigzag = _zigzag(width=100, height=80, n=60)
    sf = extract_static_features(zigzag, ex.reference_array())
    df = extract_dynamic_features(zigzag)
    old_score = compute_quality_score(sf, df)
    new_score = compute_handwriting_score(zigzag, ex.reference_array(), ex.strokes).normalized
    # Document the improvement: old scorer ~0.45-0.50, new scorer < 0.35
    assert new_score < old_score, (
        f"new scorer should be stricter on zigzag: old={old_score:.3f} new={new_score:.3f}"
    )
    assert new_score <= 0.35, f"new scorer should give <= 0.35; got {new_score:.3f}"


def test_expected_stroke_count_helper():
    assert expected_stroke_count([]) == 1
    # 'a' has 2 strokes (arc + line)
    ex = get_exercise("letter_lower_a")
    assert expected_stroke_count(ex.strokes) == 2
    # 'A' has 3 strokes
    ex = get_exercise("letter_upper_a")
    assert expected_stroke_count(ex.strokes) == 3
    # 'L' has 1 stroke
    ex = get_exercise("letter_upper_l")
    assert expected_stroke_count(ex.strokes) == 1


def test_is_closed_loop_letter_helper():
    assert is_closed_loop_letter("a")
    assert is_closed_loop_letter("o")
    assert is_closed_loop_letter("B")
    assert is_closed_loop_letter("D")
    # 'L' is not closed-loop
    assert not is_closed_loop_letter("L")
    assert not is_closed_loop_letter("i")
    assert not is_closed_loop_letter("A")  # capital A is an open triangle
    assert not is_closed_loop_letter("S")
