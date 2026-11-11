"""Phase C: quality-proxy perturbation sensitivity (SYNTHETIC data only).

Target, fixed before measurement: Spearman rho >= 0.8 between perturbation
severity and (1 - quality_score), per perturbation type, pooled over
specimens. The only exclusions allowed are geometric no-ops. These are
specimen/perturbation pairs where severity does not change the normalized
input shape, e.g. shearing a horizontal line. The test pins the exact set
so exclusions cannot grow silently.
"""
import importlib.util
import os

import pytest

from app.features.static_features import extract_static_features
from app.synthetic.handwriting_perturbations import (
    perturb_single, perturb_spacing, spacing_specimen, to_strokes, clean_specimen,
)
from app.exercises import EXERCISES

_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "run_perturbation_suite.py")


@pytest.fixture(scope="module")
def suite():
    spec = importlib.util.spec_from_file_location("run_perturbation_suite", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run()


# KNOWN FAILURE, reported as FAIL in docs/ACCEPTANCE_MATRIX.md. After the
# Phase C discrimination fixes, the pooled slant rho is 0.778, below the
# 0.8 target. Every per-template slant curve is still strictly monotone
# (rho = 1.00 each). The pooled value falls because a shallow wave hardly
# changes under shear. strict=True: if the value ever passes, this marker
# must be removed. The target itself is NOT lowered.
_SLANT_KNOWN_FAIL = pytest.mark.xfail(
    strict=True, reason="pooled slant Spearman rho 0.778 < 0.8 target (see docs/QUALITY_SCORE.md)")


@pytest.mark.parametrize("kind", ["jitter", "tremor", "scale",
                                  pytest.param("slant", marks=_SLANT_KNOWN_FAIL), "spacing"])
def test_quality_drops_monotonically_with_severity(suite, kind):
    s = suite["summary"][kind]
    assert s["spearman_rho_pooled"] >= suite["target_rho"], (kind, s["spearman_rho_pooled"])


def test_only_documented_geometric_no_ops_are_excluded(suite):
    excluded = {(d["kind"], d["specimen"]) for d in suite["no_ops"]}
    assert excluded <= {("scale", "line_easy"), ("slant", "line_easy")}


def test_perturbations_are_deterministic_given_seed():
    xy = clean_specimen(EXERCISES["loop_medium"].reference_array())
    for kind in ("jitter", "tremor", "scale", "slant"):
        a = perturb_single(xy, kind, 0.7, seed=3)
        b = perturb_single(xy, kind, 0.7, seed=3)
        assert (a == b).all()


def test_zero_severity_is_identity():
    xy = clean_specimen(EXERCISES["zigzag_hard"].reference_array())
    for kind in ("jitter", "tremor", "scale", "slant"):
        assert abs(perturb_single(xy, kind, 0.0, seed=1) - xy).max() < 1e-9


def test_regular_spacing_scores_higher_than_irregular_spacing():
    regular = extract_static_features(to_strokes(spacing_specimen()))
    irregular = extract_static_features(to_strokes(perturb_spacing(spacing_specimen(), 1.0, seed=2)))
    assert regular.spacing_accuracy > 0.95
    assert irregular.spacing_accuracy < regular.spacing_accuracy


def test_synthetic_strokes_do_not_invent_pressure():
    strokes = to_strokes([clean_specimen(EXERCISES["line_easy"].reference_array())], pressure=None)
    assert all(p.pressure is None for s in strokes for p in s.points)
