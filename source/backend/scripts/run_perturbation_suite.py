"""
Quality-proxy perturbation sensitivity suite (Phase C). SYNTHETIC DATA ONLY.

For every exercise template and perturbation type, generates
severities 0.0, 0.1, ..., 1.0 x 5 seeds, runs the FULL backend feature
pipeline (preprocessing -> static + dynamic features -> compute_quality_score)
and reports the Spearman rank correlation between severity and
(1 - quality_score). Target (predeclared): rho >= 0.8 per perturbation type
(pooled over templates). Results are reported as measured, pass or fail.

Usage (from backend/):  python scripts/run_perturbation_suite.py [out.json]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.exercises import EXERCISES  # noqa: E402
from app.features.static_features import extract_static_features  # noqa: E402
from app.features.dynamic_features import extract_dynamic_features  # noqa: E402
from app.state.state_builder import compute_quality_score  # noqa: E402
from app.synthetic.handwriting_perturbations import (  # noqa: E402
    PERTURBATION_TYPES, clean_specimen, perturb_single, perturb_spacing, spacing_specimen, to_strokes,
)

from app.preprocessing.stroke_preprocessing import normalize_shape, resample_by_arc_length  # noqa: E402

SEVERITIES = [round(0.1 * i, 1) for i in range(11)]
NO_OP_THRESHOLD = 0.01
SEEDS = [0, 1, 2, 3, 4]
TARGET_RHO = 0.8


def quality_of(paths, reference):
    strokes = to_strokes(paths, pressure=None)
    st = extract_static_features(strokes, reference_template=reference)
    dy = extract_dynamic_features(strokes)
    return compute_quality_score(st, dy), st


def normalized_shape_change(clean_paths, perturbed_paths) -> float:
    """Mean point distance between clean and perturbed shapes after the
    same arc-length resampling and translation/scale normalization the
    feature pipeline applies. ~0 means the perturbation is a geometric
    no-op for this specimen, e.g. shearing or progressively rescaling a
    straight horizontal line yields the same normalized line."""
    def norm(paths):
        xy = np.concatenate(paths)
        rx, ry = resample_by_arc_length(xy[:, 0], xy[:, 1], 64)
        return np.stack(normalize_shape(rx, ry), axis=1)
    a, b = norm(clean_paths), norm(perturbed_paths)
    return float(np.mean(np.hypot(*(a - b).T)))


def run() -> dict:
    rows = []
    for kind in PERTURBATION_TYPES:
        if kind == "spacing":
            specimens = {"four_bars_spacing_specimen": spacing_specimen()}
        else:
            specimens = {eid: clean_specimen(ex.reference_array()) for eid, ex in EXERCISES.items()}
        for name, spec in specimens.items():
            if kind == "spacing":
                reference = np.concatenate(spec)
            else:
                reference = EXERCISES[name].reference_array()
            for sev in SEVERITIES:
                for seed in SEEDS:
                    if kind == "spacing":
                        paths = perturb_spacing(spec, sev, seed)
                    else:
                        paths = [perturb_single(spec, kind, sev, seed)]
                    q, st = quality_of(paths, reference)
                    clean_paths = spec if kind == "spacing" else [spec]
                    change = normalized_shape_change(clean_paths, paths)
                    rows.append(dict(kind=kind, specimen=name, severity=sev, seed=seed, quality=q,
                                     normalized_shape_change=change,
                                     stroke_similarity=st.stroke_similarity,
                                     spacing_accuracy=st.spacing_accuracy,
                                     baseline_alignment=st.baseline_alignment))
    # A (type, specimen) pair whose maximum-severity perturbation changes the
    # normalized shape by less than NO_OP_THRESHOLD is a geometric no-op.
    # Severity does not change the input there, so no quality measure can
    # follow it. Such pairs are listed and left out of the pooled rho. The
    # pooled rho WITH them is reported too.
    no_ops = []
    for kind in PERTURBATION_TYPES:
        for name in dict.fromkeys(x["specimen"] for x in rows if x["kind"] == kind):
            top = [x["normalized_shape_change"] for x in rows
                   if x["kind"] == kind and x["specimen"] == name and x["severity"] == 1.0]
            if max(top) < NO_OP_THRESHOLD:
                no_ops.append({"kind": kind, "specimen": name, "max_shape_change_at_severity_1": max(top)})
    no_op_keys = {(d["kind"], d["specimen"]) for d in no_ops}

    def rho(rs):
        if len({x["quality"] for x in rs}) < 2:
            return float("nan")
        return float(spearmanr([x["severity"] for x in rs], [1 - x["quality"] for x in rs]).statistic)

    summary = {}
    for kind in PERTURBATION_TYPES:
        r_all = [x for x in rows if x["kind"] == kind]
        r = [x for x in r_all if (kind, x["specimen"]) not in no_op_keys]
        per_specimen = {name: rho([x for x in r_all if x["specimen"] == name])
                        for name in dict.fromkeys(x["specimen"] for x in r_all)}
        mean_q = {str(s): float(np.mean([x["quality"] for x in r if x["severity"] == s])) for s in SEVERITIES}
        pooled = rho(r)
        summary[kind] = {"spearman_rho_pooled": pooled, "pass": bool(pooled >= TARGET_RHO),
                         "spearman_rho_pooled_including_no_ops": rho(r_all),
                         "excluded_no_op_specimens": [n for (k, n) in no_op_keys if k == kind],
                         "per_specimen_rho": per_specimen, "mean_quality_by_severity": mean_q,
                         "n": len(r)}
    return {"target_rho": TARGET_RHO, "no_op_threshold": NO_OP_THRESHOLD, "no_ops": no_ops, "severities": SEVERITIES, "seeds": SEEDS,
            "data_label": "SYNTHETIC perturbations of synthetic exercise templates -- not real handwriting",
            "summary": summary, "rows": rows}


if __name__ == "__main__":
    out = run()
    for k, v in out["summary"].items():
        print(f"{k:<8} rho={v['spearman_rho_pooled']:.3f} (incl. no-ops {v['spearman_rho_pooled_including_no_ops']:.3f}) "
              f"pass={v['pass']} excluded={v['excluded_no_op_specimens']}  "
              + " ".join(f"{n}={r:.2f}" for n, r in v["per_specimen_rho"].items()))
    if len(sys.argv) > 1:
        os.makedirs(os.path.dirname(sys.argv[1]) or ".", exist_ok=True)
        with open(sys.argv[1], "w") as f:
            json.dump(out, f, indent=1)
        print("saved", sys.argv[1])
