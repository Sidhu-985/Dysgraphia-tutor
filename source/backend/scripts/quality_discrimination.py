"""
Quality-proxy discrimination check (Phase C). SYNTHETIC data only.

Clean traces of every template vs unrelated random-walk scribbles, both
through the full feature pipeline. Criteria are predeclared in
docs/QUALITY_SCORE.md: clean >= 0.90, median scribble <= 0.35, and
every clean trace above every scribble for each template.

Usage (from backend/):
  python scripts/quality_discrimination.py --seeds 0 50 [--dtw-scale 0.2] [--out f.json]
  python scripts/quality_discrimination.py --calibrate      # grid search on seeds 1000-1049
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import StaticFeatureConfig, settings  # noqa: E402
from app.exercises import EXERCISES  # noqa: E402
from app.preprocessing.stroke_preprocessing import Point, Stroke  # noqa: E402
from app.features.static_features import extract_static_features  # noqa: E402
from app.features.dynamic_features import extract_dynamic_features  # noqa: E402
from app.state.state_builder import compute_quality_score  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "live_scenarios", os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_scenarios.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)

GRID = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10]


def _q(points, ex, cfg):
    st = [Stroke(0, [Point(p["x"], p["y"], p["t"], p["pressure"]) for p in points])]
    s = extract_static_features(st, EXERCISES[ex].reference_array(), cfg=cfg)
    return compute_quality_score(s, extract_dynamic_features(st)), s


def evaluate(seed_lo, seed_hi, dtw_scale=None):
    cfg = settings.static_features.model_copy()
    if dtw_scale is not None:
        cfg = StaticFeatureConfig(**{**cfg.model_dump(), "dtw_distance_to_similarity_scale": dtw_scale})
    per = {}
    for ex, e in EXERCISES.items():
        clean, cs = _q(sc.clean_trace(e.reference_template), ex, cfg)
        scr = [_q(sc.poor_trace(seed), ex, cfg)[0] for seed in range(seed_lo, seed_hi)]
        per[ex] = {"clean": clean, "clean_similarity": cs.stroke_similarity,
                   "clean_baseline": cs.baseline_alignment,
                   "scribble_median": float(np.median(scr)), "scribble_max": float(np.max(scr)),
                   "scribble_min": float(np.min(scr))}
    crit = {
        "clean_ge_0.90": all(v["clean"] >= 0.90 for v in per.values()),
        "scribble_median_le_0.35": float(np.median([v["scribble_median"] for v in per.values()])) <= 0.35,
        "clean_above_every_scribble": all(v["clean"] > v["scribble_max"] for v in per.values()),
    }
    return {"dtw_scale": cfg.dtw_distance_to_similarity_scale, "seeds": [seed_lo, seed_hi],
            "per_template": per, "criteria": crit, "pass": all(crit.values()),
            "data_label": "SYNTHETIC traces -- not real handwriting"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 50])
    ap.add_argument("--dtw-scale", type=float, default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.calibrate:
        rows = [evaluate(1000, 1050, s) for s in GRID]
        chosen = next((r["dtw_scale"] for r in rows if r["pass"]), None)
        out = {"grid": [{"dtw_scale": r["dtw_scale"], "criteria": r["criteria"], "pass": r["pass"],
                         "per_template": r["per_template"]} for r in rows],
               "rule": "largest grid value meeting all criteria on calibration seeds 1000-1049",
               "chosen_dtw_scale": chosen}
        for r in rows:
            print(r["dtw_scale"], r["pass"], r["criteria"])
        print("chosen:", chosen)
    else:
        out = evaluate(*a.seeds, a.dtw_scale)
        for ex, v in out["per_template"].items():
            print(f"{ex:<13} clean={v['clean']:.3f} scribble median={v['scribble_median']:.3f} "
                  f"max={v['scribble_max']:.3f}")
        print("criteria:", out["criteria"], "PASS" if out["pass"] else "FAIL")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
