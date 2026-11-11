# `quality_score`: an engineering proxy, not a clinical measure

`backend/app/state/state_builder.py::compute_quality_score` is a weighted average of four features measured on one attempt:

| Term | Weight | Source |
|---|---|---|
| stroke similarity | 0.45 | Banded DTW distance between the resampled, translation/scale-normalized trace and the exercise template, mapped through `exp(-d/0.35)` |
| spacing consistency | 0.20 | Regularity of gaps between successive strokes. Neutral 0.5 when there are fewer than 3 strokes. |
| baseline alignment | 0.20 | Spread of the lowest 20% of y-values relative to the trace height |
| pressure stability | 0.15 | 1 − normalized pressure variance. A neutral 0.7 when the device reports no pressure (no pressure value is invented). |

## What it is

A reproducible, deterministic number in [0, 1]. It goes **down** when a trace departs from the template in controlled ways (see below). It feeds `prev_score`, the score history, the improvement trend and the reward.

## What it is not

* It is not validated against therapist or teacher ratings.
* It is not a dysgraphia screening or diagnostic score.
* It is not comparable across exercises. Templates of different difficulty have different attainable scores, so a gain after lowering the difficulty is not learning.
* It is not trained on any dataset. The CNN scorer in the design document is not implemented (`docs/DATA_GOVERNANCE.md` §5).

## Perturbation sensitivity (Phase C)

`backend/scripts/run_perturbation_suite.py` (test: `backend/tests/test_quality_perturbation.py`) takes clean synthetic traces of every exercise template and applies five deterministic, seeded distortions. Each runs at severities 0.0 to 1.0 in steps of 0.1, with 5 seeds per severity. The script then measures the Spearman rank correlation between severity and `1 − quality_score` through the full feature pipeline. The target was fixed before measuring: **ρ ≥ 0.8 per perturbation type**, pooled over templates.

Results are in `evidence/phase_c/`. Two defects were found and fixed:

1. **Spacing.** The original `spacing_accuracy` compared each gap with a fixed constant, 0.15 × the attempt's bounding-box diagonal. A perfectly regular row of four strokes scored **0.0**. ρ for spacing irregularity was 0.55. The feature now measures consistency against the attempt's own median gap, which matches its documented definition ("spacing consistency"). ρ is now 0.90.
2. **No change was made to the jitter, tremor, scale or slant scoring.**

One exclusion, stated openly: progressive rescaling and shear leave a **straight horizontal line** unchanged once it is normalized. The normalized shape changes by less than 0.01 even at severity 1.0, so no quality measure can track severity there. The suite detects these geometric no-ops automatically, excludes only those (type, specimen) pairs, and reports ρ both with and without them. The test fixes the allowed exclusions to exactly {(scale, line_easy), (slant, line_easy)}.

These tests show the proxy is **sensitive in the right direction to controlled synthetic distortions**. They do not show that it agrees with human judgments of real children's handwriting.

## Discrimination defect found by the live scenarios (Phase C, second round)

The struggling-writer scenario sends random-walk scribbles that have nothing to do with the template. These scribbles scored **0.52–0.68** on the quality proxy, which is close to the 0.55 "success" threshold. Clean traces scored about 0.85. So the proxy kept severity in the right order but squeezed its scale, and junk scored as "nearly fine". There are three causes, recorded in `evidence/phase_c/discrimination_before.json`:

1. **Baseline alignment did not measure alignment.** It took the spread of the lowest 20% of y-values. Those points are close together by construction, so scribbles scored ≈ 0.90–0.94.
2. **Neutral fillers counted as measurements.** A single-stroke attempt always had spacing set to 0.5, and a device without pressure always had stability set to 0.7. Both went into the weighted average, which added about 0.2 to every score.
3. **The DTW similarity scale was set without calibration.** `exp(-d/0.35)` gave unrelated scribbles a similarity of 0.29–0.65.

**Calibration criteria, written down before any change was tried:**
* Clean synthetic traces of every template: quality ≥ 0.90.
* Unrelated random-walk scribbles: median quality ≤ 0.35.
* Every clean trace scores above every scribble, for every template.
* The perturbation suite above still gives ρ ≥ 0.8 for every type.

The DTW scale is the only value tuned against data. It is picked from the grid {0.35, 0.30, 0.25, 0.20, 0.15, 0.10} as the **largest** value that meets the criteria on a **calibration** set (scribble seeds 1000–1049). It is then checked on **held-out** scribbles (seeds 0–49). The other two fixes are definition corrections, not tuning:
* Baseline alignment now compares the lower envelope of the trace with the template's lower envelope. Both are in the normalized frame, over 8 x-bins, scored as `exp(-mean|Δ|/0.1)`. The constant 0.1 is fixed in advance and not tuned.
* Terms that cannot be measured (spacing with fewer than 3 strokes, pressure on a device without pressure) are **left out** of the weighted average. They are no longer filled in with a neutral value.

## Slant: final investigation (reproduced, not fixed)

The full suite was reproduced from the packaged repository, and every row was identical (`evidence/final/baseline_quality_sensitivity.json`). It covers 5 types × 5 templates (one specimen for spacing) × 11 severities (0.0–1.0) × 5 seeds. That is 275 samples per type, or 55 for spacing. Two (type, specimen) pairs are excluded as documented no-ops: slant and scale on `line_easy`.

| Type | n (after no-op exclusion) | Spearman ρ, severity vs 1 − quality | Target 0.80 |
|---|---|---|---|
| jitter | 275 | 0.828 | PASS |
| tremor | 275 | 0.863 | PASS |
| scale | 220 | 0.810 | PASS |
| **slant** | 220 | **0.778** | **FAIL** |
| spacing | 55 | 0.899 | PASS |

Possible causes were checked against the code and the data (`evidence/final/slant_investigation.json`):
* **Is slant not measured, or normalized away?** No. Every template's quality falls strictly as slant increases (per-template ρ = 1.00). DTW is computed after translation and scale normalization, and that normalization does not undo shear.
* **Is its influence too small?** No. Per unit of normalized shape change at severity 1, the proxy's quality drop for slant on `wave_easy` is **3.97**, the largest of any template (loop 1.98, eight 2.69, zigzag 3.83).
* **Is the perturbation construction a poor fit?** **Yes, this is the cause.** Severity is defined as a shear *angle* (up to 35°). On the shallow wave (amplitude 15 over width 100), the maximum shear moves the normalized shape by only **0.027**, against 0.08–0.18 for the other templates. Pooling templates therefore mixes tiny and large real distortions at the same "severity". Measured against the actual normalized shape change, the slant ρ is **0.944**. That figure is **post hoc** and is reported only as a diagnostic.

**Decision: no change.** Changing the quality proxy so the wave scores worse under a 2.7% shape change would make it *over*-react to tiny deformations, which the evidence does not support. Redefining the severity scale after seeing the result would be manipulating perturbation severity, which the protocol forbids. The predeclared criterion stays **FAIL (0.778)**. The test keeps its strict-xfail marker, and the target is not lowered.
