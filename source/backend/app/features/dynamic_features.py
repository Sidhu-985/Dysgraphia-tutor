"""
Dynamic feature extraction ("how the child wrote").
======================================================
Fully independent of the static feature extractor, the RL agent, and the
UI. Operates on raw (already outlier-clipped) strokes with timestamps and
optional pressure, plus attempt-level context (retries so far). Session-
level accumulation (fatigue/engagement/frustration across many attempts)
is deliberately NOT done here -- that lives in `state.state_builder`, so
this module only ever reports what it can measure from the current
attempt's own signal.

Feature list (matches project spec Section 3 "Dynamic features"):
  - writing_speed        : path-length / duration, normalized
  - velocity_variation    : coefficient of variation of instantaneous speed
  - acceleration_rms      : RMS acceleration (only when sampling rate allows)
  - movement_smoothness   : dimensionless-jerk-based smoothness (Hogan &
                            Sternad's normalized jerk score)
  - pause_count / pause_total_s / pause_mean_s : pause behaviour
  - stroke_duration_s      : total attempt duration
  - inter_stroke_interval_mean_s : mean gap between successive strokes
  - pressure_mean / pressure_variance / pressure_available
  - retry_count_normalized
  - temporal_consistency  : consistency of per-stroke duration/speed within
                            the attempt

All thresholds are configurable via `backend.app.config.DynamicFeatureConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from ..config import DynamicFeatureConfig, settings
from ..preprocessing.stroke_preprocessing import Stroke, clean_strokes, smooth_series

# Reuse the canonical normalization helpers so the *same* function that
# defines "what 1.0 writing_speed means" is used everywhere (single
# source of truth, shared with the RL state module).
from ..dysgraphia_rl_bridge import normalize_speed, normalize_pressure_variance, normalize_retries


@dataclass
class DynamicFeatures:
    writing_speed: float
    velocity_variation: float
    acceleration_rms: Optional[float]
    acceleration_reliable: bool
    movement_smoothness: float
    pause_count: int
    pause_total_s: float
    pause_mean_s: float
    stroke_duration_s: float
    inter_stroke_interval_mean_s: float
    pressure_mean: float
    pressure_variance_normalized: float
    pressure_available: bool
    retry_count_normalized: float
    temporal_consistency: float

    def to_dict(self) -> dict:
        return asdict(self)


def _per_point_velocity(xs: np.ndarray, ys: np.ndarray, ts: np.ndarray) -> np.ndarray:
    dt = np.diff(ts)
    dt = np.where(dt <= 1e-6, 1e-6, dt)
    dx, dy = np.diff(xs), np.diff(ys)
    dist = np.hypot(dx, dy)
    return dist / dt  # units: coordinate-units / second


def _dimensionless_jerk(velocity: np.ndarray, dt_mean: float) -> float:
    """Log dimensionless jerk (Hogan & Sternad, 2009): a scale- and
    duration-invariant smoothness metric. Lower |value| == smoother.
    Returned here mapped to [0, 1] (1.0 == smooth) via a bounded
    transform so it can sit alongside the other [0,1] features.
    """
    if len(velocity) < 3 or dt_mean <= 0:
        return 0.5
    accel = np.diff(velocity) / dt_mean
    jerk = np.diff(accel) / dt_mean
    duration = dt_mean * len(velocity)
    peak_v = np.max(np.abs(velocity)) + 1e-9
    if len(jerk) == 0:
        return 0.5
    dimensionless_jerk = -(duration ** 3 / peak_v ** 2) * np.sum(jerk ** 2) * dt_mean
    log_djk = -np.log(abs(dimensionless_jerk) + 1e-9)
    # Squash to [0,1]; empirically log-djk for handwriting-scale motion
    # typically falls in roughly [-6, 6] -- this is a monotonic rescaling,
    # not a claim about absolute smoothness units.
    return float(np.clip((log_djk + 6.0) / 12.0, 0.0, 1.0))


def extract_dynamic_features(
    strokes: list[Stroke],
    retries_so_far: int = 0,
    cfg: Optional[DynamicFeatureConfig] = None,
) -> DynamicFeatures:
    cfg = cfg or settings.dynamic_features
    strokes = clean_strokes(strokes)
    strokes = [s for s in strokes if len(s.points) >= 1]

    if not strokes:
        return DynamicFeatures(
            writing_speed=1.0, velocity_variation=0.0, acceleration_rms=None,
            acceleration_reliable=False, movement_smoothness=0.5, pause_count=0,
            pause_total_s=0.0, pause_mean_s=0.0, stroke_duration_s=0.0,
            inter_stroke_interval_mean_s=0.0, pressure_mean=0.5,
            pressure_variance_normalized=0.3, pressure_available=False,
            retry_count_normalized=normalize_retries(retries_so_far), temporal_consistency=0.5,
        )

    all_speeds = []
    all_dts = []
    pause_events = []
    per_stroke_speed_means = []
    per_stroke_durations = []
    pressures_all = []
    total_path_length = 0.0
    total_duration = 0.0

    for s in strokes:
        xs = np.array([p.x for p in s.points])
        ys = np.array([p.y for p in s.points])
        ts = np.array([p.t for p in s.points])
        pr = np.array([p.pressure for p in s.points if p.pressure is not None])

        if len(pr):
            pressures_all.extend(pr.tolist())

        if len(xs) >= 2:
            xs_s = smooth_series(xs)
            ys_s = smooth_series(ys)
            velocity = _per_point_velocity(xs_s, ys_s, ts)
            all_speeds.extend(velocity.tolist())

            dt = np.diff(ts)
            all_dts.extend(dt.tolist())
            pause_events.extend(dt[dt > cfg.pause_threshold_s].tolist())

            per_stroke_speed_means.append(float(np.mean(velocity)))
            seg_len = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
            total_path_length += seg_len
            duration = float(ts[-1] - ts[0])
            per_stroke_durations.append(duration)
            total_duration += duration

    inter_stroke_gaps = []
    for i in range(len(strokes) - 1):
        gap = strokes[i + 1].points[0].t - strokes[i].points[-1].t
        if gap > 0:
            inter_stroke_gaps.append(gap)

    all_speeds = np.array(all_speeds) if all_speeds else np.array([0.0])
    mean_speed = float(np.mean(all_speeds))
    std_speed = float(np.std(all_speeds))
    velocity_variation = float(std_speed / mean_speed) if mean_speed > 1e-9 else 0.0

    # Convert raw speed (coordinate-units/sec) into "character-widths per
    # second" using the attempt's own bounding-box diagonal as a proxy for
    # one character's scale -- this keeps writing_speed comparable across
    # differently-sized exercises without a hardcoded pixel constant.
    # (The diagonal, rather than height alone, avoids degenerating to ~0
    # for a perfectly horizontal or perfectly vertical stroke.)
    all_xs_full = np.concatenate([[p.x for p in s.points] for s in strokes])
    all_ys_full = np.concatenate([[p.y for p in s.points] for s in strokes])
    bbox_width = float(np.max(all_xs_full) - np.min(all_xs_full)) if len(all_xs_full) else 0.0
    bbox_height = float(np.max(all_ys_full) - np.min(all_ys_full)) if len(all_ys_full) else 0.0
    char_scale = max(np.hypot(bbox_width, bbox_height), 1e-6)
    raw_speed_chars_per_sec = (total_path_length / char_scale) / max(total_duration, 1e-6)
    writing_speed = normalize_speed(raw_speed_chars_per_sec, cfg.expected_speed_chars_per_sec)

    mean_dt = float(np.mean(all_dts)) if all_dts else 0.02
    sampling_rate = 1.0 / mean_dt if mean_dt > 0 else 0.0
    acceleration_reliable = sampling_rate >= cfg.min_sampling_rate_hz
    acceleration_rms = None
    if acceleration_reliable and len(all_speeds) >= 3:
        accel = np.diff(all_speeds) / max(mean_dt, 1e-6)
        acceleration_rms = float(np.sqrt(np.mean(accel ** 2)))

    movement_smoothness = _dimensionless_jerk(all_speeds, mean_dt)

    pause_count = len(pause_events)
    pause_total_s = float(np.sum(pause_events)) if pause_events else 0.0
    pause_mean_s = float(np.mean(pause_events)) if pause_events else 0.0

    if pressures_all and len(pressures_all) >= cfg.pressure_available_min_fraction * sum(
        len(s.points) for s in strokes
    ):
        pressure_available = True
        pressure_arr = np.array(pressures_all)
        pressure_mean = float(np.mean(pressure_arr))
        pressure_variance_normalized = normalize_pressure_variance(float(np.var(pressure_arr)))
    else:
        pressure_available = False
        pressure_mean = 0.5
        pressure_variance_normalized = 0.3

    if len(per_stroke_speed_means) >= 2:
        temporal_consistency = float(np.clip(1.0 - np.std(per_stroke_speed_means) / (mean_speed + 1e-9), 0.0, 1.0))
    else:
        temporal_consistency = 0.5

    return DynamicFeatures(
        writing_speed=writing_speed,
        velocity_variation=float(np.clip(velocity_variation, 0.0, 5.0)),
        acceleration_rms=acceleration_rms,
        acceleration_reliable=acceleration_reliable,
        movement_smoothness=movement_smoothness,
        pause_count=pause_count,
        pause_total_s=pause_total_s,
        pause_mean_s=pause_mean_s,
        stroke_duration_s=total_duration,
        inter_stroke_interval_mean_s=float(np.mean(inter_stroke_gaps)) if inter_stroke_gaps else 0.0,
        pressure_mean=pressure_mean,
        pressure_variance_normalized=pressure_variance_normalized,
        pressure_available=pressure_available,
        retry_count_normalized=normalize_retries(retries_so_far),
        temporal_consistency=temporal_consistency,
    )
