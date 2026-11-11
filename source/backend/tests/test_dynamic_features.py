import numpy as np

from app.features.dynamic_features import extract_dynamic_features
from app.preprocessing.stroke_preprocessing import Point, Stroke
from .conftest import make_line_stroke


def test_faster_writing_gives_higher_writing_speed():
    fast = make_line_stroke(dt=0.005)
    slow = make_line_stroke(dt=0.05)
    f_fast = extract_dynamic_features([fast])
    f_slow = extract_dynamic_features([slow])
    assert f_fast.writing_speed > f_slow.writing_speed


def test_pauses_are_detected():
    n = 20
    pts = [Point(x=float(i), y=0.0, t=i * 0.02) for i in range(n)]
    # Inject a 2-second pause in the middle.
    pts_with_pause = pts[:10] + [Point(x=p.x, y=p.y, t=p.t + 2.0) for p in pts[10:]]
    stroke = Stroke(stroke_id=0, points=pts_with_pause)
    feats = extract_dynamic_features([stroke])
    assert feats.pause_count >= 1
    assert feats.pause_total_s >= 1.9


def test_no_pauses_for_smooth_continuous_writing():
    stroke = make_line_stroke(dt=0.02)
    feats = extract_dynamic_features([stroke])
    assert feats.pause_count == 0


def test_missing_pressure_marked_unavailable_not_fabricated():
    n = 20
    pts = [Point(x=float(i), y=0.0, t=i * 0.02, pressure=None) for i in range(n)]
    stroke = Stroke(stroke_id=0, points=pts)
    feats = extract_dynamic_features([stroke])
    assert feats.pressure_available is False
    assert feats.pressure_mean == 0.5  # documented neutral default, not a fabricated reading


def test_pressure_present_is_used_when_available():
    n = 20
    pts = [Point(x=float(i), y=0.0, t=i * 0.02, pressure=0.8) for i in range(n)]
    stroke = Stroke(stroke_id=0, points=pts)
    feats = extract_dynamic_features([stroke])
    assert feats.pressure_available is True
    assert abs(feats.pressure_mean - 0.8) < 1e-6


def test_retry_count_normalized_reflects_input():
    stroke = make_line_stroke()
    f0 = extract_dynamic_features([stroke], retries_so_far=0)
    f5 = extract_dynamic_features([stroke], retries_so_far=5)
    assert f5.retry_count_normalized > f0.retry_count_normalized


def test_acceleration_marked_unreliable_at_low_sampling_rate():
    n = 10
    pts = [Point(x=float(i), y=0.0, t=i * 1.0) for i in range(n)]  # 1 Hz -- far below threshold
    stroke = Stroke(stroke_id=0, points=pts)
    feats = extract_dynamic_features([stroke])
    assert feats.acceleration_reliable is False
    assert feats.acceleration_rms is None


def test_empty_strokes_do_not_crash():
    feats = extract_dynamic_features([])
    assert feats.pressure_available is False
    assert feats.writing_speed == 1.0


def test_numerical_stability_all_outputs_finite():
    stroke = make_line_stroke()
    feats = extract_dynamic_features([stroke])
    for k, v in feats.to_dict().items():
        if isinstance(v, float):
            assert not np.isnan(v), f"{k} is NaN"


def test_inter_stroke_interval_measured_for_multi_stroke_attempt():
    s1 = make_line_stroke(stroke_id=0)
    last_t = s1.points[-1].t
    s2_points = [Point(x=p.x, y=p.y, t=last_t + 1.5 + p.t) for p in make_line_stroke(stroke_id=1).points]
    s2 = Stroke(stroke_id=1, points=s2_points)
    feats = extract_dynamic_features([s1, s2])
    assert feats.inter_stroke_interval_mean_s >= 1.4
