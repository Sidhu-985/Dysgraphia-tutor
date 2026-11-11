"""Shared test fixtures. All 'stylus hardware' input in this suite is
synthetic (mocked) -- there is no real tablet available in CI/dev, per
project spec Section 13 ("mock the hardware input with realistic test
data")."""

from __future__ import annotations

import math
import os
import random
import tempfile

# Test traffic must never be written into the demo/app database: point
# storage at a throwaway file BEFORE app settings are first imported.
os.environ.setdefault("STORAGE_DB_PATH", os.path.join(tempfile.mkdtemp(prefix="dt_test_db_"), "test.db"))

import numpy as np
import pytest

from app.preprocessing.stroke_preprocessing import Point, Stroke


def make_line_stroke(stroke_id: int = 0, n: int = 40, length: float = 100.0,
                      y: float = 50.0, dt: float = 0.02, noise: float = 0.0,
                      pressure: float | None = 0.5, seed: int = 0) -> Stroke:
    rng = random.Random(seed)
    points = []
    for i in range(n):
        x = (i / (n - 1)) * length + (rng.uniform(-noise, noise) if noise else 0.0)
        yy = y + (rng.uniform(-noise, noise) if noise else 0.0)
        points.append(Point(x=x, y=yy, t=i * dt, pressure=pressure))
    return Stroke(stroke_id=stroke_id, points=points)


def make_loop_stroke(stroke_id: int = 0, n: int = 60, radius: float = 40.0,
                      cx: float = 50.0, cy: float = 50.0, dt: float = 0.02,
                      pressure: float | None = 0.5) -> Stroke:
    points = []
    for i in range(n):
        theta = 2 * math.pi * i / (n - 1)
        x = cx + radius * math.sin(theta)
        y = cy + radius * (1 - math.cos(theta))
        points.append(Point(x=x, y=y, t=i * dt, pressure=pressure))
    return Stroke(stroke_id=stroke_id, points=points)


def scale_translate_stroke(stroke: Stroke, scale: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> Stroke:
    pts = [Point(x=p.x * scale + dx, y=p.y * scale + dy, t=p.t, pressure=p.pressure) for p in stroke.points]
    return Stroke(stroke_id=stroke.stroke_id, points=pts)


def resample_stroke_timestamps(stroke: Stroke, factor: float) -> Stroke:
    """Simulate a different sampling rate by subsampling/duplicating
    points at a different rate while covering the same duration."""
    n = max(2, int(round(len(stroke.points) * factor)))
    old_t = np.array([p.t for p in stroke.points])
    old_x = np.array([p.x for p in stroke.points])
    old_y = np.array([p.y for p in stroke.points])
    new_t = np.linspace(old_t[0], old_t[-1], n)
    new_x = np.interp(new_t, old_t, old_x)
    new_y = np.interp(new_t, old_t, old_y)
    pts = [Point(x=float(new_x[i]), y=float(new_y[i]), t=float(new_t[i]), pressure=0.5) for i in range(n)]
    return Stroke(stroke_id=stroke.stroke_id, points=pts)


@pytest.fixture
def line_stroke():
    return make_line_stroke()


@pytest.fixture
def loop_stroke():
    return make_loop_stroke()
