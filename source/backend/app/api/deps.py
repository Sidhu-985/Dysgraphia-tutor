"""Small shared helpers for the API layer -- converting the wire format
(flat list of stroke points) into the `Stroke` objects the preprocessing/
feature-extraction modules expect, and common HTTP error helpers."""

from __future__ import annotations

from fastapi import HTTPException

from ..preprocessing.stroke_preprocessing import Point, Stroke
from ..schemas import StrokePointIn
from ..state.session_context import SessionContext
from ..storage import SessionStore


def points_to_strokes(points: list[StrokePointIn]) -> list[Stroke]:
    """Groups a flat, time-ordered list of stroke points by `stroke_id`
    into `Stroke` objects, preserving relative order of first appearance.
    The wire format sends one flat array (simpler for the tablet client to
    stream) rather than nested arrays; this is the one place that
    regroups it."""
    order: list[int] = []
    grouped: dict[int, list[Point]] = {}
    for p in points:
        if p.stroke_id not in grouped:
            grouped[p.stroke_id] = []
            order.append(p.stroke_id)
        grouped[p.stroke_id].append(Point(x=p.x, y=p.y, t=p.t, pressure=p.pressure))
    return [Stroke(stroke_id=sid, points=grouped[sid]) for sid in order]


def require_session(store: SessionStore, session_id: str) -> SessionContext:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    if session.ended:
        raise HTTPException(status_code=409, detail=f"Session {session_id} has already ended")
    return session
