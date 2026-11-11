"""Handwriting quality scoring package.

A robust, exercise-specific scorer that combines DTW shape similarity with
structural / topological checks (stroke count, aspect ratio, curvature
profile, direction sequence, loop closure) so that a malformed shape (e.g.
a zigzag drawn in place of a lowercase ``a``) cannot earn a moderately high
score merely because it has a similar bounding box, enough strokes or
similar overall dimensions.

The existing :func:`app.state.state_builder.compute_quality_score` is kept
as a backwards-compatible, reference-free quality proxy used for the
``prev_score`` component of the 12-D RL state vector. The new
:func:`compute_handwriting_score` defined here is the one actually returned
to the UI and tracked for per-letter mastery.
"""
from .handwriting_quality import (  # noqa: F401
    HandwritingScore,
    HandwritingScoreBreakdown,
    compute_handwriting_score,
    expected_stroke_count,
    is_closed_loop_letter,
)
