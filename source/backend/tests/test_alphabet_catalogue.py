"""All 52 letters (A-Z, a-z): catalogue, templates, API and sessions.
All handwriting here is synthetic (generated from the guide itself)."""

from __future__ import annotations

import math
import re
import string

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.exercises import (CATALOGUE, EXERCISES, LETTER_EXERCISES, LOWERCASE_IDS, UPPERCASE_IDS, WORD_EXERCISES,
                           exercise_for_difficulty, get_exercise, levels_in, track_of)
from app.interventions.intervention_controller import build_intervention
from app.letter_templates import (ASCENDER_TOP, BASELINE, DESCENDER, GLYPHS, LOWER_DESCENDERS, X_HEIGHT,
                                  glyph_strokes, letter_zone)
from app.main import app
from app.state.session_context import SessionContext

client = TestClient(app)
DECREASE, INCREASE = 0, 1


def _letters(case):
    return [e for e in LETTER_EXERCISES.values() if e.letter_case == case]


# ---------------------------------------------------------------- counts and ids
def test_exactly_26_uppercase_and_26_lowercase_letters():
    up, lo = _letters("upper"), _letters("lower")
    assert len(up) == 26 and len(lo) == 26
    assert sorted(e.target for e in up) == list(string.ascii_uppercase)
    assert sorted(e.target for e in lo) == list(string.ascii_lowercase)
    assert len(LETTER_EXERCISES) == 52


def test_all_52_ids_unique_stable_and_case_safe():
    ids = list(LETTER_EXERCISES)
    assert len(set(ids)) == 52
    assert UPPERCASE_IDS == [f"letter_upper_{c}" for c in string.ascii_lowercase]
    assert LOWERCASE_IDS == [f"letter_lower_{c}" for c in string.ascii_lowercase]
    assert set(UPPERCASE_IDS).isdisjoint(LOWERCASE_IDS)
    for eid in ids:
        assert re.fullmatch(r"letter_(upper|lower)_[a-z]", eid)


@pytest.mark.parametrize("eid", UPPERCASE_IDS + LOWERCASE_IDS)
def test_every_letter_record_is_complete_and_correct(eid):
    e = LETTER_EXERCISES[eid]
    char = eid[-1].upper() if "_upper_" in eid else eid[-1]
    assert e.target == char and e.category == "letter"
    assert e.letter_case == ("upper" if char.isupper() else "lower")
    assert e.strokes == glyph_strokes(char)
    assert e.prompt == f"Write the letter {char}"
    assert e.letter_zone == letter_zone(char)
    assert e.instruction and ("capital" if char.isupper() else "small") in e.instruction
    assert e.writing_lines == {"ascender": ASCENDER_TOP, "x_height": X_HEIGHT, "baseline": BASELINE,
                               "descender": DESCENDER}
    assert 0 <= e.difficulty <= 4


@pytest.mark.parametrize("char", sorted(GLYPHS))
def test_every_template_is_valid(char):
    strokes = GLYPHS[char]
    assert 1 <= len(strokes) <= 4
    for s in strokes:
        assert len(s) >= 2
        for x, y in s:
            assert math.isfinite(x) and math.isfinite(y)
            assert 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0
        steps = np.hypot(*np.diff(np.array(s), axis=0).T)
        assert steps.sum() > 3.0, "every stroke draws visible ink"


@pytest.mark.parametrize("char", sorted(GLYPHS))
def test_writing_zone_matches_the_letter(char):
    ys = np.array([p[1] for s in GLYPHS[char] for p in s])
    zone = letter_zone(char)
    if zone == "descender":
        assert ys.max() == pytest.approx(DESCENDER, abs=1)
    elif char == "Q":
        assert BASELINE < ys.max() <= BASELINE + 12           # the tail crosses the baseline a little
    else:
        assert ys.max() == pytest.approx(BASELINE, abs=1)
    if zone in ("capital", "ascender") and char != "t":
        assert ys.min() == pytest.approx(ASCENDER_TOP, abs=1)
    if zone == "x_height" or zone == "descender":
        body = [p[1] for p in GLYPHS[char][0]]
        assert min(body) == pytest.approx(X_HEIGHT, abs=1), "small letters start at the middle line"
    assert (char in LOWER_DESCENDERS) == (zone == "descender")


def test_heights_differ_between_capitals_and_small_letters():
    """Not every letter is forced into the same height."""
    def height(c):
        ys = [p[1] for s in GLYPHS[c] for p in s]
        return max(ys) - min(ys)
    assert height("c") < height("C") - 20
    assert height("p") > height("o") + 20


# ---------------------------------------------------------------- tracks and difficulty
@pytest.mark.parametrize("eid", UPPERCASE_IDS + LOWERCASE_IDS)
def test_difficulty_changes_keep_letter_category_and_case(eid):
    for action, name in ((INCREASE, "INCREASE_DIFFICULTY"), (DECREASE, "DECREASE_DIFFICULTY")):
        e = get_exercise(eid)
        s = SessionContext(session_id="s", child_id="P-0001", difficulty_level=e.difficulty, current_exercise_id=eid)
        iv = build_intervention(action, name, s)
        nxt = get_exercise(iv.params["next_exercise_id"])
        assert nxt.category == "letter" and nxt.letter_case == e.letter_case
        assert track_of(nxt.exercise_id) == track_of(eid)


def test_tracks_cover_all_levels_in_both_cases():
    assert levels_in("letter_upper") == [0, 1, 2, 3, 4]
    assert levels_in("letter_lower") == [0, 1, 2, 3, 4]
    assert exercise_for_difficulty(0, "letter_upper").target == "L"
    assert exercise_for_difficulty(0, "letter_lower").target == "i"


def test_shapes_and_words_unchanged():
    assert list(EXERCISES) == ["line_easy", "wave_easy", "loop_medium", "eight_medium", "zigzag_hard"]
    assert list(WORD_EXERCISES) == ["word_to", "word_cat", "word_sun"]
    assert track_of("loop_medium") == "shape" and track_of("word_cat") == "word"


# ---------------------------------------------------------------- API
def test_api_lists_all_52_letters():
    data = client.get("/exercises").json()
    letters = [e for e in data if e["category"] == "letter"]
    assert len(letters) == 52
    assert sorted(e["target"] for e in letters) == sorted(string.ascii_letters)


@pytest.mark.parametrize("eid", UPPERCASE_IDS + LOWERCASE_IDS)
def test_every_letter_retrievable_and_startable(eid):
    r = client.get(f"/exercise/{eid}")
    assert r.status_code == 200
    body = r.json()
    ex = CATALOGUE[eid]
    assert (body["target"], body["template_strokes"], body["letter_case"]) == (ex.target, ex.strokes, ex.letter_case)
    assert body["writing_lines"]["descender"] == DESCENDER
    start = client.post("/session/start", json={"child_id": "P-0042", "exercise_id": eid, "data_source": "synthetic_test"})
    assert start.status_code == 200, start.text
    assert start.json()["exercise"]["exercise_id"] == eid
    assert start.json()["exercise"]["target"] == ex.target


def _trace(strokes, scale=3.0):
    pts, t = [], 0.0
    for sid, s in enumerate(strokes):
        for i, (x, y) in enumerate(s):
            pts.append({"x": x * scale, "y": y * scale, "t": t, "pressure": None, "stroke_id": sid,
                        "is_stroke_start": i == 0, "is_stroke_end": i == len(s) - 1})
            t += 0.02
        t += 0.3
    return pts


@pytest.mark.parametrize("eid", UPPERCASE_IDS + LOWERCASE_IDS)
def test_every_letter_runs_through_the_real_attempt_pipeline(eid):
    start = client.post("/session/start", json={"child_id": "P-0042", "exercise_id": eid, "data_source": "synthetic_test"})
    sid, ex = start.json()["session_id"], start.json()["exercise"]
    r = client.post("/attempt/submit", json={"session_id": sid, "exercise_id": eid, "attempt_id": f"a-{eid}",
                                             "points": _trace(ex["template_strokes"])})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["policy_mode"] in ("dqn", "rule_based_fallback")
    assert body["decision_source"] in ("dqn", "rule_based_fallback", "safe_default_invalid_state",
                                       "rule_based_after_inference_error")
    nxt = body["next_exercise"]
    assert nxt["category"] == "letter" and nxt["letter_case"] == ex["letter_case"]
