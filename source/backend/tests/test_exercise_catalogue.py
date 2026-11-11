"""Exercise catalogue, API metadata and RL-intervention transitions for the
child-facing instruction feature. All traces here are synthetic."""

from __future__ import annotations

import re
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.exercises import (CATALOGUE, EXERCISES, LETTER_EXERCISES, WORD_EXERCISES, category_of,
                           exercise_for_difficulty, get_exercise, levels_in)
from app.letter_templates import GLYPHS, WORD_ADVANCE, flatten, glyph_strokes
from app.interventions.intervention_controller import build_intervention
from app.state.session_context import SessionContext
from app.synthetic.handwriting_perturbations import clean_specimen, perturb_single, to_strokes
from app.features.static_features import extract_static_features
from app.features.dynamic_features import extract_dynamic_features
from app.state.state_builder import compute_quality_score

DECREASE, INCREASE, REPEAT, CONTINUE = 0, 1, 8, 9


# ---------------------------------------------------------------- catalogue integrity
def test_every_exercise_has_explicit_consistent_metadata():
    for eid, e in CATALOGUE.items():
        assert e.exercise_id == eid
        assert e.category in ("shape", "letter", "word")
        assert e.target.strip() == e.target and e.target
        assert e.strokes and all(len(s) >= 2 for s in e.strokes), eid
        assert e.reference_template == flatten(e.strokes), "reference_template must be the strokes in order"
        assert 0 <= e.difficulty <= 4
        if e.category == "letter":
            assert re.fullmatch(r"[A-Za-z]", e.target)
            assert e.strokes == glyph_strokes(e.target), "a letter's guide is the glyph of its own target"
            assert e.prompt == f"Write the letter {e.target}"
        if e.category == "word":
            assert re.fullmatch(r"[A-Za-z]{2,12}", e.target)
            assert e.prompt == f"Write the word {e.target}"
        if e.category in ("letter", "word"):
            wl = e.writing_lines
            assert wl["ascender"] < wl["x_height"] < wl["baseline"] < wl["descender"]
            ys = [p[1] for p in e.reference_template]
            assert wl["ascender"] - 1 <= min(ys) and max(ys) <= wl["descender"] + 1
            if e.category == "word" or e.target not in "gjpqyQ":
                assert max(ys) <= wl["baseline"] + 1, eid     # only tails go below the baseline
        else:
            assert e.writing_lines is None


def test_shapes_unchanged_and_still_the_only_entries_of_EXERCISES():
    """EXERCISES keeps its original meaning (the perturbation suite iterates it)."""
    assert list(EXERCISES) == ["line_easy", "wave_easy", "loop_medium", "eight_medium", "zigzag_hard"]
    assert all(e.category == "shape" for e in EXERCISES.values())
    assert [e.difficulty for e in EXERCISES.values()] == [0, 1, 2, 3, 4]
    assert all(len(e.strokes) == 1 and e.strokes[0] == e.reference_template for e in EXERCISES.values())


def test_shape_definitions_identical_to_the_previous_commit():
    try:
        old = subprocess.run(["git", "show", "v1-baseline:backend/app/exercises.py"], capture_output=True,
                             text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("git history not available")
    ns: dict = {}
    exec(compile(old.replace("from __future__ import annotations", ""), "old_exercises", "exec"), ns)
    for k, e in ns["EXERCISES"].items():
        n = EXERCISES[k]
        assert (e.prompt, e.difficulty, e.reference_template) == (n.prompt, n.difficulty, n.reference_template)


def test_no_shape_is_labelled_a_letter_or_word():
    for e in EXERCISES.values():
        assert e.category == "shape"
    assert set(LETTER_EXERCISES) | set(WORD_EXERCISES) == set(CATALOGUE) - set(EXERCISES)


def test_word_guides_are_the_letter_glyphs_in_order():
    for e in WORD_EXERCISES.values():
        expected = []
        for i, ch in enumerate(e.target):
            expected += [[[round(x + i * WORD_ADVANCE, 2), y] for x, y in s] for s in glyph_strokes(ch)]
        assert e.strokes == expected, e.target
    # every character of every word has a guide glyph
    assert all(ch in GLYPHS for e in WORD_EXERCISES.values() for ch in e.target)


def test_ids_are_unique_and_stable():
    assert len(CATALOGUE) == 5 + 52 + 3
    assert "letter_upper_a" in CATALOGUE and "letter_lower_a" in CATALOGUE
    assert CATALOGUE["letter_upper_a"].target == "A" and CATALOGUE["letter_lower_a"].target == "a"
    assert CATALOGUE["word_cat"].target == "CAT"


# ---------------------------------------------------------------- quality proxy sanity on the new guides
@pytest.mark.parametrize("eid", sorted(set(LETTER_EXERCISES) | set(WORD_EXERCISES)))
def test_clean_trace_of_the_guide_succeeds_and_beats_a_distorted_one(eid):
    e = CATALOGUE[eid]
    paths = [clean_specimen(np.array(s) * 3, max(20, len(s) * 4)) for s in e.strokes]
    clean = to_strokes(paths)
    noisy = to_strokes([perturb_single(p, "jitter", 1.0, seed=i) for i, p in enumerate(paths)])

    def q(st):
        return compute_quality_score(extract_static_features(st, e.reference_array()), extract_dynamic_features(st))
    assert q(clean) >= 0.55, "a perfect trace must count as a success"
    assert q(clean) > q(noisy)


# ---------------------------------------------------------------- difficulty transitions (RL actions)
def _session(eid: str) -> SessionContext:
    e = get_exercise(eid)
    return SessionContext(session_id="s", child_id="P-0001", difficulty_level=e.difficulty, current_exercise_id=eid)


def test_shape_ladder_behaviour_is_unchanged():
    for level, expected in enumerate(EXERCISES):
        assert exercise_for_difficulty(level).exercise_id == expected
    s = _session("loop_medium")
    iv = build_intervention(INCREASE, "INCREASE_DIFFICULTY", s)
    assert iv.params["next_exercise_id"] == "eight_medium" and s.difficulty_level == 3
    iv = build_intervention(DECREASE, "DECREASE_DIFFICULTY", s)
    assert iv.params["next_exercise_id"] == "loop_medium" and s.difficulty_level == 2


@pytest.mark.parametrize("eid", sorted(set(LETTER_EXERCISES) | set(WORD_EXERCISES)))
def test_difficulty_actions_stay_in_the_same_category(eid):
    cat = category_of(eid)
    case = CATALOGUE[eid].letter_case
    for action, name in ((INCREASE, "INCREASE_DIFFICULTY"), (DECREASE, "DECREASE_DIFFICULTY")):
        s = _session(eid)
        iv = build_intervention(action, name, s)
        nxt = get_exercise(iv.params["next_exercise_id"])
        assert nxt.category == cat
        assert nxt.letter_case == case, "a difficulty step never switches capital <-> small letters"
        assert nxt.difficulty == s.difficulty_level, "presentation level == level of the exercise shown"
        if iv.params["delta"] == 0:
            assert nxt.exercise_id == eid, "no level change -> keep the same exercise (message says so)"


def test_word_track_skips_missing_levels_and_stops_at_its_ends():
    assert levels_in("word") == [1, 2, 3]
    s = _session("word_to")
    iv = build_intervention(DECREASE, "DECREASE_DIFFICULTY", s)
    assert iv.params["delta"] == 0 and iv.params["next_exercise_id"] == "word_to"
    assert "already the easiest" in iv.message
    s = _session("word_sun")
    iv = build_intervention(INCREASE, "INCREASE_DIFFICULTY", s)
    assert iv.params["delta"] == 0 and iv.params["next_exercise_id"] == "word_sun"


def test_letter_at_top_level_is_not_swapped_for_another_letter():
    s = _session("letter_lower_m")  # level 4, not the first level-4 letter
    iv = build_intervention(INCREASE, "INCREASE_DIFFICULTY", s)
    assert iv.params["next_exercise_id"] == "letter_lower_m"


# ---------------------------------------------------------------- API
client = TestClient(app)


def test_api_lists_full_catalogue_with_backward_compatible_fields():
    r = client.get("/exercises")
    assert r.status_code == 200
    data = {e["exercise_id"]: e for e in r.json()}
    assert set(data) == set(CATALOGUE)
    for eid, e in data.items():
        for k in ("exercise_id", "prompt", "difficulty", "reference_template"):  # original contract
            assert k in e
        ex = CATALOGUE[eid]
        assert (e["category"], e["target"]) == (ex.category, ex.target)
        assert e["template_strokes"] == ex.strokes
        assert e["reference_template"] == [p for s in e["template_strokes"] for p in s]
        assert e["writing_lines"] == ex.writing_lines


def test_letter_session_end_to_end_keeps_target_and_template_together():
    start = client.post("/session/start", json={"child_id": "P-0042", "exercise_id": "letter_upper_a",
                                                "data_source": "synthetic_test"})
    assert start.status_code == 200
    ex = start.json()["exercise"]
    assert (ex["category"], ex["target"]) == ("letter", "A")
    pts = []
    t = 0.0
    for sid, s in enumerate(ex["template_strokes"]):
        for i, (x, y) in enumerate(s):
            pts.append({"x": x * 3, "y": y * 3, "t": t, "pressure": None, "stroke_id": sid,
                        "is_stroke_start": i == 0, "is_stroke_end": i == len(s) - 1})
            t += 0.02
        t += 0.3
    r = client.post("/attempt/submit", json={"session_id": start.json()["session_id"], "exercise_id": ex["exercise_id"],
                                             "attempt_id": "att-1", "points": pts})
    assert r.status_code == 200, r.text
    nxt = r.json()["next_exercise"]
    # whatever the policy chose, the next exercise is a coherent letter record
    assert nxt["category"] == "letter" and re.fullmatch(r"[A-Za-z]", nxt["target"])
    assert nxt["template_strokes"] == CATALOGUE[nxt["exercise_id"]].strokes
    assert r.json()["presentation"]["mode"] in ("tracing", "free_writing")


def test_flutter_fixture_matches_the_api():
    """tablet_app/test/fixtures/catalogue.json is the exact /exercises response
    the Flutter consistency tests run on. Regenerate it if this fails:
    python -c "import json;from fastapi.testclient import TestClient;from app.main import app;
    open('../tablet_app/test/fixtures/catalogue.json','w').write(json.dumps(TestClient(app).get('/exercises').json(), indent=1)+'\\n')"
    """
    import json
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[2] / "tablet_app" / "test" / "fixtures" / "catalogue.json"
    if not path.exists():
        pytest.skip("tablet_app fixture not present")
    assert json.loads(path.read_text()) == client.get("/exercises").json()
