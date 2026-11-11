# A–Z alphabet upgrade

The app now covers all **52 letters**: 26 uppercase (A–Z) and 26 lowercase (a–z). Every letter has its own exercise, guide, writing lines, instruction and session path, all served by the real backend API.

The guides are **hand-authored in a school print style**. They are checked automatically against a real font. They are **not clinically validated**.

## 1. Guides (`backend/app/letter_templates.py`)

| Item | Value |
|---|---|
| Letters | 52 (`UPPERCASE` + `LOWERCASE`); a module assertion fails if one is missing |
| Coordinates | 0–100 box, y down (unchanged convention); finite; every stroke has ≥ 2 points |
| Strokes | one polyline per pen stroke, in drawing order and direction; 1–4 strokes per letter |
| Style | ball-and-stick school print; single-storey a and g; capital I with top and bottom bars, so it is distinct from l |
| Writing lines | four-line paper with three equal zones: top line 10, middle line (x-height) 40, baseline 70, descender line 100 |
| Letter zones | capital (10–70); tall lowercase b d f h k l (10–70); t rises part-way (18–70); small letters (40–70); tails g j p q y (40–100); dots of i and j just below the top line; the tail of Q crosses the baseline a little (to 80) |

**Change to the existing lines.** The box used to place the baseline at 90 and the x-height at 45, which left 10 units for descenders. The lines were moved so that tails fit inside the same 0–100 box.

- **Existing glyphs:** the 14 that existed before (A C E L N O S T U a c l m o) keep their stroke structure, order and direction. Their sizes and proportions were adjusted to the new lines: capitals are now 60 units tall instead of 80, and the round capitals C, G, O and Q are now as wide as they are tall.
- **N:** now starts at the top like every other capital.
- **Shapes:** the shape exercises are unchanged.

### Stroke-order conventions (design choice, not a clinical standard)

- Start at the top.
- Pull down before sliding across.
- Round letters start near the top and go anticlockwise: c, o, a, d, g, q, and C, G, O, Q, S.
- After a stem, a bowl goes clockwise: b, p, B, D, P, R.
- The dots of i and j and the cross bars of f and t come last.

## 2. Catalogue (`backend/app/exercises.py`)

| Item | Value |
|---|---|
| Ids | `letter_upper_a` … `letter_upper_z` and `letter_lower_a` … `letter_lower_z`: stable, unique, no case collision |
| Per letter | `target`, `strokes`, `reference_template` (concatenation, unchanged meaning), `writing_lines` (now with `descender`), `letter_case`, `letter_zone`, `instruction`, `difficulty` |
| Levels 0–4 | design choice: 0 one straight stroke (L l i); 1 straight up-down and across (F H I T t); 2 one curve (C O c o); 3 diagonals, several strokes or a tail; 4 a curve joined to a stem, humps, or several curves. The 14 earlier letters keep their levels |
| Difficulty steps | stay on the same **track**: shapes, words, capitals or small letters. A difficulty action never turns a letter into a shape or word, and never switches capital ↔ small (`track_of`) |
| Unchanged | the five shapes (ids, prompts, templates, levels) and the three words; `EXERCISES` still holds only the shapes |

### API (backward compatible)

- `/exercises` returns 60 records: 5 shapes, 52 letters and 3 words.
- `/exercise/{id}` and `/session/start` accept every letter.
- **New optional fields:** `letter_case`, `letter_zone`, `instruction`, and `writing_lines.descender`.
- `/session/start` accepts an optional `practice_mode` (`"tracing"` or `"free_writing"`). Omitted, it starts in tracing mode, as before.

## 3. Tablet app

| Screen | What it does |
|---|---|
| Dashboard (`exercise_selection_screen.dart`) | **Learn & Write the Alphabet** section with tabs **Lowercase a–z** and **Uppercase A–Z**, each a responsive grid of 26 tiles (4–9 columns depending on width). Optional practice groups: straight lines, round letters, tall letters, tail letters, humps and bumps. Words and shapes are listed below. The supervisor policy card shows the backend's real policy |
| Letter tile (`alphabet_section.dart`) | large letter, its name ("small a"), a practised badge, a selected state, a semantic label, and a tap target ≥ 64 dp |
| Letter page (`letter_detail_screen.dart`) | large reference letter, where it sits on the lines, the backend's instruction, and the guide with numbered start markers, stroke colours and direction arrows on four-line paper. Also: **Show me how** (strokes drawn in order; static under reduced motion), the journey Look → Learn the strokes → Trace → Write on your own → Feedback, **Trace** and **Write it on my own** (real backend sessions), a small ↔ capital switch for the same letter, and previous/next within A–Z or the chosen group |
| Writing (`writing_screen.dart`) | existing canvas, lined paper (now with the descender line), clear, undo, retry, submit and feedback. Starts in the mode the backend confirmed. The feedback card tells the adult who chose the next step |

**Practised badge.** It means that an attempt for that letter was accepted by the backend *in this run of the app*. It is kept in memory only and is not a stored learning record.

## 4. Verification

| Check | How | Result |
|---|---|---|
| All 52 glyphs look like their letter | `tests/test_letter_templates.py`: each guide rasterised and compared with all 52 Poppins letters placed on the same four lines | all 52 rank their own letter first, with one documented exception: the barred school-print **I** ranks third, because the sans-serif font's I has no bars; it must not rank as l |
| Distinct cases | symmetric nearest-point distance between each letter and its capital > 3 units | 26/26 |
| Lines and zones | bottom on the baseline or descender line, top on the right line, small letters start at the middle line | 52/52 |
| Catalogue and API | `tests/test_alphabet_catalogue.py`, `tests/test_exercise_catalogue.py` | pass |
| Live server | `scripts/e2e_alphabet_live.py`: real uvicorn over HTTP; every letter retrieved, started (tracing, and free-writing for every 4th), two attempts submitted, session ended | 52 letters, 104 attempts, 0 problems |
| Visual review | `scripts/render_alphabet_sheet.py` → `evidence/alphabet_dqn/alphabet_guides_sheet.png` | inspected by eye |

### Matcher changes, and why

The font matcher in the tests was **extended**, never loosened, to cope with the four-line layout:

1. **Fourth line:** the font's descender is mapped to the new descender line.
2. **Ascenders:** the font's lowercase ascenders are mapped to the top line. Poppins ascenders rise above its capitals; school paper does not.
3. **Narrow letters (l, i, I):** these keep the font's own scale. Before, they were compared in raw font pixels, which was a bug.
4. **Tolerance:** scaled with the letter box. It is 6 units for an 80-unit letter, as before; now 4.5 for 60 units.

## 5. Limitations

- The guides are hand-authored. They have not been reviewed by an occupational therapist or a teacher.
- The quality proxy was calibrated on the shapes, not on letters (`docs/QUALITY_SCORE.md`).
- The Flutter code could not be compiled or run in this environment (see the test report).
