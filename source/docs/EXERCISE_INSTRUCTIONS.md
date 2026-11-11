# Child-facing exercise instructions (letters, words, shapes)

Branch `child-instructions`, cut from `final-submission` (the attached `dysgraphia_tutor_final_submission.zip`, which is byte-identical to it).

## What the child sees

Above the canvas, a card shows:
- **what** to write, for example "Trace the letter A." or "Let's write the word SUN!";
- **how and where**, for example "Follow the guide in the writing area." or "No guide this time. Write it in the writing area.";
- "Try again! You've got this." when the same exercise comes back;
- a small model of the target and the level.

The canvas draws the guide one pen stroke at a time. Letters and words sit on lined paper: ascender, dashed x-height and baseline.

## How the pieces stay consistent

1. **One record:**
   - The backend's `ExerciseOut` carries `category` (shape / letter / word), `target` (exact shape name, character or word), `template_strokes` and, for letters and words, `writing_lines`.
   - All are optional, so older clients ignore them.
   - `reference_template` is unchanged in meaning. It is the strokes concatenated in order, and it is what the feature extractor compares against.
2. **One function:**
   - `instructionFor(exercise, mode, retryIndex)` (`tablet_app/lib/models/exercise_instruction.dart`) builds the text from category + target + mode on every build. Nothing is cached, so nothing goes stale after an exercise transition.
   - A malformed or unknown target is never shown as a letter or word. The backend's own prompt is used instead.
3. **One mode rule:** `guideVisibleFor(mode)` decides both the canvas guide and the wording ("Trace …" vs "Let's write … / Draw …").
4. **Size on the canvas:**
   - The guide is fitted to the writing lines, not to its own ink, so "c" is drawn smaller than "C".
   - Without this, every letter was stretched to the same box and "c" looked identical to "C".

## Catalogue

| Category | Exercises | New? |
|---|---|---|
| Shapes | straight line, gentle wave, loop, figure eight, zigzag (levels 0–4) | pre-existing; ids, prompts, levels and templates unchanged; explicit `category`/`target` added |
| Letters | L, l (0); T (1); C, O, c, o (2); A, E, N, S, U (3); a, m (4) | new |
| Words | TO (1), CAT (2), SUN (3) | new |

**About the letter guides:**
- They are hand-authored polylines (`backend/app/letter_templates.py`) with school-print stroke order. This is a design choice, **not clinically validated**.
- `tests/test_letter_templates.py` rasterises each guide and compares it with all 52 letters of a font that has a school-style single-storey "a" (Poppins). The guide's own letter must be the best match. The only accepted tie is "l" vs "I", which are both a plain bar in a sans-serif font.
- A mutation check confirmed the test catches mislabelled guides: O labelled C, N labelled A, C labelled c, L labelled E, U labelled m, o labelled a.

`EXERCISES` still means the five shapes. The quality-proxy perturbation suite and discrimination script iterate over it, so their predeclared scope did not change. The API serves `CATALOGUE`.

## RL integration (unchanged algorithm)

The policy still chooses among the same 10 actions. `DECREASE/INCREASE_DIFFICULTY` still mean "one level easier/harder". The intervention controller now resolves the next exercise **within the current category**:
- **shapes:** identical to before (one shape per level 0–4);
- **letters and words:** stay letters and words. A category with gaps (words: levels 1–3) moves to the next level it has. At its easiest or hardest level it keeps the current exercise, so the "already the easiest one" message is true.

## Visual inspection

`evidence/instructions/guide_templates.png` renders every guide from the backend data, with stroke order. It is drawn with matplotlib and is **not an app screenshot**.

## Limitations

- **Flutter was not run here** (no SDK; storage.googleapis.com is blocked). `flutter analyze` and `flutter test`, including the new `test/instruction_test.dart`, are written but unexecuted. Only the tree-sitter syntax check ran.
- **The quality proxy and success threshold were calibrated on the shapes only.** For letters:
  - Perfect traces score 0.78–1.00.
  - Multi-stroke items score lower: A 0.83, TO 0.79, CAT 0.78. This is because the spacing term treats stroke-to-stroke gaps as "spacing".
  - The shape similarity is size-normalised, so it cannot tell a "c" from a "C" written at the wrong size.
- **No clinical validation** of guides, stroke order or level ordering.
