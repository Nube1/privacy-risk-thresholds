# Revisions applied in this copy

`book/` is the untouched original. This copy applies the edits of the revision supplement that fit this source.

Applied: Ch 4 detectability qualifier; Ch 6 log-threshold A_i worked values; Ch 7 legal-element section; Ch 8 value-curve fit note; Ch 9 composition-gap actions; Ch 11 softened claim; Ch 13 synthetic-register box and reading-precision procedure; Ch 16 and Appendix A threshold-elements wording and gaps paragraph.

Not applicable to this source (already present or chapters absent): Ch 15 calibration-band power check, Ch 14 and Ch 17 edits, Part VII notes (Ch 19-23), which exist only in draft 5.

## Rebuild from the 304-page draft (in progress)

Chapter 18 (Continuous adversarial learning) is rebuilt as `chapters/ch18-learning.tex` from the text of the 304-page draft, in a new Part VII. The old `ch18-open.tex` is now in Part VIII and still carries its old chapter number until Chapters 19 to 22 are rebuilt.

Chapter 19 (Releases that change over time) is rebuilt as `chapters/ch19-change.tex`. The draft text had a stray fragment in Section 19.2 ("depends on how the figure is computed."); it is repaired in the source. The draft gives 0.380 for the five-year mass in Example 19.6, a matrix exponential gives 0.379, and the source uses 0.379.

Still to rebuild: Chapters 20 to 22, the updated Chapter 23, and Appendix A.5 and A.6. The module names in Chapter 18's software note (`gate`, `evidence`, `registry`, `betting`) belong to the extension code bundle and are not in `src/privacy_risk`.

## Numbers recomputed rather than copied (Chapters 18 and 19)

Recomputed with plain Python from the formulas, not taken from the extension's run: the 24 attacks of a zero-success audit; the wealth path of Example 18.4 (1.023, 1.057, 1.101, 0.705; certifies at attack 55; 4.0 at 20 and 22.2 at 37 on a clean run); the retry study (0.501 and 0.041 over 4,000 programmes); Example 19.2 (0.364, 1.44, 0.59); Example 19.4 (1.77 and 2.10 distinct patterns; 0.442 and 0.525 by formula, 0.442 and 0.526 by simulation of 200,000 classes); Example 19.6 (masses and risks by matrix exponential). The one disagreement with the draft was a rounding (0.52 against 0.525 and 0.522 against 0.526), corrected in the source.

Not yet recomputed: the extension's evidence-run counts in Table 19.1 and Figures 18.1 and 19.2, which need the extension's engine and register. The repository's `src/privacy_risk` does not contain that engine.

## Code written for Chapters 18 and 19

The modules the two chapters name did not exist in `src/privacy_risk`, so they were written from the chapters' specifications: `gate`, `betting`, `evidence`, `registry`, `dynamics` and `graph`, with `tests/test_ext_learning.py`, `tests/test_ext_models.py` (19 tests, all passing) and `scripts/extension_evidence.py`. The tests replay the checks in Algorithm 18.1, Examples 18.3, 18.4 and 19.6 and the two-loan check of Algorithm 19.1. Table 19.1 and Figures 18.1 and 19.2 still come from the extension's run; the script's own run at seed 2026 agrees in direction and size (see Chapter 19's software note).

## Gaps found and closed

- Code: `uncertainty.selected_maximum_bound` takes integer counts (new `successes` argument, tested); the `nested_variance` docstring states the normality assumption; `__version__` now matches `pyproject.toml` (0.7.0). Appendix A, "Gaps in the code", records the three as closed.
- Chapter numbering: Chapters 20, 21 and 22 are short placeholders (`\plannedchapter`) stating the question each will answer, so the old "What remains open" chapter is Chapter 23 as in the 304-page draft and the cross-references in Chapters 18 and 19 resolve by label.
- Appendix A has a new section on the six modules added for Chapters 18 and 19.

Still open: the full text of Chapters 20 to 22, the updated Chapter 23, Appendix A.5's equation-to-code map for the dynamic models, and the chapter-by-chapter check of Chapters 1 to 17 against the code.
