# Recorded discrepancies not fixed in this pass

Each needs the stored run or the paper source, and none is a book-source change.

- Item 2: census pooled releases. The fourth release coarsens experience and does not drop a key component. Fix in the paper and in `docs`.
- Item 3: `paper/neural_appendix.tex` says the neural differences lie within bootstrap intervals; most state-bootstrap intervals in `neural_comparison.json` exclude zero. Reword the appendix.
- Item 7: dispersion drift is quoted as 0.33 to 0.51 and reaches 0.59 across five sampling fractions. Update the quoted range from the stored run.
- Item 8: `docs/CLAIMS_EVIDENCE.md` quotes an older run. Replace 0.001 to 0.068 and 0.072 to 0.026 with the current 0.003 to 0.073 and 0.081 to 0.025.
- Item 10: the JPC article gives about 720 and 43,000 expected draws, the code's `expected_length_bound` gives 704 and about 39,200 at the arguments the article states. Needs the article's arguments.
- Item 11: the IFC long form calls 1.05 a ratio of interval widths, but `monitoring_study` returns a ratio of mean upper bounds. Reword.
- Item 12: the per-cell differencing posterior of 0.65 is an average of per-cell posteriors, not "at the rate of the cell". Reword.
- Item 13: `robustness.json` `fixed_sequence_subgroup` and the table's stratified floors come from different draws. Label the JSON.

## Status after Chapters 20 to 23 and Appendix A.5 (2 November 2026)

- Chapters 20, 21 and 22 are rebuilt from the draft text with every printed number recomputed by code
  (`transport`, `bayes`, `sensitivity`, `planner`, `proposals`, `betting`, `uncertainty`, `dp`).
  Where the repository's seeds give different digits from the draft, the chapter prints the repository's
  value and names the draft's. Differences: Ch 20 posterior means (one noise draw); Ch 21 flip counts
  (290/595 against 293/624); Ch 22 per-seed planner gains, calibration temperatures, the median at true
  risk 0.09 (638 against 669) and the retry study (52.2 per cent against 50.2).
- Chapter 23 was patched, not rewritten: Examples 23.2 and 23.3, the dynamics paragraph, the Part VII
  paragraphs of the closing section and the scene paragraphs were added. Its numbers that depend on the
  author's stored runs (floors by reading precision, Census search results, language-model arm) are
  unchanged and still unverified here.
- Bibliography: the entries added from memory for Chapters 20 to 22 were checked against the draft's own
  bibliography (authors, years, titles, journals); four were corrected. They have not been checked against
  the original publications.
- Still open: the six-versus-nine governance elements; the 'drisk' commands and the reviewer workflow named
  in the draft do not exist in the repository; hard-coded cross-references in Chapters 1 to 17; the
  overfull and underfull box warnings; Chapters 1 to 17 have not been re-checked against the draft's
  revisions chapter by chapter.

## Status against the second review (attachment of 2 November 2026)

Closed in code and text: boxed synthetic-register warning in Chapter 13 and a repeated one-line caution in Chapters 1, 5, 6, 7, 8, 16, 18, 19; reading-precision procedure, error-budget procedure, composition-gap decision tree, defence-of-tolerance record, test-repair procedure and the ethics questions (Appendix B, `procedures.py`, tests); uniqueness-probability transport test in `validation.py` and `pipeline.py` (limit 0.30, set after the errors were seen, so not yet prespecified); governance record holds all nine elements (ten fields) with `strict` mode; Table 16.2 first-row note; Table 13.1 noise note; Chapter 15 monotonicity standard errors and calibration-band note; Appendix A gap-to-claims table.

Not done, and why: the six-versus-nine wording in Chapters 7 and 16 is the author's decision; splitting Chapter 14, shortening Chapter 20, merging Chapters 21 and 22, moving the claim ledger to Chapter 1 and the sensitivity analysis of the synthetic parameters are structural choices for the author; the Appendix B precision table uses the repository's own attack, so its floors differ from Chapter 13's by design; the remedy-tree test inputs are illustrative because no remedy has been re-simulated; bibliography points checked in `book_revised`: the 2016 regulation appears once, the three NIST 2024 entries are each cited singly, and there is no Yang et al. entry, so those three review items apply to the author's draft and not to this copy; page references still need a check against the author's source; the literature-review appendix suggested by the review is not written.

## Chapters 4 to 17 against draft 2

Structural comparison (worked examples, propositions, algorithms) shows `book_revised` matches the draft in every chapter from 4 to 17. Chapter 4 gained Examples 4.3, 4.4 and 4.6 and Chapter 5 gained the Maurer et al. paragraph and Example 5.10 as a numbered example in this pass; the rest were already present. Every number in the draft's Examples 4.3 to 12.4 was recomputed from `uncertainty.py`, `sequential.py` or direct arithmetic and agrees with the draft to the digits printed (tests in `tests/test_ext_measures.py`). The examples the draft labels illustrative remain labelled so. Not re-derived: the figures in Chapters 13 to 15 that come from the author's stored runs (`artifacts-paper`), which this repository cannot regenerate; each carries a `% source:` comment naming its origin.
