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
