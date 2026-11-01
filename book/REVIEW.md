# Review of the repository before writing the book

Reviewed 1 November 2026 on branch `submission-package` (commit 0177aa3). This note records what the
arguments and the code support, where they are weak, and what the book must do about it.

## The argument

The core of the work holds. It separates three objects that practice runs together (a probability
model for a declared attack, a statistical estimate of it, and a governance decision) and gives each
its own rule. The decision rule (an upper confidence bound below a tolerance that has an owner, an
expiry date, a harm class, a legal basis, an access route and, for programmes, a horizon) is simple,
reproducible and auditable. The strongest single result for institutions is that no function of
per-release risks bounds a pooled attack when records keep a pseudonym across releases, together
with the measured size of that gap on simulated and public data.

Points the book has to handle with care:

1. **Novelty.** Several propositions restate known results: reciprocal frequency (Duncan and
   Lambert; Skinner and Elliot), the Boole bounds, the zero-success bound ("rule of three"), the
   implied tolerance (Abowd and Schmutte) and the pooled-risk construction (Sweeney; Ganta et al).
   The long form already says this. The book should keep that framing and place its own
   contribution in the decision record, the programme horizon, the measured gap, the no-history
   result for the credit register, fixed-sequence selection with confirmation, and the measured
   cost of monitoring an audit.
2. **Synthetic evidence.** The credit-register floors (75, 150, none for the insider) rest on chosen
   parameters, time-invariant exposure and activity independent of size. The book must say so in
   each chapter that reports them, and Part V needs a section on what a central bank would have to
   replicate on its own data.
3. **The harm weight.** The implied-tolerance result needs a relative harm weight that no committee
   currently states. Chapter 8 needs a practical method for eliciting it, or the result stays
   decorative.
4. **The model-proposed attacks line (SENTRY-R, injector study).** A local qwen3:4b arm has now been
   measured on 20 of 30 states (6.05 scenario cells against 13.65 for Sobol), and the hosted pilot
   returned no response, so there is still no result showing an LLM beats simple search; the
   executable linkage pilot is a negative result; the 23 October public-file run shows a
   targeting and coverage trade-off (31.57 of 32 high-risk settings confirmed, but 10.07 scenario
   cells covered against 13.83 for Sobol search). This belongs in one chapter with its negative
   results, not in the book's headline.
5. **"Open pipeline".** The Bulletin paper and the long form describe the pipeline as open, but the
   repository is private. Make it public, or change the wording, before either paper or the book
   circulates.
6. **Version drift.** The repository holds five manuscript lines (`main.tex` under the old title,
   `bis_ifc_full.tex`, the JPC article and supplement, the injector study, the deployment paper).
   The generated `numbers.tex` keeps figures consistent, but prose claims can drift. The book should
   draw numbers only from the generated tables.

## The code

- Tests: 194 of 195 pass in a clean Linux environment. The one failure,
  `test_shipped_manifest_matches_the_files_it_lists`, is a line-ending artefact: `.gitattributes`
  now forces LF for JSON, but this Windows working copy still has CRLF files, so
  `artifacts-paper/config.json` hashes differently from the committed blob (which matches the
  manifest). Running `git add --renormalize .` once and committing, or checking the files out
  again, removes the failure and the long list of files that show as modified.
- `governance.GovernanceThreshold` records six of the nine elements in the paper's Table 1. Legal
  basis and protected parties, access route and programme horizon are missing, so the decision
  record the code writes is thinner than the one the paper describes. `decide` also does not check
  that the bound it receives was computed at the threshold's `alpha`.
- `uncertainty.selected_maximum_bound` rebuilds success counts by rounding rates times replicates.
  It should take the counts directly.
- `uncertainty.nested_variance` gives a chi-square interval that assumes roughly normal estimates;
  the docstring should say so.
- `composition.pooled_attack` matches Proposition 9.2 and treats a target absent from a release as
  contributing zero to the marginal and not narrowing the intersection, as the paper states.
- Package metadata still says version 0.7.0 while archives were named 0.7.1.

## What the book does with this

The book keeps the long form's argument and evidence, adds full proofs beside each result, writes
for release committees and for researchers at once, and reports negative results where they occur.
Code changes above are listed here for a separate commit; the book branch does not change code.

## Discrepancies found while writing the full draft (1 November 2026)

Each chapter follows the run it names; these need fixing at the source before publication.

1. Census frontier: docs/REAL_DATA_PROVENANCE.md gives 0.276 to 0.547 (archived run); the paper and the
   current run give 0.276 to 0.942. scripts/check_realdata_paper.py checks only the archived values.
   artifacts-real/results.json is that older run and differs from artifacts-paper.
2. The census pooled releases are described as dropping one key component each; the fourth coarsens
   experience instead.
3. paper/neural_appendix.tex says the neural differences lie within bootstrap intervals; most
   state-bootstrap intervals in neural_comparison.json exclude zero.
4. realdata.py docstring says 1,989 mortgage applications; the loaded file has 1,986.
5. configs/paper.json sets the calibration band at 0.06, so the misspecified model's gap of 0.049
   passes; the long form says that model fails. The frequency-transport test checks mean cell size,
   not the uniqueness probability.
6. decision.json judges the pooled bound against 0.5 in validation and 0.12 at the gate.
7. Dispersion drift is quoted as 0.33 to 0.51; across all five sampling fractions it reaches 0.59.
8. docs/CLAIMS_EVIDENCE.md quotes figures from an older run (0.001 to 0.068, 0.072 to 0.026); the
   current run gives 0.003 to 0.073 and 0.081 to 0.025.
9. Version strings: `__init__.py` 0.3.0, pyproject 0.7.0, archives 0.7.1.
10. The JPC article gives about 720 and 43,000 expected draws; the code's expected_length_bound gives
    704 and about 39,200.
11. The IFC long form calls 1.05 a ratio of interval widths; monitoring_study returns a ratio of mean
    upper bounds.
12. The long form's per-cell differencing posterior of 0.65 is described as "at the rate of the cell";
    the code averages per-cell posteriors.
13. robustness.json `fixed_sequence_subgroup` floors differ from the table's stratified subgroup floors
    (different draws), which can confuse a reader of the JSON.
14. README still says a genuine LLM comparison is pending and describes the stylised policies as ones
    institutions operate; credit.py says a fifth of loans are originated in the panel (data: a sixth).
15. Bibliography key `ecb2022` points to a 2016 ECB regulation.
16. VALIDATION.md records 178 and 183 passing tests from earlier snapshots; a clean run on
    1 November 2026 gives 194 of 195, the failure being the CRLF manifest artefact.
17. Figures fig1_risk_distribution.pdf and sentry_r_loop.pdf separate series by colour alone and need
    hatching or line styles for black and white printing.
18. dp.py has no ceiling function or accountant tighter than basic composition; noise of scale 1/epsilon
    gives 2 epsilon under replace-one and is loan-level, not borrower-level.
