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
4. **The model-proposed attacks line (SENTRY-R, injector study).** There is still no genuine LLM
   arm; the executable linkage pilot is a negative result; the 23 October public-file run shows a
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
