# Privacy Risk Thresholds: neural release pipeline and registered audit

The main entry point is the **neural release pipeline**. A network, fitted to
development attacks on a synthetic credit register, proposes the bets and the
starting floor of a registered audit; the audit then certifies a cell-count floor
on fresh attack targets. The network never decides a release: every bet is frozen
before its outcome exists, so a poor network costs attacks, not validity.

```bash
pip install -e .
python -m privacy_risk.neural_pipeline --output artifacts-neural   # or: privacy-risk-neural
```

The run has four stages, each with its own evidence:

1. **Development attacks** on a dense grid of floors, on seeds the audit never uses.
2. **Network** (`sentry_r.DeepRiskNet`, two hidden ReLU layers, one output per declared
   attack) fitted by the binomial likelihood, with a logistic baseline on the same counts.
3. **Frozen plan**: predicted risks, bets and a start floor, written with a SHA-256 digest.
4. **Release chain** (`adaptive_audit.AdaptiveAudit.register_release_chain`) on fresh
   targets, with a reserve chain if the first floor fails.

Output: `artifacts-neural/neural_pipeline.json` (development counts, fitted weights,
frozen plans, final outcomes by seed) and
`paper/publication_versions/table_neural_pipeline.tex`. `python scripts/neural_results.py`
then draws the network's predictions against the true risks
(`paper/figures/fig_neural_predictions.pdf`) and writes the fit table
(`paper/generated/table_neural_fit.tex`). The long form (Section
"The neural release pipeline", Figure and Algorithm "neural release pipeline") and the
journal article (Section "A neural network in the loop") report the run.

## Papers and the rest of the pipeline

This repository holds the long-form manuscript *Governing privacy risk across repeated
statistical releases: probability models, programme tolerances and accountable decisions*
(`paper/bis_ifc_full.tex`), its companion journal article *Adaptive certification of
attack-conditional disclosure risk across release programmes*
(`paper/publication_versions/jpc_main.tex`), and the pipeline that produces them.
`paper/main.tex` is the earlier working copy under the first title, *Quantitative risk
assessment for data privacy: probability models, estimation uncertainty and governance
thresholds*; it carries the same factual corrections and is kept because
`scripts/make_acmart.py` builds the ACM version from it. The pipeline generates every number, table and figure the paper
reports, and it writes an auditable record of the release decision it reaches.

Two separate author publication drafts and their reproducible mathematical
extension are in `paper/publication_versions/`. The BIS/IFC manuscript focuses
on measured repeated-release findings and governance. The Journal of Privacy
and Confidentiality main paper and supplement focus on a registered sequential
audit for adaptively added attack scenarios. `adaptive_audit.py` implements
that audit as a separate, opt-in gate; existing fixed-shortlist SENTRY-R
commands remain fixed-shortlist commands. See the publication-version README
for the manuscript scope, build commands and submission limitations.

- `paper/` the manuscript source and the compiled PDF. The tables under
  `paper/generated/` are copied from a pipeline run by
  `scripts/rebuild_evidence.py --publish paper/generated`, so a table cannot disagree
  with the experiment behind it once that step has run.
- `src/privacy_risk/` the package: risk measures, the linkage attack, uncertainty,
  governance, composition, the credit register case and the real-data adapters.
- `artifacts-paper/` the run the paper reports, with its manifest of SHA-256 hashes.

It does four different jobs and labels them separately:

1. **Proof checks** verify algebraic consequences of declared assumptions: the
   reciprocal-frequency identity, the marketer identity, the sharp union bounds, the
   certification sample size, and the closed form of the risk budget.
2. **Simulation checks** estimate attack success under a declared error model, its Monte
   Carlo uncertainty, the risk--utility frontier, and the tolerance a stated relative harm weight
   implies.
3. **Falsification tests** can reject the model the risk figure rests on: linkage
   calibration, population-frequency transport, mechanism monotonicity, and adversary
   stress. A failure blocks the release decision.
4. **Governance gates** apply a prespecified upper confidence bound to a tolerance and
   write a machine-readable decision record. The programme-level finding is recorded
   beside it, because marginal risks do not bound a pooled attack.

Code cannot prove that an adversary model is realistic or that a tolerance is ethically or
legally correct. Those are empirical and policy claims. The pipeline makes them explicit
and testable.

## Run it

```bash
python -m pip install -e ".[dev]"
pytest
privacy-risk-run --config configs/base.json  --output artifacts        # fast, for CI
privacy-risk-run --config configs/paper.json --output artifacts-paper  # the paper's run
python scripts/rebuild_evidence.py --artifacts artifacts-paper --publish paper/generated  # presentation only
python scripts/check_paper_numbers.py --artifacts artifacts-paper --paper paper/bis_ifc_full.tex
python scripts/adaptive_audit_chain.py    # the journal article's release chain on the credit register
python scripts/programme_robustness.py   # floors by adversary knowledge, tolerance, register draw and noise
python scripts/robustness_table.py       # tables for the floors
python scripts/audit_process_attacks.py  # attacks on the audit process and the controls
python scripts/estimand_check.py         # the two identification tables share one estimand
python scripts/realdata_replication.py    # public-file replication, hashes and sensitivity
python scripts/realdata_neural_pilot.py   # SENTRY-R logistic/deep pilot, held-out Census states
python scripts/realdata_neural_comparison.py # richer nonlinear comparison
python scripts/plot_realdata_neural.py       # editable SVG, PDF and table
```

## Real data

`realdata.py` runs the framework on two real public microdata files that ship with the
`wooldridge` package: an extract of the US Census 2000 public use sample (29,501 records)
and the Boston Fed mortgage application study (1,986 applications). Both are already
public. The module measures what a release of that shape discloses under a declared
adversary, and it is the adapter pattern `docs/CLAIMS_EVIDENCE.md` asks for. Two things
change when the data are real: the population is not observed, so the journalist measure
rests on a fitted law with no truth to check it against, and the analyses that define
utility are the ones the file is used for.
The reproducible public-file run, its exact source hashes and exploratory
auxiliary-information sensitivity are in `artifacts-real/replication.json`.
`artifacts-real/neural_pilot.json` and two JSON model files record a public-data
neural pilot with whole states held out for validation and testing. It compares
the logistic and three-layer proposal models on one declared linkage attack;
its outputs are predictions, not release certificates. `docs/SIX_PERSPECTIVE_REVIEW.md` records the six disciplinary review lenses,
implemented corrections and open validation needs. The risk bounds on the
prespecified Census grid are simultaneous over 72 planned setting--attacker
comparisons, conditional on the fixed public file and independent simulated
attack draws. Holding out states from a single file does not establish performance
on separate institutions or live release programmes.
The expanded comparison in `artifacts-real/neural_comparison.json` (with model
weights for its first split in `artifacts-real/neural_comparison_models.json`) finds lower
test loss for one- and three-layer models than plain logistic regression on
three overlapping state splits. An explicit quadratic logistic model performs
better than both networks. This supports nonlinear modelling of the declared
attack, not a unique depth advantage or a release decision.

The credit register case of the paper runs inside the same pipeline: `credit.py` builds a
loan-level register and measures borrower identification, the predictive value of an
alleged borrower-lender relationship, attribute disclosure of arrears status,
reconstruction of a lender's position from published totals, and differencing across
monthly vintages, against one minimum cell count.

`sequential.py` covers the case where nobody stops when they said they would. It
implements the beta-binomial e-process and the confidence sequence it induces, so a
risk bound stays valid however often the run is watched; the posterior at which an
adversary with a declared payoff acts, which is one half only when the stakes are
equal and allegations are free; and the transport bound that turns a declared
neighbourhood of the attack distribution into a risk ceiling. The measured result:
an institution that peeks and stops when its fixed-sample bound first passes is
wrong far more often than its stated level, and the confidence sequence costs a few
per cent of width to fix that.

`evaluation.py` runs the attack evaluation the long-form manuscript reports in Section 13: seven
protection policies that institutions operate, from the three-contributor rule to
$k$-anonymity and noisy counts, against four attacks on the same credit register, scored
by success rate, precision on the attempts the adversary acts on, worst subgroup, the
auxiliary information the attack needs, and calibration when the adversary's own error
model is wrong. It also searches for the minimum cell count a tolerance implies, once per
release and once for the pooled programme, choosing by the exact upper bound, and it
separates the pooled excess into the part due to extra auxiliary readings and the part
due to pooling the releases. Noisy-count releases are composed across vintages, each
with fresh noise.

`configs/base.json` is a small smoke test that finishes in seconds. `configs/paper.json`
is the run the paper reports: a register of 200,000 units, a 5 per cent sample, 45
mechanism settings and around a million simulated linkage attempts. It takes a few
minutes on one core.

## Colab, and moving work between Colab and GitHub

Open `notebooks/privacy_risk_thresholds_colab.ipynb` in Google Colab and run every cell in
order. It generates only synthetic data, reproduces the paper's principal results, runs
the pooled-release attack, and writes figures and a decision record.

The first cell does the git work. It clones the repository the first time, pulls every
later time, installs the package and leaves you in the repository root:

```python
repo.push("what changed")   # commit everything, take remote changes, push
repo.pull()                 # take remote changes, keeping your edits
repo.status()               # what is different here
```

The token comes from a Colab secret named `GITHUB_TOKEN`, which you set once.
`docs/COLAB.md` has that one-time step and the cell to paste. Nothing prints the token and
no file in the repository holds it.

On your own machine the same two commands are:

```bash
python scripts/sync.py pull
python scripts/sync.py push -m "what changed"
```

Pushing to `main` also starts the `refresh evidence` workflow, which reruns the paper
configuration, refreshes `artifacts-paper/` and `paper/generated/`, checks the manuscript
against the run and commits the result back. So the numbers in the repository follow the
code without anyone remembering to rebuild them.

## The ACM TOPS version

`python scripts/make_acmart.py` rewrites the manuscript's front matter for the
current ACM `acmart` class and writes `paper/tops/`, tables and figures included.
The article version stays the working copy, so a change to the paper reaches the
submission by rerunning the script rather than by editing a second file.
`docs/TOPS.md` lists what to set before submitting.

## What a run writes

```
artifacts/
  config.json      the configuration, copied verbatim
  results.json     every experiment's output
  decision.json    the governance record: status, tolerance, owner, expiry, programme note
  figures/         the paper's five figures
  paper/           LaTeX table fragments and numbers.json for the manuscript
  manifest.json    SHA-256 of every file above
```

`artifacts-paper/` in this bundle is the run the paper reports, manifest included, so a
reader can check the tables and figures without re-running anything.

The paper `\input`s the table fragments, so a table cannot drift from the run that
produced it. `scripts/check_paper_numbers.py` has two modes. The default checks that every
value in `paper/numbers.json` appears in the long manuscript as a whole number token. The
`--mode quoted` check, for the journal article, extracts every decimal with three or more
places from the prose and fails when no declared result file produced it. Both exit
non-zero on a mismatch.

## Reproducibility contract

- Python 3.10 to 3.12 in CI (the 2026 bundle was last run on 3.11); one deterministic
  seed in the configuration.
- No confidential data and no network access are required.
- Every run writes configuration, results, validation status and SHA-256 hashes.
- A failed validation produces a non-zero exit code and cannot be presented as governed
  evidence.
- `.github/workflows/ci.yml` runs lint, unit tests, the notebook, the fast pipeline and
  the number checks. `.github/workflows/refresh-evidence.yml` reruns the paper
  configuration on a push to `main` that touches code or configuration, publishes the
  tables and commits the result.
- Figures are written without creation dates and `results.json` is strict JSON, so two
  runs with one configuration produce identical files and manifests.

## Map from paper to code

Numbers refer to `paper/bis_ifc_full.tex` as compiled in this bundle.

| Paper object | Implementation | Test |
|---|---|---|
| Register, sampling design, mechanism family | `synthetic.py` | `test_synthetic.py` |
| Key frequencies, Eq. (3) | `keys.py` | `test_keys.py` |
| Posterior linkage and the declared attack, Eqs. (9)--(11) | `linkage.py` | `test_linkage.py` |
| Scenario measures, Eqs. (12)--(19), and the frequency model | `risk.py` | `test_risk.py` |
| Monte Carlo interval and certification cost, Prop. 3, Eqs. (20)--(23) | `uncertainty.py` | `test_uncertainty.py` |
| Governance rule Eqs. (5) and (29), implied tolerance Prop. 6, risk budget Prop. 11 | `governance.py` | `test_governance.py` |
| Attainable set, frontier and fitted value curve, Eqs. (32)--(35) | `frontier.py` | `test_frontier.py` |
| Union bounds and the pooled attack, Props. 9--10, Eqs. (36)--(37) | `composition.py` | `test_composition.py` |
| What a privacy budget permits, Prop. 14, Eqs. (46)--(47) | `dp.py` | `test_dp.py` |
| Falsification protocol, Section 14 | `validation.py` | `test_validation.py` |
| Credit register case: five risks and one utility, Section 12 | `credit.py` | `test_credit.py` |
| Adversarial evaluation of the release system, Section 13 | `evaluation.py` | `test_evaluation.py` |
| Anytime certification, paid adversary, transport bound, Section 8 | `sequential.py` | `test_sequential.py` |
| Does the pooling gap survive at other register sizes, Section 10.7 | `experiments.programme_scaling` | `test_experiments.py` |
| The paper's experiments end to end | `experiments.py` | `test_experiments.py` |
| Real public microdata, Section 11 | `realdata.py` | `test_realdata.py` |
| Figures, tables and the evidence manifest | `reporting.py` | `test_reporting.py` |
| Validation, gate and evidence writing | `pipeline.py` | `test_pipeline.py` |
| SENTRY-R deep proposal, reviewed LLM intake and frozen audit, Appendices E to H | `sentry_r.py` | `test_sentry_r.py` |
| Registered sequential gate (journal article, Theorem 1) | `adaptive_audit.py` | `test_adaptive_audit.py` |

The claim-by-claim warrant, counterargument and refutation conditions are recorded in
`docs/CLAIMS_EVIDENCE.md`.

## Interpreting outcomes

`PASS` means the declared model, uncertainty bound and tolerance permit the release. It is
not proof of safety against an undeclared adversary. `FAIL` means the governed rule
refuses the release. `INVALID` means a prior validation check failed, so no release
conclusion is allowed. Every decision record carries the programme finding as well: a
pooled attack on several releases is not bounded by their marginal risks, so a programme
needs its own tolerance and its own joint simulation.

## Neural proposal model and editable diagrams

**SENTRY-R**, shorthand for scenario-enriched neural triage for release review,
is the trainable,
depth-configurable implementation of the paper's neural proposal method.
It includes a provider-neutral LLM scenario intake, measured-attack evidence
join, deep and logistic training options, JSON model export, frozen shortlist,
and a separate simultaneous final audit. Start with
[`docs/SENTRY_R.md`](docs/SENTRY_R.md); all included data are synthetic.
`paper/bis_ifc_sentry.tex` is a short companion author manuscript on the
neural method; `paper/bis_ifc_full.tex` is the full manuscript. Neither is a
BIS or IFC publication. On the synthetic benchmark, trained to convergence
over ten seeds, the logistic and deep models reach indistinguishable held-out
loss: the benchmark shows that the pipeline runs, not that depth helps.

`privacy_risk.neural_trust` provides an optional nonlinear proposal model. It
accepts attack-level development labels, ranks a fixed shortlist, and certifies
each candidate against every declared risk coordinate using fresh binomial
attacks and a simultaneous exact bound. The neural output never supplies the
release certificate. Callers must supply a final audit function whose attack
draws are independent within each declared coordinate and separate from model
training and shortlist selection. Missing dimensions and untested adversaries
cannot be inferred from the trained model.

Appendix G of `paper/bis_ifc_full.tex` (Appendix D of `paper/main.tex`) derives the
binomial bound, proves simultaneous
coverage after choosing a candidate by utility, works a zero-success audit
budget, and states the limits imposed by dependence and population shift.
`neural_trust.zero_success_budget(tolerance, alpha, total_tests)` computes the smallest
per-coordinate trial count that can pass when no attack succeeds; observed
successes can require a larger audit. It is a different function from
`adaptive_audit.zero_success_budget(tolerance, alpha, weight)`, whose third argument
is the coordinate's share of the error budget and which assumes a bet of zero.

Run the seeded two-coordinate synthetic example at any input dimension ≥ 2:

```bash
privacy-risk-neural-demo --dimension 32 --seed 19 --output neural_demo.json
```

The example is independent of `privacy-risk-run` and does not modify the paper's
historical empirical results. The paper's four explanatory diagrams are
editable SVGs in `paper/diagrams/`. Run `python scripts/build_diagrams.py`
(with Inkscape installed) to regenerate their PDF counterparts. The article
and ACM TOPS builds use those PDF counterparts; the source SVGs remain editable.

## Version 0.6 mathematics expansion

The expanded derivations are integrated into the full paper and journal supplement. See `docs/MULTISCALE_MATHEMATICS.md` for the new analytic sensitivities, multiscale proposal dictionary, TV-shift audit bounds and trial-budget planner, and `docs/EXPANSION_CHANGELOG.md` for validation and scope. Run `python scripts/multiscale_demo.py` after installing the package. Neural sensitivity bounds describe a predictor, while independent audit trials supply eligibility evidence.
