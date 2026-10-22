# Public-data experiment: six academic review perspectives

This is an internal adversarial review written from six disciplinary viewpoints.
It is **not** a review by six independent professors or a journal decision.
The changes below were applied to this revision; open items need new evidence.

## Data and protocol

`python scripts/realdata_replication.py` uses the public `wooldridge` 0.5.0
`census2000` extract (29,501 records) and the `loanapp` file (1,989 source rows;
1,986 complete cases under the declared variables). The archived source-file
SHA-256 values are in `artifacts-real/replication.json`. No individual-level
records are exported. Results condition on these **already public** files.
Target rows are sampled with replacement with fresh auxiliary noise. The grid
contains 36 prespecified combinations, 25 of which pass the minimum coverage
rule, and two attack precisions. Its upper bounds use alpha/72 for every planned
pair. This protects selection among settings and these two attacks conditional
on the fixed files, not the choice of a wholly new adversary.

| Viewpoint | Major review objection | Implemented response | Remaining limit |
|---|---|---|---|
| Statistical disclosure control | File-level key uniqueness is not population uniqueness. Auxiliary data availability and block knowledge determine identification. | The paper distinguishes file-conditional risk, model-based population frequency, and actual population claims; scenario sensitivity and explicit threat diagram were added. | No observed external attacker or population register validates the assumed auxiliary readings. |
| Mathematical statistics | Choosing the best release after looking at noisy risk estimates invalidates per-setting 95% upper limits. | Exact one-sided binomial limits now allocate 0.05 over all 72 planned setting–attacker pairs. The manuscript derives simultaneous coverage and names the conditional independence assumption. | New attacks and post-hoc grid expansion require a fresh allocation or independent audit. |
| Machine learning | The first state-held-out neural pilot showed no depth advantage; a comparison with an interpretable nonlinear model was missing. | Added 1,410 measured scenarios across 12 policies and three auxiliary-reading precisions, three state-held-out splits, and linear logistic, one-layer, three-layer and quadratic logistic comparators. The neural models beat the linear baseline in each split, but quadratic logistic did better still. Predictions never enter release bounds. | This exploratory design followed inspection of the smaller pilot; the splits overlap and all states come from one file. Independent institutional validation is still missing. |
| Official statistics and governance | A public demonstration cannot authorize publication of a confidential register. | The report labels data provenance, source hashes, decision scope and absence of a real-register gate. The manuscript separates the synthetic credit gate from the public-file experiment. | An accountable owner must set tolerances, inspect auxiliary information, and audit a proposed live release. |
| Applied econometrics | Utility computed from a public extract is not population utility and suppressing observations can change composition. | Retention and analysis-specific loss stay visible; the text now says the reference is the published extract. The 1,986 complete-case denominator is explicit. | No design weights, external outcomes or population-based utility benchmark are available for these extracts. |
| Reproducibility and software methods | Python `hash()` changes between processes, so the same seed cannot reproduce the reported frontier; prose and tables drift. | A stable `SeedSequence` replaces `hash()`. A two-process comparison under different `PYTHONHASHSEED` values matches byte-for-byte. The full paper run regenerates the table and its manifest; an executable real-data script records data hashes, version, seed and sensitivity results. | Package versions beyond `wooldridge` are not pinned; source hashes allow detecting changes. |

## Findings, with scope

- Full Census key: 25,454 singleton cells, or 86.28% of the 29,501 released records. This is **sample uniqueness**.
- Four frontier settings among 25 evaluated have familywise upper risks about 0.276–0.547 under two specified attack strengths. The most protected frontier setting retains 8,388 records (28.4%).
- The four 3-anonymous Census variants have marginal guessing probabilities summing to 0.381, while candidate-set intersection yields a pooled probability 0.709 over 5,000 sampled targets. The marginal union bound applies to the event that a **single-release** attacker succeeds, not to a new intersection attack.
- In the mortgage file the declared attack succeeds in 201 of 1,500 draws (0.134; individual one-sided upper bound 0.149). Its 1,986 complete cases occupy 150 key cells, 63 of which are singletons.
- Exploratory attacker sensitivity on the selected Census setting gives rates 0.371, 0.271 and 0.186 for log-income reading noise 0.25, 0.5 and 1.0; corresponding mortgage rates are 0.179, 0.137 and 0.093 for amount reading noise 20, 40 and 80. These are separate 3,000-target runs; their individual bounds do not form a simultaneous family.
- On nine held-out Census states, SENTRY-R logistic test log loss is 0.6323, three-layer log loss 0.6338, and a constant training-rate baseline 0.6837. Brier scores are 0.2207, 0.2212 and 0.2453 respectively. The deep model does **not** improve on logistic in this single split. These estimates are a pilot, not a significance test or a safety certificate.
- On the expanded 1,410-scenario comparison, the three-layer model improves log loss over plain logistic by 0.0025, 0.0054 and 0.0047 across three overlapping state splits. The one-layer network improves by 0.0038, 0.0062 and 0.0055. A logistic model with explicit pairwise products and squares scores better than either network in each split. The chart and generated table are in `paper/diagrams/realdata_neural_comparison.svg` and `paper/generated/table_real_neural.tex`; paired state-bootstrap intervals and representative model weights are in `artifacts-real/neural_comparison.json` and `artifacts-real/neural_comparison_models.json`. The improvement demonstrates a role for nonlinearity in this exercise, not a need for depth.

## Reproduce

```bash
python -m pip install -e '.[dev]'
python scripts/realdata_replication.py
python scripts/realdata_neural_pilot.py
python scripts/realdata_neural_comparison.py
python scripts/plot_realdata_neural.py
privacy-risk-run --config configs/paper.json --output artifacts-paper
python scripts/rebuild_evidence.py --artifacts artifacts-paper --publish paper/generated
python scripts/check_paper_numbers.py --artifacts artifacts-paper --paper paper/bis_ifc_full.tex
pytest
```

`artifacts-real/replication.json` includes the exact inputs and estimates.
`artifacts-paper/manifest.json` lists hashes for the full run's generated
evidence. The public data are supplied by an installed dependency and not
redistributed in this archive. Neither experiment measures LLM proposal quality
on real data or operates on a confidential credit register.
