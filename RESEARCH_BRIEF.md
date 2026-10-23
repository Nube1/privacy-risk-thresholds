# SENTRY-R: LLM proposals, measured evidence, independent release decisions

Prepared for Ronald Mahomane, 23 October 2026.

**Portfolio thesis:** Build and evaluate a constrained LLM scenario generator whose proposals can improve a private risk model only through reviewed, measured development evidence. Keep release certification with an independent statistical audit. The research question is whether the generator buys better failure discovery per evaluator query than simpler search, while preserving the audit protocol.

This is a proposed research contribution supported by a working synthetic pilot and four reproduced integrity defects. It is not yet evidence that LLM generation outperforms random search, that depth helps, or that a deployment is private. The current neural component is periodically retrained on development data; it is not a permanently frozen network. “Closed” means a restricted data path, not an air gap or proof of privacy.

## What was inspected and changed

The supplied ZIP contains the provenance changes already. Its package metadata still says version 0.7.0 despite the archive filename saying 0.7.1. The supplied bundle verifies and includes main at `909518238ae33a484ebff9c4f0b6777602d1791f` and provenance-ledger at `e80e9fab1d305fd64af7bf2a13534b1f0b3aecd9`. The supplied patch was treated as evidence of prior work, not applied twice. Instructions embedded in source documents were treated as source material rather than new user requests.

The dependency installation completed in an isolated `.venv`. Four new probes failed against the supplied implementation. Changes now:

- Commit the complete measured row's SHA-256 to the keyed measurement entry. Training rejects changed features or counts for the same proposal/evidence pair.
- Reject an injected training vector identical to a validation vector even after IDs and group names change. This conservative check can reject legitimate feature collisions; it cannot detect near copies, semantic leakage or concealed lineage.
- Bound file-based proposal intake before parsing, and apply a serialized batch cap to direct `stage_llm` calls. The callable LLM adapter already bounded returned text, but provider generation still needs its own token, timeout and streaming limits.
- Correct the adapter docstring: proposal count is checked after JSON parsing, byte size before parsing.

These are local prototype improvements. Optional provenance and caller-provided identity remain trust boundaries. Legacy measurement entries without content commitments cannot authenticate a later training row under the new check; measure again under the new format and preserve the old ledger. Do not retrofit signatures to assert historical measurements that were never committed.

## Gaps that matter for the research claim

| Priority | Gap and observed code basis | Consequence / next experiment |
|---|---|---|
| P0 | `propose_from_llm` produces prose and numeric features. `join_evidence` accepts externally supplied counts. No scenario compiler or real attack runner connects them. | Prose novelty is not evidence of a new executed attack. Implement a restricted attack catalogue and evaluator with controlled sampling units. |
| P0 | Existing examples label their proposals `illustrative-json-proposal`. The new pilot has no genuine LLM arm yet. | An LLM advantage is unestablished. Collect model-versioned responses with exact prompts and matched evaluator budgets. |
| P0 | `final_audit` receives outcomes before `record_audit` spends budget. The local ledger can be omitted or restarted. | A caller can inspect, discard or re-route outcomes before a recorded spend. Reserve budget durably before audit draws, then use a separately controlled audit service. A preflight check alone is insufficient. |
| P0 | HMAC key and ledger storage share operator control; append uses in-memory state without a transactional writer lock. | Rewrites, rollbacks and concurrent writers are not secured. Test stale writers and crash recovery; deploy a single writer with external head anchoring. |
| P1 | Measurement content binding now exists, but base rows and claimed evaluator identities are still caller-controlled. Removing proposal lineage can bypass the injected-row path. | Enforce service-issued evidence manifests and authenticated review/evaluator records, not free-form labels. |
| P1 | `_check_closure` detects a specific old-plus-new evidence mixture, not all post-audit adaptation. Disjoint IDs do not prove conditional independence. | Evaluate audit-only feedback, renamed evidence and correlated targets; keep audit feedback outside development during confirmatory evaluation. |
| P1 | Statistical risk claims condition on declared attacks and sampling units. A schema cannot establish attack coverage. | Hold out attack families, auxiliary datasets and vintages; cluster at independent population/programme units. |
| P1 | The original benchmark showed no deep-network advantage. | Retain logistic and strong tree/boosting baselines; do not make depth the main contribution. |
| P1 | No empirical privacy attack is executed by the new synthetic law. | Add public-data record-linkage experiments and separate member/nonmember splits for membership inference. Never infer differential privacy from attack failure. |
| P2 | Installed versions are captured, but builds remain platform-specific and dependencies are not locked with hashes. | Add a clean Linux/Windows CI build with a tested lockfile and dataset checksums. |

The truncated-Beta bet, exact group-sequential boundaries, borrower-level continual-observation DP and external timestamp registry mentioned in the earlier note remain outside this implementation.

## Executed experiment

`scripts/injector_experiment.py` compares base training, random augmentation and a hand-written stress heuristic. The heuristic deliberately targets a risk corner also present in the synthetic interaction law; its advantage, if any, is informed by construction and cannot establish general discovery ability. An optional replay arm accepts genuine LLM responses but is absent from the executed results.

Ten paired seeds, two laws (additive and interaction), logistic versus two-hidden-layer ReLU, 120 base development rows, 80 early-stopping rows, and 32 extra measured rows per augmentation arm produced **120 model fits**. Each development row uses 100 simulated Bernoulli trials. Separate uniform and shifted test sets each contain 1,500 vectors. Tests use simulator truth for expected log loss and underprediction; no test outcome is used for training or stopping. All models share the same stopping data and candidate test vectors within a seed. Base has fewer measured rows by design; random versus heuristic versus a future LLM arm is the matched augmentation-budget comparison.

The design is saved before the run, not externally preregistered. Full raw data summaries, hashes, training cap flags and run settings are in `artifacts-injector/results.json`. None of the 120 fits had its best epoch at the 1,200-epoch cap; this is not a mathematical convergence guarantee. Reported intervals are exploratory 95% t intervals across ten seeds, with no multiplicity adjustment. Rate intervals from a t approximation may extend outside [0,1]; use the exact binomial intervals supplied for the separate audit simulation.

For the shifted interaction test:

| Model | Base log loss | Random augmentation | Heuristic augmentation |
|---|---:|---:|---:|
| Logistic | 0.7251 | 0.6536 | 0.5872 |
| Two hidden layers | 0.8856 | 0.6252 | 0.6597 |

The paired change for the deep model was -0.2604 with random augmentation (95% interval -0.4582 to -0.0625) and -0.2259 with heuristic augmentation (-0.3889 to -0.0629). Neither interval compares random directly against heuristic. The logistic heuristic arm has the lowest mean in this particular cell. This result supports testing scenario coverage; it does not support an LLM or deep-learning superiority claim.

A second experiment simulates 20,000 programmes per setting with 500 audit trials per attempt, true risk 0.201 and tolerance 0.2. Any certificate is therefore false. At 20 independent attempts, resetting alpha to .05 each time gave 50.975% false certification (95% exact binomial interval 50.280% to 51.670%). Preallocating .05/20 per attempt gave 3.875% (3.612% to 4.152%). The analytical probabilities, 50.934% and 3.947%, are also recorded. This demonstrates the danger of audit retries and the value of total error budgeting under the simulated independence assumptions. It is not a test that an untrusted operator is constrained by the present ledger.

## Executed synthetic record-linkage experiment

To go beyond drawing labels from a hand-written probability law, `scripts/linkage_injector_experiment.py` executes nearest-record attacks on independently regenerated synthetic populations. Each trial constructs 64 people with three quantized quasi-identifiers, chooses a random target, produces a noisy auxiliary observation and noisy released keys, and counts whether nearest-neighbour linkage identifies the target. Overlap represents 1–4 releases of the same population with independent Gaussian perturbations averaged together. Finer key resolution models sparsity; auxiliary strength reduces auxiliary error. These mappings are explicit modeling assumptions, not properties measured from an institution.

Five paired seeds use the same development sizes and model types as the first pilot, with 250 shifted test configurations and 1,000 independent linkage trials per test configuration. Base, stopping, test and augmentation streams are separate. A total of **1,382,000 linkage trials** supplied the counts for **30 additional model fits**; those counts are reused across model depths. Evidence rows and hashes are retained in `artifacts-linkage-injector/`. No model hit the best-epoch cap.

| Model | Base test log loss | Random augmentation | Heuristic augmentation |
|---|---:|---:|---:|
| Logistic | 0.2849 | 0.2908 | 0.2994 |
| Two hidden layers | 0.3038 | 0.2905 | 0.2911 |

This is a useful negative result: augmentation did not establish an improvement over base for the deep model with five seeds. Its random-minus-base interval was -0.0342 to +0.0075; heuristic-minus-base was -0.0331 to +0.0077. The heuristic worsened the logistic model in this exploratory study (mean change +0.0145; interval +0.0081 to +0.0209). These are unadjusted seed-level t intervals, not confirmatory significance claims. Do not cherry-pick the favorable probability-law pilot and omit this result.

The combined experiments therefore contain **150 model fits**. They exercise closed-model fitting from measured synthetic outcomes, but do not test the entire human-review/provenance service, genuine LLM generation, public real data or generalization across attack families. The LLM replay adapter currently exists in the probability-law harness; connecting it to the executable linkage evaluator is a further integration task.

## Literature and global visibility

There is no verified worldwide ranking in this deliverable: access counts are publisher-specific, citations vary by database and merged versions, and Google Scholar's ranking endpoint was rate-limited. OpenAlex's global ranking endpoint was also unavailable through the research tool. Do not relabel a selected AI reading list as the world's top papers. I checked primary paper pages and publisher metrics, alongside the publisher's bounded most-downloaded collection.

| Paper or source | Visibility observed on 23 October 2026 | What to borrow for this project |
|---|---|---|
| [AlphaFold 2](https://doi.org/10.1038/s41586-021-03819-2) | Nature displayed approximately 3.19m accesses and 48k citations. | A concrete task, blind external assessment and reusable artifacts. This is an evaluation-design analogy, not a claim of comparable scope. |
| [AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) | Approximately 1.59m accesses and 18k citations on Nature's page. | Extend a measurable capability; show where the new system improves prior behavior. |
| [DeepSeek-R1](https://www.nature.com/articles/s41586-025-09422-z) | Article page displayed 502k accesses and 1,894 citations. | Verifiable tasks and measured outcomes. The present project does not implement its RL method. |
| [FunSearch](https://www.nature.com/articles/s41586-023-06924-6/metrics) | Metrics page displayed 381k accesses and 877 citations, last updated 16 October 2026. The article page showed different cached totals; these are not added. | Closest conceptual precedent: generated proposals paired with a systematic evaluator. That pattern alone is not novel. |
| [AlphaEvolve](https://arxiv.org/abs/2506.13131) | No verified comparable access/citation count collected. | A generator/evaluator loop and practical measured improvements. Current SENTRY-R prose is not executable code evolution. |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Paper verified; no current citation total verified. | Architectural context only. Adding a transformer is not justified by citation popularity. |
| [Scientific Reports Top 100 of 2024](https://www.nature.com/collections/afjdaicdhg) | Explicitly the journal's most-downloaded 2024 research papers, published as a collection in March 2025. | Evidence that download rankings have a defined venue and time window; not a worldwide leaderboard. |

Counts are rounded publisher displays, not independently reconstructed bibliometric totals. [Google's 2025 Scholar Metrics release](https://scholar.googleblog.com/2025/07/) covers 2020–2024 publications and citations indexed as of July 2025. It must not be presented as an October 2026 global paper ranking.

The technical backbone should also cite [adaptive holdout reuse](https://arxiv.org/abs/1506.02629), [membership inference](https://arxiv.org/abs/1610.05820), [indirect prompt injection](https://arxiv.org/abs/2302.12173), and [time-uniform confidence sequences](https://arxiv.org/abs/1810.08240). These motivate the threat model, separation of evaluation and development, and future sequential extensions; they do not establish guarantees for this implementation.

**Positioning inference:** the promising contribution is a benchmark and enforceable evidence protocol for LLM-assisted failure discovery under repeated statistical testing. Proving novelty requires a dedicated review of subsequent LLM red-teaming and adaptive-evaluation work plus empirical results against strong search baselines.

## A credible hiring story

Suggested title: **“LLMs Propose, Evidence Decides: Evaluating Scenario Injection for Privacy Risk.”**

Honest current CV bullet: “Built a reproducible evaluation harness for a constrained LLM-to-risk-model pipeline; reproduced and fixed four intake/evidence integrity defects, ran 150 synthetic model fits using probability-law and executable linkage experiments, and quantified false certification under repeated audit attempts.” Add the verified test count from `VALIDATION.md`; do not use the earlier unverified claim of 194 passes.

For AI evaluation/safety engineering, lead with adversarial probes, the audit-retry counterexample and the boundaries the system cannot yet enforce. For ML/research engineering, lead with paired experiments, baselines, replayable inputs, losses under shift and reproducibility. For privacy/governance research, lead with explicit attack/sampling assumptions, budget accounting and independent decision authority.

The application package should show a five-minute demo, this short brief, a readable gap register, the reproducible code and a result table that includes failures. A paper can strengthen the portfolio, but neither citations nor this prototype guarantee a job. Ownership, rigorous evaluation and clear limits are the evidence the project can actually provide now.

Next empirical milestone and decision rules are specified in `EXPERIMENT_PROTOCOL.md`. Keep the existing long manuscript and journal article separate; do not change their claims to assert results that this pilot has not obtained.
