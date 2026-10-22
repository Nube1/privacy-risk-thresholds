**Referee report, Professor 5 (learning-to-audit, selective inference, conformal, LLM-assisted testing)**

**Verdict.** The inference is sound: the Clopper-Pearson inversion, the Bonferroni gate over a shortlist fixed given D, and the zero-success budget are all correct, and they agree between the code and the paper. The problems are elsewhere. The separation of training data, shortlist and audit is only partly enforced (alpha and planned trial counts are never frozen). The one measured "deep beats logistic" result goes away once the baseline is trained to convergence. Recommendation: major revision.

Tests: 9 of 9 pass with `--no-cov`. The command given in the brief exits non-zero, but only because pyproject.toml:27 sets a repo-wide 85% coverage gate. No test fails.

### A. Verified errors

**1. MAJOR: alpha and planned trial counts are not frozen, so the audit can be tuned after the fact.** src/privacy_risk/sentry_r.py:413-417 and 429-452.
- The freeze record holds candidates, tolerances and schema, but not `alpha` and not `B_θj`.
- `final_audit` takes `alpha` from the CLI (line 501) and accepts any `trials` array.
- So an analyst can re-run with a larger alpha, or keep sampling until the audit passes. Both break the fixed-B exact bound.
- The paper says the reverse: sentry_r_section.tex:41-42 ("freeze … their planned sample sizes") and bis_ifc_sentry.tex:126 ("An audit is opened only after a shortlist and its sample sizes are fixed").

Fix, in `shortlist(bundle, candidates, size, *, alpha, trials)`:
```python
freeze.update({"alpha_family": alpha, "planned_trials": trials,
               "development_evidence_ids": bundle["evidence_ids"]})
```
Add `"evidence_ids": sorted(e for r in train_rows+valid_rows for e in [r["evidence_id"]])` in `train_model`. Then in `final_audit` drop the `alpha` argument and add:
```python
alpha = freeze["alpha_family"]
if b != freeze["planned_trials"]:
    raise ValueError("audit trials differ from the frozen plan")
if row["evidence_id"] in set(freeze["development_evidence_ids"]):
    raise ValueError("final audit reuses development evidence")
```

**2. MAJOR: the benchmark gap comes from undertraining the baseline.** scripts/sentry_r_benchmark.py:22; paper/generated/table_sentry_benchmark.tex:9-10; bis_ifc_sentry.tex:236-241.
- The logistic baseline's `best_epoch` is 80, which equals the `epochs=80` cap, so early stopping never ran.
- I re-ran the benchmark with `epochs=1000`. Logistic reaches 0.2760 at epoch 587; the three-layer ReLU reaches 0.2779. The ordering flips.
- I also drew 10 fresh replicates from the generator's own law, with a separate 2,000-row test set and 400 epochs. Logistic wins all 10 (mean 0.2562 against 0.2569; differences of 0.0003 to 0.0013).
- The reported "validation log loss" is also the early-stopping minimum on the same 100 rows. That is a post-selection, optimistic figure.
- The BIS table is typed in by hand rather than `\input`, so it will drift from the generated file.

Fix: run both models to convergence; report loss on a third programme not used for stopping; use at least 10 seeds with a spread. Replace the prose at sentry_r_section.tex:126-130 with: "Trained to convergence, the logistic and three-layer models reach indistinguishable held-out log loss on this toy law. The table shows that the pipeline runs, not that depth helps." Replace the BIS table with `\input{generated/table_sentry_benchmark}`.

**3. MAJOR: the LLM's own output supplies its provenance fields.** sentry_r.py:126-133.
- `propose_from_llm` asks the model to return `llm_model` and `prompt_hash`, then stores what comes back.
- This contradicts bis_ifc_sentry.tex:115 ("records the prompt/model source").

Fix:
```python
def propose_from_llm(call, schema, question, *, model_name: str):
    prompt = ...
    ph = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    result = json.loads(call(prompt))
    ...
    rows = [{**{k: r[k] for k in ("proposal_id","scenario","features")},
             "llm_model": model_name, "prompt_hash": ph} for r in result]
    return stage_llm(rows, schema)
```
and drop the two fields from the prompt text.

**4. MINOR: owner and expiry are claimed but not implemented.** bis_ifc_sentry.tex:265 says the decision record contains "owner and expiry". Neither `final_audit` nor the freeze records them. Either add both fields or delete the words.

**5. MINOR: the ranking rule differs between paper and code.** neural_section.tex:67 says the shortlist is ranked "by estimated risk and public utility". Both neural_trust.py:100 and sentry_r.py:409 rank on max predicted risk over tolerance alone. Replace with: "ranks candidates by their largest predicted risk-to-tolerance ratio; utility enters only the final choice among certified candidates."

**6. MINOR: the depth in the equation is not the depth in `neural_trust`.** neural_section.tex:46-53 writes a depth-L network. `neural_trust.MLP` (lines 118-161) has exactly one hidden layer. The appendix (line 102) says this correctly. Add after eq. (neural-risk): "The `neural_trust` module fixes L=1; `sentry_r` makes L configurable."

**7. MINOR: the Lipschitz bound needs the scaling factor.** sentry_r_appendix.tex:199-204 and jpc_network.tex:33. The network acts on standardised inputs (sentry_r.py:408), so the bound in raw features must include the factor max_k 1/s_k. Change the bound to "(1/4)(∏‖W_m‖₂)·max_k s_k^{-1}·‖x−x′‖₂".

### B. Checks that pass

**8. Zero-success budget.** Correct. Hand check for τ=0.12, α=0.05, m=8: log(0.00625)/log(0.88) = −5.0752/−0.12783 = 39.70, so n = 40. Then 0.88^40 = 0.00602 ≤ 0.00625, while 0.88^39 = 0.00684 > 0.00625. Other cases, code against ⌈log(α/m)/log(1−τ)⌉:

| τ | m | budget |
|---|---|---|
| 0.04 | 4 | 108 |
| 0.05 | 1 | 59 |
| 0.01 | 20 | 597 |
| 0.20 | 4 | 20 |
| 0.18 | 8 | 26 |

All match. The appendix endpoints (0.11916 at b=40, 0.12202 at b=39) also check out. The floating-point guard loop at neural_trust.py:56 is correct.

**9. Selective coverage.** Correct as stated.
- M = C·n is right, because candidates outside the shortlist are never certified.
- The inversion argument at neural_appendix.tex:40-45 is right.
- Taking the highest-utility candidate among the certified ones is covered by the deterministic implication (eq. neural-implication).
- The proof needs the shortlist, C, n, B and α to be measurable with respect to D. Finding 1 shows the code enforces C and the candidates, but not B or α.

### C. Concerns (not errors, but they limit the claims)

**10. MAJOR: the separation is procedural, not enforced.**
- `evidence_kind` is a string the caller sets itself.
- The SHA-256 freeze has no key and no timestamp, so anyone can rebuild it. It shows the record was not altered; it does not show what happened first.
- Nothing stops a second `final_audit` on the same freeze with new draws, or a new freeze made after seeing an audit.
- `propose_from_llm`'s free-text `question` can carry audit outcomes into proposals. The paper's own caveat (sentry_r_section.tex:45-46, appendix 184-186) is documentation only.

Fix: keep an append-only ledger. `final_audit(..., ledger: Path)` should refuse a `freeze_sha256` it has already recorded. `propose_from_llm` and `shortlist` should refuse to run for a schema whose ledger holds a final audit unless a new α allocation is declared. Until this exists, jpc_main.tex:116 "code-level restrictions on how neural/LLM proposals can affect audits" overstates the code. Replace with: "software checks that record, but do not enforce, the order of LLM proposals, development attacks and final audits."

**11. MAJOR: "depth-configurable" and "neural trust" rest on software alone.**
- Depth is a working parameter. The finite-difference gradient test (depth 3) and the CLI test (depth 2) exercise it.
- No measured result shows that depth, or the network at all, improves shortlists. Finding 2 suggests it does not, even on a toy law built with an x₀x₂ interaction.
- The toy "held-out programme" is drawn from the same law and feature range as training, so it is not a shift test.

**12. MINOR: rename away from "trust".** The paper itself says the network "does not define trust" (sentry_r_section.tex:11), and the "trust position" is the normalised margin 1 − U_j/τ_j. Both terms credit the network with something it does not supply. Proposed changes:
- Expand SENTRY-R as "scenario-enriched neural triage for release review".
- Rename `trust_position` to `audited_margin`, and T(θ) to "minimum audited margin".
- Retitle §sec:neural "An audited margin vector with a neural proposal model".
- Rename the module to `neural_triage.py`, keeping a deprecated alias.

**13. MINOR: tests.** No test covers a changed α, changed trial counts, audit evidence reused from development, or a second audit on one freeze. Add these alongside the fixes in 1 and 10. The budget test checks one point only; add a parametrised check against the closed form.

Relevant files:
- /home/claude/work/privacy-risk-research/src/privacy_risk/sentry_r.py
- /home/claude/work/privacy-risk-research/src/privacy_risk/neural_trust.py
- /home/claude/work/privacy-risk-research/scripts/sentry_r_benchmark.py
- /home/claude/work/privacy-risk-research/paper/generated/table_sentry_benchmark.tex
- /home/claude/work/privacy-risk-research/paper/sentry_r_section.tex
- /home/claude/work/privacy-risk-research/paper/bis_ifc_sentry.tex
- /home/claude/work/privacy-risk-research/paper/publication_versions/jpc_main.tex