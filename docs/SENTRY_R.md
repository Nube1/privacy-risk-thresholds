# SENTRY-R: neural triage for release review

SENTRY-R is shorthand for **scenario-enriched neural triage for release review**. It is a name for this implementation, not a claim of unique naming or better results. The network estimates declared attack-success probabilities to rank candidates. Only an independent, fixed-family attack audit can certify a release. The network does not turn a policy tolerance into a learned label.

## Install and reproduce the synthetic walkthrough

```bash
python -m pip install -e ".[dev]"
python scripts/sentry_r_example.py --output examples/sentry_r
sentry-r stage-llm --schema examples/sentry_r/schema.json \
  --input examples/sentry_r/llm_proposals.jsonl \
  --output examples/sentry_r/staged_llm.jsonl
sentry-r review --staged examples/sentry_r/staged_llm.jsonl \
  --decisions examples/sentry_r/review_decisions.jsonl \
  --output examples/sentry_r/approved_llm.jsonl
sentry-r join-evidence --schema examples/sentry_r/schema.json \
  --staged examples/sentry_r/approved_llm.jsonl \
  --outcomes examples/sentry_r/development_outcomes.jsonl \
  --output examples/sentry_r/verified_development.jsonl
sentry-r augment --schema examples/sentry_r/schema.json \
  --base examples/sentry_r/train.jsonl \
  --measured examples/sentry_r/verified_development.jsonl \
  --output examples/sentry_r/augmented_train.jsonl
sentry-r train --schema examples/sentry_r/schema.json \
  --train examples/sentry_r/augmented_train.jsonl \
  --validation examples/sentry_r/validation.jsonl \
  --depth 3 --width 24 --epochs 6000 --patience 150 \
  --output examples/sentry_r/deep_model.json
sentry-r shortlist --model examples/sentry_r/deep_model.json \
  --candidates examples/sentry_r/candidates.jsonl --size 4 \
  --alpha .05 --trials 900 --owner "Synthetic release committee" \
  --expiry 2099-12-31 --output examples/sentry_r/shortlist.json
python scripts/sentry_r_example.py --output examples/sentry_r \
  --audit-shortlist examples/sentry_r/shortlist.json
sentry-r audit --shortlist examples/sentry_r/shortlist.json \
  --outcomes examples/sentry_r/toy_audit_shortlist.jsonl \
  --output examples/sentry_r/audit_report.json
```

The example is **synthetic**. The audit outcome is drawn only for the frozen shortlist from a separate random stream, with exactly the trial counts frozen in `shortlist.json`. Its observed performance says nothing about a real institution or release.

The `audit` step appends one line to `examples/sentry_r/audit_ledger.jsonl` and refuses a second audit of the same freeze. The first `sentry_r_example.py` call (without `--audit-shortlist`) deletes that toy ledger, because it starts a new toy history. Do not delete a real ledger.

Run `python scripts/sentry_r_benchmark.py` to rebuild the logistic/deep comparison in the manuscript. For each of ten seeds it draws three disjoint programmes from the toy law in `scripts/sentry_r_example.py`: 320 training rows, 100 rows used only for early stopping and 2,000 test rows used only for the reported loss. Both models train until 150 epochs pass without improvement on the stopping rows; the script refuses a run whose best epoch reaches the 6,000-epoch cap. It writes `artifacts-sentry/benchmark.json` and `paper/generated/table_sentry_benchmark.tex`. On this toy law the logistic model has slightly lower test log loss than the three-layer network in every replicate. The table shows that the pipeline runs, not that depth helps.

## Data contract

`schema.json` contains ordered numeric features with predeclared minimum and maximum, ordered risk coordinates and the corresponding policy tolerances. Its SHA-256 digest is recorded in the model and shortlist. One can use any positive input dimension `d`, coordinate count `n`, width and hidden-layer depth. A final audit must cover every coordinate of every frozen candidate.

Each training or validation JSONL line has:

```json
{"record_id":"release-7","group":"programme-A","evidence_id":"attack-run-58","evidence_kind":"observed_development_attack","features":{"key_sparsity":0.2,"sample_fraction":0.1,"auxiliary_strength":0.3,"release_overlap":0.4,"access_strength":0.1,"mechanism_noise":0.2},"successes":[3,5],"trials":[100,100]}
```

`group` identifies the whole programme, population or vintage held out together. Supply **disjoint groups** for training and validation. Each count must represent independent attack trials under a declared sampling design. If an auxiliary file or target is reused in a way that induces dependence, aggregate at an independent unit or use a valid alternate confidence procedure. Never train on the final audit. Each `evidence_id` must be distinct across the training and validation sets.

A candidate row contains `candidate_id`, `features`, and finite `utility`. The model file lists every development `evidence_id` (training and validation).

`shortlist` ranks candidates by their largest predicted risk-to-tolerance ratio, smallest first, and writes a freeze record. The freeze holds the candidates, coordinates, tolerances, family-wise `alpha_family`, the `planned_trials` for each candidate and coordinate, the `tolerance_owner`, an ISO `expiry` date and the development evidence IDs. Its SHA-256 digest `freeze_sha256` covers all of these. Utility plays no part in the ranking; it enters only the final choice among certified candidates.

The audit JSONL contains one row per shortlisted candidate: `candidate_id`, `evidence_kind: "independent_final_audit"`, the exact `freeze_sha256` from its shortlist, a distinct `evidence_id`, and integer arrays `successes` and `trials`. `audit` takes alpha from the freeze (it has no `--alpha` option) and refuses the run if the freeze was edited, if today is past the expiry date, if any `trials` array differs from the frozen plan, or if an audit `evidence_id` appears among the development evidence IDs. The decision record carries the owner and expiry.

By default `audit` keeps an append-only JSON-lines ledger, `audit_ledger.jsonl`, in the directory of `--output`; `--ledger PATH` chooses another file and `--no-ledger` turns it off. Each line records the freeze digest, schema digest, decision digest, selected candidate and UTC time. A second audit of a freeze already in the ledger is refused. In Python, `final_audit(..., ledger=None)` is the default, so a caller who wants the check must pass a path.

These checks bind the audit to its freeze. They do not prove the order of events: the digest has no key or trusted timestamp, anyone can write a new freeze, and a ledger file can be deleted or a new one started. The code does not stop a new shortlist or new LLM proposals after an audit has been seen. `evidence_kind` is a string the caller sets. The user remains responsible for authentic provenance, for keeping the ledger, and for the statistical independence of externally supplied counts.

## LLM-assisted scenario injection

The file intake (`stage-llm`) takes JSONL rows with **exactly** `proposal_id`, `scenario`, `features`, `llm_model`, and `prompt_hash`; whoever writes that file sets the last two fields and is responsible for them. The proposal text can describe a new adversary, pooled release, feature configuration or auxiliary information. With the Python adapter the model returns only `proposal_id`, `scenario` and `features`; the adapter refuses any other field and sets `llm_model` to the caller's `model_name` and `prompt_hash` to the SHA-256 of the exact prompt it sent:

```python
import json
from pathlib import Path
from privacy_risk.sentry_r import ScenarioSchema, propose_from_llm, write_jsonl
schema = ScenarioSchema.load(Path("examples/sentry_r/schema.json"))

def call_my_llm(prompt: str) -> str:
    return my_provider.generate_text(prompt)  # adapt this one function to your provider

proposals = propose_from_llm(call_my_llm, schema, "Explore pooled-vintage attacks",
                             model_name="provider-model-name-and-version")
write_jsonl(Path("staged_proposals.jsonl"), proposals)
```

Send only descriptions and metadata your institution permits to leave its environment. Model output is untrusted input: the adapter parses JSON, checks numeric feature ranges, hashes each proposal and **does not execute its prose**. LLM-produced success counts are refused. A named human reviewer records an `approved` boolean and reason in `review_decisions.jsonl`; `review` passes only accepted proposals with matching digests. A separate attack evaluator must measure each accepted proposal and write `observed_development_attack` outcomes, including the proposal digest; `join-evidence` verifies the digest and review. `augment` writes a **new development training version** that includes those measured rows; train with that file, keep a disjoint validation programme and record a new model. The final audit is never appended.

The workflow is therefore `LLM proposal -> schema validation -> documented human review -> actual attack experiment -> measured development count -> new training run -> fixed shortlist -> untouched final audit`. This is the precise meaning of *LLM injection* here. It allows LLMs to expand the scenario set without accepting a generated claim about risk as evidence.

## Mathematics and interpretation

For row `i`, risk coordinate `j`, count `K_ij` and `B_ij` independent trials, SENTRY-R minimises the observed binomial negative log likelihood, omitting a parameter-independent term:

\[
L(\phi)= -\frac{1}{\sum_{ij}B_{ij}}\sum_{ij}
 [K_{ij}\log p_{ij}+(B_{ij}-K_{ij})\log(1-p_{ij})]
 +\frac{\lambda}{2}\sum_{\ell}\|W_\ell\|_F^2.
\]

With `d` input features, `L` ReLU hidden layers of width `q`, and `n` sigmoid outputs, the forward-pass parameter and arithmetic order is `O(dq+(L-1)q²+qn)`. Depth zero gives the logistic baseline. We use Adam, deterministic seeded batches, gradient clipping and early stopping on a held-out whole programme (`--patience` epochs without improvement, up to the `--epochs` cap). The model file is JSON, never an executable pickle, and contains the schema, scaling fitted on training data only, weights, seeds and dataset hashes. Predictions are for candidate ranking and must be compared with a logistic baseline on the same held-out programmes.

For `C` shortlisted candidates and `n` audited coordinates, each coordinate gets an exact binomial upper bound at one-sided error `alpha/(Cn)`. The audited margin is `z_j=1-upper_j/tau_j` (the code field is still named `trust_position`). Only candidates with every `z_j >= 0` are eligible; utility chooses among them. The finite-family union bound controls the chance of **any false certification** by `alpha` conditional on the frozen development history. Changes to candidates, attacks or sample sizes after reading final outcomes require new untouched audit data or a proven adaptive procedure.

The mathematical derivations and limitations are in `paper/neural_section.tex`, `paper/neural_appendix.tex`, and `paper/sentry_r_section.tex`. `python scripts/realdata_neural_pilot.py` trains logistic and three-layer models on measured attacks against already-public Census records and holds out entire states for testing. Its model JSON and results are in `artifacts-real/`. The richer `scripts/realdata_neural_comparison.py` finds that both one- and three-layer models improve test loss over a plain logistic model on three overlapping state splits; an explicit quadratic logistic model performs better still. The exercise is one public file and one attack, not independent programme validation or a test of LLM proposals. No neural or LLM performance on held-out real institutional release programmes has been established in this package.
