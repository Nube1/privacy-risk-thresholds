# Confirmatory protocol: does LLM injection add value?

Status: proposed follow-on experiment, not completed or externally preregistered. The synthetic baseline study is complete; the genuine LLM, real-attack and service-enforcement experiments below are pending.

## Hypothesis and estimand

Primary question: at a fixed number of evaluator trials, does a constrained LLM proposer discover more independently confirmed privacy failure cases than random search and an informed hand-written heuristic?

Primary endpoint: number of distinct attack-family-by-release failures independently confirmed per 10,000 development attack trials. Define distinctness by a frozen family taxonomy plus release identifiers, not prose similarity. Primary contrast: LLM minus the strongest non-LLM baseline chosen using development programmes only.

Secondary endpoints: held-out expected log loss, recall of risky release settings, dangerous underprediction, scenario validity/duplicate rate, confirmed novel families, wall time, total tokens, evaluator cost and cost per confirmed failure. Report token/provider costs separately from evaluator trials; equal proposal counts do not equal total cost. Use at least 30 independent programme/seed blocks for the main study and estimate a larger required sample from the pilot variance before locking the protocol. Confidence intervals across seeds alone do not imply generalization to institutions.

## Controlled arms

1. No augmentation: fixed historical development evidence.
2. Uniform random scenarios: exactly the same new evaluator budget as other augmentation arms.
3. Informed heuristic search: include the documented domain-informed stress heuristic.
4. Strong numerical search: uncertainty sampling or Bayesian optimization over the same feature domain.
5. Constrained LLM: fixed prompts, schema, model version and proposal budget; no private validation/audit outcomes in context.
6. Deliberately leaky evaluation: sandbox-only negative control showing the effect of audit feedback. Never mix this arm into a reported certificate.

Cross the augmentation arms with logistic, shallow neural and a strong tabular baseline. Fix tuning budget and stopping rules using separate development programmes. The LLM is an injector of scenarios; measured outcomes alone provide labels. Include the heuristic baseline because synthetic stress generators can appear intelligent merely by knowing the simulator.

## Real evaluator and splits

Implement a restricted scenario DSL with attack-family ID, auxiliary-data source, release mechanism, linkage keys, pooling/window choice and fixed evaluator version. Prose can explain a scenario but must not be interpreted as executable authority. Restrict the initial real experiment to record linkage on a public, licensed dataset; document data origin, checksum and redistribution terms. A subsequent membership-inference experiment needs independently sampled member and nonmember populations, a target model, shadow/attack-model training partitions and declared access to outputs.

Split by population/programme and vintage, not just record ID. Keep development fitting, early stopping, proposal evaluation, confirmatory test and final audit roles distinct. An untouched attack-family test assesses generalization; changing the declared attack family requires a fresh certificate and budget. Count evaluator samples at independent units or use cluster-aware methods. An LLM cannot certify that its own attack is correctly measured.

Reserve audit budget transactionally before independent counts are generated; record attempted/aborted audits and never refund based on observed results. Keep release-service credentials and audit outcomes unavailable to the proposer. Refuse repeated evidence IDs across the programme registry and attest counts with an evaluator-owned signature. Anchor chain heads outside operator control. Test concurrency, crashes after reservation and result withholding; local HMACs alone do not solve these problems.

## Genuine LLM replay contract

The implemented replay path avoids provider-specific credentials in the repository. Supply `--llm-replays replays/manifest.json`; each seed must have one response array containing exactly 32 proposals with precisely `proposal_id`, `scenario`, and `features`.

Manifest example (placeholders, not an experimental record):

```json
{
  "101": {
    "model": "provider-and-exact-model-version",
    "prompt_sha256": "sha256-of-the-exact-prompt-sent",
    "response_file": "seed101.json"
  }
}
```

Use `propose_from_llm` with `SCHEMA` and `QUESTION` from `scripts/injector_experiment.py` to form the exact prompt. Capture the prompt and original response at the provider boundary, provider request ID, temperature, sampling seed where supported, generation-token cap, timeout, date, usage and billed cost. The replay runner verifies the prompt hash and records response hashes. Model names remain caller assertions; this is replay integrity, not provider attestation. Do not regenerate or hand-select responses after inspecting test outcomes. Count rejected generations and their cost; specify retry policy before collection. Freeze a manifest covering every seed; incomplete manifests are errors.

The current pilot imports validated replay features directly into its synthetic evaluator and simulates their measurements. It does not exercise real human review, the complete provenance chain, or a real attack executor. Those must be included in the confirmatory integration run rather than inferred from the pilot.

## Analysis and publication gates

- Freeze the primary contrast and programme units before final test outcomes are available. Use paired programme-level intervals and a multiplicity plan for secondary comparisons. Report all arms and seeds, including invalid generations and no-improvement cases.
- Run false-certification stress tests at the boundary, under repeated attempts, correlated audit trials, hidden audit feedback and incomplete evidence registration. Distinguish statistical control under assumptions from enforcement against an operator.
- Claim LLM benefit only if the predeclared primary contrast improves with its uncertainty interval and total cost disclosed. Otherwise publish the negative finding and protocol engineering result.
- Claim formal privacy only for a separately specified and proved DP mechanism/accountant. Attack-based measurements provide conditional empirical risk evidence.
- Require an independently rerun real-data result, complete source/data/environment manifests and a tested audit-reservation service before calling the system deployment-ready.

## Five-minute interview demo

Show a valid proposal entering the schema; show that generated counts and oversized metadata are refused; show measured-row tampering rejected when recording the training artifact; explain the conservative validation-copy check; show the paired synthetic result and that random can beat an informed heuristic; then show why 20 nominal 5% audit retries can certify an unsafe case about half the time. Finish with the concrete genuine-LLM and real-evaluator work still pending. Use `VALIDATION.md` for current commands and test results.
