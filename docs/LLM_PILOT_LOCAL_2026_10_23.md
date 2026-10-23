# Local open-weight LLM pilot: two states, measured

This is the first genuine language-model comparison in the public-file study.
The hosted pilot (`docs/LLM_PILOT_STATUS_2026_10_23.md`) stopped at a provider
error. This run used an open-weight model on the author's own machine, so there
is no account, key or charge, and nothing left the machine.

## Set-up

| Item | Value |
| --- | --- |
| Model | `qwen3:4b` through Ollama 0.40.9, digest `359d7dd4bcda…`, Q4_K_M, 4.0B parameters |
| Hardware | NVIDIA RTX 5050 laptop GPU, 8 GB; the model ran fully on the GPU |
| Prompt | The saved per-state prompt from `artifacts-public-search/prompts/`, byte for byte; the search script checks its hash |
| Decoding | Thinking off, temperature 0.3, one seed per state and attempt, output constrained by Ollama structured output to the JSON shape the prompt asks for (`llm_pilot_local.response_schema()`) |
| Attempts | Up to three regenerations per state; both states were valid at the first attempt |
| Outcomes | Measured by the same executable linkage attacker as the other arms; the model supplied only tuples |
| Cost | Nil; about 25 and 34 seconds of generation per state |

A first run without the decoding constraint is kept in
`artifacts-llm-pilot-local-A-unconstrained/`. The model wrote planning prose,
then an array in the wrong shape, and hit the 8,192-token cap. That attempt was
rejected and is not used; the constraint fixes the shape of the output, not the
choice of tuples, which `replay_selection` still checks for count, distinctness
and membership in the catalogue.

## Result

Two states, 119,552 attack trials, five arms, confirmation at tolerance 0.2 with
alpha 0.05 split over 5 x 32 candidates.

| Arm | State 0: cells, confirmed/32 | State 1: cells, confirmed/32 | Mean cells |
| --- | --- | --- | --- |
| Random | 14, 28 | 11, 23 | 12.5 |
| Heuristic | 12, 32 | 12, 32 | 12.0 |
| Sobol | 14, 27 | 12, 18 | 13.0 |
| Neural search | 12, 32 | 9, 32 | 10.5 |
| Qwen3-4B replay | 7, 29 | 7, 27 | 7.0 |

Paired against Sobol, the model covers 6 fewer cells (bootstrap interval -7 to -5
over two states, which is a description of these two states, not an inference).

## What the model did

It enumerated. In both states the 32 proposals are the first 32 tuples of the
catalogue in catalogue order: `puma_width` 1 throughout, `covariate_sd` cycling
through its five levels. Those tuples are high-risk, so the confirmed yield is
high (29 and 27 of 32), but they sit in 7 scenario cells out of a possible 48.
The model read the catalogue as a list to copy, not a space to explore.

## What this settles and what it does not

It settles that a 4B open-weight model, given this prompt and this catalogue,
does not add coverage over quasi-random or neural search on these two states.
It does not settle whether a larger model, a prompt that names the coverage
goal, or a few-shot example would do better; the prompt was held fixed so that
the comparison with the baseline run stays exact. The 30-state run with the
same model is the next step, and the prompt-design question is a separate
experiment that must be declared before it is run.

## Reproduce

```bash
ollama pull qwen3:4b
python scripts/collect_llm_pilot_local.py --states 0 1
python scripts/public_injector_search.py --states 2 \
    --output artifacts-public-search-llm-local \
    --llm-replays artifacts-llm-pilot-local/manifest.json
```
