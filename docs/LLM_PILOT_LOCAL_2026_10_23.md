# Local open-weight LLM arm: 20 Census states, measured

This is the first genuine language-model comparison in the public-file study.
The hosted pilot (`docs/LLM_PILOT_STATUS_2026_10_23.md`) stopped at a provider
error. This run used an open-weight model on the author's own machine, so there
is no account, key or charge, and nothing left the machine.

## Set-up

| Item | Value |
| --- | --- |
| Model | `qwen3:4b` through Ollama 0.40.9, digest `359d7dd4bcda…`, Q4_K_M, 4.0B parameters |
| Hardware | NVIDIA RTX 5050 laptop GPU, 8 GB; the model ran fully on the GPU, about 36 s per state |
| Prompt | The saved per-state prompt from `artifacts-public-search/prompts/`, byte for byte; the search script checks its hash |
| Decoding | Thinking off, temperature 0.3, one seed per state and attempt, output constrained by Ollama structured output to the JSON shape the prompt asks for (`llm_pilot_local.response_schema()`) |
| Attempts | Up to three regenerations per state with distinct seeds |
| Outcomes | Measured by the same executable linkage attacker as the other arms; the model supplied only tuples |
| Cost | Nil |

The constraint fixes the shape of the output, not the choice of tuples:
`replay_selection` still checks count, distinctness and catalogue membership.
A first two-state run without the constraint is kept in
`artifacts-llm-pilot-local-A-unconstrained/`: the model wrote planning prose,
then an array in the wrong shape, and hit the 8,192-token cap.

## Validity of the model's lists

Over the 30 states of the baseline run: 20 valid lists within three attempts
(7 at the first attempt, 6 at the second, 7 at the third); 50 of 70 attempts
rejected, 43 for repeated or off-catalogue tuples and 7 for repeated
`proposal_id` values. States 3, 5, 6, 10, 14, 21, 22, 24, 25 and 28 produced no
valid list. Every attempt is on disk under `artifacts-llm-local-30/attempts/`.

## Result

All five arms rerun on exactly the 20 valid states (`--states-from-manifest`),
1,195,520 attack trials, confirmation at tolerance 0.2 with alpha 0.05 split
over 5 x 32 candidates per state.

| Arm | Confirmed cells (mean) | Confirmed settings of 32 (mean) | Difference from Sobol [paired 95%] |
| --- | --- | --- | --- |
| Random | 13.00 | 21.90 | -0.65 [-1.45, +0.15] |
| Domain heuristic | 12.00 | 32.00 | -1.65 [-2.20, -1.00] |
| Scrambled Sobol | 13.65 | 22.05 | 0 |
| Neural search | 10.10 | 31.55 | -3.55 [-4.35, -2.70] |
| Qwen3-4B proposals | 6.05 | 27.60 | -7.60 [-8.30, -6.80] |

The baseline ordering from the 30-state run holds on this subset.

## What the model did

It enumerated. In 19 of 20 states every tuple the model chose has
`puma_width` 1, the finest geography, and the lists follow the catalogue's own
order. Those tuples are high-risk, so the confirmed yield sits between the two
yield-maximising arms and the two space-filling arms, but they occupy 6 of the
24 scenario cells. The model read the catalogue as a list to copy, not a space
to explore.

## What this settles and what it does not

It settles that a 4B open-weight model, under this prompt, adds no coverage
over quasi-random or neural search, and that it fails the distinctness
requirement often. It does not settle what a larger model, a prompt that names
the coverage goal, or a worked example would do; the prompt was held fixed so
that the comparison with the baseline run stays exact. Any such change is a
new experiment and must be declared before it is run.

## Where it is reported

- `paper/public_search_section.tex` (shared by the IFC online appendix and the
  JPC supplement): a paragraph and Table `tab:public-search-llm`.
- `paper/bis_ifc_full.tex`, `paper/publication_versions/jpc_main.tex`: one
  sentence each where the public-file experiment is summarised.
- `paper/injector_study/main.tex`: abstract, results paragraph and table,
  limitations and next experiment.
- Numbers come from `paper/generated/numbers_llm_local.json`, written by
  `scripts/report_llm_local.py`.

## Reproduce

```bash
ollama pull qwen3:4b
python scripts/collect_llm_pilot_local.py --states 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 --output artifacts-llm-local-30
python scripts/public_injector_search.py --states 30 --states-from-manifest \
    --output artifacts-public-search-llm-local-30 \
    --llm-replays artifacts-llm-local-30/manifest.json
python scripts/report_llm_local.py
```

GPU decoding is not bit-reproducible across hardware, so a rerun may yield a
different set of valid states; the recorded responses are the evidence.
