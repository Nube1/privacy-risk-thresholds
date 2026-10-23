# Paper revision and public-file search results — 23 October 2026

The new focused draft is `paper/injector_study/main.tex`: **Measured Scenario Injection for Privacy Audits: Search Coverage, Evidence Integrity, and Independent Confirmation**. It has been compiled in the desktop LaTeX editor. The JPC supplement and long-form online appendix receive the same generated table and a shared methods/results section; their main papers receive concise context paragraphs. Existing committed PDFs predate this revision and should not be mistaken for exported revised manuscripts. Open the source in the editor for its current compiled preview.

## New measured results

This experiment performs record-linkage attacks on a real, already public Census extract, with simulated independent auxiliary-income readings. It does not claim to observe an operational attacker or estimate confidentiality in an undisclosed population.

Thirty state samples, selected before outcomes by encoded state order and minimum source size, share warm starts and use matched evaluator budgets. Each search method selects 32 additional configurations; each is independently confirmed using 300 fresh trials. The primary run executes **1,443,840 attack trials** and **120 neural fits**. The sensitivity run raises the training cap from 600 to 2,000 epochs; it repeats the same seeds and data, so it is not independent replication.

| Search | Mean confirmed high-risk cells | Mean confirmed high-risk settings out of 32 |
|---|---:|---:|
| Random | 13.20 | 22.50 |
| Domain heuristic | 11.97 | 32.00 |
| Scrambled Sobol | 13.83 | 22.03 |
| Neural search | 10.07 | 31.57 |

The primary neural-minus-Sobol difference is **−3.77 cells**, with descriptive paired-state bootstrap interval **[−4.43, −3.07]**. At the larger training cap the mean difference remains −3.77, with interval [−4.40, −3.07]. The targeting-versus-coverage tradeoff survives this sensitivity check. The simple heuristic is a necessary control: it also achieves very high targeting yield.

Four of the original 120 fits have their best epoch at the 600 cap. The 2,000-cap run has zero best-epoch cap hits, but its latest best epoch is 1,999. This is a compute-budget sensitivity check, not proof that every model has converged.

“Cells” are 24 fixed bins of declared configuration choices, not novel attack families. Some choices can produce equivalent partitions. The bootstrap intervals describe the selected empirical state blocks, not national probability-sample inference. Exact lower-bound confirmation uses alpha .05/128 across the four arms within each state; it is not a simultaneous study-wide bound across 30 states and is not a safe-release certificate.

## Reproduce

```powershell
python scripts/public_injector_search.py --states 30
python scripts/public_injector_search.py --states 30 --epochs 2000 --output artifacts-public-search-2000
python scripts/report_public_search.py
pytest
ruff check src tests scripts
```

Install the project with `python -m pip install -e ".[dev]"` first. For stable CPU behavior, set `OPENBLAS_NUM_THREADS=1`. In this workspace the project uses the isolated environment at `../privacy_injector_study/privacy-risk-thresholds/.venv`; set `PYTHONPATH` to this checkout's `src` when using that environment because its editable install points to the original study directory. Standard fresh installation in this checkout does not need that override.

`artifacts-public-search/` stores the design, public-data snapshot hash, state-level measurements and primary results. `artifacts-public-search-2000/` stores the sensitivity run. Raw person-level source records are not copied into the repository by these scripts. `paper_revision_manifest.json` hashes the revised code and evidence; the earlier `study_manifest.json` is a historical snapshot of the first study.

## Genuine LLM arm: ready to collect, not reported as completed

The current session has no OpenAI, Anthropic or Azure model credential. The experiment does not substitute hand-written or simulated proposals for genuine calls. Exact prospective prompts are saved under `artifacts-public-search/prompts/`; they contain a state code and allowed action tuples, with no individual source records. `manifest.template.json` contains their hashes and placeholder response paths.

For each state, send the exact saved prompt to the chosen model and save the original JSON response. Record model/provider version, request ID, temperature, token cap, timeout, rejection/retry policy, usage and cost. The response must contain 32 distinct executable catalogue actions, in objects with exactly `proposal_id`, `scenario`, `features`. Features are `puma_width`, `education_width`, `minimum_cell`, `covariate_sd`. Counts, risk claims and executable code are not accepted. Do not select or regenerate responses after inspecting confirmation outcomes.

Fill a copy of the manifest with genuine model versions and response paths, then run:

```powershell
python scripts/public_injector_search.py --states 30 --llm-replays PATH/manifest.json --output artifacts-public-search-llm
```

This adds the replay arm and adjusts the within-state confirmation multiplicity correction for five arms. It is a separate expanded comparison, not an amendment to the completed four-arm result. Prompt and response hashes bind the imported data but do not authenticate the model provider. This benchmark validates and executes actions directly; it does not pretend that a human review or service-controlled ledger occurred.

## What the paper can say now

The method can be studied as a constrained injector architecture, with measured labels, explicit search budgets and fresh confirmation. Neural targeting concentrates on high-risk settings but does not beat a simple space-filling method on this coverage endpoint. Evidence integrity is testable, and optional local provenance does not enforce an institutional audit boundary. LLM superiority, real confidential-register performance and deployment readiness remain unestablished.
