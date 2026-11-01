# Reproduction and validation

The supplied project was unpacked into a separate study directory. No GitHub push or publication was performed. The supplied patch was already represented in the ZIP and was not applied twice.

## Environment

Python 3.12.14 on Windows. Installation of `.[dev]` completed in `.venv`. `pip check` reports no broken requirements. `requirements-experiment-win-py312.txt` records observed versions; it is not a hashed portable lockfile. The original package still declares version 0.7.0.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MPLCONFIGDIR = "$PWD\baseline\mplconfig"
.\.venv\Scripts\python.exe scripts/injector_experiment.py --seeds 10 --epochs 1200
.\.venv\Scripts\python.exe scripts/linkage_injector_experiment.py --seeds 5
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m pytest --basetemp baseline/pytest-reproduce
```

Create `baseline` first if starting from source files without the delivered logs. On machines where a user's global Git ignore file is inaccessible, the tests that create local Git repositories need a process-local Git configuration with a readable empty exclude file. The verified run used `GIT_CONFIG_GLOBAL` pointing to `baseline/test.gitconfig`; it did not change the user's real Git settings. Regenerate its absolute exclude-file path if moving this directory. No tests require access to the private GitHub repository.

## Verification evidence

- `baseline/integrity-before.txt`: all four new integrity probes failed on the supplied implementation.
- `baseline/integrity-after.txt`: the initial 21 targeted integrity/SENTRY-R/provenance tests passed after changes.
- `baseline/full-tests-verified.txt`: **178 passed**, with **91.57% source coverage**, above the required 85%. This is the verified result for the supplied archive plus these changes; the earlier pasted claim of 194 tests was not reproduced.
- Ruff over `src`, `tests` and `scripts`: all checks passed.
- An additional evaluator sanity test checks reproducibility and distinguishes near-perfect linkage from weak auxiliary information on synthetic populations.
- Probability-law pilot: ten seeds, 120 fits, zero best-epoch cap hits. `artifacts-injector/design.json`, `results.json` and `pilot.png` contain the design, raw metrics and figure.
- Executable synthetic linkage pilot: five seeds, 30 fits, 1,382,000 independent linkage trials, zero best-epoch cap hits. Evidence rows are saved alongside `artifacts-linkage-injector/results.json`.
- Both studies disclose negative results and omit a genuine LLM arm because no provider responses were supplied.

Earlier full-suite attempts encountered restricted temporary directories and an inaccessible global Git ignore file. A temporary test configuration initially used `NUL`, which Git refused as an exclude file; using a real empty file resolved that test setup issue. Those failed logs are not the authoritative final result.

Model fits share measured evidence across depths within each arm. Statistical summaries use paired seeds and exploratory t intervals. Timing is machine-dependent, and a fixed seed does not promise bitwise equality across NumPy/BLAS versions. The exact input provenance and output hashes are recorded in `study_manifest.json`.

## What is still untested or unimplemented

Genuine LLM comparative performance; real-data attack performance; authenticated reviewer/evaluator services; external audit-budget reservation; concurrency-safe durable writes; rollback resistance; semantic contamination detection; formal differential privacy; independent replication. Content-bound measurements prevent substitution only when lineage is retained and provenance is used. A caller who can omit provenance or relabel data remains outside that guarantee.

See `RESEARCH_BRIEF.md` for findings and the hiring story, and `EXPERIMENT_PROTOCOL.md` for the confirmatory design.

## Paper revision, 23 October 2026

The updated checkout adds a public empirical-file search benchmark and genuine-response replay validation. The final revision suite reports **183 passed, 90.58% source coverage**; Ruff passes. Its full log is `baseline/paper-full-tests.txt`. The new primary run has 30 state blocks and 1,443,840 attack trials, plus a larger-cap sensitivity rerun using the same data and random streams. See `docs/PAPER_RESULTS_2026_10_23.md` and `paper_revision_manifest.json`. Earlier counts and hashes above describe the historical study snapshot.

## Run of 2 November 2026

Run on branch `peer-review-revisions` with `PYTHONPATH=src python -m pytest -o addopts="" tests --ignore=tests/test_realdata.py`: 197 passed and 8 failed. The eight failures are `ModuleNotFoundError: wooldridge`, which the real-data tests and the tests that run the full pipeline need and which was not installed in that environment. `tests/test_realdata.py` was not collected for the same reason. Coverage was not measured. The earlier counts above (178, 183 and the unreproduced 194) belong to earlier snapshots.
