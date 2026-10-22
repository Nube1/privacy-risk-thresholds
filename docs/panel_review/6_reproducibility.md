**Verdict.** The code reproduces the paper: base runs are deterministic, the toy audit JSON regenerates byte for byte, the SENTRY-R benchmark table and walkthrough rerun exactly, every hand-typed number I checked in the JPC drafts matches the stored run, and main.tex, bis_ifc_full.tex and both JPC drafts build with no undefined references. The repository breaks its own promises in three places: the manifest does not match two shipped tables, the README describes CI and automatic table refresh that do not exist, and the number checker is weaker than documented.

**Findings**

1. **MAJOR. artifacts-paper/manifest.json does not match the files it hashes.** I recomputed SHA-256 for all 22 listed files. Two fail: `paper/table_anytime.tex` (manifest 925dc634…, actual dda0398e…) and `paper/table_tolerance.tex` (manifest 9b0d8197…, actual 4b4f3924…). Both files have mtime 2026-09-29 00:22, and the manifest has mtime 2026-09-05. Rerunning `scripts/rebuild_evidence.py` on a copy produces these two tables byte-identical to the shipped ones, so the tables are current and the manifest is stale. Fix: run `python scripts/rebuild_evidence.py --artifacts artifacts-paper` and commit the new manifest. Add a test that recomputes the manifest hashes.

2. **MAJOR. The README promises CI and a "refresh evidence" workflow that do not exist (README.md:120-123, :159-160).** There is no `.github/`. No code copies `<output>/paper/*.tex` into `paper/generated/`. The only script that writes there is `scripts/sentry_r_benchmark.py`, and my runs left `paper/generated/` untouched. So README.md:18 ("written by a pipeline run, so a table cannot disagree") is false: the tables are copied by hand. Today they agree: all 12 shared tables are byte-identical between artifacts-paper/paper and paper/generated, and table_sentry_benchmark.tex exists only in paper/generated and matches a fresh rerun. Fix: add `.github/workflows/ci.yml` (ruff, pytest, base run, number check) and a refresh workflow. Or delete those claims and add a `--publish paper/generated` step to `rebuild_evidence.py`.

3. **MAJOR. `scripts/check_paper_numbers.py` does not check what README.md:139-141 says it checks.**
   - (a) It only tests whether each numbers.json value appears somewhere as a substring in the source. It never checks a number the manuscript quotes. I changed 0.81 to 0.819 and 0.4581 to 0.45812 in a copy of main.tex, and it still printed "All 229 run values quoted in the manuscript match the run", exit 0.
   - (b) It is useless for the JPC drafts. jpc_main exits 1 with 159 of 229 "missing", jpc_supplement with 197 and jpc_evidence with 162, because those drafts quote a subset by design.
   - Fix: match with numeric word boundaries (`(?<![\d.])value(?![\d])`). Add a `--subset` mode that pulls every decimal literal from the manuscript and fails when it is not in numbers.json or a declared allow-list. Point numbers.json keys at the JPC tables.

4. **MINOR. The paper-to-code map in README.md has stale equation and section numbers.** I rebuilt main.tex with `latexmk -g`, and the labels are identical to the shipped main.aux.
   - :171 "Governance rule Eq. (26)": eq:var-decomp is (26). The rule is eq:rule (5) or eq:admiss (29).
   - :172 "Eqs. (29)--(32)": util/attain/frontier/opt are (32)-(35).
   - :173 "Eqs. (33)--(35)": boole/joint are (36)-(37). Props. 9-10 is correct.
   - :174 "Prop. 12, Eqs. (36)--(38)": the DP result is Prop. 14, Eqs. (46)-(47).
   - :175 "Falsification, Section 11": it is Section 16.
   - :176 "Section 10.1": credit is 12.1.
   - :177 and :81 "Section 11": the evaluation is Section 13.
   - :178 "Section 9": sequential is Section 8.
   - :181 "Real data, Section 10": it is Section 11.
   - These are correct: Eq. (3), Eqs. (9)-(11), (12)-(19), Prop. 3 / (20)-(23), Props. 6 and 11, Section 10.7, Appendix D (app:neural = D).
   - Fix: generate the table from main.aux, or cite `\label` names.

5. **MINOR. docs/TOPS.md:49 and :53 cite the wrong sections.** "Sections 9 and 11 carry that argument" should be Sections 7, 10.6 and 13. "Sections 11.7 and 13" should be Section 13.7: there is no 11.7, because Section 11 has three subsections.

6. **MINOR. `make_acmart.py --supplement` leaves dangling cross-references.** Both modes run and write paper/tops/main.tex and supplement.tex. In supplement mode, the main file still holds 18 `\ref{app:proofs|derivations|sim}` plus `\ref{app:neural}` and `\ref{app:sentry}`, and there is no `xr`. TOPS.md tells the author to check this by hand. acmart.cls is not installed here, so I could not compile the TOPS output. Fix: have the script add `\usepackage{xr-hyper}\externaldocument{supplement}`, or rewrite those refs to "Supplement, Section X".

7. **MINOR. The pipeline is not byte-reproducible for figures, so manifests differ between identical runs.** Two base runs gave identical results.json, decision.json and tables. All five figure PDFs and manifest.json differed, and only in `/CreationDate`: pdftotext output is identical. `rebuild_evidence.py` shows the same thing. Fix: in reporting.py:63, 93, 115, 144 and 162, use `fig.savefig(path, metadata={"CreationDate": None})`, or honour SOURCE_DATE_EPOCH.

8. **MINOR. results.json is not valid strict JSON.** It holds 52 bare `NaN` tokens (DP policies with zero trials), and a strict parser rejects the file. Fix: at reporting.py:816, map non-finite values to `null` and use `allow_nan=False`.

9. **MINOR. There are two functions named `zero_success_budget` with different third arguments.** `adaptive_audit.zero_success_budget(tolerance, alpha, weight)` and `neural_trust.zero_success_budget(tolerance, alpha, total_tests)`. The example in paper/publication_versions/README.md:50 runs: `needed_if_all_fail` = 29, but with `alternative=0.03` the loop certifies at draw 38, so the variable name misleads. Fix: rename one function, and note that 29 assumes a zero bet.

10. **MINOR. README.md:154 says "Python 3.10--3.12", but pyproject.toml:9 has `>=3.10` with no upper bound.** Only 3.11 was tested here. Fix: state the tested versions, or add `<3.13`.

11. **MINOR. `privacy-risk-neural-demo --dimension 1` prints a raw traceback** (`ValueError: dimension must be >= 2`, neural_trust.py:167) instead of an argparse error. Fix: `parser.error(...)` in `main`.

12. **MINOR. jpc_supplement.tex:37 says "Four harms are evaluated beyond record identification" but names three.** Fix: name the fourth (reconstruction from totals) or say three.

**Checks that passed**
- JPC numbers: all values in jpc_evidence.tex match results.json and numbers.json after rounding:
  - the release marginals 0.0768 to 0.0168, 0.2530, 0.4581 and 0.260;
  - the credit table rows (32,581/32,513/32,351/31,615, single, pooled and precision), the cell-count floor of 50 and 20,389 records;
  - the Census values (86.3%, 0.0615, 0.2400/0.2588 at grid[24] k=10, 0.98, marginals 0.138/0.030/0.118/0.096, 0.709, 0.381, 25 settings);
  - the loans values 0.134 and 0.149;
  - the SENTRY-R values 0.3436/0.2780 and 0.0970/0.0817 with 320 and 100 rows, confirmed by a fresh benchmark rerun;
  - the toy audit values 0.9560/0.9990/98, 0.0050/0.0055/75, and 1,902 registrations with 10 crossings (0.00526);
  - the supplement's derived values 1/(αw) = 160, 0.00625 and 40 trials.
- Toy JSON: `adaptive_audit_comparison.py` output is byte-identical to the shipped `adaptive_audit_toy.json`.
- Checker on the full manuscripts: main.tex and bis_ifc_full.tex exit 0.
- Figures: paper/figures match artifacts-paper/figures in content; they differ only in PDF metadata.
- CLIs: `privacy-risk-neural-demo --dimension 4 --seed 1` works and is deterministic. `sentry-r --help` works, and the full SENTRY_R.md walkthrough runs end to end.
- Base pipeline: exits 0 in about 17 s.
- Code: no swallowed exceptions in src or scripts. Every RNG is seeded from config or the CLI, and the assessment date is pinned in config.

Scratch outputs are in /tmp/claude-0/prof6/: the repo copy, artifacts, artifacts2, ap_rebuilt, toy.json, bench.json and n.json.