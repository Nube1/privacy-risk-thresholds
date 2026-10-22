# Two publication drafts

These are editable author manuscripts, not publisher-issued documents.

| File | Intended route | Scope |
| --- | --- | --- |
| `../bis_ifc_full.tex` and `bis_ifc.pdf` | Long-form IFC Working Paper consideration | *Governing privacy risk across repeated statistical releases*: full study, credit register evaluation, decision record, security of the audit process, derivations |
| `../bis_ifc_online_appendix.tex` | Online appendix to the long form | Case studies, the neural proposal model and SENTRY-R |
| `jpc_main.tex` and `jpc_main.pdf` | Journal of Privacy and Confidentiality research draft | *Adaptive certification of attack-conditional disclosure risk across release programmes*: threat model, registered gate, audit length, release chains, the gate run on the credit register, and attacks on the audit process |
| `jpc_supplement.tex` and `jpc_supplement.pdf` | Separate journal supplementary draft | Assumptions, conditional proof, allocation, methods, reproduction and overlap disclosure |

Build locally from the repository root:

```bash
cd paper/publication_versions
latexmk -pdf -interaction=nonstopmode -halt-on-error jpc_main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error jpc_supplement.tex
cd ..
latexmk -pdf -interaction=nonstopmode -halt-on-error bis_ifc_full.tex
cp bis_ifc_full.pdf publication_versions/bis_ifc.pdf
```

The old `bis_ifc.tex` short author note is in `archive/` for provenance only.
It is not the full BIS/IFC manuscript, its PDF is superseded by `bis_ifc.pdf`, and
its SENTRY-R benchmark figures come from an undertrained baseline that the current
benchmark replaces. Do not submit it or quote from it. The journal drafts use readable temporary author formatting. Before submission
apply the JPC `jpc.cls` stylesheet, reconfirm bibliographic records and
recheck its current submission instructions. A separate supplement is part of
the journal's stated format. The BIS/IFC paper must use the chosen event's
template and word limit; the 2026 Basel conference's 5,000-word rule is not a
universal IFC rule. The 66-page source is retained as a complete working
manuscript. For an IFC Working Paper, consult its Secretariat and criteria.

Regenerate the article's evidence from the repository root:

```bash
python scripts/adaptive_audit_comparison.py --output paper/publication_versions/adaptive_audit_toy.json
python scripts/adaptive_audit_chain.py          # chain_audit.json, table_chain_audit.tex
python scripts/programme_robustness.py          # artifacts-robustness/robustness.json
python scripts/robustness_table.py              # paper/generated/table_robustness.tex and two more
python scripts/audit_process_attacks.py         # audit_process.json, table_audit_process.tex
python scripts/estimand_check.py                # artifacts-robustness/estimand_check.json
```

None of these outputs is a confidential-data measurement. The earlier two-coordinate
credit audit and the old short note are kept in `archive/` for provenance only.
Check the article's quoted figures with

```bash
python scripts/check_paper_numbers.py --artifacts artifacts-paper --mode quoted \
  --paper paper/publication_versions/jpc_main.tex \
  --source paper/publication_versions/adaptive_audit_toy.json \
           paper/publication_versions/chain_audit.json \
           paper/publication_versions/audit_process.json \
           artifacts-robustness/robustness.json artifacts-paper/results.json \
           artifacts-sentry/benchmark.json \
  --allow-file paper/publication_versions/quoted_allow.txt
```

The quoted check covers decimals with three or more places. Integers and
two-place figures were checked by hand against the same files.

The theorem is online alpha-spending applied to test supermartingales, and the
article says so. The carried-over empirical findings come from the long-form study,
and the article's first-page footnote names them. The adaptive audit has software
verification and a synthetic-register run, and no independent central-bank
validation. The cover letter should repeat the overlap statement.

An example of the opt-in registered audit interface:

```python
from privacy_risk.adaptive_audit import AdaptiveAudit, zero_success_budget

audit = AdaptiveAudit(alpha=0.05)
pooled = audit.register("release-2026-pooled-attack", tolerance=0.12, weight=0.5)
trials_if_all_fail_with_zero_bet = zero_success_budget(0.12, 0.05, 0.5)  # 29
for observed_attack_success in attack_outcomes_from_fresh_targets:
    pooled.observe(int(observed_attack_success), alternative=0.03)
    if pooled.certified:
        break
```

`attack_outcomes_from_fresh_targets` is a caller-provided stream from the
declared sampling protocol, not a shipped benchmark. The budget of 29 assumes a bet
of zero on every draw; with the bet of 0.03 used here the loop needs more draws
when every attack fails. A coordinate stops accepting outcomes once it certifies. The alternative must be
chosen before the corresponding outcome and may be updated from past results.
All required coordinates must be registered and pass `audit.eligible(names)`.
Registering only a favorable subset cannot certify the full release programme.
