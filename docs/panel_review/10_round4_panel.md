**Round 4 panel: six referees on the JPC article and the IFC long form**

Six referees reviewed the round-3 revisions, each from one field: sequential statistics, statistical disclosure control, differential privacy, security engineering, central banking, and methodology and reproducibility. Each was told to be hard on the papers. This file records what they found, most severe first, and what was done. "Done" means the change is in the manuscripts and the evidence was rerun; "Open" means it is stated as a limit or further work.

## Verdicts before the round-4 changes

| Referee | Article | Long form |
| --- | --- | --- |
| Sequential statistics | Major revision: theory sound, comparisons and novelty claims not yet | Minor revision |
| Disclosure control | Major revision: floors are properties of the generator | Major revision |
| Differential privacy | Major revision: declared attack not optimal for the declared knowledge | Major revision |
| Security engineering | Major revision of the security section | Major revision of the security section |
| Central banking | Minor to major: theory stands, application needs a rebuilt register | Major revision and a large cut |
| Methodology | Major revision, close to reject | Major revision |

## Findings that changed the results

1. **The loan panel was defective (disclosure control, central banking, methodology).** Every loan was forced to be live in month 0, a third of loans vanished in month 1 and came back in month 2, and no loan entered after month 0. That gave the history adversary a signal no real register has, and made the differencing result an identity. *Done:* the generator now has originations during the panel, a monthly exit hazard of 0.04 with no return, and a stable stock of about 27,000. All credit results were recomputed.
2. **The declared attack did not use all the declared knowledge (differential privacy).** An adversary who knows the target's history also knows when it is absent, but the attack penalised only candidates missing where the target was present. With absence used as well, risk at the old floor of 40 rose from 0.105 to 0.179. *Done:* the attack now requires the candidate's pattern of appearances to match the target's in both directions.
3. **The comparator that registered every coordinate at the start was not stopped at its first failure (sequential statistics, methodology).** This inflated its attack count and produced the claim "under a third of the attacks". *Done:* fixed; the claim is gone.
4. **The alpha split over adversaries was not needed (sequential statistics, differential privacy, disclosure control).** A release that must pass every declared attack is an intersection-union test. *Done:* the article's main design is a release chain at the full level. It also states that the alpha/3 split is the price of choosing the adversary after reading the audit.
5. **The history adversary is usually an insider who knows exposure exactly (differential privacy, security, central banking, disclosure control).** *Done:* an insider with log-exposure error 0.01 is its own row. No floor up to 300 protects against it.
6. **The utility cost was reported in loans, not credit volume (central banking, disclosure control).** *Done:* exposure share and large-firm loans are reported. The floor of 150 keeps 20 per cent of loans and 4 per cent of exposure.
7. **The audit process had no threat model (security).** An operator who can pick among seeds, run shadow audits or bet after the outcome can certify an unsafe attack. *Done:* two adversaries and five trusted parts are defined, the attacks on the process are measured (Table audit-process), and committed targets, recorded bets and a hash-chained ledger are implemented. Both papers now say that read-only Python fields are not a security control, and the long form says the same of the unkeyed manifest.

## Results after the changes

- **Floors, tolerance 0.12.** Floors are chosen by fixed sequence and confirmed on a second draw.
  - One release: 1.
  - Six releases without history: 1, but only because the generator holds exposure fixed.
  - Six releases with history: 75. Selection passed 50, but confirmation did not.
  - A reading per vintage as well: 150.
  - Insider: none up to 300.
- **Subgroup control.** The floors become 20, 200, and none up to 200.
- **Exposure to one significant figure.** This barely moves the outside adversaries' floors. It halves the insider's success, and it cuts the share of loans that can be relinked across vintages from 1.00 to 0.10.
- **Release chain over the three outside attacks.** It chose 150 in 19 of 20 streams and 100 once, against an estimated true floor of 100, with no unsafe choice. On Bernoulli streams at the estimated risks there was no unsafe choice in 400 runs.
- **Pipeline decision.** It now fails, because the upper bound on the declared four-release programme's pooled risk, 0.464, is above the release tolerance of 0.12. The decision uses the upper bound, not the point estimate.

## Other findings and their status

**Sequential statistics.**
- *Done:*
  - The chain proposition now allows the next member to depend on earlier outcomes, and online fallback, fixed-sequence, Hsu–Berger and gatekeeping work is cited.
  - Proposition 3 is presented as a Wald-identity bound, with the matching lower bound and the Krichevsky–Trofimov regret of the plug-in bet.
  - The looks design is labelled as Bonferroni over looks.
  - The operating-characteristics check was added.
  - "At no cost" was reworded, and the confirmation fallback is stated.
- *Open:* a truncated-Beta mixture in place of the capped bet; exact group-sequential boundaries; a regret-based length bound for the rule actually used.

**Disclosure control.**
- *Done:*
  - Floors at tolerances of 0.05, 0.09, 0.12 and 0.20.
  - The history adversary described as a complete information profile.
  - Rotation treated as a control on linkability, not pseudonymisation.
  - The Census analysis presented as a pipeline demonstration.
  - The research-gap sentence credits Skinner–Shlomo and Rinott–Shlomo.
- *Open:* journalist target laws; per-record risk distributions; exposure-coarsening policies beyond rounding; global recoding with local suppression; persistent noise.

**Differential privacy.**
- *Done:*
  - The hypothesis-testing bound e^ε·κ+δ is given in both papers.
  - The DP section's prior example and its independence claim are corrected.
  - The noisy-count arm is removed.
  - The KL bound is framed as a sensitivity statement with the attack held fixed.
  - The auditing literature is added.
- *Open:* a borrower-level DP release under continual observation, attacked by the same attacks; a proof that the maximum-posterior rule is optimal for each declared information set. The rule is Bayes-optimal under the stated model, but the -30 penalty only approximates exact exclusion.

**Security engineering.**
- *Done:* the security claim is stated against the operator; seed grinding, post-hoc bets and shadow audits are measured; the controls are implemented; complete mediation is stated as a condition; and the texts are honest about what a library cannot enforce.
- *Open:* a registry service with durable per-programme budgets, signing keys, external time stamps and enforced separation of duties; an adversarially chosen target law; reproducible builds with pinned dependencies.

**Central banking.**
- *Done:*
  - Cost is reported in credit volume.
  - The legal basis, protected parties and access route are added to the governance record (Regulation 2533/98 Art. 8 as amended by 951/2009; Regulation 223/2009; GDPR Recital 26 and Art. 4(5)).
  - The loan-level file is described as a stand-in for securitisation disclosures and accredited research access.
  - The conclusion that aggregates or secure access are the realistic products against a history-aware adversary is drawn.
  - An executive summary was added, and an online appendix takes the case studies and neural material.
- *Open:*
  - Programme analysis of monthly aggregate tables.
  - Lender-insider harm to other relationships.
  - Exposure that amortises.
  - Lender concentration and regional specialisation.
  - A main text below about 57 pages.

**Methodology and reproducibility.**
- *Done:*
  - The pipeline is rerun and the manifest matches.
  - The truth draws use seeds disjoint from the audit draws.
  - The estimand check is shipped as a script.
  - The READMEs, CI and the response letter are updated.
  - Orphaned files are moved to archive folders.
- *Open:* the chain audit still uses one register; the checker still covers only decimals with three or more places.
