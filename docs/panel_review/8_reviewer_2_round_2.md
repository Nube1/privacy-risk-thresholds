**Reviewer 2, round 2** (read-only; I edited no files. Both PDFs are newer than their sources and show no "??".)

**1. My first-round points**

| # | Verdict | Evidence |
|---|---|---|
| R2.1 | resolved | jpc_main:39, 50, 96; refs 151, 161, 165, 167, 170. Tian–Ramdas is SMMR 30(4), as cited. My JASA was wrong. |
| R2.2 | resolved | jpc_main:26, last sentence. No "reviewers must decide" text remains. |
| R2.3 | partly | JPC done: jpc_context:5–7, 22–27, 34, with alphabet s. The long form bis_ifc_full:901–916 still gives no shared-identifier condition and uses k as the alphabet size. |
| R2.4 | partly | jpc_evidence:24–31 runs the gate on the credit register against a frozen family. Still missing: a Census run, the beta–binomial CS comparator, an adaptive-bet comparator, and a reused-target case (jpc_main:129 admits this). |
| R2.5 | resolved | jpc_main:31, 158; supp:82 |
| R2.6 | resolved | jpc_main:122; the network section moved to jpc_network in the supplement |
| R2.7 | resolved | bis_ifc_full:1067–1073 |
| R2.8 | partly | Four contributions (102–146), line 95 fixed, overclaims removed (1696). But the paper is still 69 pages, the case studies are still in (2761–2898), and the credit register starts on p.33. |
| R2.9 | resolved | jpc_main:34–47, 139 |
| R2.10 | partly | References added. zhou2026 (jpc_main:172) still unverifiable. Long-form bibliography still out of order: Tian before HHS (3167–3171), Abowd 2023 before 2022 (2909–2916), ECB after Elliot, Dwork. Duncan et al. 2001 still has no report number (2985–2987). |
| R2.11 | resolved | jpc_main:19–23; supp:19–23 |
| R2.12 | partly | q, n and 𝓕 fixed (jpc_main:62), "at least" in the abstract. New clashes: γ is a stopping time (jpc_main:62) and also the CP level (jpc_context:37–41); K is the success count and also K_v the key (jpc_context:6, 37); U is the population and also the bound (jpc_context:3, 39); α is never defined (first used jpc_context:41). |
| R2.13 | resolved | jpc_evidence:39–50 |
| R2.14 | resolved | jpc_evidence:8–20 |
| R2.15 | resolved | jpc_context:31, 34; supp:9 |
| R2.16 | resolved | jpc_network:5; "trust position" and "rigorous separation" are gone from the JPC files |
| R2.17 | resolved | bis_ifc_full:60; jpc_main:158 |

**2. Other reviewers' major points not resolved, or resolved wrongly**

- **R3.2 (shared identifier), long form.** Neither the threat model (1873–1893) nor prop:nopool (901–916) states that one pseudonym persists across vintages. "pseudonym" does not appear in bis_ifc_full.tex.
- **R4.5 (length and order).** Not done. The reader's guide (148–163) works around the order but does not fix it.
- **R4.8 / DC.7, resolved wrongly.** Lines 1151–1153, 2021–2023 and 2239–2240 say the noisy-count releases "meet the tolerance pooled". But 1989–1991 and the tab:eval caption say those rows measure class uniqueness and cannot be compared with the loan tolerance.
- All other majors check out, including the repository items. Every manifest hash matches, .github/workflows exists, and check_paper_numbers.py:65 uses word boundaries.

**3. New problems**

1. **Grids merged (jpc_evidence:52).** It treats levels 1–100 as one grid "on that separate draw". tab:kchoice says the coarse and fine grids are separate draws. The 0.077 comes from the coarse grid; the k=30 and k=40 figures come from the fine grid.
2. **Low power read as unsafe (jpc_evidence:52).** "Under the gate… k=40 certifies in only 6 of 20" reads low power as evidence against k=40. The true pooled risk is about 0.095, below 0.12, and the frozen family also passes only 9 of 20.
3. **Adaptive registration is never tested.** Every single-release coordinate certified in every seed (Table credit-audit), so every pooled coordinate was registered. The claim at jpc_evidence:31 that the frozen family "does not" allow later additions is not tested, because this family was known in advance.
4. **"About half and half" does not add up** (jpc_evidence:50; bis 1959). The two increments are 0.053 and 0.057, which sum to 0.110 of the 0.130 gap. The remaining 0.020 is interaction; say so.
5. **Decision record (tab:record, bis 2199–2203).**
   - It marks k=40 "Pass" after a grid scan, without the caveat on selection across levels that the JPC gives.
   - It says "vintage 6", but the attack and the kept counts use the first vintage (tab:kchoice caption; jpc_evidence:25).
6. **Wrong bound compared (bis 922–923, 1466).** The two-release pooled figure (0.4114) is compared with the four-release bound (0.2530). The right bound is the sum of the first two marginals, 0.148.
7. **new_results.md §3 is itself wrong.** It says the admissible policy is still ε=5. But 0.004 and a room of 0.027 fit ε=1 (0.12 − 0.004 = √(0.027/2) ≈ 0.116). bis 1152 uses ε=1 and is correct; fix the note.
8. **Stale short version.** publication_versions/bis_ifc.tex:17, 61, 71 still has "trust position" and 0.3436/0.2780. Its PDF is byte-for-byte the size of the full paper's PDF (743,663 bytes), so the source and PDF disagree.
9. **Toy audit counts (jpc_evidence:20).** The second run gives exactly 10 of 2,000 false certifications, the same as the first run. Check whether the null stream reuses the same random draws.
10. **References.**
    - foster2008 (alpha-investing, which controls mFDR) is cited for alpha-spending (jpc_main:39).
    - mahomane2026 (jpc_main:158) gives no way to find it (no URL or DOI).
    - zhou2026 is still unverifiable.
11. **Wrong reason given (jpc_evidence:2).** It says the carried-over results are included because the pooled coordinates "rest on them". The gate does not use them.

**4. House style**

There are no em dashes, no "---", no "not only X but Y" and no comments about reviewers. "Robust risk" is a defined technical term. Phrases to change:

- jpc_evidence:29 "tell a different story"
- jpc_evidence:31 "pays nothing for monitoring"
- bis 883 "Pooling defeats the marginals"; 934 "defeats a syntactic rule"; 1698 "defeats the marginal bound"
- bis 890 "The deeper problem"
- bis 933 "actually use"
- bis 1150 "has none to spare"
- bis 1946 "stand level"
- bis 1970 "gets it cheaply"
- bis 2039 "buys less than"
- bis 2243 "should be stated plainly"
- bis 2868 "A rigorous determination"

**Recommendations**

JPC: minor revision, up from major. First fix items 1–3 at jpc_evidence:31 and 52, the γ/K/α notation, and the foster2008 and zhou2026 references.

IFC long form: still major revision. The content fixes are in, but it is still 69 pages, the noisy-count admissibility claim contradicts its own caveat, and the shared-pseudonym assumption is never stated.