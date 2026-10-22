**Reviewer 2 report: "Adaptive certification of privacy risk across multidimensional release programmes" (JPC) and "Governing privacy risk across repeated statistical releases" (IFC long form)**

**Summary.** The JPC paper treats each candidate release and attack pair as a "risk coordinate". It certifies a release only when every declared coordinate's betting e-process crosses a weighted Ville threshold, and it proves that the chance of certifying any unsafe coordinate stays at most α, even when coordinates are added after seeing earlier audits. It adds a counterexample showing that pooled-vintage risk is not bounded by marginal risks, a zero-success budget, a cost-weighted allocation and a neural "bet proposer". Its evidence is a Bernoulli toy run plus results carried over from the long-form study. The IFC long form is a 66-page study of scenario-conditional linkage risk, accountable tolerances and composition, using synthetic registers and two public files.

**Recommendations.** JPC: major revision, close to reject as it stands. The mathematics is correct but known, the paper declines to say what is new, and no experiment uses the proposed gate on a disclosure problem. IFC Working Paper: major revision. The content is useful, but it is too long, one inference step is invalid, and it claims more novelty than it shows.

---

**R2.1 MAJOR (jpc_main.tex:62–77, 88; jpc_supplement.tex:22, 29).** Theorem 1 is online familywise error control by alpha-spending over a countable sequence of hypotheses, with each hypothesis tested by a test supermartingale. The weights 6/(π²i²) and the reserve η are textbook alpha-spending. The paper cites none of Foster and Stine (2008, JRSSB, alpha-investing), Tian and Ramdas (2021, JASA, online FWER), Shafer et al. (2011, Stat. Sci., test martingales), Waudby-Smith and Ramdas (2024, JRSSB, betting confidence sequences) or Vovk and Wang (2021, e-values). Without them a reader cannot tell what the theorem adds.
*Fix:* Add a related-work paragraph that states the reduction plainly: "Theorem 1 is the online Bonferroni (alpha-spending) procedure of Tian and Ramdas (2021) applied to betting test supermartingales (Shafer et al. 2011; Waudby-Smith and Ramdas 2024). Our contribution is the choice of hypotheses, the attack-sampling condition under which they are valid, and a pooled-vintage coordinate set." Then either prove something beyond the reduction, or reframe the paper as a methods-and-application paper.

**R2.2 MAJOR (jpc_main.tex:19, 114, 116, 119).** The paper hands its novelty judgement to the referee. Examples: "priority for a distinct methodological contribution requires comparison with adjacent work"; "That work is direct warning against calling e-processes ... novel"; "The claim we offer for review"; "The journal mathematics is"; "peer reviewers must decide whether that synthesis ... constitute[s] sufficient journal novelty." These sentences are the author's job, not mine, and they weaken the paper.
*Fix:* Delete them. Replace the last sentence of the abstract with: "The gate is an application of online alpha-spending to test supermartingales; we show what it requires of attack sampling and what it costs in audit trials for disclosure tolerances." Replace line 119's final clause with: "The contribution is the coordinate definition, Assumption 1 and the audit-cost rule; the error-control mechanism is standard."

**R2.3 MAJOR (jpc_main.tex:31–37; jpc_context.tex:17).** Proposition 1 is an intersection (composition) attack. Ganta, Kasiviswanathan and Smith (2008) is not cited, yet the long form admits "Neither the phenomenon nor its direction is new" (bis_ifc_full.tex:942). The JPC paper instead calls it a "pooled-risk impossibility result" in the "journal mathematics" (line 119). The proof is also underspecified. Release v shows only the v-th coordinate of each record, so the candidate sets from different releases can only be intersected if every release carries a common record identifier or a common sensitive payload. The long form uses a different construction (release i keys on all coordinates except i; bis_ifc_full.tex:934). In addition, q stands for three things: the alphabet size here, the bets q_{i,k} and the hidden widths q_ℓ.
*Fix:* Cite Ganta et al. (2008) and Sweeney (2002). State: "Each release publishes, for every record, a persistent pseudonym and one key coordinate." Use the long form's construction in both drafts. Rename the alphabet size to s.

**R2.4 MAJOR (jpc_evidence.tex:1–58).** The proposed gate is never run on a disclosure problem. Tables 1 and 2 come from the old study and use point risks. The decision illustration at line 33 compares a point estimate with τ = 0.12, which the paper itself says is not certification. Table 3 uses i.i.d. Bernoulli laws, where validity holds by construction. The public Census and loan files (lines 40–43) are used only descriptively.
*Fix:* Add one experiment. Run the registered gate on the credit-register pipeline and the 29,501-record Census file. Register single-vintage coordinates first, then add pooled coordinates after seeing their audits. Report certification decisions, trials used and relationships released for each policy. Compare against (a) Bonferroni Clopper–Pearson on a frozen family, (b) the beta–binomial confidence sequence already in the long form (bis_ifc_full.tex:1063), and (c) a betting e-process with an adaptive bet (aGRAPA or ONS). Also report a case where Assumption 1 fails (reused targets) to show the failure the supplement describes.

**R2.5 MAJOR (jpc_main.tex:37; jpc_context.tex:4; jpc_evidence.tex:2, 20, 40; jpc_supplement.tex:57).** The overlap disclosure is inadequate. Tables 1–2, Figure 2 and all Census and loan numbers are reproduced from the long form, which was presented at the Lisbon BdP/IFC workshop (bis_ifc_full.tex:35). The main text refers to "the full manuscript", "the large working manuscript" and "the base paper" but cites none of them, and the only overlap statement is in the supplement and the README. The JPC paper also does not mention the Lisbon presentation.
*Fix:* Cite the long form as a reference. Add a footnote on page 1: "Tables 1–2, Figure 2 and Section 7.2 reproduce results first reported in Mahomane (2026), presented at the Banco de Portugal/IFC Workshop on Data Security and Privacy, Lisbon, September 2026. Theorem 1, Propositions 2–3 and Table 3 are new." Better still, cut Tables 1–2 to a single cited sentence and use the space for R2.4.

**R2.6 MAJOR (jpc_network.tex:1–35; jpc_context.tex:35–38; jpc_evidence.tex:60–61).** The neural section does not belong in a JPC article. Backpropagation formulas, parameter counts and a Lipschitz bound that "does not bound actual privacy risk" add nothing to the claim. The only link to the gate is that bets must be predictable, and that needs one paragraph. The 320-row toy (log loss 0.278 against 0.344) has no relation to disclosure.
*Fix:* Replace Section 6 with one paragraph: "Any predictor may supply q_{i,k} if it is fixed before X_{i,k} is observed; validity does not depend on its accuracy." Move the network, the SENTRY-R diagram and the toy results to the supplement or drop them.

**R2.7 MAJOR (bis_ifc_full.tex:1086–1089).** The instruction "For q targets ... selected after inspection, replace α by α/q" is invalid when q or the selection depends on the data. This is exactly the problem the JPC gate solves.
*Fix:* Replace with: "For a family of q hypotheses fixed before any outcome is seen, use α/q. If hypotheses are added after inspection, allocate α by a registered spending sequence (Tian and Ramdas 2021), as in the companion article."

**R2.8 MAJOR (bis_ifc_full.tex:98–188, 189, 1695).** Ten numbered contributions is too many for a working paper. Several of them concede prior work (Abowd and Schmutte; Ganta; Ville; Howard). The long form also makes unsupported or overstated claims: "releases at the boundary of its own tolerance about half the time" (line 95) has no source or proof, and "reproduces every result" (line 189) and "Every claim this paper makes ... reappears" (line 1695) overstate what two public files can show.
*Fix:* Cut the list to four contributions (estimand derivation, pooled bound, certification cost, credit-register evaluation). Move the case studies of Netflix, the 2020 US Census, Census 2021 and HIPAA to an annex, and aim for about 35 pages. Reword line 95 to: "A committee that releases whenever the point estimate falls below its tolerance will, for a release whose true risk equals the tolerance, release with probability near one half."

**R2.9 MAJOR (jpc_main.tex:23–38; section order).** The JPC paper is not organised as a journal article. There is no introduction with stated contributions. Related work comes in Section 8, after the results. There is no conclusion. The paper uses eq:net (jpc_context.tex:36) three sections before defining it.
*Fix:* Reorder the paper as: 1 Introduction (motivation, contributions, outline); 2 Related work (current §8 plus R2.1 and R2.10); 3 Setting and estimands; 4 Gate and theory; 5 Experiments; 6 Discussion and limitations (current §9 failure modes); 7 Conclusion.

**R2.10 MINOR (jpc_main.tex:129–136).** The reference list has eight entries, which is thin for a JPC article. Zhou et al. (2026, arXiv:2605.07002) is the "closest comparison" but I could not verify it, so confirm the record. Bates et al. is published in *J. ACM* 68(6), 2021. Fellegi–Sunter is cited for "attacker models" (line 28), which it does not treat; cite Duncan and Lambert (1989) and Elliot and Dale (1999) instead. The draft is also missing Skinner and Shlomo (2008), Hundepool et al. (2012), the privacy-auditing work (Jagielski et al. 2020; Steinke, Nasr and Jagielski 2023) and Kasiviswanathan and Smith (2014, JPC). In the long form, the bibliography is not in alphabetical order (bis_ifc_full.tex:2800–2805) and Duncan et al. (2001) needs a report number.

**R2.11 MINOR (jpc_main.tex:16; jpc_supplement.tex:7).** The affiliation line is incomplete in the JPC draft and missing from the supplement. There is also no corresponding-author email. The long form (bis_ifc_full.tex:40–45) is correct.
*Fix:* Use the long form's affiliation block in both JPC files: "Department of Computing, Mathematical and Statistical Sciences, University of Namibia, Windhoek; School of Pathology, Faculty of Health Sciences, University of the Witwatersrand, Johannesburg; Basis Energy, Victoria, Australia."

**R2.12 MINOR (notation).** The same letter means different things across sections and files:
- k is the draw index, the key and the k of k-anonymity; the supplement also uses it as the scenario index (jpc_supplement.tex:29).
- n and M both denote the family size (jpc_context.tex:20, 24).
- w_i is an error allocation in the JPC paper but a value weight in the long form (bis_ifc_full.tex:970).
- 𝓕_{before(i,k)} (line 47) is not defined.
- The abstract says "exceeds its tolerance", but the null is r ≥ τ; the supplement (line 16) itself warns against mixing strict and weak inequalities.
*Fix:* Define 𝓕_{τ_{i,k}−} as the σ-field just before the draw, and write "at or above its tolerance".

**R2.13 MINOR (jpc_evidence.tex:8–31, 33).** Tables 1 and 2 have unclear entries:
- "Unique share" (0.260) is not defined, although the caption says every entry is a success probability.
- Line 33 cites "a cell-count floor of 50, with 20,389 relationships", but that row is missing from Table 2, which shows four of the seven policies.
- k = 5 has a higher pooled risk (0.224) than no suppression (0.209). This needs an explanation.
*Fix:* Add the floor-50 row, define unique share and explain the k = 5 result.

**R2.14 MINOR (jpc_evidence.tex:46–58).** Monte Carlo standard errors are missing (about 0.0016 at 0.005). The "conditional observed rate" 10/1902 is not the quantity Theorem 1 bounds.
*Fix:* Report 10/2000 against α = 0.05 overall, with binomial SEs.

**R2.15 MINOR (figures).** Figures 1 and 2 have no \label and are never cited in the text (jpc_main.tex:104–107; jpc_evidence.tex:35–38). Label them and cite each one, or remove it.

**R2.16 MINOR (style).** Undefined terms: SENTRY-R (jpc_context.tex:36), LLM (jpc_main.tex:42) and "original trust position" (jpc_main.tex:79; original relative to what?). Puffery and meta-commentary:
- "This is the rigorous separation" (jpc_network.tex:26)
- "makes the audit claim auditable" (jpc_main.tex:52)
- "This source package does not claim journal acceptance" (jpc_supplement.tex:57)
- "Monitoring is not free, and pretending otherwise is worse" and "an honest statement" (bis_ifc_full.tex:1114, 1166)

Delete them. I found no em dashes and no "not only X but Y" constructions in any source file.

**R2.17 MINOR (consistency).** The two titles share no key terms, so readers will not see that the papers are companions. The numbers do match across drafts: 0.079/0.209, 32,581/20,389, 0.4581/0.2530, 0.709/0.381 and 0.134/0.149. The long form's status box (bis_ifc_full.tex:59) mentions "the companion focused journal article" but gives no title. Name it in both directions.

---

Notes for the main session: I made no file edits. The arithmetic in the JPC paper checks out: the 40-trial budget, the Lagrange weights, the supermartingale factor, and the toy JSON matches Table 3. The biggest fixes are R2.1–R2.5: cite online alpha-spending, drop the "reviewers must decide" sentences, cite Ganta, run the gate on the real pipeline, and disclose the overlap in the manuscript itself. I could not verify the Zhou et al. (2026) arXiv entry.