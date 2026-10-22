**Referee report, Professor 3 (formal privacy, composition, auditing)**

**Verdict.** The sequential gate (Theorem 1) is correct, and the paper says plainly that it uses known tools. But the pooled-release "impossibility result" is a restatement of the known non-composition of syntactic and marginal criteria, and the JPC version cites none of that literature. Recommendation: major revision. Reposition it as an applied methods paper with the composition-attack and privacy-auditing literature engaged, state the threat model formally, and drop the pooled proposition as a contribution.

---

**A. Verified errors**

1. **MAJOR. The JPC version omits prior art the long version acknowledges.** jpc_main.tex:30-37, 119; bibliography 128-137. The companion draft (bis_ifc_full.tex:946-955) concedes that "neither the phenomenon nor its direction is new" and cites Ganta et al. 2008. The JPC submission drops that citation and lists "a pooled-risk impossibility result" as part of "the journal mathematics" (line 119). The proposition says only that a functional of the joint law is not determined by the marginal laws. It is Sweeney's complementary-release and temporal attacks, the sequential-release problem, Ganta-Kasiviswanathan-Smith composition attacks, and in spirit Dinur-Nissim reconstruction. A JPC referee will treat leaving these out as a priority claim. Fix: at line 37 add
   `Neither the phenomenon nor the construction is new: complementary and sequential releases that are each $k$-anonymous can be joined to re-identify \cite{sweeney2002,wang2006,xiao2007}, independently protected releases can be composed into an attack \cite{ganta2008}, and many individually accurate answers permit reconstruction \cite{dinur2003}. We state it only to fix the estimand that the gate must certify.`
   At line 119, replace "coupled with a pooled-risk impossibility result" with "and uses a standard non-composition example (Proposition~\ref{prop:pool}) to motivate separate pooled coordinates". References, all real:
   - Sweeney, L. (2002). k-anonymity: a model for protecting privacy. *Int. J. Uncertainty, Fuzziness and Knowledge-Based Systems* 10(5), 557-570.
   - Wang, K. and Fung, B. C. M. (2006). Anonymizing sequential releases. *Proc. 12th ACM SIGKDD*, 414-423.
   - Xiao, X. and Tao, Y. (2007). m-invariance: towards privacy preserving re-publication of dynamic datasets. *Proc. ACM SIGMOD*, 689-700.
   - Ganta, S. R., Kasiviswanathan, S. P. and Smith, A. (2008). Composition attacks and auxiliary information in data privacy. *Proc. 14th ACM SIGKDD*, 265-273.
   - Dinur, I. and Nissim, K. (2003). Revealing information while preserving privacy. *Proc. 22nd ACM PODS*, 202-210.
   - Kifer, D. and Machanavajjhala, A. (2011). No free lunch in data privacy. *Proc. ACM SIGMOD*, 193-204.

2. **MAJOR. The pooled attack assumes a persistent record identifier that the paper never states.** jpc_main.tex:35 ("preserving the stable target universe across vintages"); bis_ifc_full.tex:2462-2478; src/privacy_risk/composition.py:74-99. The candidate sets are sets of *population indices*, and the code intersects them across releases. That only makes sense if released rows share an identifier across vintages (a stable pseudonym), or can be linked through some other published attribute. Without that, the rows of release v and release w are unrelated objects and "intersect the candidate sets" is undefined. The adversary must also know the target's full vector. Fix: before Proposition 1, add
   `\begin{assumption}[Pooled threat model] Each released row carries a pseudonym that is constant across vintages; the adversary knows the target's full quasi-identifier vector and that the target is in every release.\end{assumption}`
   Then cite it in the proposition. Also report how the Table 1 pooled risk changes if pseudonyms rotate by vintage.

3. **MINOR. The reason given for small delta is wrong.** bis_ifc_full.tex:2539-2542. The bound with delta is exact:
   `\frac{\Prob(H\mid M\in E)}{1-\Prob(H\mid M\in E)}\le\frac{p}{1-p}\Bigl(e^{\varepsilon}+\frac{\delta_{\mathrm{dp}}}{\Prob\{M(x')\in E\}}\Bigr)`
   It is not just "of order" that term. It is unbounded for rare E, so (epsilon, delta)-DP gives no uniform posterior bound. The convention delta < 1/n comes from the release-one-record-at-random mechanism being (0, 1/n)-DP, not from this term. Replace the last sentence with:
   `so no uniform posterior bound holds when $\delta_{\mathrm{dp}}>0$; with probability at least $1-O(\delta_{\mathrm{dp}}/(1-e^{-\varepsilon}))$ over outputs a bound of the form \eqref{eq:dppost} holds with a slightly inflated $\varepsilon$ \citep{kasiviswanathan2014}.`
   The delta = 0 statement (eq:dppost) is correct, and the stated numbers (0.155, 0.232, 0.451, 0.576) check out. The condition that the rest of the database has the same law under H and not-H is the right hypothesis. State which neighbouring relation (replace-one or add/remove) H uses.

4. **MINOR. One sentence overclaims what DP says, then cites the paper that refutes it.** bis_ifc_full.tex:1227-1229. "Formal privacy answers what any adversary could learn from the mechanism across a programme" is contradicted by Kifer-Machanavajjhala, cited in the next sentence: with correlated records the posterior reading fails. Replace with "Formal privacy bounds how far any adversary's odds about one record can move relative to that record's absence, under independence of records".

**B. Concerns (not errors)**

5. **MAJOR. The paper is not positioned against privacy auditing.** jpc_main.tex:111-116. The gate is a one-sided sequential test on an attack's success rate. That is the same primitive DP auditing uses with Clopper-Pearson, but pointed the other way. Auditing lower-bounds leakage (evidence of violation). This paper upper-bounds one attack's success to certify safety, which only ever speaks to the declared attack. JPC readers will expect that asymmetry stated. They will also expect the paper to use the hypothesis-testing form of DP (TPR <= e^epsilon FPR + delta): it gives DP releases such as the noisy-count policy an analytic bound on any identification attack, r <= e^epsilon r_0 + delta, so no audit is needed for them. Add a paragraph and cite:
   - Jagielski, M., Ullman, J. and Oprea, A. (2020). Auditing differentially private machine learning: how private is private SGD? *NeurIPS* 33.
   - Nasr, M., Song, S., Thakurta, A., Papernot, N. and Carlini, N. (2021). Adversary instantiation: lower bounds for differentially private machine learning. *IEEE S&P*, 866-882.
   - Steinke, T., Nasr, M. and Jagielski, M. (2023). Privacy auditing with one (1) training run. *NeurIPS* 36.
   - Wasserman, L. and Zhou, S. (2010). A statistical framework for differential privacy. *JASA* 105(489), 375-389.
   - Kairouz, P., Oh, S. and Viswanath, P. (2017). The composition theorem for differential privacy. *IEEE Trans. Inf. Theory* 63(6), 4037-4049.
   - Dwork, C., Smith, A., Steinke, T. and Vadhan, S. (2017). Exposed! A survey of attacks on private data. *Annu. Rev. Stat. Appl.* 4, 61-84.

   For "identification" as a formal event, also cite Cohen, A. and Nissim, K. (2020), Towards formalizing the GDPR's notion of singling out, *PNAS* 117(15), 8344-8352.

6. **MAJOR. What a JPC referee would demand for novelty.**
   - A formal threat model in Section 1: the adversary's knowledge, the release format, and the success event.
   - One of these two: (a) a result that goes beyond the marginal/joint point, such as a bound on pooled risk from per-release DP parameters, or a tight upper bound under a stated linkage structure; or (b) an empirical comparison with Zhou et al. 2026 and Xu et al. 2024 at equal audit budget.

   The abstract's last sentence (line 19), "priority for a distinct methodological contribution requires comparison with adjacent work", reads as the author conceding the paper has none. Line 116 ("That work is direct warning against...") has the same effect. Rewrite both as specific claims.

7. **MINOR. The title overclaims.** jpc_main.tex:13. "Certification of privacy risk" suggests a guarantee against all attacks. Use "Adaptive certification of attack-conditional disclosure risk across release programmes".

8. **MINOR. The main paper is not self-contained.** Proposition 1 (jpc_main.tex:31) uses "identification probability" before the estimand is defined in jpc_context.tex:2-4. Move Section 2 ahead of it, or define the success event inline. jpc_context.tex:4, jpc_main.tex:37 and jpc_evidence.tex rely on "the full/large working manuscript", which is uncited and unavailable. Cite it with an identifier or restate what is needed. Ville's inequality is used (jpc_main.tex:76) but not cited. Add Ville, J. (1939), *Étude critique de la notion de collectif*, Gauthier-Villars, Paris.

9. **MINOR. Notation collisions.**
   - delta means a decision rule (jpc_context.tex:2), a TV distance (jpc_network.tex:31) and a model deviation (jpc_supplement.tex:11).
   - epsilon means a clipping constant (jpc_network.tex:26) and the DP parameter (jpc_evidence.tex:33).
   - q means the bet q_{i,k}, the integer in the Prop 1 proof (jpc_main.tex:35) and the hidden widths q_l.
   - i, j index both coordinates and training rows/outputs (jpc_context.tex:36; jpc_network.tex).

   Fixes: use $\Delta_{\rm TV}$ and $\kappa_i$ for the deviations, $\eta$ for the clip, $s$ for the integer in the proof, and $h_\ell$ for widths.

10. **MINOR. The code drops delta.** src/privacy_risk/dp.py:14-21 implements only the delta = 0 bound, and `basic_composition` (lines 29-36) does not check deltas for sign or length. Nothing in the paper depends on this, but add a guard.

**C. Bibliography check (JPC)**

All eight entries are real, and the authors, years and venues are plausible.
- **zhou (verified on arXiv):** "Adaptive auditing of AI systems with anytime-valid guarantees", Zhou, Vossler, Sivaraman, Mai, Feng, arXiv:2605.07002, submitted 7 May 2026. It uses simultaneous e-processes as the paper describes. The citation is correct.
- **xu (verified):** accepted at NeurIPS 2024. Cite the proceedings rather than the arXiv number.
- **bates (verified on arXiv):** the published version is *Journal of the ACM* 68(6), 2021, article 43. Cite that.
- **howard, savi, clopper, fellegi, dwork:** volume and page numbers are right. Add issue numbers for consistency: 49(2), 38(4), 26(4), 64(328), 9(3-4).

No invented or misattributed entries found. The problem is omission (findings 1 and 5), not fabrication.

**D. Checked and correct**

- The supermartingale factor identity (jpc_main.tex:74).
- The countable-union and pathwise-weight argument.
- The zero-success budget (40 at M=8, alpha=.05, tau=.12).
- The cost-weight minimiser.
- The Boole bounds and their attainment.
- The DP posterior bound for delta = 0.

**For the main session:** no new durable facts about Ron came up in this review.