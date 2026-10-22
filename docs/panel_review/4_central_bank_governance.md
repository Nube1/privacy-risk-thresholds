**Referee report, Professor 4 (central bank statistics and IFC editorial view)**
Paper: paper/bis_ifc_full.tex, "Governing privacy risk across repeated statistical releases"

**Verdict.** The core result is useful to central bank statisticians: a tolerance set for each release does not control a monthly programme, and the paper shows what that costs on a credit register. As drafted it is not ready for IFC Working Paper consideration. The abstract buries the recommendation, the headline figures break the paper's own decision rule, the neural material in the title is not part of the evidence, and several claims about practice have no source. **Recommendation: major revision.**

**1. MAJOR. Abstract has no policy recommendation and is written for methodologists (bis_ifc_full.tex:51).** A data officer reading it cannot tell what to do differently on Monday. Replace it with:

> "Central banks publish from confidential registers every month, and an adversary can combine the vintages. This paper asks when a release programme, rather than a single release, is safe enough to publish, and who decides. Disclosure risk is a probability that depends on a declared adversary and is estimated with error. So a release decision should compare an upper confidence bound with a tolerance, and the tolerance should be a written record with a named owner, an expiry date, a harm class and a stated error probability. We prove that no function of per-release risks bounds an adversary who pools releases. On a synthetic six-vintage credit register, an attack that names the right loan for 8 per cent of targets in one release does so for about 21 per cent when vintages are pooled. At a tolerance of 0.12, calibrating on the programme rather than on one release raises the minimum cell count from 1 to 50 and cuts released relationships from 32,581 to 20,389. We recommend that central banks set tolerances for the release programme over a stated horizon, assess them by joint attack simulation, and publish a decision record with each release. The credit results are synthetic and need confirmation on a real register. An appendix sets out a neural search procedure for proposing candidate settings. It has been tested only on synthetic data and plays no part in the results."

**2. MAJOR. The headline calibration uses point estimates, which the paper says not to do (51, 2104–2111).** The abstract says "point-risk tolerance of 0.12", but eq. (admiss) at 693–698 and Prop. \ref{prop:point} say a point estimate releases at the boundary about half the time. In artifacts-paper/results.json (evaluation/programme), k=50 is also the *top* of the grid {1,5,10,20,50}, so it is the only level that passes, not "the least restrictive". **Fix:** recompute the choice with $\overline{R}_{0.95}$. Print the grid (per-release rate, pooled rate, bound, records kept) as a table. Say "the only tested level that meets it" unless a finer grid is added.

**3. MAJOR. The headline numbers come from two runs.** The abstract and Table \ref{tab:eval} give 0.079 per release and 0.209 pooled for no suppression. The calibration run in results.json gives 0.070 and 0.228 at k=1, and Table \ref{tab:credit} gives 0.0695 (1782). **Fix:** take the abstract, Section 10.4 and Table \ref{tab:eval} from one run, or state in one sentence that they are separate target draws, with their intervals.

**4. MAJOR. Title promises neural scenario search, but the body does not deliver it (31–32, 183–187, 2179–2180, 2298–2301, 2352–2357).** The paper says the method "is not used to generate the existing numerical results" and was tested only on 320 toy rows. As placed, it sits between the credit evaluation and falsification and breaks the central-bank argument. **Fix, option A (preferred):** retitle as "Governing privacy risk across repeated statistical releases: probability models, programme tolerances and accountable decisions", and move neural_section/sentry_r_section to an appendix or leave them to the companion. **Option B:** keep the title, but run the shortlist-and-audit procedure on the credit evaluation and report whether it changes the selected policy.

**5. MAJOR. Wrong length and order for an IFC Working Paper.** A central bank reader reaches the credit register at page ~40. **Proposed order:**
1. Introduction.
2. Governance rule (Sec. 6).
3. Credit register and evaluation (Secs. 12.1, 13).
4. Programme composition (Sec. 8).
5. Estimation and certification sample sizes (Sec. 5, briefly).
6. Protocol with a filled record.
7. Discussion.

**Move to appendices:** Secs. 3–4 derivations beyond Prop. 1, Sec. 9 (anytime-valid), Sec. 10 (DP), the full 200,000-record simulation (Sec. 11), the real public files (Sec. 12), the Netflix, US Census, ONS and HIPAA cases (1888–1982, none are central bank cases) and the neural material. Cut the ten-item contribution list (100–187) to four items.

**6. MAJOR. Governance is defined but never filled in (673–683, 1872–1883, 2236–2265).** The tuple $G$ is sound, but no example record exists. Line 1874 says "No tolerance has been stated for this register", while Sec. 13 uses 0.12 for the same register. Programme tolerance is also under-specified. A monthly series grows without end, pooled risk rises with the number of vintages (Table \ref{tab:comp}), and the paper never says over what horizon $\tau_{\mathrm{prog}}$ applies or what happens when month $m+1$ arrives. **Fix:**
- Add a table "Decision record for the credit register, vintage 6", with rows for the five harms. Each row gives $T$, $a$, $h$, $\tau$, $\alpha$, $\overline{R}$, $B$, mechanism setting, owner (e.g. "Chair, Statistics Release Committee"), expiry date and re-assessment trigger.
- Add to Definition \ref{def:thresh} "a declared programme horizon $m$ (number of vintages or months) for programme tolerances".
- Add a protocol step: "When a new vintage would take the programme past $m$, re-run the joint simulation before release."
- Change 1874 to "Section \ref{sec:eval} adopts $\tau=0.12$ for illustration; no institution has set a tolerance for this register."

**7. MAJOR. Claims about central bank practice are inaccurate or unsourced (1750–1761, 2032–2039, 1836, 2704).**
- AnaCredit covers loans to *legal entities* reported by credit institutions, above a debtor threshold, monthly. It does not release a loan-level file for research; research access goes through restricted environments. **Fix:** say the borrowers are firms, state the reporting threshold, and describe the three release kinds as "a stylised release set; loan-level access is normally through a secure research environment".
- The ecb2022 and boe2023 entries (2828, 2850) need part, section and URL. Reconfirm the edition and title of the Bank of England document.
- "the rule most commonly applied to supervisory aggregates" (2038) and "each of them is in operational use somewhere" (2032–2033) have no source. Cite \citet{hundepool2012} (sensitivity rules). Also note that a minimum-contributor rule usually sits alongside a dominance ($n,k$) or $p\%$ rule. Either add a dominance-rule policy to Table \ref{tab:eval} or state that it is left out.
- "which is why that rule is standard" (2704–2705): add the same citation.

**8. MAJOR. Synthetic caveat missing where results are claimed.** It is stated at 2160 and 2284, but not at 177–181, 2280–2282 or 2329–2332. Line 2281 also says "every one of them leaves the pooled programme above the tolerance". That contradicts 2118 (noisy counts, $\varepsilon=5$, meets it) and Table \ref{tab:eval} ($\varepsilon=1$, pooled 0.000). **Replace 2280–2282 with:** "On the synthetic register, every suppression policy calibrated on one release leaves the pooled programme above the 0.12 tolerance; only the noisy-count releases meet it, at a large loss of loan-level utility." Add "on the synthetic register" at 178 and 2329.

**9. MINOR. Status wording is correct in the main paper (35, 59) but risky in the companion.** bis_ifc_sentry.tex:10–14 defines `ifcred` headings in an IFC-like house style, and line 55 reads "Author-prepared paper for BIS/IFC consideration". Use neutral colours and write "Author manuscript; not a BIS or IFC publication". The companion also cites the full paper under an old title (bis_ifc_sentry.tex:74–75, "Quantitative risk assessment for data privacy..."). Use the current title.

**10. MINOR. Garbled or wrong sentences.**
- 1807–1809, "1.5 times better than naming the largest lender at random, which is 1.5 times the success rate of always naming the largest lender": pick one baseline and state it once.
- 2097, "identifies 1.000 of the unique cells" does not match Table \ref{tab:eval} (success 0.005, acts 0.005). Rewrite as "acts on 0.5 per cent of targets and is right each time".
- 2032: "7 policies" should be "Seven policies".

**11. MINOR. Prose against house rules.**
- No em dashes found; `--` appears only as an en dash in "risk--utility", which is correct.
- No "not only X but Y".
- Figures of speech and stock phrases to cut:
  - 76 and 1852, "did the work" / "doing the work".
  - 685, "earns its place".
  - 720, "pays for that control".
  - 725, "the honest response".
  - 1713, "Their purpose is sharper".
  - 1789, "the one whose distress moves markets".
  - 2057, "Four findings stand out".
  - 2104, "The headline number".
  - 2117, "is what closes the gap".
  - 2319–2320, "puts judgement in the policy quantity where it belongs".
  - 2702, "is not decoration".
- Repeated "not X but Y" set-ups to thin: 2115–2116, 2132–2133, 1995–1996.
- Replacements:
  - 2057: "Table \ref{tab:eval} gives four results."
  - 2117: "Programme-level certification (Section \ref{sec:comp}) removes the difference."
  - 685: "Each element prevents a named failure."
- "robust risk" (626 and elsewhere) is a defined technical term. Keep it.

**12. MINOR.** The Data and code availability section (2348–2352) refers to "the original sections" and "the new neural method", which is revision history. Rewrite it as a plain statement of what the pipeline produces.