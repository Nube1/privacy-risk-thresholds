**Adversarial referee report, round 3: "Adaptive certification of attack-conditional disclosure risk across release programmes" (JPC article and supplement) and "Governing privacy risk across repeated statistical releases" (IFC long form)**

**Scope and method.** I read the current jpc_main.tex (with jpc_context, jpc_network and jpc_evidence), jpc_supplement.tex and bis_ifc_full.tex with its \input files, the seven panel reports, the round-2 check and the response letter. I ran the test suite, the three number checkers, the toy audit, the credit audit and the paper pipeline. I read the gate code, the attack code and the audit scripts. This report sets out the case against the papers. Where I found no fault I say so, so that the author knows which ground is safe.

**What survives.** The gate is valid. The bet is clipped to [0, τ), it is fixed before the outcome, the null runs the right way, and the weights add to at most one. zero_success_budget(0.12, 0.05, 0.5) returns 29, which is right. The credit audit and the pipeline rerun byte for byte. Targets in the credit audit are fresh. The frozen-family comparison uses the same streams and the same cap. The certification counts, the Clopper–Pearson bounds and the decomposition 0.130 = 0.053 + 0.057 + 0.020 all check out. None of what follows is a hidden error in the maths of the gate. The problems lie in what the experiments measure and in what the text says they show.

**Recommendations.**
JPC: reject in present form, with encouragement to resubmit. The theorem is known and the paper now says so. What remains to carry the paper is the application, and the application has one synthetic register, no test of the one feature that sets the gate apart, and an outcome that a two-line drift calculation predicts.
IFC long form: major revision. Its headline number (the floor rising from 1 to 40) still mixes pooling with extra auxiliary data, the very confusion the revision claims to have removed, and it rests on an adversary assumption that the threat model does not list.

---

## Part A. Problems that sink the headline in both manuscripts

**A1 MAJOR (bis_ifc_full.tex:52, 142–147, 2155–2168; tab:kchoice; jpc_evidence.tex:25, 52).** The floor of 40 is set against the six-reading attack, not against pooling.
The abstract says: "With the same auxiliary information, pooling the six vintages alone takes the rate from 0.079 to 0.136". The next sentence says that calibrating on the programme raises the floor "from 1 to 40". But tab:kchoice uses the full pooled attack, with one exposure reading per vintage (six in all). The per-release answer of 1 uses one reading. So the 1-to-40 comparison is a one-reading adversary set against a six-reading adversary, and tab:decomp shows that more than half of that attack's excess (0.053 + 0.020 of 0.130) is not pooling at all. The floor that pooling alone needs was never computed. Under pooling alone k=10 already sits at 0.119, with an upper bound of 0.128, so it is likely well below 40. The JPC credit audit repeats the confusion: its pooled coordinate is "all six vintages and six readings" (jpc_evidence.tex:25).
The long form also breaks its own rules here: apply_section.tex:80–86 (Step 5, "holding the same auxiliary information") and worksheets_appendix.tex:76 (checklist item 5).
*Fix:* Run the k grid with the "all releases, one reading" attack and report that floor as the programme effect. Report the six-reading floor as a separate effect of the adversary's auxiliary data. In the JPC audit, register the one-reading pooled coordinate, or rename the coordinate so that it does not claim to measure pooling.

**A2 MAJOR (jpc_context.tex:6; bis_ifc_full.tex:2026–2028, 2041–2046; src/privacy_risk/evaluation.py:278–280).** The pooled attack uses knowledge that the threat model does not grant.
Assumption 1(ii) says the adversary "knows that the target is present in the first release it holds". The code does more: for every month m, `if member[m][target]: score += np.where(member[m][block], 0.0, -30.0)`. That is, the adversary knows every vintage in which the target's loan is live, and it drops candidates that are missing from any of them. Both texts say this is how pooling works: "It works here through membership" (jpc_evidence.tex:50). So the whole "pooling alone" gain of 0.057, on which the programme finding now rests, comes from a fifth piece of auxiliary data: the target's loan history. That is a strong assumption for a credit-register adversary, and it is not listed.
*Fix:* Add it to Assumption 1 and to the long form's threat model, and argue that a real adversary would know it. Report the pooled attack without it (the adversary knows presence only in the first vintage) so the reader can see how much of the programme finding survives.

**A3 MAJOR (jpc_context.tex:6, 20–34; bis_ifc_full.tex:930–938, 951, 1504–1522, 2031–2034; jpc_main.tex:131).** Under a persistent pseudonym, the pooled programme is simply one finer release.
Once every release carries the same pseudonym, releases that key on different projections of the quasi-identifiers can be joined into one table keyed on their union. Proposition 1 then says only that a finer key has higher risk. That is true, but it is not a composition result, and it points to a cheaper cure than the gate: rotate the pseudonym. Both papers concede they "did not measure pooled risk under rotation" (jpc_main.tex:131; bis_ifc_full.tex:2032–2034). Most of the release types in the long form's apply section (aggregates, payment statistics, supervisory tables) carry no loan pseudonym at all.
The long form also overstates what the proposition gives. Line 951 says "Proposition nopool does not depend on the construction", and line 1542 says two observations "follow from" it. An existence result implies neither. The abstract states the no-bound claim with no condition, although the proposition now needs one.
Sections 4.6 and 5 (the four-release simulation and the Census variants) intersect candidate sets across releases but never state the linkage assumption. Four public-use files cut from one census sample do not share record IDs. The figures 0.4581, 0.709, "1.8 times the bound" and "26.0 per cent uniquely identified" all depend on it.
*Fix:* State the pseudonym condition in the abstract, the contributions, Section 4.6 and Section 5. Report pooled risk with rotating pseudonyms, with linkage running through the key and exposure only. Discuss rotation as a policy option beside the floor.

**A4 MAJOR (bis_ifc_full.tex:2128–2135, 2155–2158, eq. admiss at 700–708; tab:credit against tab:eval).** The per-release answer of k=1 fails the long form's own subgroup control.
Line 2133 says: "With no suppression the bound is 0.122, above the tolerance, so under the supremum control no suppression already fails on a single release." Yet the abstract, the contributions and Section 13.4 give k_min = 1 as the per-release answer. tab:kchoice has no subgroup column, so the floor choice silently uses the average-target coordinate, although Section 6 requires the target set T to be explicit. Worse, the subgroup figures do not agree across tables. tab:credit gives large firms 0.185 at k=1 and 0.204 at k=5 (argmax success). tab:eval's "Worst" column gives 0.122 and 0.061 for the acted-and-correct rate at posterior 0.5 or more. Line 2132 says k=5 meets the subgroup control, while lines 1988–1990 say no setting up to k=20 does.
The Bonferroni step behind 0.122 may not be needed at all. Rule eq. admiss releases only if every target's bound is at or below τ. That is an intersection-union test (Berger 1982), and per-target level-α bounds already control it. So the "Worst" figure could move either way.
*Fix:* Pick one success event for all subgroup claims, state T for the floor choice, and recompute the Worst column both with and without the correction, saying which one governs.

**A5 MAJOR (bis_ifc_full.tex:710–716, proof 2544–2551, eq. opt 851–853, 2159–2166; tab:record 2351–2368).** Choosing the floor from the grid voids the error control.
Prop. point holds for one fixed θ. The paper picks θ with the same bounds ("smallest tested floor whose upper bound meets"). The chance that the chosen setting is unsafe can then reach the number of unsafe settings times α. The fine grid was added after the coarse grid put its only pass at its top level. k=40 is then certified by the very draw it was chosen from, and the decision record still marks the row "Pass". The long form uses Bonferroni over 72 comparisons for the Census, but no correction over 11 grid levels here.
There is a valid route that costs nothing: fixed-sequence testing from the strictest level down, stopping at the first failure (Maurer, Hothorn and Lehmacher 1995). On tab:kchoice it gives the same answer, 40. The two rules differ when bounds are not monotone in k, and here they are not: k=5 pooled is 0.224 against 0.209 unprotected.
*Fix:* State the procedure as fixed-sequence and extend Prop. point to the chosen θ, or re-audit k=40 on fresh targets.

**A6 MAJOR (bis_ifc_full.tex:1620, 1593–1616, 2910; jpc_evidence.tex:31; credit.py:64).** One register draw carries every credit headline.
The long form says "one program with a single seed" (line 1620). The simulation population, the credit register and the four scaling registers are each one draw. The nested-design check of design variance covers one mechanism on the demographic simulation only. The credit register uses one exposure-noise SD (0.6), which the attacker knows exactly, and one lender structure. On the Census, changing the noise SD moves the rate from 0.371 to 0.186 (lines 1725–1732), so this input matters. In the JPC audit the "20 seeds" vary only the attack draws. The register is fixed by the config seed.
On the credit register the simulation term at 4,000 attacks is small, so the design term, if it is as large as the nested check allows, would be about half the total. My rough transfer moves the bound at 0.136 from 0.145 to about 0.149 and at k=40 from 0.106 to about 0.110. The conclusions probably survive, but that is an assumption, not a result.
*Fix:* Redraw the register over at least 20 seeds and a range of true reading SDs, and report the spread of the chosen floor. Report bounds that include the input term.

---

## Part B. JPC article and supplement

**B1 MAJOR (jpc_main.tex:39–45, 79–97, 100–119).** With the novelty now stated plainly, little remains that is new.
The remark after Theorem 1 says it is online Bonferroni over betting supermartingales. Prop. 2 is one line from eq. (bet). Prop. 3 is a textbook Lagrangian, and the text concedes that it "ignores integer ceilings and positive success counts", that is, it optimises the one case (every attack fails) that does not arise when the risk is near the tolerance. Prop. 1 is, by the paper's own words, "neither ... new". The case for publication must therefore rest on the application in Section 5, and B2 to B4 show that it cannot yet carry that weight.
*Fix:* Add a result that the reduction does not give for free. Examples: a bound on expected audit length as a function of the gap between r and τ; an optimal or regret-bounded bet rule for this problem; or a valid treatment of reused targets with a κ that can be estimated. Otherwise, reframe the paper as a short applied note.

**B2 MAJOR (jpc_evidence.tex:31; jpc_main.tex:133).** The experiment does not test the one feature that sets the gate apart, and on the features it does test, the gate loses.
The paper admits it: "every single-release coordinate certified in every seed, so every pooled coordinate was registered, and a frozen family of all ten coordinates would have tested the same hypotheses." At equal budget the frozen family certified more often (k=50: 20 of 20 against 19 of 20; k=40: 9 of 20 against 6 of 20). The only measured gain is early stopping on single-release coordinates. That gain is measured against a fixed design of 3,000 attacks per coordinate, which is far more than a power calculation would give for a coordinate at 0.079 against 0.12. A fixed design sized by power, or a group-sequential design, is the fair comparator.
*Fix:* Build a run where adaptive registration matters. For example, register subgroup or pooled coordinates only for policies whose single-release audit leaves a narrow margin, so that the frozen family must pay for coordinates it never needed. Compare with a power-sized fixed design and with a group-sequential design, not only with the 3,000 cap.

**B3 MAJOR (jpc_evidence.tex:27–31; table_credit_audit.tex).** The result follows from arithmetic, so the experiment adds little.
With the bet capped at 0.8τ = 0.096, the expected log-growth per draw at true risk p is p log(q/τ) + (1−p) log((1−q)/(1−τ)). At p = 0.098 (k=40) this is about 0.0024 per draw, so reaching log 200 = 5.30 takes about 2,200 draws on average, near the cap of 3,000. At p = 0.079 (no suppression, single release) it is about 0.0089, or about 600 draws; the observed median is 498. The share of 6 in 20 at k=40 and the medians are what this drift calculation predicts. The cap on the bet, not the gate, sets much of the power gap with the frozen family, and the text does not say so.
Two further problems: "just below the tolerance" (jpc_evidence.tex:29) describes a rate of 0.095 to 0.098 against 0.12, which is a margin of about a fifth. And the median of 1,224 attacks at k=50 counts the one seed that failed and stopped at 3,000.
*Fix:* Give the drift calculation in the text and say what the experiment adds beyond it. Report the bet cap's effect. Report medians over certifying seeds only, or give the full distribution.

**B4 MAJOR (jpc_evidence.tex:20; scripts/adaptive_audit_comparison.py).** A claim about the software is false.
The text says: "The run shows that the implementation handles registration that depends on the history." The toy script imports only argparse, json, numpy and scipy. It never imports AdaptiveAudit; it recodes the e-process in numpy. So the toy audit does not run the shipped gate. The README's "software verification" rests on the unit tests and the credit script alone.
*Fix:* Run the toy audit through AdaptiveAudit, or delete the sentence.

**B5 MAJOR (jpc_main.tex:26, 44; jpc_evidence.tex:25).** "Five protection policies" are five levels of one policy.
Every policy in the audit is k-anonymity on the same key, with k in {1, 5, 10, 40, 50}. The long form tested noisy counts and other levers; the article does not. The abstract and contribution 4 should say "five cell-count floors".

**B6 MAJOR (jpc_main.tex:31, 177; jpc_evidence.tex:33–52).** The carried-over evidence cannot be checked by a referee.
The source of Table 4, the four-release figures and the floor choice is cited as an author manuscript "Available from the author". A journal referee cannot read it without breaking anonymity, and the main argument for why pooled coordinates fail depends on it.
*Fix:* Deposit the long form (arXiv, SSRN or an IFC working paper number) before submission and cite that record.

**B7 MAJOR (jpc_context.tex:13; jpc_evidence.tex:27).** The tolerance and the target law are not argued for.
A tolerance of 0.12 on the average chance of naming a random loan is loose by the standards of the SDC literature the paper cites, where thresholds for public release are often lower and applied to the maximum, not the average. The target law is a prosecutor law (draws from the first release). The paper names the journalist law but never runs it. Every conclusion in Table 3 depends on both choices.
*Fix:* Justify 0.12 with a source, or report the audit at two or three tolerances. Say which law a central bank should use and why.

**B8 MINOR (jpc_main.tex:50–56).** Missing prior art on audits by betting.
Risk-limiting election audits use the same tool for the same job: a betting supermartingale on sampled units, a predictable bet built from a shrunk running mean, and certification at a threshold. See Stark (2008) and later work, Waudby-Smith, Stark and Ramdas (2021, RiLACS) and Stark (2023, ALPHA). The bet rule in Section 5.2 is close to ALPHA's. Sequential releases also need Xiao and Tao (2007, m-invariance) and Byun et al. (2006). The author should check each record before citing it.

**B9 MINOR (jpc_main.tex:137).** "Immutable" overstates the code. Only CoordinateSpec is frozen. The fields crossed, log_e, trials and log, and the gate's spent and alpha, are ordinary attributes; setting coord.crossed = True certifies a coordinate. Eligibility checks only the names the caller passes, and the credit script never calls gate.eligible. Either freeze these fields or reword to "records".

**B10 MINOR (jpc_supplement.tex:77).** `python -m privacy_risk.pipeline` runs configs/base.json, a smoke test, and writes to artifacts/. It does not reproduce Table 4. The command needs `--config configs/paper.json --output artifacts-paper`.

**B11 MINOR (jpc_supplement.tex:51; jpc_network.tex).** The neural section still earns no place. Its only result is that logistic regression beat the three-layer net in 10 of 10 seeds on a toy law, and the text says it shows only "that the pipeline runs". One paragraph and a code reference would do.

**B12 MINOR (evaluation.py:208; jpc_context.tex:13).** Ties are broken by np.argmax (first index), while the threat model and Prop. 1 speak of uniform random tie-breaking. Ties are rare with continuous readings, but the text and code should agree.

---

## Part C. IFC long form

**C1 MAJOR (bis_ifc_full.tex:2195–2200; tab:ablation).** The ablation says the opposite of what the text draws from it.
The text: "The attack depends on the reading of a continuous variable far more than on the categorical key", and "Protection applied to the key achieves less than protection effort spent on the published continuous variable." The table: key without exposure 0.026; region plus exposure only 0.003; all four fields 0.072. The key alone does about nine times better than exposure without most of the key. The effect is joint. No policy that perturbs exposure was ever tested, so the second sentence has no evidence. "All four fields" (0.072, precision 0.800) and "assumed σ × 1" (0.066, precision 0.632) are the same setting and disagree without comment.
*Fix:* Describe an interaction, and drop the policy claim or test an exposure-noise policy.

**C2 MAJOR (bis_ifc_full.tex:2142–2151; results.json; evaluation.py:440–444).** The noisy-count rows cannot show anything.
The count attack has 817 trials and 4 unique classes, so its success cannot exceed 4/817 = 0.0049. "Composing ... at a total of 30 leaves the rate at 0.005" is a ceiling, not a finding. Count utility is computed on the unperturbed rows, so the noise never enters it; the reported −0.400 and −0.360 equal the fixed penalty −0.35 − 0.05/ε exactly. The text says the penalty "sets most of that difference". It sets all of it. results.json still records admissible_best = "noisy counts, ε=5", which round 2 says was removed.
*Fix:* Drop the count rows from tab:eval, or give them a loan-level event and a utility computed on the noisy output.

**C3 MAJOR (bis_ifc_full.tex:1660, 1698–1701; results.json real.census.grid).** The Census utility gives coarsening no cost.
Every k=1 setting has losses of 0.0 on all three analyses, including education in four bands against "income by education". The simulation makes coarsening costly (lines 2894–2897); the Census does not. So the Census frontier and the implied κ trade off suppression only.
*Fix:* Compute the analyses from the released bands, as in Section 4.

**C4 MAJOR (bis_ifc_full.tex:52, 1780–1835, 2423–2427; response letter 4.4 and R2.8).** The text contradicts itself on the neural work, and the letter claims a move that did not happen.
Section 5 reports two real-data neural experiments with a table, a diagram and the heading "Where a nonlinear model helps". The abstract says the neural procedure "plays no part in the results". The Discussion says it is "illustrated only on independent synthetic attack trials" with "no established ... benefit on the credit and census data". The letter says the material moved to appendices.
The real-data comparison is also weak: the linear baseline stopped at epochs 296, 108 and 50, while the SENTRY-R benchmark needed 404 to 898 for logistic convergence; no intervals are reported, although the JSON holds bootstrap intervals over nine test states; "the quadratic comparator beats both neural models in every split" fails for split 3 against three layers (interval [−0.0025, 0.0001]); and the gains are 0.0025 to 0.0062 in log loss.
*Fix:* Remove the real-data neural paragraphs from Section 5, or rewrite the abstract, Discussion and letter to match them, and train baselines to convergence.

**C5 MAJOR (decision.json; validation.json; bis_ifc_full.tex:1636, 2263–2282; apply Step 6).** The pipeline passes what the paper says should fail.
decision.json reads PASS with "within_programme_tolerance": true while pooled risk is 0.458 against a release tolerance of 0.12, because the programme check uses τ_prog = 0.5 and measures releases A–D at k=3, not the released setting. validation.json passes frequency_transport and mechanism_monotonicity, yet the text reports negative binomial misfit (test 2 says the journalist figure "should be replaced by a bound"; it is not), and the tables are not monotone (kanon worst target 0.744 at k=10 and 0.768 at k=20; pooled 0.209, 0.224 and 0.225). Test 4 says such breaks "must be investigated rather than smoothed". The long form does not investigate them.
*Fix:* Make the gate fail on programme risk above the release tolerance, and act on tests 2 and 4 as written.

**C6 MAJOR (bis_ifc_full.tex:2139).** The attribute attack has no baseline. "A confident reading of arrears status is right 0.944 of the time", but arrears occur in about 7 per cent of loans, so guessing "no arrears" for everyone is right about 93 per cent of the time. The JSON's lift of 13.58 divides by the wrong base rate.
*Fix:* Report accuracy against the majority-class guess, and for disclosure in the harmful direction only.

**C7 MAJOR (bis_ifc_full.tex:1123–1133, 1196–1200).** The shift bound is loose, and "no more" is false.
By data processing, D_KL(Q‖P) ≤ ρ gives kl(R_Q‖R_P) ≤ ρ for the Bernoulli success indicator, which is far tighter than Pinsker near zero. For the ε=1 count release, R_P = 0.004 reaches 0.12 only at kl = 0.299, eleven times the 0.027 quoted. The bound is also applied to point estimates, not to upper bounds. And line 1133 says shift is "measured here"; it is a bound under an assumed ρ.
*Fix:* Use the Bernoulli-KL inversion on the upper bound, and reword "measured".

**C8 MINOR (bis_ifc_full.tex:1064–1071; proof 2633–2635).** Prop. attack-threshold is wrong twice. The threshold (c_f + c_a)/(h + c_f) equals one half whenever h = c_f + 2c_a, not "only" when h = c_f and c_a = 0. And when c_a > h the payoff at p = 1 is h − c_a < 0, so the adversary never acts, whereas the proposition says it acts at p ≥ 1. Set p* = +∞ in that case.

**C9 MINOR (bis_ifc_full.tex:2849–2853, 1956–1966; credit.py:365).** The printed differencing formula leaves out the (A=0, D=1) term, so as printed it cannot exceed 1/2 and could not give 0.61; the code has the factor of 2. The same paragraph gives "4.7 movements per cell" and "9.3 across all cells" for one rate. The model gives 0.61 where the realised share is 1.00, which fails test 1; the text treats this as an illustration rather than a rejection.

**C10 MINOR (bis_ifc_full.tex:1667, 1759).** Unique shares are given for the wrong unit. "27,390 classes and 86.3 per cent of them are unique": 0.8628 is 25,454 unique records over 29,501, a record share; the class share is 92.9 per cent. "150 classes, 3.2 per cent of them are unique" is 63 of 1,986 records; 42 per cent of classes are unique.

**C11 MINOR (bis_ifc_full.tex:1472–1477, 1545–1552).** The frontier is not concave (slopes from 6.0 to 0.006, out of order), yet a concave power curve is fitted and the "4.5 per cent" loss reduction is computed on it, from the single-release grid. Optimise over the discrete settings, or call it a sketch.

**C12 MINOR (bis_ifc_full.tex:2121–2126; tab:eval, tab:credit, tab:ablation).** Precision claims rest on a few dozen decisions. At k=5, 46 of 4,000 attempts pass the threshold and 35 are right: about 114 attempts per correct identification, not "few attempts". The k=5 per-release rate appears as 0.062, 0.067, 0.0722, 0.072 and 0.066 in different tables, and tab:credit's k=10 figure of 0.0653 lies above tab:eval's upper bound of 0.061. That points to different estimands, not noise. State the estimand in each table.

**C13 MINOR (bis_ifc_full.tex:1109–1113, 730–733, 985, 1176, 2400).** Smaller slips. The α/q advice for certification is too strict for the release decision (see A4). "Proposition cert gives the required sample size" is false; it covers zero successes only. Line 985 calls governing the union "the conservative route", which Prop. nopool shows it is not. 0.165 is 1.96 times 0.084, not "more than double". The k-anonymity failure holds by construction (the key leaves out a variable the attacker holds), yet the Discussion lists it as one of two main negative results.

**C14 MINOR (bis_ifc_full.tex:1751, 1839).** "Larger on real data" rests on 1.86 against 1.81 from one draw each. The causal accounts at 1741–1743 and 1752–1753 are never tested. Drop them or test them.

**C15 MINOR (bis_ifc_full.tex:2229, 2232–2235; apply_section.tex:173–174).** "A lower bound on what is possible" does not fit an attacker given an exact key, a persistent pseudonym, the true error model and the loan's full history. "Without modification" on a real register is not shown: Section 5 ran only the demographic path on real files.

---

## Part D. Reproducibility

**D1 MAJOR (tests/test_reporting.py:75; artifacts-paper/manifest.json:26).** One of 147 tests fails. The shipped manifest lists run.log, which does not exist, with the same hash as decision.json. A fresh run writes no run.log entry. So the shipped manifest was edited by hand or is stale, which contradicts the README's "manifest included".

**D2 MINOR (scripts/check_paper_numbers.py).** The quoted-mode checker is weaker than the long form says (lines 1632–1636). It checks only decimals with three or more places, and a number passes if it matches any value in any source file (0.077 matches about 400 entries in results.json). It ignores integers and two-place figures such as 29, 40, 3,000, 0.30 and "19 of 20". Twelve of 159 quoted decimals in the long form match no declared result, including all the neural pilot figures. I checked the JPC integers by hand and found them right, apart from the median rounding in B3 (498.5 and similar are cut down by int()).

---

## Part E. Response-letter claims that the text does not bear out

| Letter item | Claim | What the text shows |
| --- | --- | --- |
| Summary 1 | Pooled excess split into its two causes | Split in tab:decomp, but the 1-to-40 headline still uses the six-reading attack (A1) |
| R2.3 / R3.2 | Pseudonym condition stated in both manuscripts | Stated at 932 and 2031 only; missing from Sections 4.6 and 5 and from the abstract (A3) |
| 2.10 | ρ renamed | tab:measures still reads "Journalist ρ"; fig. 5 caption and lines 1370, 2262, 2266 unchanged |
| 4.4, R2.8 | Neural material moved to appendices | Two real-data neural experiments remain in Section 5 (C4) |
| Round 2 | "Noisy counts admissible" removed | results.json still names ε=5 as admissible_best (C2) |
| Round 2 | Two-release figure compared with its own bound | Falsification test 5 (line 2289) still says "a factor of 1.6 at two releases"; the right figure is 2.8 |
| R3.1 | Sequential-release literature added to both | Long form cites none of Wang and Fung, Xiao and Tao, or Byun et al. |
| SDC 16 | Sample-size anomaly | tab:scaling still has n = 2,362 at N = 50,000, 2.8 SD below expectation, with no comment |

The letter is out of date in the author's favour in one place: the long form now gives Duncan et al. (2001) a report number.

---

## What would change my recommendation

For the JPC article: (1) a pooled coordinate that measures pooling alone, under a threat model that lists what the adversary knows; (2) a run in which adaptive registration saves audits that a frozen family must spend, against a power-sized fixed design; (3) replication over register draws; (4) one result beyond the online-Bonferroni reduction.

For the long form: (1) the floor recomputed for pooling alone and for the listed adversary, with a valid selection rule; (2) the subgroup event unified and the per-release answer restated; (3) the count rows and the neural real-data section removed or repaired; (4) the pipeline made to fail where the text says it should.
