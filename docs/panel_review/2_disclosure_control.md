**Referee report, Professor 2 (official statistics and SDC).** Scope: paper/bis_ifc_full.tex, the generated tables, and the code in src/privacy_risk/{risk,realdata,evaluation,experiments}.py.

**Verdict.** The algebra in my area is correct, but most of it is already in the SDC literature, and the paper should say so. Several empirical claims either disagree with the paper's own tables or rest on code that does not do what the prose says. Recommendation: major revision. Findings 1 to 7 must be fixed before circulation.

## A. Verified errors

**1. MAJOR. The falsification list quotes stale figures (lines 2194-2195 and 2201).**
- The paper says the misspecified model's gap is 0.053 and the two risks differ by 0.001. Section 9 (line 1368), the Figure 3 caption and numbers.tex all give 0.049 and 0.007 (0.4642 against 0.4571).
- Item 2 says the journalist error is "between 13 and 18 per cent". Lines 457 and 1338 and numbers.tex give 8 to 25.
- Fix: "failing for a misspecified covariate precision with a gap of \runcalibrationgapwrong{}, while the two risk estimates differ by 0.007" and "the journalist risk to between \runjournalisterrorlow{} and \runjournalisterrorhigh{} per cent". Also add this list to the script that checks prose figures against the run.

**2. MAJOR. The census comparison cites the wrong row (line 1656).**
- "The same comparison ran to 0.74 against 0.10" is the k=10 point estimate. The census case is at k_min=5.
- Fix: "the same comparison at $k_{\min}=5$ ran to 0.868, with a corrected lower bound of 0.778, against 0.20".

**3. MAJOR. The credit record breaks the paper's own rule (line 1877).**
- The text says "at $\tau = 0.10$ would admit only the $k_{\min} = 20$ setting". Line 1791 reports that setting's interval as [0.044, 0.158], so its upper bound fails eq. (admiss).
- Fix: "at $\tau=0.10$ the point estimate admits only $k_{\min}=20$, but its upper bound of 0.158 does not, so no setting in Table~\ref{tab:credit} passes \eqref{eq:admiss}".

**4. MAJOR. The census adversary does not misread the key (lines 1638 and 1703, and the tab:real caption).**
- In realdata.attack the observed key is the true key[position]. The value `error` enters only the adversary's likelihood.
- So the adversary reads the key exactly and only *assumes* a 10 per cent error rate.
- Also, the "Simulated" column is the maximum over the declared and the twice-as-precise adversary (the `robust` value), not "the declared adversary" as the caption says.
- Fix: either corrupt the observed key, or write "observes the key exactly but models a misreading probability of 0.10". Relabel the column "robust (max of two adversaries)".

**5. MAJOR. The census journalist figure rests on an undisclosed sampling fraction (lines 1628-1632).**
- experiments.real_data calls journalist_estimate with sampling_fraction=0.05 by default.
- The fitted population cell mean of 0.886 and ρ̂=0.225 follow directly from that assumed π.
- The wooldridge extract is a subsample of a PUMS file, so its fraction of the US population is far below 0.05.
- Fix: state the assumed π, report ρ̂ over a plausible range of π, or drop the figure.

**6. MAJOR. The Section 11 programme gap comes from the adversary's extra readings, not from pooling releases (lines 2004-2018, 2068-2074, 2104-2119).**
- In evaluation.identification_attack, exposure is fixed per loan. The pooled attack adds six independent readings of the same number, which is the same as one reading with standard deviation divided by √6.
- The per-release attack gets one reading. The gap therefore measures a larger auxiliary file, not what the releases disclose jointly.
- Fix: give the per-release adversary the same six readings, or state plainly that the gap is driven by repeated auxiliary observation.
- Separately, programme_parameter picks k by the point estimate (`match_rate <= tolerance`), not by the upper bound in eq. (admiss). The "k=1 against k=50" headline should be recomputed with the bound.
- Also, the "Worst" column for no suppression is 0.122, which exceeds τ=0.12. So under the paper's own supremum control, no suppression already fails per release.

**7. MAJOR. The noisy-counts rows are not composed across vintages (lines 2096-2100 and 2118).**
- For counts, `pooled = single`, and counts_attack uses month 0 only. So "pooled" for noisy counts is one release.
- Six vintages at ε=5 give ε=30 under basic composition.
- The target set also differs: the trials are unique cells, not loans. So 0.005 is not comparable with the other rows.
- "Identifies 1.000 of the unique cells" contradicts the table (Release 0.005, Acts 0.005).
- The utility figure uses a penalty the authors chose (−0.35 − 0.05/ε, in utility_of), and that penalty alone decides the ranking.
- Fix: compose across vintages, state the total ε, report the penalty, and drop the claim that ε=5 is the "highest utility policy meeting the tolerance".

**8. MINOR. The "Worst" column measures a different event (line 2083).**
- The "worst size class" 0.022 is below the average of 0.062 because the subgroup counts only correct guesses the adversary acted on (posterior ≥ 0.5). "Release" counts argmax matches.
- Fix: redefine the column, or say "acted-and-correct rate".

## B. Concerns (not errors, but they change the reading)

**9. MAJOR. The k-anonymity "refutation" overreaches (lines 1383-1402, 496-498, proof at 2393).**
- The adversary holds a noisy reading of the released y, so y is a quasi-identifier. A release that is k-anonymous on a key that leaves out a variable the intruder knows is not k-anonymous in Sweeney's sense.
- This is a key-selection failure, and well known (Hundepool et al. 2012, ch. 3). It is not a failure of the criterion.
- Reword: "k-anonymity on a declared key does not deliver 1/k when the intruder also knows a released continuous variable".
- The proposition's "or Assumption \ref{as:unif}" is not proved. If the adversary's information is exchangeable, a non-uniform choice cannot raise the class-average success above 1/f_k. Delete "or \ref{as:unif}".

**10. MAJOR. The journalist measure is misnamed (lines 430-437, tab:measures, line 1345).**
- ρ=P(F=1|f=1) is the population-uniqueness probability.
- The journalist's linkage success for a sample unique is E[1/F_k | f_k=1] ≥ ρ.
- The Skinner-Elliot (2002) θ = Σ I(f_k=1)/Σ F_k I(f_k=1) is a third quantity.
- In El Emam and Dankar (2008), "journalist risk" is 1/min F_k. That paper coined the prosecutor/journalist/marketer terms, and it is not cited.
- Fix: rename ρ "probability that a sample unique is population unique", and report E[1/F|f=1] alongside it.
- "Using π as the risk is a common shortcut" (line 1339) needs a source or should be deleted.

**11. MINOR. The negative binomial fit fails its own closure check.**
- results.json gives a fitted dispersion that rises from 0.33 at π=0.02 to 0.51 at π=0.2.
- If the negative binomial model held, thinning would leave the dispersion unchanged. The drift is evidence of misfit and belongs in falsification test 2.
- Line 1332 ("the same dispersion") should say it.

**12. MAJOR. The contributor rules do not follow SDC practice (lines 1828-1837, 2037-2042, 2704).**
- The threshold rule is a rule for magnitude tables. Applying it to microdata key cells is non-standard.
- With lender shares up to 23 per cent, the dominance (n,k) and p% rules are the relevant primary rules (Hundepool et al. 2012, ch. 4). One contributor can estimate another's amount within p% even with three or more contributors, so "Requiring three contributors … removes both" understates the risk.
- "Complementary suppression" as coded (withhold the smallest surviving cell in each region) is not audited against the other margins (sector, size) and is not secondary suppression in the Cox (1980) or τ-ARGUS sense.
- Fix: relabel it "heuristic secondary suppression (unaudited)", or run an LP audit.

**13. MINOR. Novelty and citations.**
- Proposition recip, identity (marksamp), Proposition plugin, the binomial-thinning formula, the threshold rule and differencing are all standard. Proposition recip is not "routinely used as a definition" without derivation: see Duncan and Lambert (1989), Lambert (1993, JOS 9:313-331) and Skinner and Elliot (2002).
- Line 208 misattributes the uniqueness measures. Add Skinner and Holmes (1998, JOS 14:361-372), Elamir and Skinner (2006, JOS 22:79-95) and El Emam and Dankar (2008, JAMIA 15:627-637).
- Line 946: the linked-tables and differencing problem is old in SDC as well as in DP.
- What is new: the governance tuple, pricing the certification budget, and the finding that no function of marginal risks bounds pooled risk, stated in the terms a certifying body uses. That last finding is weakly new, since its construction is the classic intersection attack.

**14. MINOR. Overreach about real registers (lines 1999 and 2114).**
- "A release system of the kind supervisors run every month" and "an ordinary register" describe a synthetic register with chosen parameters.
- Replace with "on a synthetic register built to share the structure of a supervisory collection", matching line 2160.

**15. MINOR. Notation clashes.**
- k is both a key value and the k-anonymity parameter, inside one proposition (line 493). Use κ_min or ℓ for one of them; κ is also taken, so pick with care.
- γ is both a comparison vector and a power exponent.
- q is prior weights, a target count and specificity.
- λ is the utility weight and the Poisson mean.
- B is replicates, the event B_t, and 𝓑_a.
- a is the scenario and a constant in φ(r)=a+br^γ.

**16. MINOR. Other text and data points.**
- Lines 1807-1809 repeat themselves in a way that makes no sense. Replace with: "An allegation that is right a third of the time is 1.5 times the success rate of always naming the largest lender (share 0.23), and wrong about two borrowers in three."
- In tab:scaling, n=2,362 at N=50,000 is 2.8 SD below the Bernoulli expectation of 2,500. Check the generator.

**17. MINOR. Bibliography.**
- My-area entries are correct in authors, years and venues, except these two.
- Rinott and Shlomo (2006) lacks "LNCS 4302, Springer".
- Duncan et al. (2001) is better cited as the NISS technical report, or the published chapter.
- Entries are missing for Lambert (1993), Skinner and Holmes (1998), El Emam and Dankar (2008), Cox (1980) and Dalenius and Reiss (1982). The last is needed for the swapping lever at line 1310.