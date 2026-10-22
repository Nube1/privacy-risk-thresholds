# Claims, evidence and limits

This document prevents computational evidence from being used for a claim stronger than it
supports. Each row names what kind of statement the claim is, what the code establishes,
and what would refute it.

| Claim | Mathematical status | Computational evidence | Possible refutation |
|---|---|---|---|
| Exact exchangeable matching gives risk `1/f_k` | Deductive under A1--A4 | Unit test builds equal-posterior candidates; the attack with no error and no covariate returns the reciprocal-frequency mean | Show unequal likelihoods, informative side data, or a non-exchangeable attack |
| Sample marketer risk equals `D_I/n` | Algebraic identity for the released-candidate experiment | Property checked over synthetic key classes and in the paper run to seven digits | Change the attack population or the conditioning event; the identity then answers a different experiment |
| Sample-frequency risk dominates population-frequency risk | Pathwise inequality plus Jensen | `plugin_ordering` checks both directions on the paper run | Cannot be refuted without changing the estimand; the two answer different questions |
| Journalist risk needs a population-frequency model, not just `pi` | Consequence of Bayes' theorem | Negative binomial law fitted to sample cell counts, compared with the known truth at five sampling fractions | External counts reject the frequency model, or the thinning assumption fails |
| Measurement error does not simply reduce risk | Empirical claim about one release | Key error alone lowers success; one noisy covariate restores it | A design where the covariate adds nothing; then error does reduce risk |
| `k`-anonymity is the rule `sup R <= 1/k` under A1--A4, and fails otherwise | Deductive, plus a construction | Per-target risks in minimum-size cells exceed `1/k` at every `k`, with Bonferroni-corrected lower bounds | Deny the adversary any information beyond the key; the bound then holds |
| Zero observed attacks do not prove zero risk | Exact binomial inference | Test reproduces `1-alpha^(1/B)`, the rule of three, and the certification table | Reject independence or Bernoulli replication; then use an appropriate dependent interval |
| A point estimate is insufficient for a threshold | Statistical decision result | The gate uses a one-sided bound; the run reports settings the bound rejects and the estimate accepts | Interval coverage fails under the fitted data-generating process |
| A stated relative harm weight implies a tolerance | Deductive on a differentiable frontier | Implied tolerance computed over the nondominated settings for six relative harm weights | Show a frontier where the maximiser is not interior, or refuse to assign a relative harm weight |
| Marginal release risks weakly bound union risk | Sharp probability theorem | Tests verify both bounds and their attainment | Cannot be refuted without changing the event |
| Marginal risks do not bound a pooled attack | Deductive construction, plus simulation | Four 3-anonymous releases with no unique record admit a pooled attack far above the union bound | Deny the adversary the ability to combine releases |
| Independence gives `1-prod(1-R_i)` | Conditional identity, not a default | Displayed as a counterfactual only | Dependence diagnostics or shared records invalidate the premise |
| A programme tolerance should be allocated by value | Deductive under a concave value curve | Numerical optimum matches the closed form for the power family | Show a value curve outside the family, or a programme where the union event is the wrong control |
| Stronger protection reduces risk | Empirical claim, mechanism-specific | Prespecified monotonicity test over the suppression levers | A statistically material upward movement rejects the claim |
| A stronger adversary raises measured risk | Empirical claim about the declared scenario | Stress test compares two capability levels across every setting | A stronger capability that lowers risk indicates a misdeclared scenario or a coding error |
| Differential privacy bounds posterior movement, not linkage probability | Deductive, prior-dependent | Posterior bound table across budgets and priors | None; the bound is exact. The refutable part is any claim that one epsilon equals one tolerance |
| One protection lever does not govern five credit-register risks | Empirical claim about one register | The minimum cell count halves large-borrower identification, moves the relationship predictive value by four per cent and leaves reconstruction untouched | A register where the same lever moves all five, which would be a property of that register rather than a general result |
| An alleged lending relationship is a base-rate statement | Deductive, checked as an identity | The four-cell table reproduces the predictive value to four decimals | None; the refutable part is any claim that high specificity alone makes an allegation reliable |
| Published totals leak by residual arithmetic | Deductive | Single- and two-contributor cells counted on the simulated register | A publication rule that removes both, which is the point of the three-contributor convention |
| A differencing finding is a posterior, not a disclosure | Deductive under a Poisson movement model | Unit-change share and the posterior computed on six monthly vintages | Evidence that movements are not Poisson, which changes the posterior but not its being one |
| The binomial interval covers only simulation error | Deductive, measured | Nested design separates within and between replicate variance | None; the measured split is design-specific and reported as such |
| Per-release calibration does not govern a programme | Empirical claim about one synthetic release system, backed by Proposition 10 | Seven policies attacked per release and pooled over six vintages. Choosing by the exact upper bound at a tolerance of 0.12, the per-release rule selects no suppression and the pooled rule a minimum cell count of 40 on the finer grid (50 on the coarse one). About half the pooled excess is extra auxiliary readings; with one reading, pooling the vintages alone takes no suppression from 0.079 to 0.136 | A release system where the pooled attack with one auxiliary reading gains nothing over the best single release, which would need a register whose membership does not change between vintages |
| The continuous variable, not the key, carries the attack | Empirical, measured by ablation | Withdrawing the exposure reading drops identification from 0.072 to 0.026 and leaves the adversary unable to pass its own decision threshold | A register whose key alone identifies, which is the case a minimum cell count is written for |
| A misspecified adversary is miscalibrated, not less successful | Empirical, measured under shift | Misstating the dispersion moves the success rate by less than a thousandth and the calibration gap from 0.001 to 0.068 | Evidence that the maximum-posterior choice does move under rescaling, which would contradict the invariance it rests on |
| A fixed-sample bound does not survive monitoring | Deductive, measured | Peeking twenty times over four thousand attacks makes the fixed bound exclude the truth in a fifth of runs at a nominal five per cent; the confidence sequence in none | None; the confidence sequence is valid by Ville's inequality. The refutable part is the measured error rate, which depends on the peeking schedule |
| The adversary's acting threshold is a payoff, not one half | Deductive | The threshold follows from the expected payoff; the same attack read at three payoff structures gives three different coverage and precision pairs | None; the refutable part is any claim that a particular payoff structure describes a real adversary |
| Calibration does not transport without a stated neighbourhood | Deductive under Pinsker's inequality | The declared divergence gives a risk ceiling, and the release that meets the tolerance can absorb only a small one | None; the bound is exact. The refutable part is whether a KL ball describes the shift an institution will face |
| The pooling gap is not an artefact of the register size chosen | Empirical, measured across four sizes | Larger registers are safer release by release while the ratio of pooled risk to the marginal bound rises from 1.57 to 2.05 between fifty thousand and four hundred thousand units | A register size at which the pooled attack gains nothing over the best single release, which the trend runs against over the range tested |
| The governance protocol improves auditability | Institutional argument | Machine-readable decision record, generated tables and hashes demonstrate implementability | Comparative operational evidence shows no improvement or unacceptable decision costs |

## What a measurement corrected

The differencing posterior was first computed with one movement rate pooled over all
published cells, which returned 0.003 and would have dismissed the attack. The rate that
belongs in it is the rate of the cell where the change appears: those cells are the quiet
ones, and the per-cell calculation returns 0.61 against a realised share of 1.00. The
three figures span the range, so the reported quantity is a statement about a fitted
movement model, and the pipeline reports all three rather than the most convenient one.

## What was tried and dropped

A district record swapping lever was implemented and removed. It reduced simulated risk,
but only against an adversary that takes the released district at face value; the declared
scenario gives no reason to assume that. Reporting its protection would have treated a
modelling assumption as if it were a property of the release. The removal is itself an
instance of the stress test doing its job, and it is reported in the paper rather than
hidden in a commit.

## Argument supported by the pipeline

The code establishes that the framework is internally coherent and operational: the
principal quantities can be computed, uncertainty can be carried into the release
decision, repeated releases can be tested jointly, a tolerance can be tied to a stated
relative harm weight, and every decision can leave a reproducible record. It also demonstrates
why a rule based only on a point estimate, on sample uniqueness, on `k`-anonymity, or on
separate marginal releases is incomplete.

## Argument not established by the pipeline

Synthetic experiments do not establish that a real adversary has been modelled correctly,
that a given tolerance is lawful or ethically acceptable, or that a protected release is
harmless. Real deployment requires held-out linkage outcomes, population-frequency
evidence, a declared harm analysis, legal review, and sign-off by the named owner. The
repository refuses to convert a failed model validation into a release decision.

## Required real-data extension

Implement an organisation-specific adapter outside this synthetic package. It must supply
de-identified key-class counts, held-out predicted match probabilities and outcomes,
external frequency benchmarks, and aligned attack-success indicators for every release. Do
not commit direct identifiers, auxiliary registers, secrets, or confidential row-level
data. The adapter should emit only the aggregate inputs that the validation and governance
modules consume.
