"""Recompute the numbers printed in Chapter 22 (run: PYTHONPATH=src python3 scripts/chapter22_evidence.py)."""
import math, json
import numpy as np
from scipy.stats import beta, binom
from privacy_risk.uncertainty import binomial_upper, simultaneous_upper_bounds, beta_posterior_upper
from privacy_risk.betting import run_audit, predictable_bets
from privacy_risk import planner as pl

out = {}
tau = 0.12
# credible vs exact after zero successes
def first(f):
    return next(n for n in range(1, 200) if f(n) <= tau)
out["zero_exact"] = first(lambda n: binomial_upper(0, n))
out["zero_uniform"] = first(lambda n: beta_posterior_upper(0, n, 1, 1))
out["zero_b19"] = first(lambda n: beta_posterior_upper(0, n, 1, 9))
out["surv_23"] = (1 - tau) ** 23
out["surv_15"] = (1 - tau) ** 15
out["prior_3_97_upper"] = beta_posterior_upper(0, 0, 3, 97)
# one success in 30
out["one_in_30"] = dict(exact=binomial_upper(1, 30), uniform=beta_posterior_upper(1, 30, 1, 1),
                        b19=beta_posterior_upper(1, 30, 1, 9), b397=beta_posterior_upper(1, 30, 3, 97),
                        exact_zero_in_40=binomial_upper(0, 40) , exact_1_in_40=binomial_upper(1, 40))
out["zero_in_40"] = binomial_upper(0, 40)
# simultaneous
out["zero_counts"] = {m: next(n for n in range(1, 300) if binomial_upper(0, n, 0.05 / m) <= tau) for m in (1, 2, 4, 8, 16)}
# rounding example
out["round"] = (round(150 * 0.05), round(150 * 0.047), 7 / 150)
# example 22.2
def ce(p, k, b): return (b * math.log(1 + math.exp(math.log(p / (1 - p)))) - k * math.log(p / (1 - p)))
N = 1010
out["ex22_2"] = dict(g1=(10 * .05 - 1) / N, g2=(1000 * .05 - 100) / N,
                     l05=(ce(.05, 1, 10) + ce(.05, 100, 1000)) / N, l10=(ce(.1, 1, 10) + ce(.1, 100, 1000)) / N)
# example 22.7
def cert_at(prior_a, prior_b, xs):
    o = [0] * 400
    for i in xs: o[i - 1] = 1
    r = run_audit(o, a=prior_a, b=prior_b)
    return r["attacks"] if r["certified"] else None
out["ex22_7"] = dict(forecast_clean=cert_at(.6, 19.4, []), default_clean=cert_at(1, 9, []),
                     f_s4=cert_at(.6, 19.4, [4]), d_s4=cert_at(1, 9, [4]),
                     f_s4_21=cert_at(.6, 19.4, [4, 21]), d_s4_21=cert_at(1, 9, [4, 21]))
bets = list(predictable_bets([0, 0, 0, 1, 0], tau, .6, 19.4))
out["bets"] = [round(b, 4) for b in bets]
w = 1.0
for x, q in zip([0, 0, 0, 1], bets[:4]): w *= q / tau if x else (1 - q) / (1 - tau)
out["wealth_after4"] = w
w3 = np.prod([(1 - q) / (1 - tau) for q in bets[:3]]); out["wealth_after3"] = w3
out["first_bet_factors"] = ((1 - .03) / .88, (1 - .1) / .88)
# betting simulations
rng = np.random.default_rng(2026)
def sim(p, n, cap=1500):
    c = 0; lens = []; clens = []
    for _ in range(n):
        r = run_audit((rng.random(cap) < p).astype(int), cap and tau, 0.05, cap)
        lens.append(r["attacks"]); c += r["certified"]
        if r["certified"]: clens.append(r["attacks"])
    return dict(cert=c / n, median=float(np.median(lens)), median_cert=float(np.median(clens)) if clens else None)
out["sim_false"] = sim(0.12, 1000)
for p in (0.03, 0.06, 0.09): out[f"sim_{p}"] = sim(p, 200)
out["fixed24"] = {p: (1 - p) ** 24 for p in (0.03, 0.06, 0.09)}
out["false_ci"] = float(binom.ppf(0.5, 1, 0.5))
# retry study
rng = np.random.default_rng(2026)
true, tol, B, L = 0.201, 0.2, 500, 20
def retry(share, trials=2000):
    hit = 0
    for _ in range(trials):
        for _ in range(L):
            k = rng.binomial(B, true)
            if binomial_upper(int(k), B, share) <= tol: hit += 1; break
    return hit / trials
out["retry_full"] = retry(0.05); out["retry_split"] = retry(0.05 / L)
# planner five seeds
res = []
for seed in range(5):
    r = np.random.default_rng(seed)
    x = r.uniform(-1, 1, (240, 4)); groups = np.repeat(np.arange(120), 2)
    true_logit = -2 + .7 * x[:, 0] - .5 * x[:, 1] + .9 * x[:, 2] * x[:, 3]
    pt = 1 / (1 + np.exp(-true_logit)); b = np.full(240, 100); k = r.binomial(b, pt)
    tr, ca, te = pl.grouped_split(groups, seed=seed)
    fl = pl.fit(x[tr], k[tr], b[tr], hidden=(), seed=seed); fn = pl.fit(x[tr], k[tr], b[tr], hidden=(12, 8), seed=seed)
    rec = {}
    for name, m in (("log", fl), ("net", fn)):
        z = pl.logits(m, x); T = pl.calibrate_temperature(z[ca], k[ca], b[ca])
        rec[name] = dict(ce=pl.binomial_cross_entropy(z[te], k[te], b[te]), T=T,
                         ce_cal=pl.binomial_cross_entropy(z[te] / T, k[te], b[te]), conv=m["converged"])
    rec["floor"] = pl.binomial_cross_entropy(true_logit[te], k[te], b[te])
    # true-probability floor: expected CE under truth
    rec["floor_exp"] = float((b[te] * (np.logaddexp(0, true_logit[te]) - pt[te] * true_logit[te])).sum() / b[te].sum())
    res.append(rec)
out["planner"] = res
print(json.dumps(out, indent=1, default=float))
