"""Synthetic adaptive-audit comparison for the focused journal draft.

Runs Bernoulli attack streams under known laws. This verifies operating
characteristics in a toy experiment, not privacy risk on a real release.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from privacy_risk.adaptive_audit import AdaptiveAudit


def simulate(seed: int = 2709, replications: int = 2000, cap: int = 300) -> dict:
    if replications < 1 or cap < 1:
        raise ValueError("replications and cap must be positive")
    rng = np.random.default_rng(seed)
    tau, alpha, m, q = .12, .05, 8, .03
    individual_level = alpha / m
    threshold = np.log(1 / individual_level)
    lp = np.log(q / tau)
    ln = np.log((1 - q) / (1 - tau))
    scenarios = {}
    for name, p in (("safe", .04), ("boundary", .12)):
        # Realizations are independent across trials and replication rows.
        draw = rng.binomial(1, p, (replications, cap)).astype(np.int16)
        cumulative = np.cumsum(np.where(draw == 1, lp, ln), axis=1)
        hit = cumulative >= threshold
        ever = hit.any(axis=1)
        first = hit.argmax(axis=1) + 1
        counts = draw.sum(axis=1)
        cp_upper = beta.ppf(1 - individual_level, counts + 1, cap - counts)
        cp_upper[counts == cap] = 1
        scenarios[name] = {
            "risk": p,
            "replications": replications,
            "anytime_pass_rate": round(float(ever.mean()), 5),
            "fixed_cp_pass_rate_at_cap": round(float((cp_upper <= tau).mean()), 5),
            "median_first_pass_if_pass": int(np.median(first[ever])) if ever.any() else None,
            "cap": cap,
        }

    # A second, separate null simulation registers an additional coordinate
    # only when a first safe stream has already passed. It runs through the
    # shipped gate (AdaptiveAudit), so it exercises the implementation's
    # registration after an earlier certification, on independent streams.
    registrations = 0
    false_passes = 0
    for _ in range(replications):
        gate = AdaptiveAudit(alpha)
        safe = gate.register("safe", tau, 1 / m)
        initial = rng.binomial(1, .04, cap)
        for outcome in initial.tolist():
            if safe.observe(int(outcome), q):
                break
        if not safe.certified:
            continue
        registrations += 1
        null = gate.register("null-after-safe", tau, 1 / m)
        new_null = rng.binomial(1, tau, cap)
        for outcome in new_null.tolist():
            if null.observe(int(outcome), q):
                break
        false_passes += int(null.certified)
    return {
        "seed": seed, "tolerance": tau, "family_alpha": alpha,
        "local_weight": 1 / m, "alternative": q,
        "scenario_results": scenarios,
        "adaptive_registration": {
            "runs": replications, "registrations": registrations,
            "null_crossings": false_passes,
            "conditional_rate": round(false_passes / registrations, 5) if registrations else None,
        },
        "scope": "Independent Bernoulli toy laws; not an evaluation on real releases",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2709)
    parser.add_argument("--replications", type=int, default=2000)
    parser.add_argument("--cap", type=int, default=300)
    args = parser.parse_args()
    result = simulate(args.seed, args.replications, args.cap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
