"""Choose a cell-count floor with the registered gate on the synthetic credit register.

The agency declares three outside adversaries before any audit: one release with one
exposure reading; the six vintages pooled with one reading and knowledge of the
loan's history (the vintages in which it is live); and the same with one reading per
live vintage. A floor is acceptable only if all three attacks meet the tolerance.

The main design is a release chain (``AdaptiveAudit.register_release_chain``): floors
are audited from the strictest down, the three coordinates of a floor are opened only
after the floor before it has passed, and the chain stops at the first floor that does
not pass. Each coordinate certifies at the full threshold 1/alpha. Choosing the last
passing floor is a fixed sequence of intersection-union tests, so the chance of
choosing a floor at which any declared attack succeeds at or above the tolerance is at
most alpha (the release-chain proposition of the article).

Five other designs receive the same draws:

* one chain per adversary at alpha/3, the familywise control over coordinates that
  lets the agency pick an adversary after reading the results;
* the gate with all coordinates registered at the start (Bonferroni over 39), stopped
  at the first floor that does not pass;
* fixed-sequence Clopper-Pearson intersection-union tests at level alpha, with 3,000
  and with 1,000 attacks per coordinate;
* Clopper-Pearson looks every 500 attacks, alpha split evenly over the six looks
  (Bonferroni over looks, a simple valid sequential design).

True risks come from separate draws of 100,000 targets. Validity is checked
separately on Bernoulli streams at those risks (``--oc-replications``).

    python scripts/adaptive_audit_chain.py

The register is synthetic. Nothing here is a confidential-data measurement.
"""
from __future__ import annotations

import argparse
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import beta

from privacy_risk.adaptive_audit import AdaptiveAudit
from privacy_risk.credit import make_credit_register
from privacy_risk.evaluation import Policy, programme_attack, released_rows

LEVELS = [300, 200, 150, 100, 75, 50, 40, 30, 20, 15, 10, 5, 1]
ADVERSARIES = {
    "single": (1, 1, True),
    "pooling_alone": (6, 1, True),
    "full_pooled": (6, "live", True),
}
METHODS = [("release_chain", "Gate, release chain at $\\alpha$"),
           ("adversary_chains", "Gate, one chain per attack at $\\alpha/3$"),
           ("frozen_gate", "Gate, all 39 registered at the start"),
           ("fixed_cp_cap", "Fixed sequence, 3{,}000 per coordinate"),
           ("fixed_cp_planned", "Fixed sequence, 1{,}000 per coordinate"),
           ("looks", "Looks every 500, Bonferroni over looks")]


def policy(k: int) -> Policy:
    return Policy("none" if k == 1 else f"k_anonymity_{k}", "microdata", k)


def cp_upper(successes: int, trials: int, level: float) -> float:
    if successes >= trials:
        return 1.0
    return float(beta.ppf(1 - level, successes + 1, trials - successes))


def bet(successes: int, trials: int, cap: float) -> float:
    """The shrunk running rate from earlier draws of the same coordinate, capped."""
    return float(min((successes + 0.5) / (trials + 1.0), cap))


def run_gate_stream(stream: list[int], tau: float, threshold: float, cap: int,
                    bet_cap: float) -> tuple[bool, int]:
    """One coordinate's e-process on a stream, at a given log threshold."""
    log_e, successes = 0.0, 0
    for n, h in enumerate(stream[:cap]):
        q = bet(successes, n, bet_cap)
        log_e += math.log(q / tau) if h else math.log((1 - q) / (1 - tau))
        successes += int(h)
        if log_e >= threshold:
            return True, n + 1
    return False, min(len(stream), cap)


def streams_for(task: tuple) -> dict:
    config, seed, adversary, k, n = task
    register = make_credit_register(config)
    sd = float(config["credit"].get("covariate_sd", 0.6))
    vintages, readings, history = ADVERSARIES[adversary]
    result = programme_attack(register, policy(k), sd, n, np.random.default_rng(int(seed)),
                              vintages=vintages, readings=readings,
                              target_history=history)
    return {"adversary": adversary, "k": k, "seed": int(seed),
            "hits": result["hits"].astype(int).tolist()}


def run_seed(hits: dict, alpha: float, tau: float, cap: int, bet_cap: float,
             planned: int) -> dict:
    out = {}
    names = list(ADVERSARIES)

    # 1. Release chain: IUT at each floor, fixed sequence over floors, full alpha.
    gate = AdaptiveAudit(alpha)
    chain = gate.register_release_chain("release", tau, 1.0, [str(k) for k in LEVELS],
                                        names)
    used, audited = 0, []
    level = chain.open_next()
    while level is not None:
        k = int(next(iter(level.values())).name.split(":")[1])
        passed = True
        for name in names:
            coord = level[name]
            for h in hits[(name, k)][:cap]:
                if coord.observe(int(h), bet(coord.successes, coord.trials, bet_cap)):
                    break
            used += coord.trials
            if not coord.certified:
                passed = False
                break  # the floor has failed; the other attacks need not be audited
        audited.append({"k": k, "passed": passed})
        if not passed:
            break
        level = chain.open_next()
    floor = chain.floor()
    out["release_chain"] = {"floor": int(floor) if floor else None, "attacks": used,
                            "audited": audited}

    # 2. One chain per adversary at alpha/3; the floor is the strictest of the three.
    total, floors = 0, []
    for name in names:
        threshold = -math.log(alpha / len(names))
        floor_a = None
        for k in LEVELS:
            ok, n = run_gate_stream(hits[(name, k)], tau, threshold, cap, bet_cap)
            total += n
            if not ok:
                break
            floor_a = k
        floors.append(floor_a)
    out["adversary_chains"] = {"floor": None if None in floors else max(floors),
                               "attacks": total,
                               "per_adversary": dict(zip(names, floors))}

    # 3. All coordinates registered at the start, Bonferroni, stopped at first failure.
    threshold = -math.log(alpha / (len(names) * len(LEVELS)))
    total, floor = 0, None
    for k in LEVELS:
        passed = True
        for name in names:
            ok, n = run_gate_stream(hits[(name, k)], tau, threshold, cap, bet_cap)
            total += n
            if not ok:
                passed = False
                break
        if not passed:
            break
        floor = k
    out["frozen_gate"] = {"floor": floor, "attacks": total}

    # 4 and 5. Fixed-sequence Clopper-Pearson IUT at level alpha.
    for label, n in (("fixed_cp_cap", cap), ("fixed_cp_planned", planned)):
        total, floor = 0, None
        for k in LEVELS:
            passed = True
            for name in names:
                draws = hits[(name, k)][:n]
                total += n
                if cp_upper(int(sum(draws)), n, alpha) > tau:
                    passed = False
                    break
            if not passed:
                break
            floor = k
        out[label] = {"floor": floor, "attacks": total}

    # 6. Looks every 500 with alpha split over the looks, IUT, fixed sequence.
    looks = list(range(500, cap + 1, 500))
    total, floor = 0, None
    for k in LEVELS:
        passed = True
        for name in names:
            draws = hits[(name, k)]
            ok, stop = False, cap
            for n in looks:
                if cp_upper(int(sum(draws[:n])), n, alpha / len(looks)) <= tau:
                    ok, stop = True, n
                    break
            total += stop
            if not ok:
                passed = False
                break
        if not passed:
            break
        floor = k
    out["looks"] = {"floor": floor, "attacks": total}
    return out


def operating_characteristics(truth: dict, alpha: float, tau: float, cap: int,
                              bet_cap: float, replications: int, seed: int) -> dict:
    """The release chain on Bernoulli streams at the estimated risks.

    Given the register and the floor, attack outcomes on fresh targets are
    independent Bernoulli draws at the coordinate's risk, so this reproduces the
    chain's error and power without rerunning the attack. The risks are also shifted
    up, to show how the chance of an unsafe choice behaves when every coordinate is
    closer to the tolerance than estimated.
    """
    rng = np.random.default_rng(seed)
    names = list(ADVERSARIES)
    threshold = -math.log(alpha)
    out = {}
    for shift in (0.0, 0.01, 0.02):
        floors, unsafe = [], 0
        for _ in range(replications):
            floor = None
            for k in LEVELS:
                passed = True
                for name in names:
                    p = min(truth[name][str(k)]["rate"] + shift, 1.0)
                    stream = (rng.random(cap) < p).astype(int).tolist()
                    ok, _ = run_gate_stream(stream, tau, threshold, cap, bet_cap)
                    if not ok:
                        passed = False
                        break
                if not passed:
                    break
                floor = k
            floors.append(floor)
            if floor is not None and any(
                    min(truth[n][str(k)]["rate"] + shift, 1.0) >= tau
                    for n in names for k in LEVELS if k >= floor):
                unsafe += 1
        values = [f if f is not None else 10_000 for f in floors]
        uniq, counts = np.unique(values, return_counts=True)
        out[f"{shift:g}"] = {"replications": replications,
                             "unsafe_choices": unsafe,
                             "unsafe_rate": unsafe / replications,
                             "floors": {str(int(u)) if u < 10_000 else "none": int(c)
                                        for u, c in zip(uniq, counts)}}
    return out


def floor_text(floors: list) -> str:
    values = ["none" if f is None else f for f in floors]
    unique = sorted(set(values), key=lambda v: (v == "none", v if v != "none" else 0))
    return ", ".join(f"{v} ({values.count(v)})" for v in unique)


def thousands(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def write_table(payload: dict, path: Path) -> None:
    truth_row = ("Estimated true floor & " + str(payload["true_floor"]) + " & -- \\\\\n"
                 "\\midrule\n")
    body = ""
    for method, label in METHODS:
        rows = payload["summary"][method]
        body += (f"{label} & {floor_text(rows['floors'])} & "
                 f"{thousands(rows['median_attacks'])} \\\\\n")
    caption = (
        "Choosing the cell-count floor on the synthetic credit register over "
        f"{payload['seeds']} attack streams, tolerance {payload['tolerance']:.2f}, "
        f"$\\alpha={payload['alpha']:.2f}$. The agency declares three attacks before the "
        "audit (one release with one reading; six releases with one reading and the "
        "loan's history; six releases with a reading per live vintage and the history), "
        "and a floor passes only when all three meet the tolerance. Each design audits "
        "the floors from 300 down and stops at the first that fails. Entries give each "
        "floor chosen with the number of streams in brackets, and the median total "
        "number of attacks. The estimated true floor uses 100{,}000 separate targets "
        "per coordinate. All designs see the same draws.")
    text = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
            "\\label{tab:chain-audit}\n\\small\n\\begin{tabular}{lrr}\n\\toprule\n"
            "Design & Floor chosen & Attacks \\\\\n\\midrule\n" + truth_row + body
            + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", default="paper/publication_versions/chain_audit.json")
    parser.add_argument("--table",
                        default="paper/publication_versions/table_chain_audit.tex")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--cap", type=int, default=3000)
    parser.add_argument("--planned", type=int, default=1000)
    parser.add_argument("--truth", type=int, default=100_000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=0.12)
    parser.add_argument("--bet-cap", type=float, default=0.11)
    parser.add_argument("--oc-replications", type=int, default=400)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--table-only", action="store_true")
    args = parser.parse_args()
    if args.table_only:
        write_table(json.loads(Path(args.output).read_text()), Path(args.table))
        return

    config = json.loads(Path(args.config).read_text())
    base = int(config["seed"]) + 3_000_000
    register = make_credit_register(config)
    names = list(ADVERSARIES)
    # truth streams live far from the audit streams, so no seed is shared
    truth_tasks = [(config, base + 90_000_000 + 1_000 * c + k, name, k, args.truth)
                   for c, name in enumerate(names) for k in LEVELS]
    seed_tasks = [(config, base + 100_000 * s + 1_000 * c + k, name, k, args.cap)
                  for s in range(args.seeds) for c, name in enumerate(names)
                  for k in LEVELS]
    with Pool(args.workers) as pool:
        truth_rows = pool.map(streams_for, truth_tasks)
        seed_rows = pool.map(streams_for, seed_tasks)

    truth = {}
    for row in truth_rows:
        n, s = len(row["hits"]), int(sum(row["hits"]))
        truth.setdefault(row["adversary"], {})[str(row["k"])] = {
            "rate": s / n, "trials": n,
            "lower": float(beta.ppf(0.025, s, n - s + 1)) if s else 0.0,
            "upper": float(beta.ppf(0.975, s + 1, n - s))}

    # the estimated true floor: the last level from the top at which every attack's
    # rate is below the tolerance; levels whose 95% interval straddles it are listed
    true_floor, unresolved = None, []
    for k in LEVELS:
        rates = [truth[n][str(k)] for n in names]
        if any(r["lower"] <= args.tolerance <= r["upper"] for r in rates):
            unresolved.append(k)
        if max(r["rate"] for r in rates) >= args.tolerance:
            break
        true_floor = k

    per_seed = []
    for s in range(args.seeds):
        lo = base + 100_000 * s
        hits = {(row["adversary"], row["k"]): row["hits"] for row in seed_rows
                if lo <= row["seed"] < lo + 100_000}
        per_seed.append(run_seed(hits, args.alpha, args.tolerance, args.cap,
                                 args.bet_cap, args.planned))

    def unsafe(floor: int | None) -> bool:
        return floor is not None and any(truth[n][str(k)]["rate"] >= args.tolerance
                                         for n in names for k in LEVELS if k >= floor)

    summary = {}
    for method, _ in METHODS:
        floors = [r[method]["floor"] for r in per_seed]
        attacks = [r[method]["attacks"] for r in per_seed]
        summary[method] = {"floors": floors,
                           "at_true_floor": sum(f == true_floor for f in floors),
                           "unsafe_choices": sum(unsafe(f) for f in floors),
                           "median_attacks": float(np.median(attacks)),
                           "mean_attacks": float(np.mean(attacks))}
    per_adversary = {name: [r["adversary_chains"]["per_adversary"][name]
                            for r in per_seed] for name in names}

    oc = operating_characteristics(truth, args.alpha, args.tolerance, args.cap,
                                   args.bet_cap, args.oc_replications,
                                   int(config["seed"]) + 4_000_000)

    released = {str(k): int(released_rows(register, policy(k), 0).size) for k in LEVELS}
    payload = {"alpha": args.alpha, "tolerance": args.tolerance, "cap": args.cap,
               "planned": args.planned, "bet_cap": args.bet_cap, "seeds": args.seeds,
               "levels": LEVELS, "released": released, "truth": truth,
               "true_floor": true_floor, "unresolved_levels": unresolved,
               "summary": summary, "per_adversary_floors": per_adversary,
               "operating_characteristics": oc, "per_seed": per_seed}
    Path(args.output).write_text(json.dumps(payload, indent=1))
    write_table(payload, Path(args.table))
    print(json.dumps({"true_floor": true_floor, "unresolved": unresolved,
                      "summary": {m: (v["floors"], v["unsafe_choices"],
                                      v["median_attacks"]) for m, v in summary.items()},
                      "oc": oc}, indent=1))


if __name__ == "__main__":
    main()
