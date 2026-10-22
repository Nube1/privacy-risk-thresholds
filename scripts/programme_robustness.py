"""What drives the programme floor on the synthetic credit register.

The long-form study chose the minimum cell count for the programme against an
attack that pools six vintages, holds one exposure reading per live vintage and
knows in which vintages the target's loan is live. This script separates those
pieces of knowledge, chooses the floor by a valid selection rule, confirms it on
fresh targets, and repeats the choice over register draws and reading noise.

Nothing here reads confidential data. Run from the repository root:

    python scripts/programme_robustness.py --output artifacts-robustness/robustness.json
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import beta

from privacy_risk.credit import make_credit_register
from privacy_risk.evaluation import Policy, programme_attack, relink_share, released_rows

ALPHA = 0.05
TOLERANCE = 0.12
LEVELS = [1, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300]
REPLICATION_LEVELS = [1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300]
TOLERANCES = [0.05, 0.09, 0.12, 0.20]
INSIDER_SD = 0.01
VARIANTS = {
    # label: (vintages, readings, knows the target's live vintages,
    #         reading SD (None = the declared outsider SD), exposure digits)
    "single_one": (1, 1, True, None, None),
    "pooled_one_no_history": (6, 1, False, None, None),
    "pooled_one_history": (6, 1, True, None, None),
    "pooled_live_no_history": (6, "live", False, None, None),
    "pooled_live_history": (6, "live", True, None, None),
    "insider_pooled": (6, 1, True, INSIDER_SD, None),
    "single_one_rounded": (1, 1, True, None, 1),
    "pooled_one_history_rounded": (6, 1, True, None, 1),
    "pooled_live_history_rounded": (6, "live", True, None, 1),
    "insider_pooled_rounded": (6, 1, True, INSIDER_SD, 1),
}
REPLICATION_VARIANTS = ["single_one", "pooled_one_history", "pooled_live_history"]


def upper(successes: int, trials: int, level: float) -> float:
    """Exact one-sided Clopper-Pearson upper bound."""
    if trials == 0:
        return 1.0
    if successes >= trials:
        return 1.0
    return float(beta.ppf(1 - level, successes + 1, trials - successes))


def policy(k: int) -> Policy:
    return Policy("none" if k == 1 else f"k_anonymity_{k}", "microdata", k)


def summarise(result: dict, classes: int) -> dict:
    hits, size = result["hits"], result["size_class"]
    n = int(hits.size)
    out = {"trials": n, "successes": int(hits.sum()),
           "rate": float(hits.mean()) if n else float("nan"),
           "upper": upper(int(hits.sum()), n, ALPHA), "released": result["released"],
           "by_class": {}}
    for c in range(classes):
        mask = size == c
        m = int(mask.sum())
        s = int(hits[mask].sum())
        out["by_class"][str(c)] = {
            "trials": m, "successes": s,
            "rate": s / m if m else float("nan"),
            # every class must pass for the release to pass: an intersection-union
            # test, valid at level alpha with no correction
            "upper_iut": upper(s, m, ALPHA),
            # simultaneous bounds, needed only to report the worst class as a bound
            "upper_bonferroni": upper(s, m, ALPHA / classes)}
    classes_seen = [v for v in out["by_class"].values() if v["trials"]]
    out["worst_rate"] = max(v["rate"] for v in classes_seen)
    out["worst_upper_iut"] = max(v["upper_iut"] for v in classes_seen)
    out["worst_upper_bonferroni"] = max(v["upper_bonferroni"] for v in classes_seen)
    return out


def smallest_passing(levels: list[int], passed: dict) -> int | None:
    for k in sorted(levels):
        if passed[k]:
            return k
    return None


def fixed_sequence(levels: list[int], passed: dict) -> int | None:
    """Test from the strictest level down and stop at the first failure.

    Each level is tested at the full level alpha; the procedure controls the chance
    of choosing an unsafe level at alpha whatever the true risks, because the first
    unsafe level reached must be falsely passed for any unsafe level to be chosen
    (Maurer, Hothorn and Lehmacher 1995).
    """
    chosen = None
    for k in sorted(levels, reverse=True):
        if not passed[k]:
            break
        chosen = k
    return chosen


def grid(register, covariate_sd: float, replicates: int, seed: int, levels: list[int],
         variants: list[str], draw: int) -> dict:
    out: dict = {}
    for v_index, name in enumerate(variants):
        vintages, readings, history, sd, digits = VARIANTS[name]
        out[name] = {}
        for k in levels:
            stream = seed + 900_000 + 10_000 * draw + 1_000 * v_index + k
            result = programme_attack(register, policy(k), sd or covariate_sd,
                                      replicates, np.random.default_rng(stream),
                                      vintages=vintages, readings=readings,
                                      target_history=history, exposure_digits=digits)
            out[name][str(k)] = summarise(result, register.n_size_classes)
    return out


def floors(table: dict, levels: list[int]) -> dict:
    out = {}
    for name, rows in table.items():
        average = {k: rows[str(k)]["upper"] <= TOLERANCE for k in levels}
        subgroup = {k: rows[str(k)]["worst_upper_iut"] <= TOLERANCE for k in levels}
        out[name] = {"smallest_passing": smallest_passing(levels, average),
                     "fixed_sequence": fixed_sequence(levels, average),
                     "fixed_sequence_subgroup": fixed_sequence(levels, subgroup)}
    return out


def replicate(task: tuple) -> dict:
    config, register_seed, covariate_sd, replicates = task
    local = copy.deepcopy(config)
    local["seed"] = int(register_seed)
    register = make_credit_register(local)
    table = grid(register, covariate_sd, replicates, int(register_seed),
                 REPLICATION_LEVELS, REPLICATION_VARIANTS, draw=7)
    return {"register_seed": int(register_seed), "covariate_sd": covariate_sd,
            "floors": floors(table, REPLICATION_LEVELS),
            "rates": {name: {k: rows[k]["rate"] for k in rows}
                      for name, rows in table.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", default="artifacts-robustness/robustness.json")
    parser.add_argument("--replicates", type=int, default=4000)
    parser.add_argument("--replication-replicates", type=int, default=2000)
    parser.add_argument("--registers", type=int, default=20)
    parser.add_argument("--sds", default="0.3,0.6,1.0")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    seed = int(config["seed"])
    covariate_sd = float(config["credit"].get("covariate_sd", 0.6))
    register = make_credit_register(config)
    names = list(VARIANTS)

    selection = grid(register, covariate_sd, args.replicates, seed, LEVELS, names, draw=1)
    chosen = floors(selection, LEVELS)
    confirmation = grid(register, covariate_sd, args.replicates, seed, LEVELS, names,
                        draw=2)
    confirmed = {}
    for name, rule in chosen.items():
        k = rule["fixed_sequence"]
        row = confirmation[name][str(k)] if k is not None else None
        confirmed[name] = {"level": k,
                           "rate": row["rate"] if row else None,
                           "upper": row["upper"] if row else None,
                           "passes": bool(row and row["upper"] <= TOLERANCE)}

    # Subgroup control: every size class must pass. Uniform target draws leave the
    # largest class with a few hundred targets, so it is audited on its own
    # stratified draw; the other classes use the uniform draws above.
    largest = register.n_size_classes - 1
    stratified = {}
    for v_index, name in enumerate(["single_one", "pooled_one_history",
                                    "pooled_live_history"]):
        vintages, readings, history, sd, digits = VARIANTS[name]
        rows = {}
        for k in LEVELS:
            result = programme_attack(register, policy(k), sd or covariate_sd,
                                      args.replicates,
                                      np.random.default_rng(seed + 950_000
                                                            + 1_000 * v_index + k),
                                      vintages=vintages, readings=readings,
                                      target_history=history, target_class=largest,
                                      exposure_digits=digits)
            n, s_ = int(result["hits"].size), int(result["hits"].sum())
            others = max(selection[name][str(k)]["by_class"][str(c)]["upper_iut"]
                         for c in range(largest))
            rows[str(k)] = {"trials": n, "successes": s_, "rate": s_ / n if n else None,
                            "upper": upper(s_, n, ALPHA), "other_classes_upper": others}
        # a class with no released units has no target to attack, so it passes
        passes = {k: max(rows[str(k)]["upper"] if rows[str(k)]["trials"] else 0.0,
                         rows[str(k)]["other_classes_upper"]) <= TOLERANCE
                  for k in LEVELS}
        stratified[name] = {"rows": rows, "fixed_sequence": fixed_sequence(LEVELS, passes)}

    relink = {str(k): relink_share(register, policy(k)) for k in LEVELS}
    relink_rounded = {str(k): relink_share(register, policy(k), 1) for k in LEVELS}
    released = {str(k): int(released_rows(register, policy(k), 0).size) for k in LEVELS}
    # what a floor costs, in the terms a central bank uses
    first = np.flatnonzero(register.live[:, 0])
    total_exposure = float(register.exposure[first].sum())
    large = register.n_size_classes - 1
    multi = np.bincount(register.borrower[first], minlength=register.n_borrowers) > 1
    coverage = {}
    for k in LEVELS:
        rows_k = released_rows(register, policy(k), 0)
        kept_borrowers = np.bincount(register.borrower[rows_k],
                                     minlength=register.n_borrowers)
        coverage[str(k)] = {
            "loans": int(rows_k.size), "loan_share": rows_k.size / first.size,
            "exposure_share": float(register.exposure[rows_k].sum()) / total_exposure,
            "large_firm_loan_share": (float((register.size_class[rows_k] == large).sum())
                                      / float((register.size_class[first] == large)
                                              .sum())),
            "multi_lender_borrowers_kept": int(((kept_borrowers > 1) & multi).sum()),
            "multi_lender_borrowers": int(multi.sum())}
    tolerance_floors = {}
    for tau in TOLERANCES:
        tolerance_floors[f"{tau:g}"] = {
            name: fixed_sequence(LEVELS, {k: rows[str(k)]["upper"] <= tau
                                          for k in LEVELS})
            for name, rows in selection.items()}

    sds = [float(s) for s in args.sds.split(",")]
    tasks = [(config, seed + 1 + r, sd, args.replication_replicates)
             for r in range(args.registers) for sd in sds]
    with Pool(args.workers) as pool:
        replications = pool.map(replicate, tasks)

    summary = {}
    for sd in sds:
        runs = [r for r in replications if r["covariate_sd"] == sd]
        summary[f"{sd:g}"] = {}
        for name in REPLICATION_VARIANTS:
            picks = [r["floors"][name]["fixed_sequence"] for r in runs]
            numeric = [p if p is not None else 10_000 for p in picks]
            rates = {k: [r["rates"][name][str(k)] for r in runs]
                     for k in REPLICATION_LEVELS}
            summary[f"{sd:g}"][name] = {
                "floors": picks,
                "median_floor": float(np.median(numeric)),
                "min_floor": min(numeric), "max_floor": max(numeric),
                "none_passed": sum(p is None for p in picks),
                "between_register_sd": {str(k): float(np.std(v, ddof=1))
                                        for k, v in rates.items()},
                "mean_rate": {str(k): float(np.mean(v)) for k, v in rates.items()}}

    payload = {"alpha": ALPHA, "tolerance": TOLERANCE, "levels": LEVELS,
               "replicates": args.replicates, "variants": {k: list(map(str, v))
                                                           for k, v in VARIANTS.items()},
               "released": released, "relink_share": relink,
               "relink_share_rounded_1_digit": relink_rounded, "coverage": coverage,
               "tolerance_floors": tolerance_floors, "insider_sd": INSIDER_SD,
               "selection": selection, "floors": chosen, "confirmation": confirmed,
               "subgroup_stratified": stratified, "confirmation_grid": confirmation,
               "replication": {"registers": args.registers, "sds": sds,
                               "replicates": args.replication_replicates,
                               "levels": REPLICATION_LEVELS, "summary": summary,
                               "runs": replications}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=1))
    print(json.dumps({"floors": chosen, "confirmation": confirmed,
                      "subgroup": {n: v["fixed_sequence"] for n, v in stratified.items()},
                      "replication": {sd: {n: (v["median_floor"], v["min_floor"],
                                               v["max_floor"])
                                           for n, v in s.items()}
                                      for sd, s in summary.items()}}, indent=1))


if __name__ == "__main__":
    main()
