"""The neural release pipeline: a network proposes, the registered gate decides.

This is the package's main entry point for choosing a release setting on the
synthetic credit register:

    python -m privacy_risk.neural_pipeline --output artifacts-neural

It runs four stages, and keeps their evidence apart.

1. **Development attacks.** The declared attacks are run on a dense grid of
   cell-count floors, with a small number of targets per floor and seeds that the
   final audit never uses. These counts are the only training data.
2. **Network.** A dense ReLU network (``sentry_r.DeepRiskNet``) is fitted to the
   counts by the binomial likelihood, one output per declared attack, with floors
   held out for early stopping. A logistic model (the same network with no hidden
   layer) is fitted to the same counts as a baseline.
3. **Proposal.** Before any final outcome is drawn, the fitted network predicts the
   success probability of every declared attack at every floor of the release grid.
   The predictions supply the bets of the gate and the floor at which the release
   chain starts, and both are written to a frozen plan with its digest.
4. **Final audit.** The release chain of ``AdaptiveAudit.register_release_chain``
   audits the plan on fresh targets. Only these outcomes can certify a floor. If the
   chain that starts at the proposed floor fails at its first floor, a reserve share
   of the error budget runs a second chain from the strictest floor.

The network's accuracy does not affect the error bound, because every bet is fixed
before its outcome and the chain order is fixed at registration. A poor network
costs attacks, not validity. The register is synthetic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .adaptive_audit import AdaptiveAudit
from .credit import CreditRegister, make_credit_register
from .evaluation import Policy, programme_attack, released_rows
from .sentry_r import DeepRiskNet

LEVELS = [300, 200, 150, 100, 75, 50, 40, 30, 20, 15, 10, 5, 1]
DEVELOPMENT_LEVELS = [1, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 50, 60, 75,
                      90, 100, 125, 150, 175, 200, 250, 300]
ADVERSARIES = {
    "single": (1, 1, True),
    "pooling_alone": (6, 1, True),
    "full_pooled": (6, "live", True),
}
FEATURES = ["log floor", "share of loans kept", "log mean released cell size",
            "share of released loans not live in every vintage"]


def policy(k: int) -> Policy:
    return Policy("none" if k == 1 else f"k_anonymity_{k}", "microdata", k)


def floor_features(register: CreditRegister, k: int) -> list[float]:
    """Features of a floor that the agency can compute without running an attack."""
    first = np.flatnonzero(register.live[:, 0])
    rows = released_rows(register, policy(k), 0)
    if rows.size == 0:
        return [math.log(k), 0.0, 0.0, 0.0]
    key = register.key()[rows]
    _, counts = np.unique(key, return_counts=True)
    not_always = float(np.mean(~register.live[rows].all(axis=1)))
    return [math.log(k), rows.size / first.size, math.log(float(np.mean(counts))),
            not_always]


def attack_hits(task: tuple) -> dict:
    config, seed, adversary, k, n = task
    register = make_credit_register(config)
    sd = float(config["credit"].get("covariate_sd", 0.6))
    vintages, readings, history = ADVERSARIES[adversary]
    result = programme_attack(register, policy(k), sd, n, np.random.default_rng(int(seed)),
                              vintages=vintages, readings=readings,
                              target_history=history)
    return {"adversary": adversary, "k": k, "seed": int(seed),
            "hits": result["hits"].astype(int).tolist()}


def fit(x: np.ndarray, k: np.ndarray, b: np.ndarray, train: np.ndarray,
        valid: np.ndarray, depth: int, width: int, seed: int) -> tuple:
    mean = x[train].mean(axis=0)
    scale = np.where(x[train].std(axis=0) < 1e-9, 1.0, x[train].std(axis=0))
    z = (x - mean) / scale
    net = DeepRiskNet(x.shape[1], k.shape[1], width=width, depth=depth, seed=seed)
    metrics = net.fit((z[train], k[train], b[train]), (z[valid], k[valid], b[valid]),
                      epochs=4000, batch_size=16, lr=0.01, patience=300, l2=1e-4,
                      seed=seed)
    return net, mean, scale, metrics


def bet_from(prediction: float, tau: float, cap: float) -> float:
    return float(min(max(prediction, 1e-4), cap, tau - 1e-6))


def plug_in(successes: int, trials: int, cap: float) -> float:
    return float(min((successes + 0.5) / (trials + 1.0), cap))


def release_chain(gate: AdaptiveAudit, name: str, hits: dict, levels: list[int],
                  names: list[str], tau: float, weight: float, cap: int,
                  bets: dict | None, bet_cap: float) -> dict:
    """Run one release chain; bets come from the plan, or from the running rate."""
    chain = gate.register_release_chain(name, tau, weight, [str(k) for k in levels], names)
    used, audited = 0, []
    level = chain.open_next()
    while level is not None:
        k = int(next(iter(level.values())).name.split(":")[1])
        passed = True
        for adversary in names:
            coord = level[adversary]
            for h in hits[(adversary, k)][:cap]:
                q = (bets[(adversary, k)] if bets is not None
                     else plug_in(coord.successes, coord.trials, bet_cap))
                if coord.observe(int(h), q):
                    break
            used += coord.trials
            if not coord.certified:
                passed = False
                break
        audited.append({"k": k, "passed": passed})
        if not passed:
            break
        level = chain.open_next()
    floor = chain.floor()
    return {"floor": int(floor) if floor else None, "attacks": used, "audited": audited}


def run(config: dict, output: Path, seeds: int = 20, development_targets: int = 400,
        cap: int = 3000, alpha: float = 0.05, tau: float = 0.12, bet_cap: float = 0.11,
        reserve: float = 0.2, workers: int = 2, truth: dict | None = None) -> dict:
    register = make_credit_register(config)
    names = list(ADVERSARIES)
    base = int(config["seed"]) + 3_000_000

    # 1. development attacks, on seeds the final audit never uses
    dev_tasks = [(config, base + 200_000_000 + 1_000 * c + k, a, k, development_targets)
                 for c, a in enumerate(names) for k in DEVELOPMENT_LEVELS]
    final_tasks = [(config, base + 100_000 * s + 1_000 * c + k, a, k, cap)
                   for s in range(seeds) for c, a in enumerate(names) for k in LEVELS]
    with Pool(workers) as pool:
        dev_rows = pool.map(attack_hits, dev_tasks)
        final_rows = pool.map(attack_hits, final_tasks)
    x = np.array([floor_features(register, k) for k in DEVELOPMENT_LEVELS])
    counts = {(r["adversary"], r["k"]): int(sum(r["hits"])) for r in dev_rows}
    kk = np.array([[counts[(a, k)] for a in names] for k in DEVELOPMENT_LEVELS], float)
    bb = np.full_like(kk, float(development_targets))
    positions = np.arange(len(DEVELOPMENT_LEVELS))
    train, valid = positions[positions % 3 != 1], positions[positions % 3 == 1]

    # 2. network and logistic baseline, fitted to development counts only
    models = {}
    for label, depth, width in (("network", 2, 16), ("logistic", 0, 1)):
        net, mean, scale, metrics = fit(x, kk, bb, train, valid, depth, width, seed=19)
        models[label] = {"net": net, "mean": mean, "scale": scale, "metrics": metrics}

    # 3. frozen plan: predictions, bets and start floor for every declared attack
    xl = np.array([floor_features(register, k) for k in LEVELS])
    plans = {}
    for label, m in models.items():
        pred = m["net"].predict((xl - m["mean"]) / m["scale"])
        predictions = {(a, k): float(pred[i, j]) for i, k in enumerate(LEVELS)
                       for j, a in enumerate(names)}
        predicted_floor = None
        for k in LEVELS:
            if max(predictions[(a, k)] for a in names) >= tau:
                break
            predicted_floor = k
        # start one level stricter than the predicted floor, as a margin
        if predicted_floor is None:
            start = LEVELS[0]
        else:
            idx = LEVELS.index(predicted_floor)
            start = LEVELS[max(idx - 1, 0)]
        plan = {"predictions": {f"{a}:{k}": v for (a, k), v in predictions.items()},
                "bets": {f"{a}:{k}": bet_from(v, tau, bet_cap)
                         for (a, k), v in predictions.items()},
                "predicted_floor": predicted_floor, "start_floor": start}
        plan["sha256"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()
                                        ).hexdigest()
        plans[label] = plan

    # 4. final audit on fresh targets, the same draws for every design
    designs = {}
    per_seed = []
    for s in range(seeds):
        lo = base + 100_000 * s
        hits = {(r["adversary"], r["k"]): r["hits"] for r in final_rows
                if lo <= r["seed"] < lo + 100_000}
        row = {}
        gate = AdaptiveAudit(alpha)
        row["plug_in"] = release_chain(gate, "plugin", hits, LEVELS, names, tau, 1.0,
                                       cap, None, bet_cap)
        for label in ("network", "logistic"):
            plan = plans[label]
            bets = {(a, k): plan["bets"][f"{a}:{k}"] for a in names for k in LEVELS}
            gate = AdaptiveAudit(alpha)
            row[f"{label}_bets"] = release_chain(gate, label, hits, LEVELS, names, tau,
                                                 1.0, cap, bets, bet_cap)
            gate = AdaptiveAudit(alpha)
            start = LEVELS.index(plan["start_floor"])
            first = release_chain(gate, f"{label}start", hits, LEVELS[start:], names,
                                  tau, 1.0 - reserve, cap, bets, bet_cap)
            if first["floor"] is None:
                second = release_chain(gate, f"{label}reserve", hits, LEVELS, names,
                                       tau, reserve, cap, None, bet_cap)
                first = {"floor": second["floor"],
                         "attacks": first["attacks"] + second["attacks"],
                         "audited": first["audited"] + second["audited"],
                         "used_reserve": True}
            else:
                first["used_reserve"] = False
            row[f"{label}_plan"] = first
        per_seed.append(row)

    names_d = ["plug_in", "network_bets", "logistic_bets", "network_plan", "logistic_plan"]
    unsafe_levels = set()
    if truth:
        for k in LEVELS:
            if any(truth[a][str(k)]["rate"] >= tau for a in names):
                unsafe_levels.add(k)

    def unsafe(floor: int | None) -> bool:
        return floor is not None and any(k in unsafe_levels for k in LEVELS if k >= floor)

    for d in names_d:
        floors = [r[d]["floor"] for r in per_seed]
        attacks = [r[d]["attacks"] for r in per_seed]
        designs[d] = {"floors": floors, "median_attacks": float(np.median(attacks)),
                      "unsafe_choices": sum(unsafe(f) for f in floors) if truth else None,
                      "reserve_used": sum(bool(r[d].get("used_reserve")) for r in per_seed)}

    calibration = {}
    if truth:
        for label, plan in plans.items():
            errors = [abs(plan["predictions"][f"{a}:{k}"] - truth[a][str(k)]["rate"])
                      for a in names for k in LEVELS]
            calibration[label] = {"mean_absolute_error": float(np.mean(errors)),
                                  "max_absolute_error": float(np.max(errors))}
        for label, plan in plans.items():
            calibration[label]["predicted_floor"] = plan["predicted_floor"]

    payload = {
        "method": "neural release pipeline", "alpha": alpha, "tolerance": tau,
        "cap": cap, "bet_cap": bet_cap, "reserve": reserve, "seeds": seeds,
        "development": {"levels": DEVELOPMENT_LEVELS, "targets": development_targets,
                        "features": FEATURES,
                        "train_levels": [DEVELOPMENT_LEVELS[i] for i in train],
                        "validation_levels": [DEVELOPMENT_LEVELS[i] for i in valid],
                        "counts": {f"{a}:{k}": counts[(a, k)] for a in names
                                   for k in DEVELOPMENT_LEVELS}},
        "models": {label: {"metrics": m["metrics"], "network": m["net"].to_dict(),
                           "mean": m["mean"].tolist(), "scale": m["scale"].tolist()}
                   for label, m in models.items()},
        "plans": plans, "designs": designs, "calibration": calibration,
        "per_seed": per_seed,
        "scope": ("Synthetic register. The network proposes bets and a start floor; "
                  "only fresh audit outcomes certify.")}
    output.mkdir(parents=True, exist_ok=True)
    (output / "neural_pipeline.json").write_text(json.dumps(payload, indent=1))
    return payload


def write_table(payload: dict, path: Path) -> None:
    labels = [("plug_in", "Running-rate bets, chain from 300"),
              ("logistic_bets", "Logistic bets, chain from 300"),
              ("network_bets", "Network bets, chain from 300"),
              ("logistic_plan", "Logistic bets and start floor"),
              ("network_plan", "Network bets and start floor")]

    def floors(values: list) -> str:
        vals = ["none" if v is None else v for v in values]
        uniq = sorted(set(vals), key=lambda v: (v == "none", v if v != "none" else 0))
        return ", ".join(f"{v} ({vals.count(v)})" for v in uniq)

    body = "".join(
        f"{label} & {floors(payload['designs'][d]['floors'])} & "
        f"{payload['designs'][d]['median_attacks']:,.0f}".replace(",", "{,}")
        .replace("({,}", "(").replace("{,} ", ", ") + " \\\\\n"
        for d, label in labels)
    caption = (
        "The neural release pipeline on the synthetic credit register, "
        f"{payload['seeds']} attack streams, tolerance {payload['tolerance']:.2f}, "
        f"$\\alpha={payload['alpha']:.2f}$. Every design is a release chain over the "
        "three declared attacks on the same final draws; they differ only in where the "
        "bets and the start floor come from. The network and the logistic model are "
        "fitted to development attacks on separate seeds and fixed before the final "
        "audit. Entries give each floor chosen with the number of streams in brackets, "
        "and the median number of attacks.")
    path.write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
        "\\label{tab:neural-pipeline}\n\\small\n\\begin{tabular}{lrr}\n\\toprule\n"
        "Source of bets and start & Floor chosen & Attacks \\\\\n\\midrule\n" + body
        + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", default="artifacts-neural")
    parser.add_argument("--truth", default="paper/publication_versions/chain_audit.json",
                        help="estimated true risks, used only to score the designs")
    parser.add_argument("--table",
                        default="paper/publication_versions/table_neural_pipeline.tex")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    truth = None
    if Path(args.truth).is_file():
        truth = json.loads(Path(args.truth).read_text()).get("truth")
    payload = run(config, Path(args.output), seeds=args.seeds, workers=args.workers,
                  truth=truth)
    write_table(payload, Path(args.table))
    print(json.dumps({"designs": {d: (v["floors"], v["median_attacks"], v["unsafe_choices"])
                                  for d, v in payload["designs"].items()},
                      "calibration": payload["calibration"],
                      "models": {k: v["metrics"] for k, v in payload["models"].items()}},
                     indent=1))


if __name__ == "__main__":
    main()
