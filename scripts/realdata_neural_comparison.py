"""Prespecified richer public-file comparison of logistic and deep proposal models.

All labels are simulated attack outcomes on real Census records. Whole states are
held out. Predictions are only proposal scores; this is not a release audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from privacy_risk.realdata import (RealFile, attack, build_release, key_structure,
                                   load_census2000)
from privacy_risk.sentry_r import DeepRiskNet, ScenarioSchema, matrix, train_model


PLANS = tuple((puma, education, minimum)
              for puma in (1, 10, 100) for education in (1, 4)
              for minimum in (1, 5))
PRECISIONS = (.15, .5, 1.5)
SPLITS = (20260904, 20260905, 20260906)


def build_rows(seed: int, trials: int) -> list[dict]:
    census = load_census2000()
    rows = []
    for state, frame in census.frame.groupby("state_code", sort=True):
        if len(frame) < 100:
            continue
        subset = RealFile(f"state-{state}", frame.reset_index(drop=True),
                          census.block, census.key_columns, census.covariate)
        for puma, education, minimum in PLANS:
            release = build_release(subset, {"puma": puma, "education": education,
                                             "experience": 10}, minimum)
            if release.size < 60:
                continue
            structure = key_structure(subset, release)
            income = subset.frame[subset.covariate].to_numpy()[release.index]
            for j, sd in enumerate(PRECISIONS):
                rng = np.random.default_rng(np.random.SeedSequence(
                    [seed, int(state), puma, education, minimum, j]))
                outcome = attack(subset, release, .1, sd, trials, rng)
                key = f"state-{state}-p-{puma}-e-{education}-k-{minimum}-sd-{j}"
                rows.append({"record_id": key, "evidence_id": key,
                             "evidence_kind": "observed_development_attack",
                             "group": f"state-{state}",
                             "features": {"retained": release.size/subset.size,
                                          "reciprocal": structure[
                                              "mean_reciprocal_frequency"],
                                          "log_records": float(np.log1p(subset.size)),
                                          "puma_width": float(np.log10(puma)),
                                          "education_width": float(education),
                                          "minimum_cell": float(minimum),
                                          "covariate_sd": sd,
                                          "income_spread": float(np.std(income))},
                             "successes": [outcome["successes"]],
                             "trials": [trials]})
    return rows


SCHEMA = ScenarioSchema(
    ("retained", "reciprocal", "log_records", "puma_width", "education_width",
     "minimum_cell", "covariate_sd", "income_spread"),
    (0, 0, 0, 0, 1, 1, .1, 0),
    (1, 1, 12, 3, 10, 20, 3, 30),
    ("linkage",), (.4,))


def score(model: dict, rows: list[dict], schema: ScenarioSchema = SCHEMA) -> dict:
    x, k, n = matrix(rows, schema)
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    p = np.clip(DeepRiskNet.from_dict(model["network"]).predict(z), 1e-10, 1-1e-10)
    losses = -(k*np.log(p)+(n-k)*np.log1p(-p)).reshape(-1)
    state_losses = {}
    for i, row in enumerate(rows):
        state = row["group"]
        current = state_losses.setdefault(state, [0., 0])
        current[0] += float(losses[i])
        current[1] += int(n[i, 0])
    return {"log_loss": float(losses.sum()/n.sum()),
            "brier": float(np.sum(k*(1-p)**2+(n-k)*p*p)/n.sum()),
            "observed_rate": float(k.sum()/n.sum()),
            "state_loss_and_trials": state_losses}


def paired_interval(reference: dict, contender: dict, seed: int) -> dict:
    states = sorted(reference["state_loss_and_trials"])
    a = np.array([reference["state_loss_and_trials"][s] for s in states])
    b = np.array([contender["state_loss_and_trials"][s] for s in states])
    if not np.array_equal(a[:, 1], b[:, 1]):
        raise ValueError("paired models evaluated on different targets")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(states), (5000, len(states)))
    differences = (a[picks, 0].sum(axis=1)-b[picks, 0].sum(axis=1)) / (
        a[picks, 1].sum(axis=1))
    return {"reference_minus_contender": reference["log_loss"]-contender["log_loss"],
            "state_bootstrap_95_interval": np.quantile(differences,
                                                       [.025, .975]).tolist(),
            "resampled_states": len(states)}


PAIRS = tuple((a, b) for i, a in enumerate(SCHEMA.features)
              for b in SCHEMA.features[i:])
POLY_SCHEMA = ScenarioSchema(
    SCHEMA.features + tuple(f"{a}__x__{b}" for a, b in PAIRS),
    SCHEMA.lower + (0.,)*len(PAIRS),
    SCHEMA.upper + (1000.,)*len(PAIRS),
    SCHEMA.coordinates, SCHEMA.tolerances)


def polynomial_rows(rows: list[dict]) -> list[dict]:
    return [{**row, "features": {**row["features"], **{
        f"{a}__x__{b}": row["features"][a]*row["features"][b]
        for a, b in PAIRS}}} for row in rows]


def run(seed: int = 20260904, trials: int = 60) -> tuple[dict, dict]:
    rows = build_rows(seed, trials)
    states = sorted({row["group"] for row in rows})
    results = []
    representative_models = {}
    for split_seed in SPLITS:
        groups = list(states)
        np.random.default_rng(split_seed).shuffle(groups)
        n = len(groups)
        a, b = int(.65*n), int(.8*n)
        parts = [set(groups[:a]), set(groups[a:b]), set(groups[b:])]
        train, valid, test = ([r for r in rows if r["group"] in selected]
                              for selected in parts)
        scores = {}
        for depth, name in ((0, "logistic"), (1, "one_layer"), (3, "three_layer")):
            model = train_model(train, valid, SCHEMA, width=24, depth=depth,
                                epochs=1500, patience=100, seed=split_seed+depth)
            scores[name] = score(model, test)
            if split_seed == SPLITS[0]:
                representative_models[name] = model
            scores[name]["validation_log_loss"] = model["metrics"]["validation_log_loss"]
            scores[name]["best_epoch"] = model["metrics"]["best_epoch"]
        polynomial = train_model(polynomial_rows(train), polynomial_rows(valid),
                                 POLY_SCHEMA, width=24, depth=0,
                                 epochs=1500, patience=100, seed=split_seed)
        scores["quadratic_logistic"] = score(polynomial, polynomial_rows(test),
                                              POLY_SCHEMA)
        if split_seed == SPLITS[0]:
            representative_models["quadratic_logistic"] = polynomial
        scores["quadratic_logistic"]["validation_log_loss"] = (
            polynomial["metrics"]["validation_log_loss"])
        scores["quadratic_logistic"]["best_epoch"] = (
            polynomial["metrics"]["best_epoch"])
        comparisons = {name: paired_interval(scores["logistic"], scores[name],
                                              split_seed + i)
                       for i, name in enumerate(("one_layer", "three_layer",
                                                  "quadratic_logistic"))}
        comparisons["quadratic_vs_three_layer"] = paired_interval(
            scores["quadratic_logistic"], scores["three_layer"], split_seed + 3)
        results.append({"split_seed": split_seed,
                        "states": {"train": sorted(parts[0]),
                                   "validation": sorted(parts[1]),
                                   "test": sorted(parts[2])},
                        "rows": {"train": len(train), "validation": len(valid),
                        "test": len(test)}, "scores": scores,
                        "paired_comparisons": comparisons,
                        "test_sha256": hashlib.sha256(
                            json.dumps(test, sort_keys=True).encode()).hexdigest()})
    return ({"status": "public Census state-held-out comparison, not a certificate",
             "scenario_plans": [list(plan) for plan in PLANS],
             "precisions": list(PRECISIONS), "attacks_per_scenario": trials,
             "source_seed": seed, "total_rows": len(rows),
             "split_seeds": list(SPLITS), "results": results},
            representative_models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts-real/neural_comparison.json"))
    args = parser.parse_args()
    result, models = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.output.with_name("neural_comparison_models.json").write_text(
        json.dumps({"representative_split": SPLITS[0], "models": models},
                   indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; {result['total_rows']} rows")
    for split in result["results"]:
        print(split["split_seed"], {name: round(score["log_loss"], 5)
                                    for name, score in split["scores"].items()})


if __name__ == "__main__":
    main()
