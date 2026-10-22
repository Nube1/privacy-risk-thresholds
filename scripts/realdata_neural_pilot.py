"""State-held-out public-Census pilot for SENTRY-R; never a release certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from privacy_risk.realdata import (RealFile, attack, build_release, key_structure,
                                   load_census2000)
from privacy_risk.sentry_r import DeepRiskNet, ScenarioSchema, matrix, train_model


def rows_by_state(seed: int = 20260904, trials: int = 80) -> list[dict]:
    census = load_census2000()
    rows = []
    for state, group in census.frame.groupby("state_code", sort=True):
        if len(group) < 100:
            continue
        subset = RealFile(f"census-state-{state}", group.reset_index(drop=True),
                          census.block, census.key_columns, census.covariate)
        for width, kmin in ((1, 1), (10, 3), (100, 5)):
            release = build_release(subset, {"puma": width, "education": 4,
                                             "experience": 10}, kmin)
            if release.size < 60:
                continue
            structure = key_structure(subset, release)
            for strength, sd in enumerate((.25, .75)):
                outcome = attack(subset, release, .1, sd, trials,
                                 np.random.default_rng(np.random.SeedSequence(
                                     [seed, int(state), width, kmin, strength])))
                key = f"state-{state}-width-{width}-k-{kmin}-sd-{strength}"
                rows.append({"record_id": key, "evidence_id": key,
                             "evidence_kind": "observed_development_attack",
                             "group": f"state-{state}",
                             "features": {"retained": release.size/subset.size,
                                          "mean_reciprocal": structure[
                                              "mean_reciprocal_frequency"],
                                          "log_records": float(np.log1p(subset.size)),
                                          "log_puma_width": float(np.log10(width)),
                                          "minimum_cell": float(kmin),
                                          "covariate_sd": sd},
                             "successes": [outcome["successes"]],
                             "trials": [trials]})
    return rows


def evaluate(model: dict, rows: list[dict], schema: ScenarioSchema) -> dict:
    x, k, n = matrix(rows, schema)
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    net = DeepRiskNet.from_dict(model["network"])
    p = np.clip(net.predict(z), 1e-10, 1-1e-10)
    return {"log_loss": float(-np.sum(k*np.log(p)+(n-k)*np.log1p(-p))/n.sum()),
            "brier": float(np.sum(k*(1-p)**2+(n-k)*p*p)/n.sum()),
            "mean_predicted": float(np.mean(p)),
            "observed_success_rate": float(k.sum()/n.sum())}


def run(seed: int = 20260904, trials: int = 80) -> tuple[dict, dict]:
    rows = rows_by_state(seed, trials)
    groups = sorted({r["group"] for r in rows})
    # Split whole states deterministically, before looking at attack outcomes.
    np.random.default_rng(seed + 1).shuffle(groups)
    n = len(groups)
    train_groups, valid_groups, test_groups = (set(groups[:int(.65*n)]),
                                              set(groups[int(.65*n):int(.8*n)]),
                                              set(groups[int(.8*n):]))
    train = [r for r in rows if r["group"] in train_groups]
    valid = [r for r in rows if r["group"] in valid_groups]
    test = [r for r in rows if r["group"] in test_groups]
    schema = ScenarioSchema(
        ("retained", "mean_reciprocal", "log_records", "log_puma_width",
         "minimum_cell", "covariate_sd"), (0, 0, 0, 0, 1, .1),
        (1, 1, 12, 3, 20, 2), ("linkage",), (.4,))
    trained = {}
    scores = {}
    for depth, name in ((0, "logistic"), (3, "deep_3_layer")):
        model = train_model(train, valid, schema, width=16, depth=depth,
                            epochs=1500, patience=100, seed=seed + depth)
        trained[name] = model
        scores[name] = evaluate(model, test, schema)
        scores[name]["best_epoch"] = model["metrics"]["best_epoch"]
    baseline = sum(r["successes"][0] for r in train)/sum(r["trials"][0] for r in train)
    k = sum(r["successes"][0] for r in test)
    nt = sum(r["trials"][0] for r in test)
    scores["constant_train_rate"] = {
        "log_loss": float(-(k*np.log(baseline)+(nt-k)*np.log1p(-baseline))/nt),
        "brier": float((k*(1-baseline)**2+(nt-k)*baseline**2)/nt)}
    result = {"status": "public Census state-held-out pilot; no cross-institution validation",
              "seed": seed, "trials_per_scenario": trials,
              "split_groups": {"train": sorted(train_groups),
                               "validation": sorted(valid_groups),
                               "test": sorted(test_groups)},
              "rows": {"train": len(train), "validation": len(valid),
                       "test": len(test)}, "scores": scores,
              "test_sha256": hashlib.sha256(json.dumps(test, sort_keys=True).encode()).hexdigest(),
              "limits": "Predicts one declared attack on public Census states; "
                        "models were never trained on final-audit outcomes and "
                        "their predictions are not risk upper bounds."}
    return result, trained


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts-real/neural_pilot.json"))
    args = parser.parse_args()
    result, trained = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for name, model in trained.items():
        args.output.with_name(f"neural_{name}.json").write_text(
            json.dumps(model, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}; {result['scores']}")


if __name__ == "__main__":
    main()
