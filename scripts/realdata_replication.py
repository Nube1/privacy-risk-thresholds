"""Reproduce the public-file experiment, sensitivity checks and source hashes.

No individual-level records are written. Results are conditional on the released
files, a target sampled with replacement, and the specified attacker protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import wooldridge

from privacy_risk import realdata as rd
from privacy_risk.experiments import real_data
from privacy_risk.uncertainty import binomial_upper


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(config: dict) -> dict:
    result = real_data(config)
    seed = int(config["seed"])
    census = rd.load_census2000()
    loans = rd.load_loan_applications()
    chosen = next(row for row in result["census"]["grid"]
                  if row["label"] == result["census"]["chosen"])
    release = rd.build_release(census, chosen["coarsening"], chosen["min_cell_count"])
    loan_release = rd.build_release(loans, {}, 1)
    scenarios = []
    for file, released, scales, label in ((census, release, (0.25, 0.5, 1.0), "census"),
                                           (loans, loan_release, (20., 40., 80.), "loans")):
        for i, scale in enumerate(scales):
            # Distinct random streams per scenario; these are stress checks, not
            # additional confirmatory comparisons in the prespecified grid.
            outcome = rd.attack(file, released, .1, scale, 3000,
                                np.random.default_rng(seed + 900 + 10*i
                                                      + (100 if label == "loans" else 0)))
            scenarios.append({"file": label, "covariate_sd": scale,
                              "successes": outcome["successes"],
                              "trials": outcome["trials"], "rate": outcome["rate"],
                              "upper_95_individual": binomial_upper(
                                  outcome["successes"], outcome["trials"], .05)})
    package = Path(wooldridge.__file__).parent / "datasets"
    return {"design": "Public released microdata; targets sampled with replacement; "
                      "risk conditional on these files and declared attacker",
            "source": {"wooldridge_version": importlib.metadata.version("wooldridge"),
                       "census2000_sha256": digest(package / "census2000.csv.bz2"),
                       "loanapp_sha256": digest(package / "loanapp.csv.bz2")},
            "seed": seed, "real_config": config["real"],
            "attack_replicates": config["attack_replicates"]["real"],
            "results": result, "exploratory_sensitivity": scenarios}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts-real/replication.json"))
    args = parser.parse_args()
    payload = run(json.loads(args.config.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                      allow_nan=False) + "\n")
    print(f"wrote {args.output}; SHA-256 {digest(args.output)}")


if __name__ == "__main__":
    main()
