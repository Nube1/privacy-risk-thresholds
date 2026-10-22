"""Check that two tables of the long form share one identification estimand.

Table tab:credit (credit.borrower_identification) and Table tab:eval
(evaluation.programme_attack for one release and one reading) are computed by
different functions. At 20,000 targets they should agree up to sampling error.

    python scripts/estimand_check.py --output artifacts-robustness/estimand_check.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from privacy_risk.credit import borrower_identification, make_credit_register, released_loans
from privacy_risk.evaluation import Policy, programme_attack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.json")
    parser.add_argument("--output", default="artifacts-robustness/estimand_check.json")
    parser.add_argument("--targets", type=int, default=20000)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    register = make_credit_register(config)
    sd = float(config["credit"].get("covariate_sd", 0.6))
    rows = []
    for k in (5, 10, 20):
        a = borrower_identification(register, released_loans(register, k), sd,
                                    args.targets, np.random.default_rng(11 + k))["rate"]
        b = programme_attack(register, Policy(f"k_anonymity_{k}", "microdata", k), sd,
                             args.targets, np.random.default_rng(12 + k))["hits"].mean()
        se = float(np.sqrt(a * (1 - a) / args.targets + b * (1 - b) / args.targets))
        rows.append({"k": k, "credit_table": float(a), "evaluation_table": float(b),
                     "difference": float(a - b), "se_difference": se})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"targets": args.targets, "rows": rows},
                                            indent=1))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
