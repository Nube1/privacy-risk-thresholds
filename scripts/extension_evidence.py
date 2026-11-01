"""Regenerate the pooled-attack table of Chapter 19 from privacy_risk.dynamics.

    python scripts/extension_evidence.py --output artifacts-extension/table_19_1.json

Three synthetic panels of 3,000 loans in 12 key classes, the first 400 loans live in
the first vintage as targets in each, readings with standard deviation 0.3 on the log
scale. Counts are successes out of 1,200; bounds are exact one-sided 95 per cent.
The book's Table 19.1 came from the extension's own run, so counts agree with this
script to within sampling error and not digit for digit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from privacy_risk.dynamics import PanelModel, pooled_attack, simulate_panel


def upper(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k == n else float(beta.ppf(1 - alpha, k + 1, n - k))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="artifacts-extension/table_19_1.json")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {"seed": args.seed, "targets": 1200, "rows": []}
    panels = {name: [simulate_panel(mk(), rng) for _ in range(3)]
              for name, mk in (("frozen", PanelModel.static), ("changing", PanelModel))}
    for w in range(1, 7):
        row = {"vintages": w}
        for name, plist in panels.items():
            for label, (know, s) in {"key": (False, 0.0), "history": (True, 0.0),
                                     "history_readings": (True, 0.3)}.items():
                if label == "key" and name == "changing":
                    continue
                k = sum(pooled_attack(p, w, np.flatnonzero(p["live"][:, 0])[:400], know, s, rng)[0]
                        for p in plist)
                row[f"{name}_{label}"] = {"successes": int(k), "upper95": round(upper(k, 1200), 3)}
        out["rows"].append(row)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    for r in out["rows"]:
        print(r["vintages"], {k: v["successes"] for k, v in r.items() if k != "vintages"})


if __name__ == "__main__":
    main()
