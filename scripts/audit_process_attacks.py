"""Attacks on the audit process, and what the controls in adaptive_audit do to them.

The gate's error bound assumes an operator who cannot predict attack targets at
registration, cannot choose a bet after seeing the outcome, and runs every attack
through one registry with one budget. This script measures how often an unsafe
coordinate is certified when an operator breaks each assumption, and how often when
the matching control is in place. Streams are Bernoulli at a stated risk, which is
what fresh-target attack outcomes are given the register.

    python scripts/audit_process_attacks.py --output paper/publication_versions/audit_process.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TAU = 0.12
ALPHA = 0.05
CAP = 3000
BET_CAP = 0.11


def certified(draws: np.ndarray, threshold: float) -> np.ndarray:
    """Vectorised gate with the plug-in bet, one row per stream."""
    n = draws.shape[1]
    before = np.concatenate([np.zeros((draws.shape[0], 1)),
                             np.cumsum(draws, axis=1)[:, :-1]], axis=1)
    trials = np.arange(n)[None, :]
    q = np.minimum((before + 0.5) / (trials + 1.0), BET_CAP)
    step = np.where(draws == 1, np.log(q / TAU), np.log((1 - q) / (1 - TAU)))
    return (np.cumsum(step, axis=1) >= threshold).any(axis=1)


def honest_rate(rng, risk: float, streams: int, threshold: float) -> float:
    out = []
    for start in range(0, streams, 2000):
        size = min(2000, streams - start)
        out.append(certified((rng.random((size, CAP)) < risk).astype(np.int8), threshold))
    return float(np.concatenate(out).mean())


def post_hoc(rng, risk: float, runs: int, threshold: float, cap: int = 300) -> float:
    """The operator bets just below the tolerance on successes and zero on failures."""
    wins = 0
    for _ in range(runs):
        log_e = 0.0
        for x in (rng.random(cap) < risk):
            q = (TAU - 1e-4) if x else 0.0
            log_e += math.log(q / TAU) if x else math.log((1 - q) / (1 - TAU))
            if log_e >= threshold:
                wins += 1
                break
    return wins / runs


LABELS = {"seed grinding or shadow audits": "Seed grinding or shadow audits",
          "bet chosen after the outcome": "Bet chosen after the outcome"}


def write_table(payload: dict, path: Path) -> None:
    body = "".join(
        f"{LABELS[r['attack']]} & {r['risk']:.2f} & {r['tries']} & "
        f"{r['without_control']:.3f} & {r['with_control']:.3f} \\\\\n"
        for r in payload["rows"])
    caption = (
        "Attacks on the audit process. Share of runs in which an unsafe or boundary "
        "coordinate is certified, with one coordinate at the full level "
        f"$\\alpha={payload['alpha']:.2f}$ and tolerance {payload['tolerance']:.2f}. Seed "
        "grinding rows use at most 3{,}000 draws and the capped plug-in bet; post-hoc rows "
        "use at most 300 draws and the operator's bet without the control, and the capped "
        "plug-in bet with it. Tries is the number of seeds or shadow "
        "audits the operator can choose among. Without the control, the rate for several "
        "tries is $1-(1-f)^{\\rm tries}$ for the single-try rate $f$; with it, the "
        "operator has one stream. Script: \\nolinkurl{scripts/audit_process_attacks.py}.")
    path.write_text(
        "\\begin{table}[t]\\centering\\small\n\\caption{" + caption + "}"
        "\\label{tab:audit-process}\n\\begin{tabular}{lrrrr}\n\\toprule\n"
        "Attack on the audit & True risk & Tries & Without control & With control"
        "\\\\\\midrule\n" + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="paper/publication_versions/audit_process.json")
    parser.add_argument("--streams", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--table",
                        default="paper/publication_versions/table_audit_process.tex")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    threshold = -math.log(ALPHA)
    rows = []
    for risk in (0.12, 0.14):
        f = honest_rate(rng, risk, args.streams, threshold)
        for tries in (1, 10, 100):
            rows.append({"attack": "seed grinding or shadow audits", "risk": risk,
                         "tries": tries,
                         "without_control": 1 - (1 - f) ** tries,
                         "with_control": f,
                         "control": "targets from a committed secret and a later beacon; "
                                    "one registry and one budget per programme"})
    for risk in (0.14, 0.20):
        rows.append({"attack": "bet chosen after the outcome", "risk": risk, "tries": 1,
                     "without_control": post_hoc(rng, risk, 2000, threshold),
                     "with_control": honest_rate(rng, risk, args.streams, threshold),
                     "control": "bet committed to the ledger before the draw"})
    payload = {"tolerance": TAU, "alpha": ALPHA, "weight": 1.0, "cap": CAP,
               "bet_cap": BET_CAP, "streams": args.streams, "rows": rows}
    Path(args.output).write_text(json.dumps(payload, indent=1))
    write_table(payload, Path(args.table))
    for r in rows:
        print(r["attack"], r["risk"], r["tries"], round(r["without_control"], 4),
              round(r["with_control"], 4))


if __name__ == "__main__":
    main()
