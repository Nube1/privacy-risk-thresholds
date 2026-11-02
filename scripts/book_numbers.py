"""Compute the numbers the book's worked examples and Appendix B print, and check the text.

Run:  PYTHONPATH=src python3 scripts/book_numbers.py [--check]

Writes book_revised/generated/book_numbers.json and appendix_b_precision.tex. With --check
it also confirms that each printed figure appears in the chapter that quotes it, so a
number edited in the text but not in the code, or the reverse, is caught.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

from privacy_risk.sequential import anytime_upper, attack_threshold
from privacy_risk.uncertainty import binomial_lower as lo, binomial_upper as up

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "book_revised"


def numbers() -> list[tuple[str, str, str]]:
    """(chapter file, name, text as printed)."""
    r: list[tuple[str, str, str]] = []
    add = lambda ch, name, text: r.append((ch, name, text))
    f3 = lambda x: f"{x:.3f}"
    # Chapter 4
    add("ch04-estimate", "ex4.3 exact bound", f"{up(127, 1200):.4f}")
    add("ch04-estimate", "ex4.3 largest passing count",
        str(max(x for x in range(100, 140) if up(x, 1200) <= 0.12)))
    add("ch04-estimate", "ex4.4 bound 337/4000", f3(up(337, 4000)))
    add("ch04-estimate", "ex4.4 bound 458/4000", f3(up(458, 4000)))
    add("ch04-estimate", "ex4.6 24 attacks", f3(1 - 0.05 ** (1 / 24)))
    add("ch04-estimate", "ex4.6 one success in 24", f3(up(1, 24)))
    # Chapter 5
    for k in (2, 5, 10):
        add("ch05-selection", f"scan error {k}", f3(1 - 0.95 ** k))
    add("ch05-selection", "ex5.10 bound 384/4000", f3(up(384, 4000)))
    for c in (44, 52, 60):
        add("ch05-selection", f"ex5.7 bound {c}/400", f3(up(c, 400)))
    # Chapter 6
    add("ch06-monitoring", "ex6.2 bound at 84", f"{up(84, 500):.4f}")
    add("ch06-monitoring", "ex6.2 bound at 85", f"{up(85, 500):.4f}")
    add("ch06-monitoring", "ex6.2 split level count",
        str(max(x for x in range(200) if up(x, 500, 0.05 / 20) <= 0.2)))
    add("ch06-monitoring", "ex6.7 anytime 20/300", f3(anytime_upper(20, 300)))
    add("ch06-monitoring", "ex6.7 anytime 40/600", f3(anytime_upper(40, 600)))
    # Chapter 7
    add("ch07-thresholds", "ex7.3 bonferroni large", f3(up(2, 60, 0.05 / 3)))
    add("ch07-thresholds", "ex7.5 broker threshold", f3(attack_threshold(10, 1, 0.5)))
    add("ch07-thresholds", "ex7.5 journalist threshold", f3(attack_threshold(1, 5, 0.5)))
    # Chapter 8
    e = 1 / 1.23
    add("ch08-tolerance", "ex8.7 series share", f3(0.2 / (1 + 0.25 ** e)))
    # Chapter 10
    odds = lambda p, eps: math.exp(eps) * p / (math.exp(eps) * p + 1 - p)
    add("ch10-dp", "ex10.6 four loans", f3(odds(0.1, 4)))
    add("ch10-dp", "ex10.6 one third four loans", f3(odds(1 / 3, 4)))
    # Chapter 11
    add("ch11-simulation", "ex11.3 corrected lower bound", f3(lo(186, 250, 0.05 / 230)))
    # Chapter 12
    add("ch12-public", "ex12.2 plain bound", f3(up(359, 1500)))
    add("ch12-public", "ex12.2 corrected bound", f3(up(359, 1500, 0.05 / 72)))
    return r


def precision_table() -> tuple[list[dict], str]:
    """Appendix B table: floors by reading precision, from the paper configuration."""
    from privacy_risk.credit import make_credit_register
    from privacy_risk.procedures import floor_by_precision
    cfg = json.loads((ROOT / "configs" / "paper.json").read_text())
    reg = make_credit_register(cfg)
    floors = [150, 75, 50, 20, 10, 5, 1]
    rows = floor_by_precision(reg, [0.3, 0.6, 1.0], floors, 0.12, 0.05, 1500, seed=cfg["seed"])
    lines = []
    for row in rows:
        by = {d["floor"]: d["bound"] for d in row["detail"]}
        cells = [f"{by[f]:.3f}" if f in by else "---" for f in floors]
        sel = "none" if row["floor"] == 1 else str(row["floor"])
        lines.append(f"{row['precision']:.1f} & " + " & ".join(cells) + f" & {sel} \\\\%")
    head = ("\\begin{tabular}{@{}lrrrrrrrl@{}}\n\\toprule\n"
            "$s$ & 150 & 75 & 50 & 20 & 10 & 5 & 1 & Selected \\\\\n\\midrule\n")
    tail = "\\bottomrule\n\\end{tabular}\n"
    return rows, head + "\n".join(l.removesuffix("%") for l in lines) + "\n" + tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    out = BOOK / "generated"
    out.mkdir(exist_ok=True)
    rows = numbers()
    (out / "book_numbers.json").write_text(json.dumps(
        [{"chapter": c, "name": n, "text": t} for c, n, t in rows], indent=1) + "\n")
    detail, body = precision_table()
    (out / "appendix_b_precision.tex").write_text(body)
    (out / "appendix_b_precision.json").write_text(json.dumps(detail, indent=1) + "\n")
    print(f"wrote {len(rows)} numbers and the Appendix B table to {out}")
    if not args.check:
        return 0
    bad = []
    for c, n, t in rows:
        text = (BOOK / "chapters" / f"{c}.tex").read_text()
        if not re.search(r"(?<![\d.])" + re.escape(t) + r"(?!\d)", text):
            bad.append(f"{c}: {n} = {t} not found")
    app = (BOOK / "appendix-procedures.tex").read_text()
    if "generated/appendix_b_precision" not in app:
        bad.append("appendix-procedures does not input the generated precision table")
    for b in bad:
        print("MISMATCH", b)
    print("all numbers agree with the text" if not bad else f"{len(bad)} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
