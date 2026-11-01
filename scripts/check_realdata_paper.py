"""Check public-data claims: current-run values in the manuscript, archived values in the provenance report.

Two census runs exist. The manuscript reports the current run in
`artifacts-paper/results.json`; the archived exploratory run in `artifacts-real/` is
reported only in `docs/REAL_DATA_PROVENANCE.md`, under its own heading. Mixing them is
the error this check guards against.
"""
from __future__ import annotations

import json
from pathlib import Path


def frontier_range(census: dict) -> list[str]:
    frontier = [census["grid"][i]["risk_upper"] for i in census["frontier"]]
    return [f"{min(frontier):.3f}", f"{max(frontier):.3f}"]


def main() -> None:
    manuscript = (Path("paper/bis_ifc_full.tex").read_text(encoding="utf-8")
                  + Path("paper/generated/table_real_neural.tex").read_text(encoding="utf-8"))
    provenance = Path("docs/REAL_DATA_PROVENANCE.md").read_text(encoding="utf-8")
    current_section, archived_section = provenance.split("## Archived exploratory run", 1)

    current = json.loads(Path("artifacts-paper/results.json").read_text())["real"]["census"]
    current_required = frontier_range(current)
    missing = [v for v in current_required
               if v not in manuscript or v not in current_section]

    real = json.loads(Path("artifacts-real/replication.json").read_text())
    pilot = json.loads(Path("artifacts-real/neural_pilot.json").read_text())
    comparison = json.loads(Path("artifacts-real/neural_comparison.json").read_text())
    archived_required = frontier_range(real["results"]["census"])
    archived_required += [f"{row['rate']:.3f}" for row in real["exploratory_sensitivity"]]
    for key in ("logistic", "deep_3_layer", "constant_train_rate"):
        score = pilot["scores"][key]
        archived_required += [f"{score['log_loss']:.4f}", f"{score['brier']:.4f}"]
    archived_required += [f"{row['scores']['three_layer']['log_loss']:.4f}"
                          for row in comparison["results"]]
    missing += [v for v in archived_required if v not in archived_section]

    # the archived upper end must not be quoted as the manuscript's frontier
    stale = archived_required[1]
    if stale != current_required[1] and f"{current_required[0]} to {stale}" in manuscript:
        missing.append(f"manuscript quotes the archived frontier {current_required[0]} to {stale}")
    if missing:
        raise SystemExit(f"public-data values missing or misplaced: {missing}")
    print(f"verified {len(current_required)} current and {len(archived_required)} archived "
          "public-data values")


if __name__ == "__main__":
    main()
