"""Check archived public-data claims separately from current manuscript claims."""
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    paper = (Path("paper/bis_ifc_full.tex").read_text()
             + Path("paper/generated/table_real_neural.tex").read_text()
             + Path("docs/REAL_DATA_PROVENANCE.md").read_text())
    real = json.loads(Path("artifacts-real/replication.json").read_text())
    pilot = json.loads(Path("artifacts-real/neural_pilot.json").read_text())
    comparison = json.loads(Path("artifacts-real/neural_comparison.json").read_text())
    census = real["results"]["census"]
    frontier = [census["grid"][i]["risk_upper"] for i in census["frontier"]]
    required = [f"{min(frontier):.3f}", f"{max(frontier):.3f}"]
    required += [f"{row['rate']:.3f}" for row in real["exploratory_sensitivity"]]
    for key in ("logistic", "deep_3_layer", "constant_train_rate"):
        score = pilot["scores"][key]
        required += [f"{score['log_loss']:.4f}", f"{score['brier']:.4f}"]
    required += [f"{row['scores']['three_layer']['log_loss']:.4f}"
                 for row in comparison["results"]]
    missing = [value for value in required if value not in paper]
    if missing:
        raise SystemExit(f"real-data values missing from manuscript and archived provenance report: {missing}")
    print(f"verified {len(required)} real-data values across manuscript and archived provenance report")


if __name__ == "__main__":
    main()
