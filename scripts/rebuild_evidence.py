"""Rewrite figures, tables, numbers and the manifest from a stored run.

Use this when only the presentation changes. It never re-simulates, so the
results it publishes are the ones the recorded run produced.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from privacy_risk.reporting import write_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts-paper")
    parser.add_argument("--publish", default=None,
                        help="also copy the tables to this directory, e.g. paper/generated,"
                             " and the figures to its sibling figures/")
    arguments = parser.parse_args()
    root = Path(arguments.artifacts)
    results = json.loads((root / "results.json").read_text())
    config = json.loads((root / "config.json").read_text())
    decision = json.loads((root / "decision.json").read_text())
    write_evidence(results, config, decision, root)
    print(f"rebuilt the evidence in {root} from its stored results")
    if arguments.publish:
        publish(root, Path(arguments.publish))


def publish(root: Path, tables: Path) -> None:
    """Copy the run's tables and figures to where the manuscripts input them."""
    tables.mkdir(parents=True, exist_ok=True)
    for table in sorted((root / "paper").glob("table_*.tex")):
        shutil.copy2(table, tables / table.name)
    figures = tables.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for figure in sorted((root / "figures").glob("*.pdf")):
        shutil.copy2(figure, figures / figure.name)
    print(f"published tables to {tables} and figures to {figures}")


if __name__ == "__main__":
    main()
