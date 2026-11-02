#!/usr/bin/env python3
"""Regenerate every number, table, figure and the book from the code in this repository.

    python reproduce.py --fast     # scratch directory, reduced replicates, nothing committed is touched
    python reproduce.py --full     # the paper's settings, rewrites the committed evidence in place
    python reproduce.py --full --only pipeline,book

Stages run in order. A failing stage is reported and the run continues, so one broken
step does not hide the rest; the exit status is non-zero if any stage failed. In --full
mode the run ends by listing which committed files changed, which is the test of the
reproducibility claim: an unchanged tree means the stored evidence is what the code
produces. Figures embed no dates and the results are strict JSON, so identical inputs
should give identical files.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
PY = sys.executable


def stages(fast: bool, work: Path) -> list[tuple[str, list[list[str]]]]:
    art = str(work / "artifacts-paper") if fast else "artifacts-paper"
    cfg = "configs/base.json" if fast else "configs/paper.json"
    gen = str(work / "paper-generated") if fast else "paper/generated"
    rob = str(work / "robustness.json") if fast else "artifacts-robustness/robustness.json"
    return [
        ("pipeline", [[PY, "-m", "privacy_risk.pipeline", "--config", cfg, "--output", art]]),
        ("evidence", [[PY, "scripts/rebuild_evidence.py", "--artifacts", art, "--publish", gen]]),
        ("robustness", [[PY, "scripts/programme_robustness.py", "--config", "configs/paper.json",
                         "--output", rob, "--replicates", "600" if fast else "4000"]]
                       + ([] if fast else [[PY, "scripts/robustness_table.py"]])),
        ("extension", [[PY, "scripts/extension_evidence.py"] + (
            ["--output", str(work / "table_19_1.json")] if fast else [])]),
        ("chapter22", [[PY, "scripts/chapter22_evidence.py"]]),
        ("worked", [[PY, "scripts/worked_examples.py"] + (
            ["--output", str(work / "worked_examples.json")] if fast else [])]),
        ("audit", [[PY, "scripts/audit_process_attacks.py", "--streams", "2000" if fast else "20000"]
                   + (["--output", str(work / "audit_process.json")] if fast else [])]),
        ("book-numbers", [[PY, "scripts/book_numbers.py", "--check"]]),
        ("tests", [[PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "tests",
                    "--ignore=tests/test_realdata.py"]]),
        ("book", [["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
                  ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"]]),
    ]


def run(cmd: list[str], cwd: Path, accept: tuple[int, ...] = (0,)) -> int:
    """Run one command. The pipeline exits 2 when its decision is not a PASS, which is a
    result and not a failure of the run, so that stage accepts 2."""
    print("   $", " ".join(cmd), flush=True)
    code = subprocess.run(cmd, cwd=cwd, env=ENV).returncode
    return 0 if code in accept else code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fast", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated stage names")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    work = ROOT / "build"
    if a.fast:
        work.mkdir(exist_ok=True)
    plan = stages(a.fast, work)
    if a.list:
        print("\n".join(name for name, _ in plan))
        return 0
    only = {s for s in a.only.split(",") if s}
    results = []
    for name, cmds in plan:
        if only and name not in only:
            continue
        if name == "book" and not shutil.which("latexmk"):
            results.append((name, "skipped (no latexmk)", 0.0))
            continue
        print(f"== {name}", flush=True)
        t0, status = time.time(), "ok"
        for cmd in cmds:
            cwd = ROOT / "book_revised" if name == "book" else ROOT
            if run(cmd, cwd, (0, 2) if name == "pipeline" else (0,)) != 0:
                status = "FAILED"
                break
        results.append((name, status, time.time() - t0))
    print("\nstage          status    seconds")
    for n, s, t in results:
        print(f"{n:<14} {s:<9} {t:7.1f}")
    failed = [n for n, s, _ in results if s == "FAILED"]
    if a.full and shutil.which("git"):
        diff = subprocess.run(["git", "status", "--porcelain", "--", "artifacts-paper",
                               "artifacts-robustness", "artifacts-extension", "paper/generated",
                               "book_revised/generated"], cwd=ROOT, capture_output=True, text=True).stdout
        print("\nCommitted evidence changed by this run:" if diff.strip()
              else "\nCommitted evidence unchanged: the stored files are what the code produces.")
        print(diff)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
