"""Compare the figures a manuscript quotes with the runs that produced them.

The paper `\\input`s the generated tables, so those cannot drift. This script covers
the prose, in two modes.

``--mode full`` (the default, for the long manuscripts) reads `paper/numbers.json`
from an artifacts directory and checks that each value appears in the manuscript as a
whole number token, so 0.81 does not match inside 0.819.

``--mode quoted`` (for drafts that quote a subset, such as the journal article) works
the other way round. It extracts every decimal with three or more places from the
manuscript prose and checks that each one equals, after rounding to the same number
of places, a value in one of the declared result files. A figure typed by hand that
no run produced fails the check.

Both modes exit non-zero on a mismatch, so a stale number fails the build rather than
reaching a referee.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Values that are not quoted in the prose, or that appear only inside a generated table.
SKIP = {"cells_possible", "released_cells", "budget_equal_share", "utility_span",
        "misspecified_rate", "programme_kmin", "mean_occupied_cell",
        # written by the pipeline but not quoted in the prose; the older substring
        # match passed them only because they occur inside longer numbers
        "credit_posterior_single", "credit_status_accuracy", "credit_status_arrears",
        "credit_status_confident", "loan_single_cells", "nested_share_high"}
DECIMAL = re.compile(r"(?<![\d.])(\d+\.\d{3,})(?![\d])")


def normalise(text: str) -> str:
    """Strip LaTeX thousands separators and collapse whitespace."""
    return re.sub(r"\s+", " ", text.replace("{,}", ",").replace("~", " "))


def strip_comments(text: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", text)


def read_with_inputs(path: Path, generated: bool = True) -> str:
    """Read a manuscript together with the files it inputs.

    With ``generated=False`` the pipeline's own tables are left out, because they
    cannot drift and would otherwise dominate the quoted-number check.
    """
    text = strip_comments(path.read_text())
    for name in re.findall(r"\\input\{([^}]+)\}", text):
        if not generated and "generated" in name:
            continue
        target = path.parent / name
        if target.suffix != ".tex":
            target = target.with_suffix(".tex")
        if target.exists():
            text += "\n" + read_with_inputs(target, generated)
    return text


def token_present(value: str, paper: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(value)}(?![\d])", paper) is not None


def flatten(value) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        try:
            return [float(value.replace(",", ""))]
        except ValueError:
            return [float(v) for v in re.findall(r"\d+\.\d+", value)]
    if isinstance(value, dict):
        return [x for v in value.values() for x in flatten(v)]
    if isinstance(value, list):
        return [x for v in value for x in flatten(v)]
    return []


def full_mode(numbers: dict, paper: str, skip: set[str]) -> int:
    missing = [(name, value) for name, value in sorted(numbers.items())
               if name not in skip and not token_present(normalise(value), paper)]
    checked = len(numbers) - len(skip & set(numbers))
    if missing:
        print(f"{len(missing)} of {checked} run values are not in the manuscript:")
        for name, value in missing:
            print(f"  {name}: {value}")
        return 1
    print(f"All {checked} run values quoted in the manuscript match the run.")
    return 0


def quoted_mode(paper: str, sources: list[Path], allow: set[str]) -> int:
    known: list[float] = []
    for source in sources:
        known += flatten(json.loads(source.read_text()))
    quoted = sorted(set(DECIMAL.findall(paper)))
    unmatched = []
    for literal in quoted:
        if literal in allow:
            continue
        places = len(literal.split(".")[1])
        target = float(literal)
        if not any(abs(round(v, places) - target) < 10 ** -(places + 3) for v in known):
            unmatched.append(literal)
    if unmatched:
        print(f"{len(unmatched)} of {len(quoted)} quoted decimals match no declared result:")
        for literal in unmatched:
            print(f"  {literal}")
        return 1
    print(f"All {len(quoted)} quoted decimals match a declared result.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts-paper")
    parser.add_argument("--paper", required=True, help="path to the manuscript .tex")
    parser.add_argument("--mode", choices=["full", "quoted"], default="full")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--source", nargs="*", default=[],
                        help="extra JSON result files for --mode quoted")
    parser.add_argument("--allow", nargs="*", default=[],
                        help="decimals that are inputs or constants, not run results")
    parser.add_argument("--allow-file", default=None,
                        help="file with one allowed decimal per line; # starts a comment")
    arguments = parser.parse_args()
    allow = set(arguments.allow)
    if arguments.allow_file:
        for line in Path(arguments.allow_file).read_text().splitlines():
            token = line.split("#", 1)[0].strip()
            if token:
                allow.add(token)

    artifacts = Path(arguments.artifacts)
    numbers = json.loads((artifacts / "paper" / "numbers.json").read_text())
    if arguments.mode == "full":
        paper = normalise(read_with_inputs(Path(arguments.paper)))
        return full_mode(numbers, paper, SKIP | set(arguments.skip))
    paper = normalise(read_with_inputs(Path(arguments.paper), generated=False))
    sources = [artifacts / "paper" / "numbers.json", artifacts / "results.json",
               *[Path(s) for s in arguments.source]]
    return quoted_mode(paper, sources, allow)


if __name__ == "__main__":
    sys.exit(main())
