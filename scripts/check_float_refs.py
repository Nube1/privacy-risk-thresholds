"""Check that every table, figure and algorithm is labelled and cited in the text.

    python scripts/check_float_refs.py paper/bis_ifc_full.tex [more.tex ...]

The manuscript is flattened through \\input. For each float environment the script
finds its label and counts references to that label outside the float. A float with
no label, or a label that the text never cites, fails. It checks citation, not
whether the citing sentence explains the float; that needs reading.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

FLOATS = ("table", "table*", "figure", "figure*", "algorithm")


def flatten(path: Path, seen: set | None = None) -> str:
    seen = seen or set()
    if path in seen:
        return ""
    seen.add(path)
    text = path.read_text()
    text = re.sub(r"(?<!\\)%.*", "", text)

    def repl(m: re.Match) -> str:
        name = m.group(1)
        for base in (path.parent, Path("paper"), Path("paper/publication_versions")):
            cand = (base / name)
            for c in (cand, cand.with_suffix(".tex")):
                if c.is_file():
                    return flatten(c.resolve(), seen)
        return ""
    return re.sub(r"\\input\{([^}]*)\}", repl, text)


def check(path: Path) -> list[str]:
    text = flatten(path.resolve())
    problems = []
    pattern = re.compile(r"\\begin\{(" + "|".join(re.escape(f) for f in FLOATS)
                         + r")\}(.*?)\\end\{\1\}", re.S)
    spans = []
    for m in pattern.finditer(text):
        body = m.group(2)
        labels = re.findall(r"\\label\{([^}]*)\}", body)
        spans.append((m.start(), m.end(), m.group(1), labels))
    outside = text
    for start, end, *_ in reversed(spans):
        outside = outside[:start] + " " * (end - start) + outside[end:]
    for start, _, kind, labels in spans:
        if not labels:
            problems.append(f"{kind} at offset {start}: no label")
            continue
        label = labels[0]
        refs = len(re.findall(r"\\(?:ref|cref|Cref|autoref|pageref)\{[^}]*\b"
                              + re.escape(label) + r"\b[^}]*\}", outside))
        if refs == 0:
            problems.append(f"{kind} {label}: never cited in the text")
    print(f"{path}: {len(spans)} floats, {len(problems)} problems")
    return problems


def main() -> None:
    bad = []
    for name in sys.argv[1:]:
        bad += check(Path(name))
    for line in bad:
        print("  " + line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
