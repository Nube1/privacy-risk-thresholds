"""Build editable SVG diagrams and their PDF counterparts for the manuscript."""
from __future__ import annotations

import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1] / "paper" / "diagrams"


def box(x: int, y: int, width: int, height: int, lines: list[str], heavy=False) -> str:
    rect = (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" '
            f'fill="white" stroke="#21334a" stroke-width="{3 if heavy else 2}"/>')
    start = y + height / 2 - (len(lines) - 1) * 14
    labels = "".join(f'<text x="{x + width/2}" y="{start + 28*i}" '
                     f'text-anchor="middle">{escape(line)}</text>' for i, line in enumerate(lines))
    return rect + labels


def line(x1: int, y1: int, x2: int, y2: int) -> str:
    return (f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="#21334a" '
            'stroke-width="2.5" marker-end="url(#arrow)"/>')


def svg(width: int, height: int, contents: str) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img">'
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
            'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            '<path d="M0 0 L10 5 L0 10 Z" fill="#21334a"/></marker></defs>'
            '<style>text{font:20px Arial,sans-serif;fill:#182b41}</style>'
            + contents + '</svg>\n')


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    pool = (box(65, 45, 340, 115, ["Release 1", "A1 = {target, u, v}", "risk = 1/3"])
            + box(595, 45, 340, 115, ["Release 2", "A2 = {target, w, x}", "risk = 1/3"])
            + line(235, 160, 440, 215) + line(765, 160, 560, 215)
            + box(355, 215, 290, 105, ["Pooled intersection", "A1 ∩ A2 = {target}", "risk = 1"], True)
            + '<text x="500" y="365" text-anchor="middle">The marginal union limit, 2/3, does not bound pooled risk.</text>')
    trust = (box(35, 45, 220, 95, ["Coordinate z1", "adversary A"])
             + box(275, 45, 220, 95, ["Coordinate z2", "group B"])
             + box(515, 45, 220, 95, ["Coordinates …", "other risks"])
             + box(755, 45, 220, 95, ["Coordinate zn", "pooled attack"])
             + line(505, 145, 505, 205)
             + box(255, 210, 500, 100, ["Trust position = minimum audited margin", "release only if every coordinate passes"], True))
    workflow = (box(30, 40, 255, 90, ["Past attack scenarios", "development data"])
                + box(372, 40, 255, 90, ["Neural model", "rank candidates"])
                + box(715, 40, 255, 90, ["Fixed shortlist", "no final audit data"])
                + box(715, 245, 255, 90, ["Fresh attack trials", "all C × n pairs"])
                + box(372, 245, 255, 90, ["Exact upper bounds", "simultaneous audit"])
                + box(30, 245, 255, 90, ["Release decision", "all margins ≥ 0"], True)
                + line(285, 85, 372, 85) + line(627, 85, 715, 85)
                + line(842, 130, 842, 245) + line(715, 290, 627, 290)
                + line(372, 290, 285, 290))
    real_audit = (box(30, 30, 285, 100, ["Public microdata", "Census + loan applications"])
                  + box(365, 30, 285, 100, ["Declared releases", "coarsen + suppress"])
                  + box(700, 30, 285, 100, ["Attack simulation", "fresh sampled targets"])
                  + line(315, 80, 365, 80) + line(650, 80, 700, 80)
                  + line(842, 130, 842, 205)
                  + box(700, 205, 285, 100, ["72 exact bounds", "family error 0.05"])
                  + box(365, 205, 285, 100, ["Selected frontier", "conditional file risk"])
                  + box(30, 205, 285, 100, ["Sensitivity + limits", "no live register audit"])
                  + line(700, 255, 650, 255) + line(365, 255, 315, 255))
    for name, width, height, content in [
        ("pooling", 1000, 390, pool), ("trust_vector", 1010, 340, trust),
        ("neural_workflow", 1000, 370, workflow),
        ("realdata_audit", 1020, 340, real_audit),
    ]:
        path = ROOT / f"{name}.svg"
        path.write_text(svg(width, height, content), encoding="utf-8")
        subprocess.run(["inkscape", str(path), "--export-type=pdf",
                        f"--export-filename={ROOT / (name + '.pdf')}"], check=True,
                       capture_output=True, text=True)
        print(path)


if __name__ == "__main__":
    main()
