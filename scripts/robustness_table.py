"""Write the LaTeX tables for the programme robustness study.

    python scripts/robustness_table.py

Reads artifacts-robustness/robustness.json and writes paper/generated/table_robustness.tex,
table_tolerance_floors.tex and table_replication.tex. It also records in the JSON the
floor after confirmation and the subgroup floors from a grid that starts at k=200.
"""
from __future__ import annotations

import json
from pathlib import Path

PATH = Path("artifacts-robustness/robustness.json")
ROWS = [("single_one", "One release, one reading"),
        ("pooled_one_no_history", "Six releases, one reading, no history"),
        ("pooled_one_history", "Six releases, one reading, history"),
        ("pooled_live_no_history", "Six releases, reading per vintage, no history"),
        ("pooled_live_history", "Six releases, reading per vintage, history"),
        ("insider_pooled", "Insider: exact exposure, history"),
        ("single_one_rounded", "One release, one reading"),
        ("pooled_one_history_rounded", "Six releases, one reading, history"),
        ("pooled_live_history_rounded", "Six releases, reading per vintage, history"),
        ("insider_pooled_rounded", "Insider: exact exposure, history")]
SUBGROUP_START = 200


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def pct(x: float) -> str:
    return f"{100 * x:.0f}"


def confirmed_floor(data: dict, name: str) -> tuple[int | None, dict | None]:
    """The selected floor if its confirmation passes, else the next stricter one that does.

    Moving to a stricter level after a failed confirmation keeps the error bound,
    because every stricter level also passed at selection.
    """
    levels = sorted(data["levels"])
    chosen = data["floors"][name]["fixed_sequence"]
    if chosen is None:
        return None, None
    tau = data["tolerance"]
    for k in [x for x in levels if x >= chosen]:
        row = data["confirmation_grid"][name][str(k)] if "confirmation_grid" in data \
            else None
        if row is None:
            break
        if row["upper"] <= tau:
            return k, row
    return None, None


def main() -> None:
    data = json.loads(PATH.read_text())
    tau = data["tolerance"]
    coverage = data["coverage"]

    # subgroup floors, fixed sequence from SUBGROUP_START down
    levels = [k for k in sorted(data["levels"], reverse=True) if k <= SUBGROUP_START]
    subgroup = {}
    for name, v in data["subgroup_stratified"].items():
        rows = v["rows"]
        floor = None
        for k in levels:
            r = rows[str(k)]
            worst = max(r["upper"] if r["trials"] else 0.0, r["other_classes_upper"])
            if worst > tau:
                break
            floor = k
        subgroup[name] = floor
    data["subgroup_floors_from_200"] = subgroup

    confirmed = {}
    for name, _ in ROWS:
        k, row = confirmed_floor(data, name)
        confirmed[name] = {"level": k, "rate": row["rate"] if row else None,
                           "upper": row["upper"] if row else None}
    data["confirmed_floors"] = confirmed
    PATH.write_text(json.dumps(data, indent=1))

    body = ""
    for i, (name, label) in enumerate(ROWS):
        if i == 6:
            body += ("\\midrule\n\\multicolumn{6}{l}{\\emph{Exposure published to one "
                     "significant figure}}\\\\\n")
        c = confirmed[name]
        if c["level"] is None:
            body += f"{label} & none & -- & -- & -- & -- \\\\\n"
            continue
        cov = coverage[str(c["level"])]
        sg = subgroup.get(name, "--") if name in subgroup else "--"
        sg = "none" if sg is None else sg
        body += (f"{label} & {c['level']} & {c['rate']:.3f} ({c['upper']:.3f}) & "
                 f"{pct(cov['loan_share'])} & {pct(cov['exposure_share'])} & {sg} \\\\\n")
    caption = (
        f"What sets the programme floor on the synthetic credit register, tolerance {tau:.2f}. "
        "History: the adversary knows in which vintages the target's loan is live. The "
        "floor is chosen by fixed-sequence testing from $k=300$ down on one draw of "
        f"{thousands(data['replicates'])} targets and confirmed on a second, independent draw "
        "(rate, with exact 95 per cent upper bound); where confirmation fails the next "
        "stricter level that passes is shown. Loans and exposure are the percentages of "
        "the first release kept at that floor. The subgroup floor requires every size "
        "class to meet the tolerance, with the largest firms on a stratified draw, from "
        f"$k={SUBGROUP_START}$ down. The insider reads exposure with error "
        f"{data['insider_sd']:g} on the log scale. In the Floor column ``none'' means no level up to 300 "
        "passes, and in the Subgroup column no level up to "
        f"{SUBGROUP_START}.")
    header = ("Adversary & Floor & Confirmation & Loans (\\%) & Exposure (\\%) & Subgroup")
    table = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
             "\\label{tab:robust}\n\\small\\setlength{\\tabcolsep}{3pt}\n"
             "\\begin{tabular}{lrrrrr}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
             + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    Path("paper/generated/table_robustness.tex").write_text(table)

    names = [("single_one", "One release"), ("pooled_one_history", "Pooling, history"),
             ("pooled_live_history", "Full pooled"), ("insider_pooled", "Insider")]
    body = ""
    for t, row in data["tolerance_floors"].items():
        body += f"{float(t):.2f} & " + " & ".join(
            "none" if row[n] is None else str(row[n]) for n, _ in names) + " \\\\\n"
    caption = ("The floor chosen at four tolerances by the selection draw of "
               "Table~\\ref{tab:robust}, before confirmation; at 0.12 pooling with the "
               "history therefore shows 50, where Table~\\ref{tab:robust} shows the "
               "confirmed 75. ``None'' means no level up to 300 passes.")
    table = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
             "\\label{tab:tolerance-floors}\n\\small\n\\begin{tabular}{lrrrr}\n\\toprule\n"
             "Tolerance & " + " & ".join(label for _, label in names) + " \\\\\n\\midrule\n"
             + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    Path("paper/generated/table_tolerance_floors.tex").write_text(table)

    rep = data["replication"]
    names = [("single_one", "One release, one reading"),
             ("pooled_one_history", "Six releases, one reading, history"),
             ("pooled_live_history", "Six releases, reading per vintage, history")]
    body = ""
    for sd in [f"{s:g}" for s in rep["sds"]]:
        cells = []
        for key, _ in names:
            v = rep["summary"][sd][key]
            hi = v["max_floor"]
            hi_text = "none" if hi >= 10_000 else str(hi)
            cell = f"{v['median_floor']:g} [{v['min_floor']}, {hi_text}]"
            if v.get("none_passed"):
                cell += f"$^{{{v['none_passed']}}}$"
            cells.append(cell)
        body += f"{sd} & " + " & ".join(cells) + " \\\\\n"
    caption = (
        f"The floor over {rep['registers']} independent register draws and three levels "
        "of reading noise (the standard deviation of the adversary's log-exposure "
        f"reading), {thousands(rep['replicates'])} targets per level. Median floor, with the "
        "smallest and largest in brackets; ``none'' means no level up to 300 passed, and a "
        "superscript counts the register draws in which that happened.")
    header = "Reading SD & " + " & ".join(label for _, label in names)
    table = ("\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
             "\\label{tab:replication}\n\\small\n"
             "\\begin{tabular}{lrrr}\n\\toprule\n" + header + " \\\\\n\\midrule\n"
             + body + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    Path("paper/generated/table_replication.tex").write_text(table)
    print(json.dumps({"confirmed": confirmed, "subgroup": subgroup}, indent=1))


if __name__ == "__main__":
    main()
