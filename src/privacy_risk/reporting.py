"""Figures, LaTeX fragments and the evidence manifest for one pipeline run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - backend must be set before pyplot

INK, ACCENT, MID, LIGHT = "#1A1A1A", "#7B2D26", "#6E7B8B", "#B8BEC6"
STYLE = {"font.family": "serif", "font.size": 9, "axes.linewidth": 0.6,
         "axes.edgecolor": "#333333", "axes.grid": True, "grid.color": "#DDDDDD",
         "grid.linewidth": 0.5, "axes.axisbelow": True, "figure.dpi": 200,
         "savefig.bbox": "tight", "legend.frameon": False}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value: float | None, places: int = 3) -> str:
    """Format a figure for a table, printing a dash when the run has none."""
    if value is None or (isinstance(value, float) and value != value):
        return "--"
    return f"{value:.{places}f}"


def thousands(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", "{,}")


def figures(results: dict, output: Path) -> list[Path]:
    """Draw the paper's five figures. Restrained colours, serif type, no decoration."""
    output.mkdir(parents=True, exist_ok=True)
    written = []
    with plt.rc_context(STYLE):
        written.append(_risk_distribution(results, output))
        written.append(_frontier(results, output))
        written.append(_calibration(results, output))
        written.append(_composition(results, output))
        written.append(_journalist(results, output))
    return written


def _risk_distribution(results: dict, output: Path) -> Path:
    styles = [(INK, "-"), (MID, "-"), (ACCENT, "-"), (MID, "--"), (INK, ":")]
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    for (level, risks), (colour, dash) in zip(results["risk_distribution"].items(), styles):
        values = np.asarray(risks)
        ax.step(values, np.arange(1, values.size + 1) / values.size, where="post",
                color=colour, ls=dash, lw=1.1,
                label=f"$k_{{\\min}}={level}$ ({values.size:,} records)")
    ax.set_xscale("log")
    ax.set_xlabel("prosecutor risk $1/f_k$")
    ax.set_ylabel("share of released records")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=7.5)
    path = output / "fig1_risk_distribution.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def _frontier(results: dict, output: Path) -> Path:
    grid = results["grid"]
    settings = grid["settings"]
    risk = np.array([s["risk_upper"] for s in settings])
    value = np.array([s["utility"] for s in settings])
    front = np.asarray(grid["frontier"], dtype=int)
    fit = grid["fit"]
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    ax.scatter(risk, value, s=9, facecolor="none", edgecolor=LIGHT, lw=0.6,
               label=f"attainable settings ({len(settings)})")
    ax.plot(risk[front], value[front], color=INK, lw=1.2, marker="o", ms=3.2,
            label="nondominated frontier")
    curve = np.linspace(risk[front].min(), risk[front].max(), 200)
    ax.plot(curve, fit["a"] + fit["b"] * curve ** fit["gamma"], color=MID, ls="--", lw=0.9,
            label=f"fitted value curve ($R^2={fit['r_squared']:.3f}$)")
    marks = grid["implied_tolerances"]
    chosen = [marks[0], marks[len(marks) // 2], marks[-1]]
    for row in chosen:
        ax.plot([row["tolerance"]], [row["utility"]], marker="s", ms=5, color=ACCENT)
        ax.annotate(f"$\\kappa={row['kappa']:g}$", (row["tolerance"], row["utility"]),
                    textcoords="offset points", xytext=(6, -9), fontsize=7.5, color=ACCENT)
    ax.set_xlabel("upper 95% bound on scenario risk, $\\overline{R}_{1-\\alpha}$")
    ax.set_ylabel("utility (negative normalised loss)")
    ax.legend(loc="lower right", fontsize=7.5)
    path = output / "fig2_frontier.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def _calibration(results: dict, output: Path) -> Path:
    marks = {"declared model correct": ("o", INK, "-"),
             "covariate precision misspecified": ("^", ACCENT, "--")}
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for curve in results["attack"]["calibration"]:
        marker, colour, dash = marks[curve["name"]]
        ax.plot([b["mean_posterior"] for b in curve["bins"]],
                [b["empirical"] for b in curve["bins"]], marker=marker, ms=4,
                color=colour, ls=dash, lw=1.0,
                label=f"{curve['name']}\n(mean gap {curve['gap']:.3f})")
    ax.plot([0, 1], [0, 1], color=MID, lw=0.7, ls=":")
    ax.set_xlabel("mean maximum posterior in bin")
    ax.set_ylabel("empirical success rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=7)
    path = output / "fig3_calibration.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def _composition(results: dict, output: Path) -> Path:
    programme = results["programme"]
    marginal = programme["marginal"]
    pooled = programme["pooled"]
    steps = np.arange(1, len(marginal) + 1)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(steps, marginal, width=0.34, color=LIGHT, edgecolor=INK, lw=0.6,
           label="marginal risk of release $i$")
    ax.plot(steps, pooled, color=ACCENT, marker="o", ms=4, lw=1.3,
            label="pooled attack, releases 1 to $m$")
    ax.axhspan(programme["marginal_lower"], programme["union_upper"], color="#EDEDED",
               zorder=0, label="bounds implied by the marginals")
    ax.axhline(programme["independence"], color=MID, ls="--", lw=0.9,
               label="value under independence")
    ax.axhline(1 / programme["kmin"], color=INK, ls=":", lw=0.9,
               label=f"$1/k$ with $k={programme['kmin']}$")
    ax.set_xticks(steps)
    ax.set_xticklabels([name.split()[0] for name in
                        [r["name"] for r in programme["releases"]]])
    ax.set_xlabel("release")
    ax.set_ylabel("probability of correct identification")
    ax.set_ylim(0, max(max(pooled) * 1.2, 0.55))
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    path = output / "fig4_composition.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def _journalist(results: dict, output: Path) -> Path:
    rows = results["measures"]["journalist"]
    fractions = [r["pi"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot(fractions, [r["truth"] for r in rows], marker="o", ms=4, color=INK, lw=1.1,
            label="true $\\rho$")
    ax.plot(fractions, [r["model"] for r in rows], marker="s", ms=4, color=ACCENT,
            ls="--", lw=1.1, label="model-based estimate")
    ax.plot(fractions, fractions, color=MID, ls=":", lw=0.9, label="sampling fraction $\\pi$")
    ax.set_xlabel("sampling fraction $\\pi$")
    ax.set_ylabel("$P(F_k=1\\mid f_k=1)$")
    ax.legend(loc="upper left", fontsize=7.5)
    path = output / "fig5_journalist.pdf"
    fig.savefig(path, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    return path


def _table(body: str, caption: str, label: str, columns: str, header: str,
           small: bool = False) -> str:
    size = "\\small\n" if small else ""
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n{size}\\begin{{tabular}}{{{columns}}}\n\\toprule\n"
            f"{header} \\\\\n\\midrule\n{body}\\bottomrule\n\\end{{tabular}}\n"
            "\\end{table}\n")


def paper_tables(results: dict, output: Path) -> list[Path]:
    """Write the paper's tables as LaTeX fragments, so the text cannot drift from the run."""
    output.mkdir(parents=True, exist_ok=True)
    keys, measures = results["keys"], results["measures"]
    written = []

    rows = [
        ("Prosecutor, mean $1/f_k$", "uniform choice among released candidates",
         measures["prosecutor_mean"]),
        ("Prosecutor, $\\max 1/f_k$", "worst released record", measures["prosecutor_max"]),
        ("Share of records with $f_k = 1$", "sample uniqueness",
         measures["released_unique_share"]),
        ("Marketer, $D_I/n$", "identity \\eqref{eq:marksamp}, same experiment",
         measures["marketer_sample"]),
        ("Plug-in $n^{-1}\\sum_i 1/F_{K(i)}$", "population frequencies assumed known",
         measures["plugin_population_frequency"]),
        ("Marketer, unconditional \\eqref{eq:markpop}", "membership unknown, $\\pi$ applied",
         measures["marketer_unconditional"]),
        ("$\\pi$ times the plug-in measure", "neither experiment; reported anyway",
         measures["pi_times_plugin"]),
    ]
    design = [r for r in measures["journalist"] if abs(r["pi"] - keys["sampling_fraction"]) < 1e-9]
    if design:
        rows += [("Uniqueness probability $\\rho$, truth", "$\\Prob(F_k = 1 \\mid f_k = 1)$, computed",
                  design[0]["truth"]),
                 ("Uniqueness probability $\\rho$, model-based", "fitted negative binomial for $F_k$",
                  design[0]["model"]),
                 ("Journalist, naive $\\pi$", "sampling fraction used as the risk",
                  keys["sampling_fraction"])]
    body = "".join(f"{name} & {note} & {value:.4f} \\\\\n" for name, note, value in rows)
    caption = ("Conventional risk summaries for the unprotected release "
               f"($N = {thousands(keys['population'])}$, "
               f"$\\pi = {keys['sampling_fraction']:.2f}$, "
               f"$n = {thousands(keys['released'])}$).")
    path = output / "table_measures.tex"
    path.write_text(_table(body, caption, "tab:measures", "llr",
                           "Measure & Experiment described & Value"))
    written.append(path)

    body = "".join(
        f"{row['k']} & {thousands(row['released'])} & {row['retained']:.3f} & "
        f"{row['bound']:.4f} & {row['average']:.4f} & {row['average_upper']:.4f} & "
        f"{row['targets']} & {row['worst']:.3f} ({row['worst_lower']:.3f}) & "
        f"{row['share_lower_above_bound']:.3f} \\\\\n"
        for row in results["k_anonymity"])
    caption = ("Minimum cell counts against the simulated risk of the declared adversary. "
               "The average uses "
               f"{thousands(results['k_anonymity'][0]['average_trials'])} "
               "uniformly drawn targets with $\\overline{R}_{0.95}$ its one-sided "
               "Clopper--Pearson bound. The worst target is taken over the listed number of "
               "targets in minimum-size cells, each estimated from "
               f"{results['k_anonymity'][0]['replicates_per_target']} replicates, and is "
               "reported with a one-sided 95 per cent lower bound carrying a Bonferroni "
               "correction for the number of targets examined. The last column is the share "
               "of those targets whose individual lower bound exceeds $1/k$.")
    header = ("$k_{\\min}$ & Kept & Share & $1/k$ & Average & $\\overline{R}_{0.95}$ &\n"
              "Targets & Worst (bound) & Share above $1/k$")
    path = output / "table_kanonymity.tex"
    path.write_text(_table(body, caption, "tab:kanon", "rrrrrrrrr", header, small=True))
    written.append(path)

    body = "".join(
        f"{row['kappa']:.2f} & {row['tolerance']:.3f} & {row['setting']} & "
        f"{row['retained']:.2f} \\\\\n"
        for row in results["grid"]["implied_tolerances"])
    body = body.replace("k_min = ", "$k_{\\min} = ").replace(" &\n", " &\n")
    body = "".join(line.replace("groups, $k_{\\min} = ", "groups, $k_{\\min} = ")
                   for line in body.splitlines(keepends=True))
    body = body.replace(", $k_{\\min} = 1 &", ", $k_{\\min} = 1$ &")
    for value in (1, 3, 5, 10, 20):
        body = body.replace(f"$k_{{\\min}} = {value} &", f"$k_{{\\min}} = {value}$ &")
    caption = ("Tolerance implied by the relative harm weight, computed by maximising "
               "$U - \\kappa \\overline{R}_{0.95}$ over the nondominated settings. The "
               "selected setting reports age band width in years, occupation groups and "
               "minimum cell count.")
    path = output / "table_tolerance.tex"
    path.write_text(_table(body, caption, "tab:tau", "rrlr",
                           "$\\kappa = c_h/\\lambda$ & Implied $\\tau^\\star$ & "
                           "Selected setting & Records kept"))
    written.append(path)

    programme = results["programme"]
    body = "".join(
        f"Release {row['name']} & {thousands(row['released'])} & {row['marginal']:.4f} & "
        f"{row['unique_share']:.3f} \\\\\n" for row in programme["releases"])
    body += "\\midrule\n\\multicolumn{4}{l}{\\emph{Programme-level quantities}} \\\\\n"
    body += (f"Lower bound $\\max_i R_i$ & & {programme['marginal_lower']:.4f} & \\\\\n"
             f"Upper bound $\\min(1, \\sum_i R_i)$ & & {programme['union_upper']:.4f} & \\\\\n"
             f"Value under independence & & {programme['independence']:.4f} & \\\\\n")
    for step, (risk, unique) in enumerate(zip(programme["pooled"],
                                              programme["unique_share"]), start=1):
        if step == 1:
            continue
        body += f"Pooled attack, $m = {step}$ & & {risk:.4f} & {unique:.3f} \\\\\n"
    caption = (f"Four {programme['kmin']}-anonymous releases from one sample, "
               f"{thousands(programme['targets'])} target replicates. No single release "
               "contains a unique record. Marginal risks are unconditional; a target "
               "suppressed from a release contributes zero to that release and does not "
               "narrow the intersection. The pooled adversary intersects the candidate sets "
               "of releases 1 to $m$ and selects uniformly from the intersection.")
    path = output / "table_programme.tex"
    path.write_text(_table(body, caption, "tab:comp", "lrrr",
                           "& Records released & Marginal risk & Unique share"))
    written.append(path)

    credit = results.get("credit")
    if credit:
        body = "".join(
            f"{row['min_cell_count']} & {number(row['retained'])} & "
            f"{number(row['identification'], 4)} & "
            f"{number(row['identification_upper'], 4)} & "
            f"{number(row['identification_largest'])} & "
            f"{number(row['relationship_ppv'])} & "
            f"{number(row['status_homogeneous_largest'])} & "
            f"{number(row['utility'])} \\\\\n"
            for row in credit["settings"])
        caption = (
            "Credit register: five risks and one utility under a minimum cell count on "
            f"the borrower key. The register holds {thousands(credit['register']['loans'])} "
            f"relationships between {thousands(credit['register']['borrowers'])} borrowers "
            f"and {credit['register']['lenders']} lenders. Identification is the average "
            "over released relationships with its one-sided bound, and again for the "
            "largest size class alone. The predictive value is that of an alleged "
            "borrower-lender relationship, and the homogeneity share is the proportion of "
            "large borrowers whose key class discloses their arrears status. Utility is the "
            "negative weighted loss across credit growth, concentration, the arrears ratio "
            "and lender-level supervision.")
        header = ("$k_{\\min}$ & Kept & Identification & $\\overline{R}_{0.95}$ & "
                  "Large firms & Relationship PPV & Status disclosed & Utility")
        path = output / "table_credit.tex"
        path.write_text(_table(body, caption, "tab:credit", "rrrrrrrr", header, small=True))
        written.append(path)

    evaluation = results.get("evaluation")
    if evaluation:
        body = ""
        for row in evaluation["policies"]:
            if row["kind"] != "microdata":
                # count releases measure class uniqueness, a different event
                continue
            single, pooled = row["identification"], row["identification_pooled"]
            body += (
                f"{row['label']} & {thousands(row['released'])} & "
                f"{number(row['utility'])} & "
                f"{number(single.get('match_rate', single['success']), 3)} & "
                f"{number(pooled.get('match_rate', pooled['success']), 3)} & "
                f"{number(single['precision'])} & "
                f"{number(single['decisions'] / single['trials'] if single['trials'] else None)} & "
                f"{number(single.get('worst_class_rate'), 3)} & "
                f"{number(single.get('worst_class_upper'), 3)} \\\\\n")
        caption = (
            "Attack outcomes by protection policy, on the same credit register and the "
            "same adversary. The release and pooled columns give the share of targets "
            "the maximum-posterior linkage names correctly, for one release and pooled "
            "over the "
            f"{evaluation['register']['months']} vintages. Precision is measured over the "
            "attempts the adversary acts on, those whose posterior passes "
            f"{evaluation['policies'][0]['identification'].get('threshold', 0.5):g}, and "
            "the column headed acts is how often it acts at all. The last two columns use "
            "the same success event as the release column for the size class with the "
            "highest rate, with that class's exact 95 per cent upper bound; a release "
            "passes the subgroup control when every class's bound meets the tolerance, "
            "which needs no correction for the number of classes. Utility is the "
            "negative weighted loss on the four analyses the register is collected for. "
            "Noisy-count releases are not compared: their targets are key classes, not "
            "loans.")
        header = ("Policy & Kept & Utility & Release & Pooled & Precision & Acts & "
                  "Worst class & Upper")
        path = output / "table_evaluation.tex"
        path.write_text(_table(body, caption, "tab:eval", "lrrrrrrrr", header, small=True))
        written.append(path)

        decomposition = evaluation.get("decomposition") or []
        if decomposition:
            body = ""
            for row in decomposition:
                cells = [row[key] for key in ("one_release_one_reading",
                                              "one_release_all_readings",
                                              "all_releases_one_reading",
                                              "all_releases_all_readings")]
                body += (f"{row['label']} & " + " & ".join(
                    f"{number(cell['match_rate'], 3)} ({number(cell['upper'], 3)})" for cell in cells)
                    + " \\\\\n")
            months = evaluation["register"]["months"]
            caption = (
                "Where the pooled excess comes from. Share of targets named correctly, "
                "with the exact 95 per cent upper bound in brackets. The adversary attacks "
                f"one release or all {months} vintages, holding one exposure reading or "
                f"one per vintage ({months}). The first and last columns are the release "
                "and pooled columns of Table~\\ref{tab:eval}; the middle two change one "
                "factor at a time on the same targets.")
            header = ("Policy & One release, one reading & One release, "
                      f"{months} readings & All releases, one reading & "
                      f"All releases, {months} readings")
            path = output / "table_decomposition.tex"
            path.write_text(_table(body, caption, "tab:decomp", "lrrrr", header, small=True))
            written.append(path)

        grid = [(evaluation["programme"], "coarse")]
        if evaluation.get("programme_fine"):
            grid.append((evaluation["programme_fine"], "fine"))
        body = ""
        for table, label in grid:
            for k in table["levels"]:
                key = str(k)
                body += (f"{k} & {label} & {thousands(table['per_release_records'][key])} & "
                         f"{number(table['per_release_rate'][key], 3)} & "
                         f"{number(table['per_release_upper'][key], 3)} & "
                         f"{number(table['pooled_rate'][key], 3)} & "
                         f"{number(table['pooled_upper'][key], 3)} \\\\\n")
        caption = (
            "Choosing the minimum cell count. Each level applies $k$-anonymity with that "
            "floor; kept is the number of loans released in the first vintage. The "
            "coarse and fine grids use separate target draws. A level meets the "
            f"tolerance of {evaluation['tolerance']:.2f} when the exact upper bound, not "
            "the point estimate, is at or below it.")
        header = ("$k_{\\min}$ & Grid & Kept & Release & Upper & Pooled & Upper")
        path = output / "table_programme_choice.tex"
        path.write_text(_table(body, caption, "tab:kchoice", "rlrrrrr", header, small=True))
        written.append(path)

        short = {4: "sector, region, size, exposure", 3: "one field withdrawn",
                 2: "region and exposure only"}
        names = {("sector", "region", "size", "exposure"): "all four fields",
                 ("sector", "region", "size"): "no exposure reading",
                 ("sector", "region", "exposure"): "no size class",
                 ("region", "exposure"): "region and exposure only"}
        body = "".join(
            f"{names.get(tuple(row['fields']), short.get(row['held'], 'other'))} & "
            f"{row['held']} & "
            f"{number(row.get('match_rate'))} & {number(row['precision'])} & "
            f"{number(row.get('calibration_error'))} \\\\\n"
            for row in evaluation["ablations"])
        body += "\\midrule\n"
        body += "".join(
            f"assumed $\\sigma$ $\\times$ {row['factor']:g} & 4 & "
            f"{number(row.get('match_rate'))} & {number(row['precision'])} & "
            f"{number(row.get('calibration_error'))} \\\\\n"
            for row in evaluation["shift"])
        caption = (
            "What the attack needs, and what happens when the adversary's model is "
            "wrong. The upper block withdraws auxiliary information from the adversary "
            "one field at a time; the lower block keeps all four fields and misstates the "
            "dispersion of its exposure reading. Calibration error is the gap between the "
            "average posterior the adversary reports and the share of its guesses that "
            "are right. Both blocks use $k$-anonymity at $k=5$.")
        header = "Adversary holds & Fields & Risk & Precision & Calibration error"
        path = output / "table_ablation.tex"
        path.write_text(_table(body, caption, "tab:ablation", "lrrrr", header, small=True))
        written.append(path)

    scaling = results.get("scaling")
    if scaling:
        body = "".join(
            f"{thousands(row['population_size'])} & {thousands(row['sample_size'])} & "
            f"{number(row['largest_marginal'])} & {number(row['pooled_two'])} & "
            f"{number(row['pooled'])} & {number(row['union_bound'])} & "
            f"{number(row['ratio_to_bound'], 2)} \\\\\n" for row in scaling)
        caption = (
            "Does the pooling gap survive at other register sizes? The same four "
            "releases, the same keys and the same minimum cell count, rebuilt on "
            "registers of four sizes with the sampling fraction held fixed. Larger "
            "registers are safer release by release, because the key classes grow. The "
            "pooled attack falls more slowly, so the ratio of the pooled risk to the "
            "bound the marginals imply rises with size rather than falling. The gap the "
            "paper is about is not an artefact of the size chosen for the main study.")
        header = ("Population & Sample & Largest marginal & Pooled, 2 releases & "
                  "Pooled, 4 & Boole bound & Ratio")
        path = output / "table_scaling.tex"
        path.write_text(_table(body, caption, "tab:scaling", "rrrrrrr", header, small=True))
        written.append(path)

    adaptive = results.get("adaptive")
    if adaptive:
        body = "".join(
            f"{row['benefit']:g} & {row['false_cost']:g} & {row['action_cost']:g} & "
            f"{number(row['threshold'])} & {number(row['coverage'])} & "
            f"{number(row['precision'])} & {number(row['yield'])} \\\\\n"
            for row in adaptive["payoffs"])
        caption = (
            "Where a paid adversary acts. The payoff of a correct allegation, the cost "
            "of a false one and the cost of making any allegation fix the posterior "
            "threshold of Proposition \\ref{prop:attack-threshold}. Coverage is the share "
            "of targets the adversary acts on at that threshold, precision the share of "
            "those allegations that are true, and yield the product, which is what the "
            "adversary collects. The attack is the pooled one against "
            f"{adaptive['reference_policy'].replace('_', ' ')}. One half, the threshold "
            "usually assumed, is the middle case of equal stakes and free allegations.")
        header = ("Benefit $h$ & False cost $c_f$ & Action cost $c_a$ & $p^\\star$ & "
                  "Coverage & Precision & Yield")
        path = output / "table_adversary.tex"
        path.write_text(_table(body, caption, "tab:adversary", "rrrrrrr", header))
        written.append(path)

        body = "".join(
            f"{row['tolerance']:g} & {row['alpha']:g} & {thousands(row['fixed'])} & "
            f"{thousands(row['anytime'])} & "
            f"{number(row['anytime'] / row['fixed'], 1)} \\\\\n"
            for row in adaptive["certification"])
        caption = (
            "The cost of watching the run. The fixed-sample column is the number of "
            "successless attacks that certifies the tolerance at one prearranged sample "
            "size. The anytime column is the number that certifies it while the "
            "institution monitors and may stop whenever it likes, using the confidence "
            "sequence of Proposition \\ref{prop:anytime} with a uniform prior. The last "
            "column is the ratio. It is the price of that mixture, which stays able to "
            "certify after a success is seen, and not of monitoring itself: a predeclared "
            "point alternative at zero risk is also anytime-valid and certifies after the "
            "fixed-sample count, but can never certify once a success occurs.")
        header = ("Tolerance $\\tau$ & $\\alpha$ & Fixed sample & Anytime & Ratio")
        path = output / "table_anytime.tex"
        path.write_text(_table(body, caption, "tab:anytime", "rrrrr", header))
        written.append(path)

    real = results.get("real")
    if real:
        census = real["census"]
        rows = [census["grid"][i] for i in census["frontier"]]
        body = "".join(
            f"{row['coarsening']['puma']} & {row['coarsening']['education']} & "
            f"{row['coarsening']['experience']} & {row['min_cell_count']} & "
            f"{number(row['retained'])} & {number(row['exact_matching'], 4)} & "
            f"{number(row['risk'], 4)} & {number(row['risk_upper'], 4)} & "
            f"{number(row['utility'])} \\\\\n" for row in rows)
        caption = (
            "US Census 2000 public use sample: the nondominated releases. The file "
            f"holds {thousands(census['full_key']['records'])} records, and the columns "
            "report the width of the area, schooling and experience bands, the minimum "
            "cell count, the share of records kept, the risk under exact matching, the "
            "simulated risk with its simultaneous one-sided bound over the 72 "
            "planned setting--adversary comparisons, and the utility across the three "
            "income analyses. The simulated risk is the larger of two adversaries, the "
            "declared one and one with a covariate reading twice as precise. Both read "
            "the key exactly but model a misreading probability of 0.10, and both hold "
            "a noisy reading of log weekly income.")
        header = ("Area & Schooling & Experience & $k_{\\min}$ & Kept & Exact "
                  "matching & Simulated & $\\overline{R}_{0.95}$ & Utility")
        path = output / "table_real_census.tex"
        path.write_text(_table(body, caption, "tab:real", "rrrrrrrrr", header, small=True))
        written.append(path)

    alphas = sorted({row["alpha"] for row in results["certification"]}, reverse=True)
    taus = sorted({row["tolerance"] for row in results["certification"]}, reverse=True)
    lookup = {(row["tolerance"], row["alpha"]): row["replicates"]
              for row in results["certification"]}
    body = ""
    for tau in taus:
        label = f"${tau:g}$" if tau >= 0.01 else f"$10^{{{int(round(np.log10(tau)))}}}$"
        cells = " & ".join(thousands(lookup[(tau, a)]) for a in alphas)
        body += f"{label} & {cells} \\\\\n"
    header = " & ".join(["$\\tau$"] + [f"$B$ at $\\alpha = {a:g}$" for a in alphas])
    caption = ("Independent simulated attacks required to certify "
               "$\\overline{R}_{1-\\alpha} \\leq \\tau$ with no observed success, from "
               "\\eqref{eq:cert}.")
    path = output / "table_certification.tex"
    path.write_text(_table(body, caption, "tab:cert", "r" * (len(alphas) + 1), header))
    written.append(path)
    return written


def paper_numbers(results: dict) -> dict[str, str]:
    """Every figure the paper quotes in prose, formatted as the paper prints it."""
    keys, measures = results["keys"], results["measures"]
    attack, grid = results["attack"], results["grid"]
    programme, budget = results["programme"], results["budget"]
    settings = grid["settings"]
    front = np.asarray(grid["frontier"], dtype=int)
    design = [r for r in measures["journalist"]
              if abs(r["pi"] - keys["sampling_fraction"]) < 1e-9][0]
    errors = [abs(r["model"] - r["truth"]) / r["truth"] for r in measures["journalist"]]
    naive = [r["pi"] / r["truth"] for r in measures["journalist"]]
    stress = [s["risk_stronger"] / s["risk_declared"] for s in settings
              if s["risk_declared"] > 0]
    kanon = {row["k"]: row for row in results["k_anonymity"]}
    numbers = {
        "population": thousands(keys["population"]),
        "released": thousands(keys["released"]),
        "cells_possible": thousands(keys["cells_possible"]),
        "cells_occupied": thousands(keys["cells_occupied"]),
        "population_unique_cells": thousands(keys["population_unique_cells"]),
        "population_unique_share": f"{100 * keys['population_unique_share']:.2f}",
        "released_cells": thousands(keys["released_cells"]),
        "released_unique_cells": thousands(keys["released_unique_cells"]),
        "released_unique_share": f"{100 * keys['released_unique_share']:.1f}",
        "prosecutor_mean": f"{measures['prosecutor_mean']:.4f}",
        "plugin_population": f"{measures['plugin_population_frequency']:.4f}",
        "marketer_unconditional": f"{measures['marketer_unconditional']:.4f}",
        "plugin_ratio": f"{measures['plugin_ordering']['ratio']:.1f}",
        "unconditional_ratio":
            f"{measures['prosecutor_mean'] / measures['marketer_unconditional']:.0f}",
        "journalist_truth": f"{design['truth']:.4f}",
        "journalist_model": f"{design['model']:.4f}",
        "journalist_error_low": f"{100 * min(errors):.0f}",
        "journalist_error_high": f"{100 * max(errors):.0f}",
        "journalist_naive_design": f"{naive[1] if len(naive) > 1 else naive[0]:.2f}",
        "fitted_mean_cell": f"{design['fitted_mean_cell']:.2f}",
        "true_mean_cell": f"{measures['true_mean_population_cell']:.2f}",
        "mean_occupied_cell": f"{measures['mean_occupied_cell']:.2f}",
        "attack_theory": f"{attack['theory_exact_matching']:.4f}",
        "attack_key_only": f"{attack['key_error_only']:.4f}",
        "attack_covariate": f"{attack['with_covariate']:.4f}",
        "attack_upper": f"{attack['upper_bound']:.4f}",
        "attack_understatement": f"{100 * attack['understatement']:.0f}",
        "calibration_gap_correct": f"{attack['calibration'][0]['gap']:.3f}",
        "calibration_gap_wrong": f"{attack['calibration'][1]['gap']:.3f}",
        "misspecified_rate": f"{attack['misspecified_rate']:.4f}",
        "grid_settings": str(len(settings)),
        "frontier_points": str(front.size),
        "frontier_risk_low": f"{grid['risk_span'][0]:.3f}",
        "frontier_risk_high": f"{grid['risk_span'][1]:.3f}",
        "frontier_fit_r2": f"{grid['fit']['r_squared']:.3f}",
        "frontier_gamma": f"{grid['fit']['gamma']:.2f}",
        "frontier_slope_high": f"{grid['concavity']['largest_slope']:.1f}",
        "frontier_slope_low": f"{grid['concavity']['smallest_slope']:.3f}",
        "stress_low": f"{100 * (min(stress) - 1):.0f}",
        "stress_high": f"{100 * (max(stress) - 1):.0f}",
        "stress_median": f"{100 * (float(np.median(stress)) - 1):.0f}",
        "programme_kmin": str(programme["kmin"]),
        "programme_union_upper": f"{programme['union_upper']:.4f}",
        "programme_lower": f"{programme['marginal_lower']:.4f}",
        "programme_independence": f"{programme['independence']:.4f}",
        "pooled_two": f"{programme['pooled'][1]:.4f}",
        "pooled_all": f"{programme['pooled'][-1]:.4f}",
        "pooled_two_ratio": f"{programme['pooled'][1] / programme['union_upper']:.1f}",
        "pooled_all_ratio": f"{programme['pooled'][-1] / programme['union_upper']:.1f}",
        "pooled_unique_share": f"{100 * programme['unique_share'][-1]:.1f}",
        "budget_exponent": f"{budget['exponent']:.2f}",
        "budget_gain": f"{100 * budget['loss_reduction']:.1f}",
        "budget_agreement_decimals": str(budget["agreement_decimals"]),
        "nested_between_share": f"{100 * results['design_variance']['between_share']:.0f}",
        "nested_share_high":
            f"{100 * results['design_variance']['between_share_interval'][1]:.0f}",
        "nested_within_sd": f"{results['design_variance']['within_sd']:.4f}",
        "nested_between_sd": f"{results['design_variance']['between_sd']:.4f}",
        "nested_naive_se": f"{results['design_variance']['naive_standard_error']:.4f}",
        "nested_se": f"{results['design_variance']['nested_standard_error']:.4f}",
        "nested_outer": str(results["design_variance"]["outer_replicates"]),
        "budget_equal_share": f"{budget['equal_split'][0]:.3f}",
        "budget_allocation": ", ".join(f"{x:.3f}" for x in budget["allocation"]),
        "utility_span": f"{grid['utility_span'][0] - grid['utility_span'][1]:.4f}",
    }
    real = results.get("real")
    if real:
        census, loans = real["census"], real["loans"]
        key = census["full_key"]
        composition = census["composition"]
        worst = census["worst_target"]
        numbers.update({
            "real_records": thousands(key["records"]),
            "real_cells": thousands(key["cells"]),
            "real_unique_share": f"{100 * key['unique_share']:.1f}",
            "real_mean_recip": f"{key['mean_reciprocal_frequency']:.3f}",
            "real_key_space": thousands(census["key_space"]),
            "real_rho_model": f"{census['journalist']['rho_model']:.3f}",
            "real_fitted_cell": f"{census['journalist']['fitted_mean_cell']:.3f}",
            "real_settings": str(len(census["grid"])),
            "real_frontier": str(len(census["frontier"])),
            "real_risk_low": f"{census['grid'][census['frontier'][0]]['risk_upper']:.3f}",
            "real_risk_high": f"{census['grid'][census['frontier'][-1]]['risk_upper']:.3f}",
            "real_worst": f"{worst['maximum']:.2f}",
            "real_worst_lower": f"{worst['maximum_lower_bound']:.3f}",
            "real_worst_cell": str(worst["smallest_cell"]),
            "real_worst_targets": str(worst["targets"]),
            "real_worst_mean": f"{worst['mean']:.3f}",
            "real_union_upper": f"{composition['union_upper']:.3f}",
            "real_pooled": f"{composition['pooled'][-1]:.3f}",
            "real_pooled_ratio":
                f"{composition['pooled'][-1] / composition['union_upper']:.2f}",
            "real_pooled_unique": f"{100 * composition['unique_share'][-1]:.1f}",
            "loan_records": thousands(loans["structure"]["records"]),
            "loan_cells": str(loans["structure"]["cells"]),
            "loan_unique_share": f"{100 * loans['structure']['unique_share']:.1f}",
            "loan_identification": f"{loans['identification']['rate']:.3f}",
            "loan_identification_upper": f"{loans['identification']['upper']:.3f}",
            "loan_denied_homogeneous":
                f"{100 * loans['denied']['homogeneous_share']:.1f}",
            "loan_denied_positive":
                f"{100 * loans['denied']['homogeneous_positive_share']:.2f}",
            "loan_late_homogeneous": f"{100 * loans['late']['homogeneous_share']:.1f}",
            "loan_late_positive":
                f"{100 * loans['late']['homogeneous_positive_share']:.2f}",
            "loan_single_cells": f"{100 * loans['residual'][0]['single_record_share']:.0f}",
            "loan_exposed_amount":
                f"{100 * loans['residual'][0]['exposed_amount_share']:.1f}",
        })

    credit = results.get("credit")
    if credit:
        first, last = credit["settings"][0], credit["settings"][-1]
        # A size class can draw no targets in a short run, and then it has no interval.
        first_bounds = first["identification_largest_bounds"] or [None, None]
        last_bounds = last["identification_largest_bounds"] or [None, None]
        table = first["relationship_table"]
        credit_numbers = {
            "credit_loans": thousands(credit["register"]["loans"]),
            "credit_borrowers": thousands(credit["register"]["borrowers"]),
            "credit_lenders": str(credit["register"]["lenders"]),
            "credit_key_cells": thousands(credit["register"]["key_cells"]),
            "credit_identification": f"{first['identification']:.4f}",
            "credit_identification_borrower": f"{first['identification_borrower']:.4f}",
            "credit_identification_large_low":
                number(first_bounds[0]),
            "credit_identification_large_high":
                number(first_bounds[1]),
            "credit_identification_large_strict_low":
                number(last_bounds[0]),
            "credit_identification_large_strict_high":
                number(last_bounds[1]),
            "credit_identification_large": number(first["identification_largest"]),
            "credit_identification_large_strict":
                number(last["identification_largest"]),
            "credit_ppv": f"{first['relationship_ppv']:.3f}",
            "credit_ppv_strict": f"{last['relationship_ppv']:.3f}",
            "credit_prevalence": f"{table['prevalence']:.4f}",
            "credit_sensitivity": f"{table['sensitivity']:.3f}",
            "credit_specificity": f"{table['specificity']:.3f}",
            "credit_ppv_identity": f"{table['base_rate_identity']:.3f}",
            "credit_ppv_lift": f"{table['lift_over_market_share']:.1f}",
            "credit_status_large": number(first["status_homogeneous_largest"]),
            "credit_status_large_strict":
                number(last["status_homogeneous_largest"]),
            "credit_status_arrears":
                f"{100 * first['status_homogeneous_arrears']:.2f}",
            "credit_status_confident": f"{100 * first['status_confident_share']:.0f}",
            "credit_status_accuracy": f"{100 * first['status_confident_accuracy']:.0f}",
            "credit_single_contributor":
                f"{100 * credit['reconstruction'][0]['single_contributor_share']:.1f}",
            "credit_participant_recoverable":
                f"{100 * credit['reconstruction'][0]['recoverable_by_a_participant_share']:.1f}",
            "credit_exposed_amount":
                f"{100 * credit['reconstruction'][0]['exposed_amount_share']:.2f}",
            "credit_recoverable_amount":
                f"{100 * credit['reconstruction'][0]['recoverable_amount_share']:.1f}",
            "credit_unit_change":
                f"{100 * credit['differencing'][0]['unit_change_share']:.1f}",
            "credit_unit_change_strict":
                f"{100 * credit['differencing'][-1]['unit_change_share']:.1f}",
            "credit_posterior_single":
                f"{credit['differencing'][0]['posterior_single_move']:.2f}",
            "credit_moves_per_cell":
                f"{credit['differencing'][0]['moves_per_cell']:.1f}",
            "credit_empirical_single":
                f"{credit['differencing'][0]['empirical_single_move_share']:.2f}",
            "credit_posterior_per_cell":
                f"{credit['differencing'][0]['posterior_single_move_per_cell']:.2f}",
            "credit_moves_unit_cell":
                f"{credit['differencing'][0]['moves_per_unit_change_cell']:.1f}",
            "credit_utility_strict": f"{last['utility']:.3f}",
            "credit_retained_strict": f"{last['retained']:.3f}",
        }
        numbers.update({name: value for name, value in credit_numbers.items()
                        if "--" not in value and "nan" not in value})

    scaling = results.get("scaling")
    if scaling:
        small, large = scaling[0], scaling[-1]
        numbers.update({
            "scale_small_population": thousands(small["population_size"]),
            "scale_large_population": thousands(large["population_size"]),
            "scale_small_marginal": number(small["largest_marginal"]),
            "scale_large_marginal": number(large["largest_marginal"]),
            "scale_small_pooled": number(small["pooled"]),
            "scale_large_pooled": number(large["pooled"]),
            "scale_small_ratio": number(small["ratio_to_bound"], 2),
            "scale_large_ratio": number(large["ratio_to_bound"], 2),
            "scale_sizes": str(len(scaling)),
        })

    adaptive = results.get("adaptive")
    if adaptive:
        study = adaptive["monitoring"]
        equal, paid, cautious = adaptive["payoffs"][0], adaptive["payoffs"][1], \
            adaptive["payoffs"][-1]
        transport = adaptive["transport"]
        adaptive_numbers = {
            "seq_risk": number(adaptive["risk"]),
            "seq_fixed_upper": number(adaptive["fixed_upper"]),
            "seq_anytime_upper": number(adaptive["anytime_upper"]),
            "seq_runs": thousands(study["runs"]),
            "seq_looks": str(study["looks"]),
            "seq_monitor_replicates": thousands(study["replicates"]),
            "seq_fixed_error": f"{100 * study['fixed_error_rate']:.0f}",
            "seq_anytime_error": f"{100 * study['anytime_error_rate']:.0f}",
            "seq_width_ratio": number(study["width_ratio"], 2),
            "seq_paid_threshold": number(paid["threshold"]),
            "seq_paid_coverage": number(paid["coverage"]),
            "seq_paid_precision": number(paid["precision"]),
            "seq_paid_yield": number(paid["yield"]),
            "seq_equal_yield": number(equal["yield"]),
            "seq_cautious_threshold": number(cautious["threshold"]),
            "seq_cautious_precision": number(cautious["precision"]),
            "seq_cautious_coverage": number(cautious["coverage"]),
            "seq_admissible_divergence": f"{adaptive['admissible_divergence']:.4f}",
            "seq_admissible_room": (f"{adaptive['admissible_room']:.3f}"
                                    if adaptive.get("admissible_room") is not None
                                    else "--"),
            "seq_admissible_risk": number(adaptive.get("admissible_risk")),
            "seq_admissible_policy": str(adaptive.get("admissible_policy")),
            "seq_transport_small": number(transport[0]["bound"]),
            "seq_transport_large": number(transport[-1]["bound"]),
            "seq_divergence_large": f"{transport[-1]['divergence']:g}",
        }
        for row in adaptive["certification"]:
            tag = f"{row['tolerance']:g}".replace(".", "").replace("0", "0", 1)
            adaptive_numbers[f"seq_cert_fixed_{tag}"] = thousands(row["fixed"])
            adaptive_numbers[f"seq_cert_anytime_{tag}"] = thousands(row["anytime"])
        numbers.update({name: value for name, value in adaptive_numbers.items()
                        if "--" not in value and "nan" not in value})

    evaluation = results.get("evaluation")
    if evaluation:
        by_name = {row["policy"]: row for row in evaluation["policies"]}
        programme = evaluation["programme"]

        def rate(row: dict, pooled: bool = False) -> float:
            entry = row["identification_pooled" if pooled else "identification"]
            value = entry.get("match_rate")
            return float(value if value is not None else entry["success"])

        best_gap = max(evaluation["policies"],
                       key=lambda row: rate(row, True) - rate(row))
        counts = [row for row in evaluation["policies"] if row["kind"] == "counts"]
        evaluation_numbers = {
            "eval_replicates": thousands(evaluation["replicates"]),
            "eval_tolerance": f"{evaluation['tolerance']:.2f}",
            "eval_policies": str(len(evaluation["policies"])),
            "eval_none_release": f"{rate(by_name['none']):.3f}",
            "eval_none_pooled": f"{rate(by_name['none'], True):.3f}",
            "eval_min3_release": f"{rate(by_name['min_contributors']):.3f}",
            "eval_min3_pooled": f"{rate(by_name['min_contributors'], True):.3f}",
            "eval_k5_release": f"{rate(by_name['k_anonymity_5']):.3f}",
            "eval_k5_pooled": f"{rate(by_name['k_anonymity_5'], True):.3f}",
            "eval_k10_release": f"{rate(by_name['k_anonymity_10']):.3f}",
            "eval_k10_pooled": f"{rate(by_name['k_anonymity_10'], True):.3f}",
            "eval_k5_precision": number(by_name["k_anonymity_5"]["identification"]["precision"]),
            "eval_k5_decisions": number(
                by_name["k_anonymity_5"]["identification"]["decisions"]
                / max(by_name["k_anonymity_5"]["identification"]["trials"], 1)),
            "eval_k5_worst": number(
                by_name["k_anonymity_5"]["identification"].get("worst_class_rate")),
            "eval_k5_worst_upper": number(
                by_name["k_anonymity_5"]["identification"].get("worst_class_upper")),
            "eval_none_worst": number(
                by_name["none"]["identification"].get("worst_class_rate")),
            "eval_none_worst_upper": number(
                by_name["none"]["identification"].get("worst_class_upper")),
            "eval_attribute_majority": number(
                by_name["k_anonymity_5"]["attribute"].get("majority_accuracy")),
            "eval_k5_utility": f"{by_name['k_anonymity_5']['utility']:.3f}",
            "eval_largest_gap_policy": best_gap["label"],
            "eval_largest_gap": f"{rate(best_gap, True) - rate(best_gap):.3f}",
            "eval_per_release_choice": str(programme["per_release_choice"]),
            "eval_pooled_choice": str(programme["pooled_choice"]),
            "eval_per_release_kept": thousands(
                programme["per_release_records"][str(programme["per_release_choice"])])
                if programme["per_release_choice"] else "--",
            "eval_pooled_kept": thousands(
                programme["per_release_records"][str(programme["pooled_choice"])])
                if programme["pooled_choice"] else "--",
            "eval_admissible": str(evaluation["admissible_best"]),
            "eval_ablation_full": number(evaluation["ablations"][0].get("match_rate")),
            "eval_ablation_minimal": number(evaluation["ablations"][-1].get("match_rate")),
            "eval_ablation_noexposure": number(evaluation["ablations"][1].get("match_rate")),
            "eval_calibration_true": number(evaluation["shift"][1].get("calibration_error")),
            "eval_calibration_worst": number(
                max(row.get("calibration_error", 0.0) for row in evaluation["shift"])),
            "eval_relationship_ppv": number(
                by_name["k_anonymity_5"]["relationship"]["posterior"]),
            "eval_attribute_accuracy": number(
                by_name["k_anonymity_5"]["attribute"]["posterior"]),
            "eval_differencing_precision": number(
                by_name["k_anonymity_5"]["differencing"]["posterior"]),
            "eval_per_release_choice_upper": str(programme.get("per_release_choice_upper")),
            "eval_pooled_choice_upper": str(programme.get("pooled_choice_upper")),
        }
        fine = evaluation.get("programme_fine")
        if fine and fine.get("pooled_choice_upper"):
            choice = str(fine["pooled_choice_upper"])
            evaluation_numbers.update({
                "eval_fine_pooled_choice_upper": choice,
                "eval_fine_pooled_kept": thousands(fine["per_release_records"][choice]),
                "eval_fine_pooled_upper": f"{fine['pooled_upper'][choice]:.3f}",
                "eval_fine_pooled_rate": f"{fine['pooled_rate'][choice]:.3f}"})
        for row in evaluation.get("decomposition") or []:
            if row["policy"] == "none":
                evaluation_numbers.update({
                    "eval_decomp_release_readings":
                        f"{row['one_release_all_readings']['match_rate']:.3f}",
                    "eval_decomp_pooled_one_reading":
                        f"{row['all_releases_one_reading']['match_rate']:.3f}",
                    "eval_decomp_pooled_one_reading_upper":
                        f"{row['all_releases_one_reading']['upper']:.3f}"})
        if counts:
            entry = counts[-1]
            evaluation_numbers.update({
                "eval_counts_epsilon":
                    f"{entry['identification']['epsilon']:g}",
                "eval_counts_precision": number(entry["identification"]["precision"]),
                "eval_counts_recall": number(
                    entry["identification"]["true_positive"]
                    / max(entry["identification"]["positives"], 1)),
                "eval_counts_utility": f"{entry['utility']:.3f}"})
        numbers.update({name: value for name, value in evaluation_numbers.items()
                        if "--" not in value and "nan" not in value and value != "None"})

    for level, row in kanon.items():
        # values printed in the generated table
        numbers[f"kanon_{level}_average"] = f"{row['average']:.4f}"
        numbers[f"kanon_{level}_upper"] = f"{row['average_upper']:.4f}"
        numbers[f"kanon_{level}_worst"] = f"{row['worst']:.3f}"
        numbers[f"kanon_{level}_worst_lower"] = f"{row['worst_lower']:.3f}"
        numbers[f"kanon_{level}_retained"] = f"{row['retained']:.3f}"
        # the rounder forms the prose quotes
        numbers[f"kanon_{level}_worst_lower_prose"] = f"{row['worst_lower']:.2f}"
    for level in (10, 20):
        if level in kanon:
            numbers[f"kanon_{level}_retained_pct"] = f"{100 * kanon[level]['retained']:.1f}"
    if 10 in kanon:
        numbers["kanon_10_worst_prose"] = f"{kanon[10]['worst']:.2f}"
    for row in grid["implied_tolerances"]:
        numbers[f"tolerance_kappa_{row['kappa']:g}".replace(".", "_")] = (
            f"{row['tolerance']:.3f}")
    return numbers


def write_evidence(results: dict, config: dict, decision: dict, output: Path) -> dict:
    """Write results, figures, LaTeX fragments and a hash manifest for the whole run."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "results.json").write_text(
        json.dumps(_finite(results), indent=2, default=float, allow_nan=False) + "\n")
    (output / "decision.json").write_text(
        json.dumps(_finite(decision), indent=2, default=float, allow_nan=False) + "\n")
    figures(results, output / "figures")
    paper = output / "paper"
    paper_tables(results, paper)
    numbers = paper_numbers(results)
    (paper / "numbers.json").write_text(json.dumps(numbers, indent=2, sort_keys=True) + "\n")
    (paper / "numbers.tex").write_text(
        "".join(f"\\newcommand{{\\run{name.replace('_', '')}}}{{{value}}}\n"
                for name, value in sorted(numbers.items())))
    manifest = {str(path.relative_to(output)): sha256(path)
                for path in sorted(output.rglob("*")) if path.is_file()
                and path.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _finite(value):
    """Replace NaN and infinities by None so the file is strict JSON."""
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value
