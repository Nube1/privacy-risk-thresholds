"""Figure and table of what the neural network predicted, against the true risks.

    python scripts/neural_results.py

Reads artifacts-neural/neural_pipeline.json (the network's frozen predictions) and
paper/publication_versions/chain_audit.json (true risks from 100,000 separate
targets per coordinate). Writes paper/figures/fig_neural_predictions.pdf and
paper/generated/table_neural_fit.tex. The figure marks each series twice, by shade
and by line style, so it reads in black and white.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LEVELS = [1, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300]
ATTACKS = [("single", "(a) One release, one reading"),
           ("pooling_alone", "(b) Pooled, history"),
           ("full_pooled", "(c) Pooled, history, readings")]


def main() -> None:
    neural = json.loads(Path("artifacts-neural/neural_pipeline.json").read_text())
    truth = json.loads(Path("paper/publication_versions/chain_audit.json").read_text())["truth"]
    tau = neural["tolerance"]
    dev_levels = neural["development"]["levels"]
    counts = neural["development"]["counts"]
    n_dev = neural["development"]["targets"]

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "FreeSerif", "DejaVu Serif"],
                         "mathtext.fontset": "stix", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=True)
    for ax, (key, title) in zip(axes, ATTACKS):
        t = [truth[key][str(k)]["rate"] for k in LEVELS]
        net = [neural["plans"]["network"]["predictions"][f"{key}:{k}"] for k in LEVELS]
        log = [neural["plans"]["logistic"]["predictions"][f"{key}:{k}"] for k in LEVELS]
        dev = [counts[f"{key}:{k}"] / n_dev for k in dev_levels]
        ax.plot(dev_levels, dev, linestyle="none", marker="x", color="0.55", ms=3.5,
                label="development counts")
        ax.plot(LEVELS, t, color="0.0", linestyle=":", linewidth=1.4,
                label="true risk (100,000 targets)")
        ax.plot(LEVELS, net, color="0.15", linestyle="-", marker="o", ms=3,
                label="network prediction")
        ax.plot(LEVELS, log, color="0.5", linestyle="--", marker="s", ms=3,
                markerfacecolor="white", label="logistic prediction")
        ax.axhline(tau, color="0.3", linewidth=0.8, linestyle="-.")
        ax.text(1.1, tau + 0.004, "tolerance", fontsize=7, color="0.3")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100, 300])
        ax.set_xticklabels(["1", "10", "100", "300"])
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("cell-count floor $k$")
    axes[0].set_ylabel("success probability")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.tight_layout()
    out = Path("paper/figures/fig_neural_predictions.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)

    rows = []
    for label, name in (("network", "Network, two hidden layers"),
                        ("logistic", "Logistic model")):
        m = neural["models"][label]["metrics"]
        c = neural["calibration"][label]
        plan = neural["plans"][label]
        rows.append(f"{name} & {m['training_log_loss']:.4f} & "
                    f"{m['validation_log_loss']:.4f} & {m['best_epoch']} & "
                    f"{c['mean_absolute_error']:.3f} & {c['max_absolute_error']:.3f} & "
                    f"{plan['predicted_floor']} & {plan['start_floor']} \\\\\n")
    caption = (
        "What the two models learned from the development attacks. Log losses are the "
        "binomial log loss per development attack on the training and held-out floors; "
        "the epoch is where held-out loss was lowest. Errors compare each model's frozen "
        "prediction with the true risk, estimated from 100{,}000 separate targets, over "
        "the 39 coordinates of the release grid. The predicted floor is the last floor, "
        "from $k=300$ down, at which every predicted risk lies below the tolerance; the "
        "start floor is one step stricter.")
    Path("paper/generated/table_neural_fit.tex").write_text(
        "\\begin{table}[t]\n\\centering\n\\caption{" + caption + "}\n"
        "\\label{tab:neural-fit}\n\\small\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
        " & \\multicolumn{2}{c}{Log loss} & & \\multicolumn{2}{c}{Error against truth} "
        "& \\multicolumn{2}{c}{Floor}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}"
        "\\cmidrule(lr){7-8}\n"
        "Model & Training & Held out & Epoch & Mean & Largest & Predicted & Start \\\\\n"
        "\\midrule\n" + "".join(rows) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(out)


if __name__ == "__main__":
    main()
