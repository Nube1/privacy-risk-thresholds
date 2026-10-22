"""Render the public-file neural comparison without editing its measurements."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    result = json.loads(Path("artifacts-real/neural_comparison.json").read_text())
    splits = result["results"]
    x = list(range(1, len(splits)+1))
    styles = (("one_layer", "One hidden layer", "#286C76", -.23),
              ("three_layer", "Three hidden layers", "#AA5546", 0),
              ("quadratic_logistic", "Quadratic logistic", "#242B35", .23))
    fig, ax = plt.subplots(figsize=(6.2, 3.5), layout="constrained")
    for key, label, color, offset in styles:
        reduction = [split["scores"]["logistic"]["log_loss"]
                     - split["scores"][key]["log_loss"] for split in splits]
        ax.bar([i+offset for i in x], reduction, width=.21,
               color=color, label=label)
    ax.set_xticks(x, [f"Split {i}" for i in x])
    ax.set_ylabel("Log-loss reduction vs linear logistic")
    ax.set_ylim(0, .0085)
    ax.grid(axis="y", linewidth=.5, alpha=.4)
    ax.legend(ncol=3, fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    root = Path("paper/diagrams")
    root.mkdir(parents=True, exist_ok=True)
    fig.savefig(root / "realdata_neural_comparison.svg", metadata={"Date": None})
    fig.savefig(root / "realdata_neural_comparison.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{State-held-out binomial log loss on public Census records. "
             r"The three splits overlap, so these are descriptive replications, "
             r"not independent studies. Lower is better.}",
             r"\label{tab:real-neural}", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Split & Linear logistic & One layer & Three layers & Quadratic logistic \\",
             r"\midrule"]
    for i, split in enumerate(splits, 1):
        s = split["scores"]
        lines.append(f"{i} & {s['logistic']['log_loss']:.4f} & "
                     f"{s['one_layer']['log_loss']:.4f} & "
                     f"{s['three_layer']['log_loss']:.4f} & "
                     f"{s['quadratic_logistic']['log_loss']:.4f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    Path("paper/generated/table_real_neural.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
