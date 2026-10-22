"""Rebuild the synthetic SENTRY-R logistic/deep comparison over seeded replicates.

For each seed the script draws three disjoint toy programmes from the law in
``sentry_r_example.py``: a training programme, a stopping programme used only
for early stopping, and a test programme used only for the reported loss. Both
models train until the patience rule stops them; a run whose best epoch reaches
the cap is refused, so no reported model is cut off by the cap.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from privacy_risk.sentry_r import DeepRiskNet, ScenarioSchema, digest, matrix, train_model

SEEDS = tuple(range(101, 111))
MODELS = ((0, "Logistic"), (3, "Three-layer ReLU"))
TRAIN_ROWS, STOP_ROWS, TEST_ROWS = 320, 100, 2000
EPOCH_CAP, PATIENCE, WIDTH = 6000, 150, 24


def load_example_module():
    path = Path(__file__).with_name("sentry_r_example.py")
    spec = importlib.util.spec_from_file_location("sentry_r_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(model: dict, rows: list[dict], schema: ScenarioSchema) -> tuple[float, float]:
    x, k, b = matrix(rows, schema)
    net = DeepRiskNet.from_dict(model["network"])
    z = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    p = net.predict(z)
    log_loss = net.loss_grads(z, k, b, 0.)[0]
    brier = float(np.sum(k*(1-p)**2 + (b-k)*p*p) / np.sum(b))
    return log_loss, brier


def run(root: Path) -> dict:
    schema = ScenarioSchema.load(root / "schema.json")
    toy = load_example_module()
    replicates = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        train = toy.toy_development(rng, TRAIN_ROWS, f"bench-{seed}-train")
        stop = toy.toy_development(rng, STOP_ROWS, f"bench-{seed}-stop")
        test = toy.toy_development(rng, TEST_ROWS, f"bench-{seed}-test")
        result = {"seed": seed, "test_sha256": digest(test)}
        for depth, name in MODELS:
            model = train_model(train, stop, schema, width=WIDTH, depth=depth,
                                epochs=EPOCH_CAP, patience=PATIENCE, seed=seed)
            best = model["metrics"]["best_epoch"]
            if best >= EPOCH_CAP:
                raise RuntimeError(f"seed {seed} {name}: best epoch hit the cap")
            log_loss, brier = evaluate(model, test, schema)
            result[name] = {"test_log_loss": log_loss, "test_brier": brier,
                            "best_epoch": best,
                            "stopping_log_loss": model["metrics"]["validation_log_loss"]}
        replicates.append(result)
    summary = []
    for _, name in MODELS:
        loss = np.array([r[name]["test_log_loss"] for r in replicates])
        brier = np.array([r[name]["test_brier"] for r in replicates])
        epochs = [r[name]["best_epoch"] for r in replicates]
        summary.append({"model": name, "mean_test_log_loss": float(loss.mean()),
                        "sd_test_log_loss": float(loss.std(ddof=1)),
                        "mean_test_brier": float(brier.mean()),
                        "sd_test_brier": float(brier.std(ddof=1)),
                        "best_epoch_min": int(min(epochs)),
                        "best_epoch_max": int(max(epochs))})
    diff = np.array([r["Three-layer ReLU"]["test_log_loss"] - r["Logistic"]["test_log_loss"]
                     for r in replicates])
    return {"status": "synthetic only; not a comparison on real releases",
            "seeds": list(SEEDS), "train_rows": TRAIN_ROWS, "stopping_rows": STOP_ROWS,
            "test_rows": TEST_ROWS, "epoch_cap": EPOCH_CAP, "patience": PATIENCE,
            "width": WIDTH, "schema_sha256": digest(schema.as_dict()),
            "summary": summary,
            "paired_difference_deep_minus_logistic": {
                "mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                "min": float(diff.min()), "max": float(diff.max()),
                "seeds_deep_lower": int(np.sum(diff < 0))},
            "replicates": replicates}


def table(result: dict) -> str:
    d = result["paired_difference_deep_minus_logistic"]
    caption = (
        "Synthetic SENTRY-R comparison over "
        f"{len(result['seeds'])} seeded replicates of the toy law. Each replicate "
        f"draws {result['train_rows']} training rows, {result['stopping_rows']} rows "
        "used only for early stopping (patience "
        f"{result['patience']} epochs, cap {result['epoch_cap']}) and "
        f"{result['test_rows']} test rows used only for the reported losses. "
        "Entries are mean (standard deviation) across replicates; smaller is better. "
        "The last row is the paired deep-minus-logistic test log loss "
        f"(range {d['min']:.4f} to {d['max']:.4f}; deep lower in "
        f"{d['seeds_deep_lower']} of {len(result['seeds'])}). "
        "These are not results on real release programmes.")
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\caption{" + caption + "}",
             r"\label{tab:sentry-synthetic}", r"\begin{tabular}{lrrr}", r"\toprule",
             r"Proposal model & Test log loss & Test Brier score & Best epoch (range) \\",
             r"\midrule"]
    for row in result["summary"]:
        lines.append(f'{row["model"]} & {row["mean_test_log_loss"]:.4f} '
                     f'({row["sd_test_log_loss"]:.4f}) & {row["mean_test_brier"]:.4f} '
                     f'({row["sd_test_brier"]:.4f}) & {row["best_epoch_min"]}--'
                     f'{row["best_epoch_max"]} \\\\')
    lines.append(r"\midrule")
    lines.append(f'Deep minus logistic & {d["mean"]:.4f} ({d["sd"]:.4f}) & & \\\\')
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("examples/sentry_r"),
                        help="directory holding schema.json")
    parser.add_argument("--output", type=Path, default=Path("artifacts-sentry/benchmark.json"))
    parser.add_argument("--table", type=Path,
                        default=Path("paper/generated/table_sentry_benchmark.tex"))
    args = parser.parse_args()
    result = run(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(table(result))
    print(f"wrote {args.output} and {args.table}")


if __name__ == "__main__":
    main()
