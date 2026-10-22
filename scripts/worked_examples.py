"""Recompute every number in the long form's worked examples (Section sec:worked).

    python scripts/worked_examples.py --output artifacts-robustness/worked_examples.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scipy.stats import beta, norm

TAU, ALPHA = 0.12, 0.05


def upper(y: int, n: int, a: float = ALPHA) -> float:
    return float(beta.ppf(1 - a, y + 1, n - y))


def length(p: float, q: float, weight: float = 1.0) -> float:
    mu = p * math.log(q / TAU) + (1 - p) * math.log((1 - q) / (1 - TAU))
    return (math.log(1 / (ALPHA * weight)) + math.log((1 - q) / (1 - TAU))) / mu


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts-robustness/worked_examples.json")
    args = parser.parse_args()
    rob = json.loads(Path("artifacts-robustness/robustness.json").read_text())
    conf = rob["confirmation_grid"]
    sel = rob["selection"]["pooled_one_history"]
    ex1 = conf["single_one"]["1"]
    ex2 = {"selection_upper": {k: sel[k]["upper"] for k in
                               ["300", "200", "150", "100", "75", "50", "40"]},
           "confirm_50": conf["pooled_one_history"]["50"],
           "confirm_75": conf["pooled_one_history"]["75"]}
    reading, sd = 12.30, 0.6
    cell = {"t": 12.0, "a": 12.7, "b": 12.25, "c": 11.6}
    dens = {k: norm.pdf(reading, v, sd) for k, v in cell.items()}
    total = sum(dens.values())
    without = {k: v / total for k, v in dens.items()}
    kept = dens["t"] + dens["a"]
    with_history = {"t": dens["t"] / kept, "a": dens["a"] / kept}
    q = 0.05
    fail, success = (1 - q) / (1 - TAU), q / TAU
    payload = {
        "example_bound": {"successes": ex1["successes"], "trials": ex1["trials"],
                          "rate": ex1["rate"], "upper": upper(ex1["successes"],
                                                              ex1["trials"]),
                          "margin": 1 - upper(ex1["successes"], ex1["trials"]) / TAU},
        "example_selection": ex2,
        "example_history": {"reading": reading, "sd": sd, "cell": cell,
                            "posterior_without_history": without,
                            "posterior_with_history": with_history},
        "example_zero": {"full_weight": math.ceil(math.log(1 / ALPHA) / -math.log(1 - TAU)),
                         "third_weight": math.ceil(math.log(3 / ALPHA)
                                                   / -math.log(1 - TAU)),
                         "log_20": math.log(20), "log_60": math.log(60),
                         "step": -math.log(1 - TAU)},
        "example_gate": {"failure_factor": fail, "success_factor": success,
                         "failures_to_certify": math.log(20) / math.log(fail),
                         "failures_per_success": math.log(1 / success) / math.log(fail),
                         "drift_at_0.05": 0.05 * math.log(success)
                         + 0.95 * math.log(fail),
                         "drift_at_0.116": 0.116 * math.log(success)
                         + 0.884 * math.log(fail)},
        "example_length": {"k150": length(0.0913, 0.0913), "k100": length(0.1162, 0.1162)},
        "example_grinding": {"single": 0.0381, "ten": 1 - (1 - 0.0381) ** 10,
                             "hundred": 1 - (1 - 0.0381) ** 100},
        "example_dp": {"bound": math.exp(1) * 0.03 + 1e-6},
    }
    Path(args.output).write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1)[:1500])


if __name__ == "__main__":
    main()
