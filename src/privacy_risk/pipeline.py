"""Run every experiment, apply the governance gate and write the evidence.

The run does four separable jobs and labels them: proof checks, simulation checks,
falsification tests, and the governance gate. A failed falsification test cannot be
presented as a release decision: the gate returns INVALID and the process exits with
a non-zero status.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .experiments import error_model, run_all
from .frontier import choose_max_utility
from .governance import GovernanceThreshold, decide
from .reporting import write_evidence
from .validation import (
    composition_test,
    frequency_transport_error,
    full_validation_report,
    monotonicity_violations,
    stress_ratios,
)


def validate(results: dict, config: dict) -> dict:
    """Run the falsification tests that can reject the model behind the risk figure."""
    attack = results["attack"]
    measures = results["measures"]
    settings = results["grid"]["settings"]
    design = [row for row in measures["journalist"]
              if abs(row["pi"] - results["keys"]["sampling_fraction"]) < 1e-9][0]
    ece = float(attack["calibration"][0]["gap"])
    transport = frequency_transport_error(design["fitted_mean_cell"],
                                          measures["true_mean_population_cell"])
    protection = np.array([s["min_cell_count"] for s in settings], dtype=float)
    risk = np.array([s["risk"] for s in settings], dtype=float)
    finest = [s for s in settings
              if s["age_width"] == min(x["age_width"] for x in settings)
              and s["occupation_groups"] == max(x["occupation_groups"] for x in settings)]
    violations = monotonicity_violations(
        np.array([s["min_cell_count"] for s in finest], dtype=float),
        np.array([s["risk"] for s in finest], dtype=float),
        float(config.get("monotonicity_tolerance", 0.025)))
    stress = stress_ratios(np.array([s["risk_declared"] for s in settings]),
                           np.array([s["risk_stronger"] for s in settings]),
                           float(config.get("stress_tolerance", 0.05)))
    programme = results["programme"]
    composition = composition_test(programme["pooled"][-1], programme["union_upper"],
                                   config.get("programme", {}).get("tau_programme"),
                                   programme.get("pooled_upper"))
    report = full_validation_report(ece, float(config.get("calibration_ece_limit", 0.06)),
                                    transport,
                                    float(config.get("frequency_mae_limit", 1.5)),
                                    violations, stress, composition)
    report["metrics"]["protection_levels"] = protection.tolist()
    report["metrics"]["risk_by_setting"] = risk.tolist()
    return report


def gate(results: dict, config: dict, validation: dict) -> dict:
    """Apply the prespecified tolerance to the bound, not to the point estimate."""
    settings = results["grid"]["settings"]
    labels = np.array([s["label"] for s in settings])
    upper = np.array([s["risk_upper"] for s in settings])
    utility = np.array([s["utility"] for s in settings])
    tolerance = float(config["risk_tolerance"])
    chosen = choose_max_utility(labels, upper, utility, tolerance)
    threshold = GovernanceThreshold(tolerance, str(config["owner"]),
                                    str(config["expiry_date"]),
                                    "declared key error plus one noisy covariate",
                                    str(config["harm_class"]),
                                    float(config.get("alpha", 0.05)))
    if chosen is None:
        record = decide(1.0, threshold, validation["passed"],
                        config.get("assessment_date", "2026-09-04"))
        record["chosen_setting"] = None
    else:
        index = int(np.flatnonzero(labels == chosen)[0])
        record = decide(float(upper[index]), threshold, validation["passed"],
                        config.get("assessment_date", "2026-09-04"))
        record["chosen_setting"] = str(chosen)
        record["point_estimate"] = float(settings[index]["risk"])
        record["replicates"] = int(settings[index]["replicates"])
    programme = validation["programme"]
    pooled_final = results["programme"].get("pooled_upper")
    record["release_status"] = record["status"]
    record["release_reason"] = record["reason"]
    if (record["status"] == "PASS" and pooled_final is not None
            and pooled_final > tolerance):
        # a per-release pass does not carry over to the programme: the declared
        # programme's pooled risk is above the release tolerance, so the gate fails
        record["status"] = "FAIL"
        record["reason"] = ("per-release upper bound is within tolerance, but the upper "
                            "bound on the pooled risk of the declared programme exceeds "
                            "the release tolerance")
    record["programme"] = {
        "pooled_risk": programme["pooled"],
        "union_upper_bound": programme["union_upper"],
        "marginals_bound_the_programme": programme["marginals_bound_the_programme"],
        "upper_within_programme_tolerance": programme.get(
            "upper_within_programme_tolerance"),
        "programme_tolerance_used_by_validation": config.get("programme", {}).get(
            "tau_programme"),
        "pooled_upper_bound": pooled_final,
        "pooled_upper_exceeds_release_tolerance": (pooled_final is not None
                                                   and pooled_final > tolerance),
        "note": ("Marginal risks do not bound a pooled attack. A programme needs its own "
                 "tolerance and its own joint simulation."),
    }
    record["adversary"] = str(error_model(config))
    record["interpretation"] = ("Synthetic evidence only. A PASS is conditional on the "
                               "declared adversary and the stated tolerance; it is not "
                               "authority to release real data.")
    return record


def run(config: dict, output: Path) -> dict:
    results = run_all(config)
    validation = validate(results, config)
    decision = gate(results, config, validation)
    results["validation"] = validation
    results["decision"] = decision
    manifest = write_evidence(results, config, decision, Path(output))
    results["manifest"] = manifest
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--output", default="artifacts")
    arguments = parser.parse_args()
    config = json.loads(Path(arguments.config).read_text())
    result = run(config, Path(arguments.output))
    print(json.dumps(result["decision"], indent=2, default=float))
    if not result["validation"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
