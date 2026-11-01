import json
from pathlib import Path

import numpy as np

from privacy_risk.pipeline import gate, run, validate

CONFIG = json.loads(Path("configs/base.json").read_text())
SMALL = {**CONFIG, "population_size": 3000, "sample_fraction": 0.25,
         "journalist_fractions": [0.25], "kanonymity_levels": [3],
         "risk_distribution_levels": [1, 3],
         "grid": {"age_widths": [10, 20], "occupation_groups": [12, 3],
                  "min_cell_counts": [1, 5]},
         "attack_replicates": {"headline": 200, "key_only": 100, "kanon": 150,
                               "kanon_key": 100, "grid": 60, "per_target": 20,
                               "composition_targets": 200},
         "per_target_cap": 10, "minimum_release_size": 20,
         "frequency_mae_limit": 50.0, "calibration_ece_limit": 0.5,
         # the loose test band cannot reject the misspecified model, so the power
         # requirement is switched off here; test_calibration_band_needs_power checks it
         "require_calibration_power": False,
         "monotonicity_tolerance": 0.5, "stress_tolerance": 0.5,
         "credit": {"borrowers": 800, "lenders": 12, "months": 3,
                    "min_cell_counts": [1, 5], "minimum_contributors": [3]},
         "nested": {"outer_replicates": 4, "inner_replicates": 40},
         "real": {"puma_widths": [100], "education_widths": [4],
                  "experience_widths": [10], "min_cell_counts": [1, 5],
                  "minimum_release_size": 500, "per_target_replicates": 20,
                  "per_target_cap": 8, "composition_targets": 200,
                  "composition_kmin": 3}}


def test_end_to_end(tmp_path: Path):
    result = run(SMALL, tmp_path)
    assert (tmp_path / "manifest.json").exists()
    assert result["decision"]["status"] in {"PASS", "FAIL", "INVALID"}
    assert "programme" in result["decision"]


def test_a_failed_falsification_test_blocks_the_decision(tmp_path: Path):
    strict = {**SMALL, "calibration_ece_limit": 0.0}
    result = run(strict, tmp_path)
    assert not result["validation"]["passed"]
    assert result["decision"]["status"] == "INVALID"


def test_the_gate_uses_the_bound_and_can_refuse_every_setting(tmp_path: Path):
    result = run(SMALL, tmp_path)
    validation = validate(result, SMALL)
    tight = gate(result, {**SMALL, "risk_tolerance": 1e-6}, validation)
    assert tight["status"] == "FAIL" and tight["chosen_setting"] is None
    generous = gate(result, {**SMALL, "risk_tolerance": 1.0}, validation)
    assert generous["upper_risk"] <= 1.0
    assert generous["point_estimate"] <= generous["upper_risk"]


def test_validation_records_the_programme_finding(tmp_path: Path):
    result = run(SMALL, tmp_path)
    programme = result["validation"]["programme"]
    assert np.isfinite(programme["ratio"])
    assert "marginals_bound_the_programme" in programme


def test_calibration_band_needs_power(tmp_path: Path):
    """A band wide enough to pass the misspecified model invalidates the decision."""
    result = run({**SMALL, "require_calibration_power": True}, tmp_path)
    checks = result["validation"]["checks"]
    assert "calibration_band_has_power" in checks
    assert checks["calibration_band_has_power"] is False
    assert result["decision"]["status"] == "INVALID"


def test_programme_check_and_gate_use_one_tolerance(tmp_path: Path):
    result = run({**SMALL, "programme": {**SMALL["programme"], "pooled_tolerance": 0.05}},
                 tmp_path)
    record = result["decision"]["programme"]
    assert record["programme_tolerance"] == 0.05
    assert result["validation"]["programme"]["programme_tolerance"] == 0.05
    assert "programme_tolerance_used_by_validation" not in record
