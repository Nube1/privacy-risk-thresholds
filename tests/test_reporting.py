import json
from pathlib import Path

import pytest

from privacy_risk.pipeline import run
from privacy_risk.reporting import paper_numbers, thousands

FAST = {
    "seed": 20260904, "population_size": 4000, "sample_fraction": 0.25,
    "journalist_fractions": [0.1, 0.25], "kanonymity_levels": [3, 5],
    "risk_distribution_levels": [1, 3],
    "grid": {"age_widths": [10, 20], "occupation_groups": [12, 3], "min_cell_counts": [1, 5]},
    "attack_replicates": {"headline": 200, "key_only": 100, "kanon": 150, "kanon_key": 100,
                          "grid": 60, "per_target": 20, "composition_targets": 200},
    "per_target_cap": 10, "minimum_release_size": 20, "kappas": [0.5, 5.0],
    "programme": {"kmin": 3, "weights": [1.0, 0.5], "tau_programme": 0.4},
    "scaling": {"population_sizes": [2000, 4000], "targets": 150},
    "certification": {"taus": [0.01], "alphas": [0.05]},
    "dp": {"epsilons": [1.0], "priors": [0.1]},
    "credit": {"borrowers": 800, "lenders": 12, "months": 3,
               "min_cell_counts": [1, 5], "minimum_contributors": [3]},
    "nested": {"outer_replicates": 4, "inner_replicates": 40},
    "real": {"puma_widths": [100], "education_widths": [4], "experience_widths": [10],
             "min_cell_counts": [1, 5], "minimum_release_size": 500,
             "per_target_replicates": 20, "per_target_cap": 8,
             "composition_targets": 200, "composition_kmin": 3},
    "risk_tolerance": 0.9, "alpha": 0.05, "assessment_date": "2026-09-04",
    "owner": "Owner", "expiry_date": "2027-09-04", "harm_class": "test",
    "calibration_ece_limit": 0.5, "frequency_mae_limit": 50.0,
    "monotonicity_tolerance": 0.5, "stress_tolerance": 0.5,
}


def test_thousands_uses_latex_separators():
    assert thousands(1234567) == "1{,}234{,}567"


def test_run_writes_every_piece_of_evidence(tmp_path: Path):
    results = run(FAST, tmp_path)
    for name in ("config.json", "results.json", "decision.json", "manifest.json"):
        assert (tmp_path / name).exists()
    for number in range(1, 6):
        assert list((tmp_path / "figures").glob(f"fig{number}_*.pdf"))
    tables = sorted(p.name for p in (tmp_path / "paper").glob("table_*.tex"))
    assert tables == ["table_ablation.tex", "table_adversary.tex",
                      "table_anytime.tex", "table_certification.tex",
                      "table_credit.tex", "table_evaluation.tex",
                      "table_kanonymity.tex", "table_measures.tex",
                      "table_programme.tex", "table_programme_choice.tex",
                      "table_real_census.tex",
                      "table_scaling.tex",
                      "table_tolerance.tex"]
    numbers = json.loads((tmp_path / "paper" / "numbers.json").read_text())
    assert numbers["population"] == "4{,}000"
    assert results["decision"]["status"] in {"PASS", "FAIL", "INVALID"}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "results.json" in manifest and len(next(iter(manifest.values()))) == 64


def test_tables_are_valid_latex_fragments(tmp_path: Path):
    run(FAST, tmp_path)
    for path in (tmp_path / "paper").glob("table_*.tex"):
        text = path.read_text()
        assert text.count("\\begin{table}") == text.count("\\end{table}") == 1
        assert text.count("\\begin{tabular}") == text.count("\\end{tabular}") == 1
        assert "\\label{tab:" in text and "\\caption{" in text


def test_paper_numbers_need_a_full_result_set():
    with pytest.raises(KeyError):
        paper_numbers({"keys": {}})


def test_shipped_manifest_matches_the_files_it_lists():
    import hashlib
    root = Path(__file__).resolve().parent.parent / "artifacts-paper"
    manifest = json.loads((root / "manifest.json").read_text())
    entries = manifest.get("files", manifest)
    assert entries
    for name, digest in entries.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name


def test_shipped_results_are_strict_json():
    root = Path(__file__).resolve().parent.parent / "artifacts-paper"
    json.loads((root / "results.json").read_text(),
               parse_constant=lambda token: pytest.fail(f"non-finite value {token}"))
