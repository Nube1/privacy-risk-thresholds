import numpy as np
import pytest

from privacy_risk.linkage import (
    exact_match_probabilities,
    maximum_posterior_success,
    posterior_linkage,
)


def test_uniform_candidates_recover_reciprocal_frequency():
    p = posterior_linkage(np.ones(4), np.ones(4))
    assert np.allclose(p, 0.25) and maximum_posterior_success(p) == pytest.approx(0.25)

def test_exact_match():
    assert np.allclose(exact_match_probabilities([1, 2, 3], [1, 1, 2]), [0.5, 1, 0])

def test_bad_posterior_refuses():
    with pytest.raises(ValueError):
        posterior_linkage([0], [0])



def test_attack_recovers_reciprocal_frequency_without_extra_information():
    from privacy_risk.linkage import ErrorModel, attack, minimum_cell_targets, per_target_risk
    from privacy_risk.risk import prosecutor_risk
    from privacy_risk.synthetic import identity_release, make_register
    register = make_register({"seed": 11, "population_size": 3000, "sample_fraction": 0.3})
    release = identity_release(register)
    blind = ErrorModel(age_error=0.0, occupation_error=0.0)
    result = attack(register, release, blind, 800, np.random.default_rng(3),
                    use_covariate=False)
    theory = prosecutor_risk(release.key).mean()
    assert result.rate == pytest.approx(theory, abs=0.06)
    assert result.trials == 800 and set(result.as_dict()) == {"rate", "successes", "trials"}
    informed = attack(register, release, ErrorModel(), 800, np.random.default_rng(4))
    assert informed.rate > 0
    targets = minimum_cell_targets(release, 1, 5, np.random.default_rng(5))
    per_target = per_target_risk(register, release, blind, targets, 20,
                                 np.random.default_rng(6))
    assert per_target.size == targets.size
    assert per_target_risk(register, release, blind, np.array([]), 5,
                           np.random.default_rng(7)).size == 0


def test_error_model_refuses_impossible_settings():
    from privacy_risk.linkage import ErrorModel
    with pytest.raises(ValueError):
        ErrorModel(age_error=1.0)
    with pytest.raises(ValueError):
        ErrorModel(covariate_sd=0)
