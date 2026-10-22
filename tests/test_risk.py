import numpy as np
import pytest

from privacy_risk.risk import journalist_unique_probability, marketer_sample_risk, prosecutor_risk


def test_prosecutor_and_marketer_identity():
    keys = np.array([1, 1, 2, 3, 3, 3])
    assert np.allclose(prosecutor_risk(keys), [0.5, 0.5, 1, 1/3, 1/3, 1/3])
    assert marketer_sample_risk(keys) == pytest.approx(3/6)
    assert prosecutor_risk(keys).mean() == pytest.approx(marketer_sample_risk(keys))

def test_journalist_census():
    assert journalist_unique_probability(1, {1: .2, 2: .8}) == pytest.approx(1)



def test_frequency_model_recovers_the_population_mean():
    import numpy as np

    from privacy_risk.risk import fit_frequency_model, journalist_from_fit
    rng = np.random.default_rng(1)
    population = rng.negative_binomial(2, 2 / (2 + 12), size=4000) + 0
    sample = rng.binomial(population, 0.25)
    fit = fit_frequency_model(sample[sample > 0], 4000, 0.25)
    assert 6 < fit.mean < 24
    assert 0 < journalist_from_fit(0.25, fit) < 1


def test_unconditional_marketer_applies_the_sampling_fraction():
    import numpy as np

    from privacy_risk.risk import marketer_unconditional_risk, prosecutor_risk
    keys = np.array([1, 1, 2, 3])
    assert marketer_unconditional_risk(keys, 0.5) == pytest.approx(
        0.5 * prosecutor_risk(keys).mean())
    with pytest.raises(ValueError):
        marketer_unconditional_risk(keys, 0)


def test_journalist_truth_uses_the_known_population():
    import numpy as np

    from privacy_risk.risk import journalist_truth
    population = np.array([1, 1, 2, 3])
    assert journalist_truth(np.array([2, 3]), population) == 1.0
    assert journalist_truth(np.array([1]), population) == 0.0
