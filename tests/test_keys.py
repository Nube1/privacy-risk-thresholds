import numpy as np

from privacy_risk.keys import key_frequencies, sample_population_frequencies


def test_frequencies_and_order():
    f, F = sample_population_frequencies(np.array([1, 1, 2, 3]), np.array([0, 2]))
    assert f == {1: 1, 2: 1} and F == {1: 2, 2: 1, 3: 1}
    assert key_frequencies(np.array([4, 4])) == {4: 2}

