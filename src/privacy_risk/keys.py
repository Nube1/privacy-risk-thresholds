from __future__ import annotations

import numpy as np


def key_frequencies(keys: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(np.asarray(keys), return_counts=True)
    return {int(k): int(v) for k, v in zip(values, counts)}


def sample_population_frequencies(population_keys: np.ndarray, sample_indices: np.ndarray):
    population_keys = np.asarray(population_keys)
    sample_indices = np.asarray(sample_indices, dtype=int)
    return key_frequencies(population_keys[sample_indices]), key_frequencies(population_keys)

