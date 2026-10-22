"""Synthetic register, sampling design and protection mechanisms.

The register is the one described in the paper's simulation section. No real or
confidential data is read anywhere in this package.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def skewed_probabilities(categories: int, exponent: float) -> np.ndarray:
    weights = 1.0 / np.arange(1, categories + 1) ** exponent
    return weights / weights.sum()


@dataclass(frozen=True)
class Register:
    """A finite population with quasi-identifiers and one confidential variable."""

    frame: pd.DataFrame
    sample_index: np.ndarray
    age_min: int
    age_max: int
    fine_band_width: int
    districts: int
    occupations: int
    sampling_fraction: float

    @property
    def size(self) -> int:
        return len(self.frame)

    @property
    def released(self) -> pd.DataFrame:
        return self.frame.loc[self.sample_index]

    @property
    def fine_bands(self) -> int:
        return (self.age_max - self.age_min) // self.fine_band_width

    def bands(self, age: np.ndarray, width: int) -> np.ndarray:
        count = (self.age_max - self.age_min) // width
        return np.minimum((np.asarray(age) - self.age_min) // width, count - 1)

    def collapse(self, occupation: np.ndarray, groups: int) -> np.ndarray:
        occupation = np.asarray(occupation)
        return occupation % groups if groups < self.occupations else occupation


def make_register(config: dict) -> Register:
    """Draw the population and the confidential sample from one seeded generator."""
    rng = np.random.default_rng(int(config["seed"]))
    n = int(config["population_size"])
    districts = int(config.get("districts", 40))
    occupations = int(config.get("occupations", 12))
    age_min, age_max = int(config.get("age_min", 18)), int(config.get("age_max", 88))

    district = rng.choice(districts, size=n, p=skewed_probabilities(districts, 0.9))
    sex = rng.integers(0, 2, size=n)
    age = rng.integers(age_min, age_max, size=n)
    keep = rng.random(n) < np.exp(-0.02 * (age - age_min))
    age = np.where(keep, age, rng.integers(age_min, 60, size=n))
    occupation = rng.choice(occupations, size=n, p=skewed_probabilities(occupations, 0.8))

    district_effect = rng.normal(0, 0.45, districts)
    occupation_effect = rng.normal(0, 0.35, occupations)
    outcome = (8.0 + district_effect[district] + occupation_effect[occupation]
               + 0.018 * (age - 50) + 0.10 * sex + rng.normal(0, 0.60, n))

    fraction = float(config["sample_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must be in (0, 1]")
    included = rng.random(n) < fraction

    frame = pd.DataFrame({"district": district, "sex": sex, "age": age,
                          "occupation": occupation, "outcome": outcome})
    return Register(frame=frame, sample_index=np.flatnonzero(included),
                    age_min=age_min, age_max=age_max,
                    fine_band_width=int(config.get("fine_band_width", 5)),
                    districts=districts, occupations=occupations,
                    sampling_fraction=fraction)


def resample(register: Register, seed: int) -> Register:
    """Draw a fresh sample from the same finite population.

    The population is held fixed and only the sampling design is redrawn, which is
    what separates design variance from Monte Carlo variance.
    """
    rng = np.random.default_rng(seed)
    included = rng.random(len(register.frame)) < register.sampling_fraction
    return Register(frame=register.frame, sample_index=np.flatnonzero(included),
                    age_min=register.age_min, age_max=register.age_max,
                    fine_band_width=register.fine_band_width,
                    districts=register.districts, occupations=register.occupations,
                    sampling_fraction=register.sampling_fraction)


def pack_key(district: np.ndarray, sex: np.ndarray, band: np.ndarray, group: np.ndarray,
             bands: int, groups: int) -> np.ndarray:
    """Injective encoding of a key tuple into one integer."""
    if bands < 1 or groups < 1:
        raise ValueError("bands and groups must be positive")
    return ((np.asarray(district) * 2 + np.asarray(sex)) * bands
            + np.asarray(band)) * groups + np.asarray(group)


@dataclass(frozen=True)
class Mechanism:
    """A point in the mechanism family: key granularity plus a minimum cell count."""

    age_width: int = 5
    occupation_groups: int = 12
    min_cell_count: int = 1

    def label(self) -> str:
        return (f"{self.age_width}-year bands, {self.occupation_groups} groups, "
                f"k_min = {self.min_cell_count}")


@dataclass(frozen=True)
class Release:
    """The records a mechanism releases, with their released key components."""

    index: np.ndarray
    district: np.ndarray
    sex: np.ndarray
    band: np.ndarray
    group: np.ndarray
    bands: int
    groups: int

    @property
    def size(self) -> int:
        return int(self.index.size)

    @property
    def key(self) -> np.ndarray:
        return pack_key(self.district, self.sex, self.band, self.group, self.bands, self.groups)


def suppress_small_cells(release: Release, minimum: int) -> Release:
    """Drop every record whose released key class holds fewer than `minimum` records."""
    if minimum <= 1:
        return release
    key = release.key
    values, counts = np.unique(key, return_counts=True)
    keep = np.isin(key, values[counts >= minimum])
    return Release(index=release.index[keep], district=release.district[keep],
                   sex=release.sex[keep], band=release.band[keep], group=release.group[keep],
                   bands=release.bands, groups=release.groups)


def apply_mechanism(register: Register, mechanism: Mechanism) -> Release:
    """Coarsen the key as the mechanism dictates, then suppress small cells."""
    released = register.released
    bands = (register.age_max - register.age_min) // mechanism.age_width
    release = Release(index=register.sample_index,
                      district=released.district.to_numpy(),
                      sex=released.sex.to_numpy(),
                      band=register.bands(released.age.to_numpy(), mechanism.age_width),
                      group=register.collapse(released.occupation.to_numpy(),
                                              mechanism.occupation_groups),
                      bands=bands, groups=mechanism.occupation_groups)
    return suppress_small_cells(release, mechanism.min_cell_count)


def identity_release(register: Register) -> Release:
    """The unprotected release: the finest key and no suppression."""
    return apply_mechanism(register, Mechanism(register.fine_band_width,
                                               register.occupations, 1))


def make_population(n: int, seed: int) -> pd.DataFrame:
    """Backwards-compatible helper retained for the earlier public interface."""
    register = make_register({"seed": seed, "population_size": n, "sample_fraction": 1.0})
    frame = register.frame.copy()
    frame.insert(0, "unit", np.arange(len(frame)))
    frame["key"] = pack_key(frame.district.to_numpy(), frame.sex.to_numpy(),
                            register.bands(frame.age.to_numpy(), register.fine_band_width),
                            frame.occupation.to_numpy(), register.fine_bands,
                            register.occupations)
    return frame


def sample_population(population: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("sample fraction must be in (0, 1]")
    return population.sample(frac=fraction, replace=False, random_state=seed).sort_index()
