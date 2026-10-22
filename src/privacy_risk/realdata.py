"""Running the framework on real public microdata.

Two files ship with the `wooldridge` package and are used here as released
microdata, not as confidential holdings: an extract of the US Census 2000 public
use sample, and the Boston Fed mortgage application study. Both are already
public, and this module measures what a file of that shape discloses under a
declared adversary. It does not read confidential data, and it is the adapter
pattern that `docs/CLAIMS_EVIDENCE.md` asks for.

Two things change when the data are real. The population behind the file is not
observed, so a population-frequency measure and the true journalist risk cannot be
computed and a fitted model has to stand in their place. And the analyses that
define utility are the ones the file is actually used for, so they are declared
here rather than invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from .composition import candidate_sets_from_keys, pooled_attack_from_sets
from .risk import fit_frequency_model, journalist_from_fit, prosecutor_risk
from .uncertainty import binomial_upper, selected_maximum_bound


@dataclass(frozen=True)
class RealFile:
    """A real microdata file with a declared key, covariate and analyses."""

    name: str
    frame: pd.DataFrame
    block: str
    key_columns: tuple[str, ...]
    covariate: str
    sensitive: tuple[str, ...] = ()
    source: str = ""

    @property
    def size(self) -> int:
        return int(len(self.frame))


def load_census2000() -> RealFile:
    """US Census 2000 public use sample: 29,501 records with geography and schooling."""
    import wooldridge

    frame = wooldridge.data("census2000").copy()
    frame["state_code"] = pd.factorize(frame["state"])[0]
    frame["experience"] = frame["exper"].astype(int)
    frame["education"] = frame["educ"].astype(int)
    frame["income"] = frame["lweekinc"].astype(float)
    frame = frame.reset_index(drop=True)
    return RealFile(name="census2000", frame=frame, block="state_code",
                    key_columns=("state_code", "puma", "education", "experience"),
                    covariate="income",
                    source="US Census 2000 public use microdata sample, "
                           "distributed with the wooldridge package")


def load_loan_applications() -> RealFile:
    """Boston Fed mortgage application study: 1,989 real applications."""
    import wooldridge

    frame = wooldridge.data("loanapp").copy()
    frame = frame.dropna(subset=["appinc", "loanamt", "married", "dep", "occ", "msa",
                                 "approve"]).reset_index(drop=True)
    frame["income_band"] = (frame["appinc"] // 20).astype(int)
    frame["loan_band"] = (frame["loanamt"] // 40).astype(int)
    frame["dependants"] = frame["dep"].clip(upper=3).astype(int)
    frame["married_flag"] = frame["married"].astype(int)
    frame["occupancy"] = frame["occ"].astype(int)
    frame["area"] = pd.factorize(frame["msa"])[0]
    frame["denied"] = (1 - frame["approve"]).astype(int)
    frame["late"] = ((frame.get("mortlat1", 0) + frame.get("mortlat2", 0)) > 0).astype(int)
    frame["amount"] = frame["loanamt"].astype(float)
    return RealFile(name="loan_applications", frame=frame, block="area",
                    key_columns=("area", "income_band", "married_flag", "dependants",
                                 "occupancy"),
                    covariate="amount", sensitive=("denied", "late"),
                    source="Boston Fed mortgage application study, distributed with "
                           "the wooldridge package")


def coarsen(values: np.ndarray, width: int) -> np.ndarray:
    """Group a numeric column into bands of the given width."""
    if width <= 1:
        return np.asarray(values)
    return (np.asarray(values) // width).astype(int)


@dataclass
class RealRelease:
    """One released version of a real file."""

    index: np.ndarray
    key: np.ndarray
    block: np.ndarray
    label: str
    dropped: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.index.size)


def encode(columns: list[np.ndarray]) -> np.ndarray:
    """Pack several key components into one integer code."""
    frame = pd.DataFrame({f"c{i}": np.asarray(c) for i, c in enumerate(columns)})
    return pd.factorize(pd.MultiIndex.from_frame(frame))[0]


def build_release(file: RealFile, coarsening: dict[str, int], minimum_cell: int,
                  drop_columns: tuple[str, ...] = ()) -> RealRelease:
    """Apply coarsening and a minimum cell count to a real file."""
    frame = file.frame
    parts = []
    used = []
    for column in file.key_columns:
        if column in drop_columns:
            continue
        width = int(coarsening.get(column, 1))
        parts.append(coarsen(frame[column].to_numpy(), width))
        used.append(f"{column}/{width}" if width > 1 else column)
    key = encode(parts)
    index = np.arange(len(frame))
    if minimum_cell > 1:
        values, counts = np.unique(key, return_counts=True)
        keep = np.isin(key, values[counts >= minimum_cell])
        index, key = index[keep], key[keep]
    label = ", ".join(used) + f", k_min = {minimum_cell}"
    return RealRelease(index=index, key=key, block=frame[file.block].to_numpy()[index],
                       label=label, dropped=int(len(frame) - index.size),
                       detail={"coarsening": dict(coarsening),
                               "minimum_cell": int(minimum_cell),
                               "dropped_columns": list(drop_columns)})


def key_structure(file: RealFile, release: RealRelease) -> dict:
    """What the key alone separates, which on a real file is all that can be checked."""
    values, counts = np.unique(release.key, return_counts=True)
    risks = prosecutor_risk(release.key)
    return {"file": file.name, "records": file.size, "released": release.size,
            "cells": int(values.size), "unique_cells": int((counts == 1).sum()),
            "unique_share": float(counts[counts == 1].sum() / release.size),
            "mean_reciprocal_frequency": float(risks.mean()),
            "max_reciprocal_frequency": float(risks.max()),
            "median_cell_size": float(np.median(counts))}


def key_space(file: RealFile, nested: tuple[tuple[str, str], ...] = ()) -> int:
    """Size of the key space, counting nested columns by their observed pairs."""
    frame = file.frame
    counted = set()
    total = 1
    for parent, child in nested:
        if parent in file.key_columns and child in file.key_columns:
            total *= int(frame.groupby([parent, child]).ngroups)
            counted.update({parent, child})
    for column in file.key_columns:
        if column not in counted:
            total *= int(frame[column].nunique())
    return total


def journalist_estimate(release: RealRelease, sampling_fraction: float,
                        cells: int) -> dict:
    """The journalist measure on a real file, where no truth is available.

    The population frequencies are not observed, so the fitted law is the only
    route to the measure, and its error cannot be checked against a truth. The key
    space has to be supplied, since the fit needs the empty cells as well.
    """
    _, counts = np.unique(release.key, return_counts=True)
    fit = fit_frequency_model(counts, max(int(cells), counts.size + 1),
                              sampling_fraction)
    return {"sampling_fraction": sampling_fraction, "key_space": int(cells),
            "fitted_mean_cell": fit.mean, "dispersion": fit.dispersion,
            "rho_model": journalist_from_fit(sampling_fraction, fit),
            "truth_available": False}


def attack(file: RealFile, release: RealRelease, error: float, covariate_sd: float,
           replicates: int, rng: np.random.Generator,
           targets: np.ndarray | None = None) -> dict:
    """Linkage against a real release, blocked on the file's block column.

    The adversary observes the key with a stated error rate on one component and a
    noisy reading of the covariate, and picks the maximum-posterior record.
    """
    if release.size == 0:
        return {"rate": 0.0, "trials": 0}
    covariate = file.frame[file.covariate].to_numpy()[release.index]
    key, block = release.key, release.block
    order = np.argsort(block, kind="stable")
    key_s, cov_s, idx_s, blk_s = key[order], covariate[order], np.arange(key.size)[order], block[order]
    values, first = np.unique(blk_s, return_index=True)
    ends = np.append(first[1:], blk_s.size)
    spans = {int(b): (int(a), int(e)) for b, a, e in zip(values, first, ends)}
    drawn = (rng.choice(key.size, size=int(replicates), replace=True)
             if targets is None else np.asarray(targets))
    success = np.zeros(drawn.size)
    posterior = np.zeros(drawn.size)
    for i, position in enumerate(drawn.tolist()):
        start, end = spans[int(block[position])]
        candidate_key, candidate_cov = key_s[start:end], cov_s[start:end]
        candidate_id = idx_s[start:end]
        matched = (candidate_key == key[position])
        likelihood = np.where(matched, 1.0 - error, error / max(candidate_key.size - 1, 1))
        reading = covariate[position] + rng.normal(0, covariate_sd)
        likelihood = likelihood * norm.pdf(reading, loc=candidate_cov, scale=covariate_sd)
        total = likelihood.sum()
        if total <= 0:
            continue
        weights = likelihood / total
        best = int(np.argmax(weights))
        posterior[i] = weights[best]
        success[i] = float(candidate_id[best] == position)
    return {"rate": float(success.mean()), "trials": int(drawn.size),
            "successes": int(success.sum()), "mean_posterior": float(posterior.mean())}


def per_target(file: RealFile, release: RealRelease, error: float, covariate_sd: float,
               replicates: int, cap: int, rng: np.random.Generator,
               alpha: float = 0.05) -> dict:
    """The worst released record among those in the smallest key classes."""
    values, counts = np.unique(release.key, return_counts=True)
    smallest = int(counts.min())
    positions = np.flatnonzero(np.isin(release.key, values[counts == smallest]))
    if positions.size > cap:
        positions = rng.choice(positions, size=cap, replace=False)
    if positions.size == 0:
        return {"targets": 0}
    rates = _per_target_rates(file, release, error, covariate_sd, positions,
                              replicates, rng)
    summary = selected_maximum_bound(rates, replicates, alpha)
    summary["smallest_cell"] = smallest
    return summary


def _per_target_rates(file: RealFile, release: RealRelease, error: float,
                      covariate_sd: float, positions: np.ndarray, replicates: int,
                      rng: np.random.Generator) -> np.ndarray:
    rates = np.zeros(positions.size)
    for i, position in enumerate(positions.tolist()):
        repeated = np.full(replicates, position)
        outcome = attack(file, release, error, covariate_sd, 0, rng, targets=repeated)
        rates[i] = outcome["rate"]
    return rates


def attribute_disclosure(file: RealFile, release: RealRelease, column: str) -> dict:
    """Disclosure of a binary attribute by class homogeneity."""
    values = file.frame[column].to_numpy()[release.index]
    homogeneous = 0
    homogeneous_positive = 0
    for cell in np.unique(release.key).tolist():
        members = release.key == cell
        rate = float(values[members].mean())
        if rate in (0.0, 1.0):
            homogeneous += int(members.sum())
        if rate == 1.0:
            homogeneous_positive += int(members.sum())
    return {"column": column, "released": release.size,
            "homogeneous_share": homogeneous / release.size,
            "homogeneous_positive_share": homogeneous_positive / release.size,
            "base_rate": float(values.mean())}


def residual_disclosure(file: RealFile, columns: tuple[str, ...],
                        minimum: int = 3) -> dict:
    """Cells of a published aggregate that give one record away by subtraction."""
    frame = file.frame
    key = encode([frame[c].to_numpy() for c in columns])
    values, counts = np.unique(key, return_counts=True)
    amounts = frame[file.covariate].to_numpy()
    single = counts == 1
    exposed = float(np.sum([amounts[key == v].sum() for v in values[single]]))
    return {"cells": int(values.size), "single_record_cells": int(single.sum()),
            "single_record_share": float(single.sum() / values.size),
            "cells_below_minimum": int((counts < minimum).sum()),
            "cells_below_minimum_share": float((counts < minimum).mean()),
            "exposed_amount_share": exposed / float(amounts.sum()),
            "columns": list(columns), "minimum": minimum}


def utility(file: RealFile, release: RealRelease, analyses: list[dict]) -> dict:
    """Loss on the analyses the file is actually used for.

    The analyst sees only what is released. When the grouping column is published in
    bands, the estimate for each fine level is the mean of its band in the release,
    so coarsening costs accuracy as well as suppression does.
    """
    frame = file.frame
    widths = release.detail.get("coarsening", {}) if release.detail else {}
    losses = {}
    for analysis in analyses:
        column, outcome = analysis["by"], analysis["outcome"]
        truth = frame.groupby(column)[outcome].mean()
        width = int(widths.get(column, 1))
        kept = frame.iloc[release.index]
        if width > 1:
            band_mean = kept.groupby(coarsen(kept[column].to_numpy(), width))[outcome].mean()
            levels = truth.index.to_numpy()
            released = pd.Series(band_mean.reindex(coarsen(levels, width)).to_numpy(),
                                 index=truth.index)
        else:
            released = kept.groupby(column)[outcome].mean()
        joined = truth.to_frame("truth").join(released.to_frame("released"), how="left")
        covered = joined["released"].notna()
        if not covered.any():
            losses[analysis["name"]] = 10.0
            continue
        error = float(np.sqrt(np.mean((joined.loc[covered, "released"]
                                       - joined.loc[covered, "truth"]) ** 2)))
        scale = float(frame[outcome].std())
        losses[analysis["name"]] = error / scale + 0.5 * float(1 - covered.mean())
    total = float(np.mean(list(losses.values())))
    return {"utility": -total, "losses": losses,
            "retained": release.size / file.size}


def composition(file: RealFile, plans: list[dict], minimum_cell: int,
                targets: np.ndarray) -> dict:
    """Several releases from one real file, attacked separately and then pooled."""
    sets = []
    names = []
    sizes = []
    for plan in plans:
        release = build_release(file, plan.get("coarsening", {}), minimum_cell,
                                tuple(plan.get("drop", ())))
        sets.append(candidate_sets_from_keys(release.index, release.key))
        names.append(plan["name"])
        sizes.append(release.size)
    result = pooled_attack_from_sets(sets, targets)
    payload = result.as_dict()
    payload["releases"] = [{"name": n, "released": s, "marginal": m,
                            "unique_share": u}
                           for n, s, m, u in zip(names, sizes, result.marginal,
                                                 result.marginal_unique_share)]
    payload["minimum_cell"] = minimum_cell
    return payload


def bound(rate: float, trials: int, alpha: float = 0.05) -> float:
    return binomial_upper(int(round(rate * trials)), trials, alpha) if trials else 1.0
