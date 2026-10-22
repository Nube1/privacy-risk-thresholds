"""A granular credit register, and the disclosure risks specific to one.

The register follows the shape of a supervisory loan-level collection: borrowers
with a sector, region and size class, lenders with unequal market shares, one
exposure and one arrears status per relationship, observed monthly. Nothing here
reads real data.

Five risks are measured, because a credit register carries risks that a
demographic file does not:

* borrower identification, the linkage problem of the main study;
* relationship disclosure, the claim that a named borrower banks with a named
  lender, scored by its positive predictive value rather than its recall;
* status inference, the disclosure of arrears or default without identification;
* reconstruction of one lender's contribution from published totals;
* differencing across consecutive months and revisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, poisson

from .synthetic import skewed_probabilities


@dataclass(frozen=True)
class CreditRegister:
    """Loan-level relationships between borrowers and lenders, observed monthly."""

    borrower: np.ndarray
    lender: np.ndarray
    sector: np.ndarray
    region: np.ndarray
    size_class: np.ndarray
    exposure: np.ndarray
    arrears: np.ndarray
    month: np.ndarray
    live: np.ndarray
    n_borrowers: int
    n_lenders: int
    n_sectors: int
    n_regions: int
    n_size_classes: int
    months: int

    @property
    def loans(self) -> int:
        return int(self.borrower.size)

    def key(self) -> np.ndarray:
        """The quasi-identifier an outside party can reconstruct for a borrower."""
        return ((self.sector * self.n_regions + self.region) * self.n_size_classes
                + self.size_class)

    def month_mask(self, month: int) -> np.ndarray:
        return (self.month == month) & self.live[:, month]


def make_credit_register(config: dict) -> CreditRegister:
    """Draw the register from one seeded generator."""
    settings = config.get("credit", {})
    rng = np.random.default_rng(int(config["seed"]) + 5_000)
    borrowers = int(settings.get("borrowers", 20_000))
    lenders = int(settings.get("lenders", 60))
    sectors = int(settings.get("sectors", 18))
    regions = int(settings.get("regions", 12))
    sizes = int(settings.get("size_classes", 4))
    months = int(settings.get("months", 6))

    sector = rng.choice(sectors, size=borrowers, p=skewed_probabilities(sectors, 0.7))
    region = rng.choice(regions, size=borrowers, p=skewed_probabilities(regions, 0.8))
    size_class = rng.choice(sizes, size=borrowers, p=np.array([0.62, 0.24, 0.11, 0.03]))

    # most borrowers have one lender; larger firms hold several
    relationships = 1 + rng.binomial(3, 0.10 + 0.20 * size_class)
    borrower_index = np.repeat(np.arange(borrowers), relationships)
    share = skewed_probabilities(lenders, 1.05)
    lender = rng.choice(lenders, size=borrower_index.size, p=share)

    loan_sector = sector[borrower_index]
    loan_region = region[borrower_index]
    loan_size = size_class[borrower_index]
    scale = np.array([11.0, 12.5, 14.0, 16.0])[loan_size]
    exposure = np.exp(rng.normal(scale, 0.8))
    arrears_rate = 0.03 + 0.02 * (loan_sector % 3) + 0.01 * (3 - loan_size)
    arrears = rng.random(borrower_index.size) < arrears_rate

    # A monthly panel with a stable stock. Most loans are outstanding when the panel
    # opens; the rest are originated in a later month. Every loan then leaves with a
    # constant monthly hazard, and a loan that has left does not return. With the
    # default settings about a fifth of the loans ever observed are originated during
    # the panel and the stock changes by a few per cent a month, as in a register.
    exit_hazard = float(settings.get("exit_hazard", 0.04))
    entry_share = float(settings.get("entry_share", 1.0 - 1.0 / (1.0 + exit_hazard
                                                              * (months - 1))))
    n = borrower_index.size
    entering = rng.random(n) < entry_share
    origin = np.where(entering, rng.integers(1, max(months, 2), size=n), 0)
    # months survived after origination, geometric with the exit hazard
    survive = rng.geometric(exit_hazard, size=n)
    live = np.zeros((n, months), dtype=bool)
    for m in range(months):
        live[:, m] = (origin <= m) & (m < origin + survive)

    return CreditRegister(borrower=borrower_index, lender=lender, sector=loan_sector,
                          region=loan_region, size_class=loan_size, exposure=exposure,
                          arrears=arrears, month=np.zeros(borrower_index.size, dtype=int),
                          live=live, n_borrowers=borrowers, n_lenders=lenders,
                          n_sectors=sectors, n_regions=regions, n_size_classes=sizes,
                          months=months)


def released_loans(register: CreditRegister, minimum_cell: int,
                   month: int = 0) -> np.ndarray:
    """Rows a loan-level release keeps under a minimum cell count on the key."""
    live = np.flatnonzero(register.live[:, month])
    if minimum_cell <= 1:
        return live
    key = register.key()[live]
    values, counts = np.unique(key, return_counts=True)
    return live[np.isin(key, values[counts >= minimum_cell])]


def borrower_identification(register: CreditRegister, rows: np.ndarray,
                            covariate_sd: float, replicates: int,
                            rng: np.random.Generator) -> dict:
    """Linkage on the borrower key, with a noisy reading of the exposure.

    The adversary blocks on region, scores candidates by the likelihood of the
    observed sector and size class and by the density of its exposure reading, and
    selects the maximum-posterior loan.
    """
    if rows.size == 0:
        return {"rate": 0.0, "trials": 0, "mean_reciprocal_frequency": float("nan")}
    key = register.key()[rows]
    values, counts = np.unique(key, return_counts=True)
    size = dict(zip(values.tolist(), counts.tolist()))
    reciprocal = float(np.mean([1.0 / size[int(k)] for k in key.tolist()]))
    log_exposure = np.log(register.exposure)
    region = register.region[rows]
    targets = rng.choice(rows.size, size=int(replicates), replace=True)
    success = np.zeros(int(replicates))
    borrower_success = np.zeros(int(replicates))
    borrower = register.borrower
    for i, position in enumerate(targets.tolist()):
        block = np.flatnonzero(region == region[position])
        candidates = rows[block]
        reading = log_exposure[rows[position]] + rng.normal(0, covariate_sd)
        same_key = (key[block] == key[position]).astype(float)
        likelihood = np.where(same_key > 0, 1.0, 1e-9) * norm.pdf(
            reading, loc=log_exposure[candidates], scale=covariate_sd)
        total = likelihood.sum()
        if total <= 0:
            continue
        chosen = candidates[int(np.argmax(likelihood))]
        success[i] = float(chosen == rows[position])
        borrower_success[i] = float(borrower[chosen] == borrower[rows[position]])
    by_class = {}
    counts_by_class = {}
    size_of_target = register.size_class[rows][targets]
    for band in range(register.n_size_classes):
        members = size_of_target == band
        if members.sum() >= 50:
            by_class[str(band)] = float(success[members].mean())
            counts_by_class[str(band)] = {"successes": int(success[members].sum()),
                                          "trials": int(members.sum())}
    cell_size = np.array([size[int(k)] for k in key.tolist()], dtype=float)
    thin = {str(band): float(np.mean(1.0 / cell_size[register.size_class[rows] == band]))
            for band in range(register.n_size_classes)
            if (register.size_class[rows] == band).any()}
    return {"rate": float(success.mean()), "trials": int(replicates),
            "borrower_rate": float(borrower_success.mean()),
            "mean_reciprocal_frequency": reciprocal,
            "rate_by_size_class": by_class,
            "counts_by_size_class": counts_by_class,
            "reciprocal_frequency_by_size_class": thin}


def relationship_disclosure(register: CreditRegister, rows: np.ndarray) -> dict:
    """Positive predictive value of an alleged borrower--lender relationship.

    The adversary names, for each borrower in the release, the lender holding the
    largest share of that borrower's key class. Recall is not the quantity of
    interest: the harm follows from allegations that are acted on, so the measure is
    the share of allegations that are true. The four cells of the decision table are
    reported as well, because the value of the measure is a base-rate statement and
    should be checkable as one.
    """
    if rows.size == 0:
        return {"positive_predictive_value": float("nan"), "allegations": 0}
    key = register.key()[rows]
    lender = register.lender[rows]
    borrower = register.borrower[rows]
    modal: dict[int, int] = {}
    for cell in np.unique(key).tolist():
        members = lender[key == cell]
        counts = np.bincount(members, minlength=register.n_lenders)
        modal[int(cell)] = int(np.argmax(counts))
    relationships = {(int(b), int(bank))
                     for b, bank in zip(borrower.tolist(), lender.tolist())}
    alleged = set()
    cell_of_borrower: dict[int, int] = {}
    for b, cell in zip(borrower.tolist(), key.tolist()):
        cell_of_borrower[int(b)] = int(cell)
    for b, cell in cell_of_borrower.items():
        alleged.add((b, modal[cell]))
    borrowers = len(cell_of_borrower)
    pairs = borrowers * register.n_lenders
    true_positive = len(alleged & relationships)
    false_positive = len(alleged) - true_positive
    positives = len({pair for pair in relationships if pair[0] in cell_of_borrower})
    false_negative = positives - true_positive
    true_negative = pairs - true_positive - false_positive - false_negative
    prevalence = positives / pairs
    sensitivity = true_positive / positives if positives else float("nan")
    specificity = (true_negative / (pairs - positives)
                   if pairs > positives else float("nan"))
    ppv = true_positive / len(alleged) if alleged else float("nan")
    identity_denominator = prevalence * sensitivity + (1 - prevalence) * (1 - specificity)
    share = np.bincount(lender, minlength=register.n_lenders) / lender.size
    return {"positive_predictive_value": ppv, "allegations": len(alleged),
            "borrowers": borrowers, "pairs": pairs,
            "true_positive": true_positive, "false_positive": false_positive,
            "false_negative": false_negative, "true_negative": true_negative,
            "prevalence": prevalence, "sensitivity": sensitivity,
            "specificity": specificity,
            "base_rate_identity": (prevalence * sensitivity / identity_denominator
                                   if identity_denominator > 0 else float("nan")),
            "largest_lender_share": float(share.max()),
            "lift_over_market_share": (ppv / float(share.max())
                                       if share.max() > 0 else float("inf"))}


def status_inference(register: CreditRegister, rows: np.ndarray,
                     threshold: float = 0.9) -> dict:
    """Disclosure of arrears status without identification.

    A key class that is homogeneous in status discloses that status for every
    member, whatever the class size, which is the failure that a minimum cell count
    does not address.
    """
    if rows.size == 0:
        return {"homogeneous_share": float("nan")}
    key = register.key()[rows]
    arrears = register.arrears[rows]
    homogeneous = 0
    homogeneous_arrears = 0
    confident = 0
    correct = 0
    for cell in np.unique(key).tolist():
        members = key == cell
        rate = float(arrears[members].mean())
        count = int(members.sum())
        if rate in (0.0, 1.0):
            homogeneous += count
        if rate == 1.0:
            homogeneous_arrears += count
        posterior = max(rate, 1 - rate)
        if posterior >= threshold:
            confident += count
            correct += int((arrears[members] == (rate >= 0.5)).sum())
    total = int(rows.size)
    by_class = {}
    for band in range(register.n_size_classes):
        members = register.size_class[rows] == band
        if members.sum() < 20:
            continue
        sub_key = key[members]
        sub_arrears = arrears[members]
        share = 0
        for cell in np.unique(sub_key).tolist():
            inside = sub_key == cell
            rate = float(sub_arrears[inside].mean())
            if rate in (0.0, 1.0):
                share += int(inside.sum())
        by_class[str(band)] = share / int(members.sum())
    return {"released": total, "homogeneous_share": homogeneous / total,
            "homogeneous_arrears_share": homogeneous_arrears / total,
            "homogeneous_share_by_size_class": by_class,
            "confident_share": confident / total,
            "confident_accuracy": correct / confident if confident else float("nan"),
            "threshold": threshold, "base_rate": float(arrears.mean())}


def reconstruction_from_totals(register: CreditRegister, month: int = 0,
                               minimum_contributors: int = 3) -> dict:
    """How much of one lender's position is recoverable from published totals.

    Aggregates are published by region and sector. A lender's contribution to a
    cell follows by residual arithmetic whenever the cell has one contributor, or
    two of which one is the adversary itself.
    """
    live = np.flatnonzero(register.live[:, month])
    cell = ((register.region[live] * register.n_sectors + register.sector[live])
            * register.n_size_classes + register.size_class[live])
    lender = register.lender[live]
    exposure = register.exposure[live]
    exposed_cells = 0
    exposed_amount = 0.0
    recoverable_amount = 0.0
    self_known = 0
    total_amount = float(exposure.sum())
    cells = np.unique(cell)
    suppressed = 0
    contributions = 0
    for value in cells.tolist():
        members = cell == value
        contributors = np.unique(lender[members])
        contributions += int(contributors.size)
        if contributors.size < minimum_contributors:
            suppressed += 1
        if contributors.size == 1:
            exposed_cells += 1
            exposed_amount += float(exposure[members].sum())
            recoverable_amount += float(exposure[members].sum())
        elif contributors.size == 2:
            self_known += 1
            recoverable_amount += float(exposure[members].sum())
    return {"cells": int(cells.size), "single_contributor_cells": exposed_cells,
            "single_contributor_share": exposed_cells / cells.size,
            "two_contributor_cells": self_known,
            "recoverable_by_a_participant_share": (exposed_cells + self_known) / cells.size,
            "exposed_amount_share": exposed_amount / total_amount if total_amount else 0.0,
            "recoverable_amount_share": (recoverable_amount / total_amount
                                         if total_amount else 0.0),
            "cells_below_minimum": suppressed,
            "cells_below_minimum_share": suppressed / cells.size,
            "lender_cell_contributions": contributions,
            "minimum_contributors": minimum_contributors}


def differencing_risk(register: CreditRegister, minimum_cell: int = 3) -> dict:
    """Differencing across consecutive monthly releases.

    Counts are published per key class per month. When a class changes by exactly
    one loan, the change itself identifies that loan's contribution. The adversary
    does not know that only one loan moved, so the inference carries the posterior
    probability that a unit change was a single arrival or departure rather than
    offsetting moves.
    """
    key = register.key()
    unit_changes = 0
    comparisons = 0
    exposed_loans = 0
    moves = []
    cell_moves: dict[int, list[int]] = {}
    unit_change_cells: list[int] = []
    for month in range(register.months - 1):
        before = register.live[:, month]
        after = register.live[:, month + 1]
        for cell in np.unique(key).tolist():
            members = key == cell
            count_before = int((members & before).sum())
            count_after = int((members & after).sum())
            if min(count_before, count_after) < minimum_cell:
                continue
            comparisons += 1
            moved_total = int((members & (before ^ after)).sum())
            moves.append(moved_total)
            cell_moves.setdefault(int(cell), []).append(moved_total)
            if abs(count_after - count_before) == 1:
                unit_changes += 1
                unit_change_cells.append(int(cell))
                moved = np.flatnonzero(members & (before ^ after))
                if moved.size == 1:
                    exposed_loans += 1
    # the posterior needs the arrival and departure rate of the cell in question,
    # not one rate pooled over cells: movement is heterogeneous, and the pooled
    # calculation answers a question about the average cell instead
    def posterior(rate: float) -> float:
        lam = max(rate, 1e-9)
        both = 2 * poisson.pmf(1, lam) * poisson.pmf(0, lam)
        odd_total = sum(poisson.pmf(a, lam) * poisson.pmf(d, lam)
                        for a in range(0, 40) for d in range(0, 40) if abs(a - d) == 1)
        return float(both / odd_total) if odd_total > 0 else float("nan")

    lam = max(float(np.mean(moves)) / 2 if moves else 0.0, 1e-9)
    single = 2 * poisson.pmf(1, lam) * poisson.pmf(0, lam)
    odd = sum(poisson.pmf(a, lam) * poisson.pmf(d, lam)
              for a in range(0, 40) for d in range(0, 40) if abs(a - d) == 1)
    per_cell = [posterior(float(np.mean(cell_moves[cell])) / 2)
                for cell in unit_change_cells if cell_moves.get(cell)]
    return {"comparisons": comparisons, "unit_changes": unit_changes,
            "unit_change_share": unit_changes / comparisons if comparisons else 0.0,
            "single_move_confirmed": exposed_loans,
            "empirical_single_move_share": (exposed_loans / unit_changes
                                            if unit_changes else float("nan")),
            "posterior_single_move": float(single / odd) if odd > 0 else float("nan"),
            "posterior_single_move_per_cell": (float(np.mean(per_cell)) if per_cell
                                               else float("nan")),
            "moves_per_cell": float(np.mean(moves)) if moves else 0.0,
            "moves_per_unit_change_cell": (float(np.mean(
                [np.mean(cell_moves[c]) for c in unit_change_cells if cell_moves.get(c)]))
                if unit_change_cells else 0.0),
            "poisson_rate": lam}


def central_bank_utility(register: CreditRegister, rows: np.ndarray,
                         weights: dict[str, float] | None = None) -> dict:
    """Loss on the four analyses a credit register is collected for.

    Credit growth by sector, concentration by region, the arrears ratio, and
    lender-level exposure by sector. Each is computed from the release and compared
    with the register, and the losses are combined with declared weights.
    """
    weights = weights or {"growth": 0.3, "concentration": 0.2, "stability": 0.3,
                          "supervision": 0.2}
    if rows.size == 0:
        return {"utility": -10.0}
    live = np.flatnonzero(register.live[:, 0])
    exposure = register.exposure

    def sector_totals(index: np.ndarray) -> np.ndarray:
        return np.bincount(register.sector[index], weights=exposure[index],
                           minlength=register.n_sectors)

    def region_concentration(index: np.ndarray) -> np.ndarray:
        out = np.zeros(register.n_regions)
        for region in range(register.n_regions):
            members = register.region[index] == region
            if not members.any():
                continue
            amounts = np.bincount(register.lender[index][members],
                                  weights=exposure[index][members],
                                  minlength=register.n_lenders)
            total = amounts.sum()
            out[region] = float(((amounts / total) ** 2).sum()) if total else 0.0
        return out

    def lender_sector(index: np.ndarray) -> np.ndarray:
        table = np.zeros((register.n_lenders, register.n_sectors))
        np.add.at(table, (register.lender[index], register.sector[index]),
                  exposure[index])
        return table

    truth_growth = sector_totals(live)
    release_growth = sector_totals(rows) / max(rows.size / live.size, 1e-9)
    growth_loss = float(np.mean(np.abs(release_growth - truth_growth)
                                / np.maximum(truth_growth, 1.0)))
    concentration_loss = float(np.mean(np.abs(region_concentration(rows)
                                              - region_concentration(live))))
    stability_loss = float(abs(register.arrears[rows].mean()
                               - register.arrears[live].mean())
                           / max(register.arrears[live].mean(), 1e-9))
    truth_table, release_table = lender_sector(live), lender_sector(rows)
    scale = max(truth_table.sum(), 1.0)
    supervision_loss = float(np.abs(release_table / max(rows.size / live.size, 1e-9)
                                    - truth_table).sum() / scale)
    losses = {"growth": growth_loss, "concentration": concentration_loss,
              "stability": stability_loss, "supervision": supervision_loss}
    total = float(sum(weights[name] * value for name, value in losses.items()))
    return {"utility": -total, "losses": losses, "weights": weights,
            "retained": float(rows.size / live.size)}
