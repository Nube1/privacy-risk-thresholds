"""The procedures Appendix B sets out: precision, error budget, remedy choice, defence record.

Each function is arithmetic on declared inputs. None of them estimates a quantity the
institution has not supplied, and none of them replaces the institution's judgement about
where the inputs come from.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .uncertainty import binomial_upper


# ---- B.1 Arguing for the reading precision ---------------------------------------------

def combine_readings(sds: list[float]) -> float:
    """Standard deviation of the best linear combination of independent unbiased readings.

    Each reading of the same log exposure carries an error with standard deviation s_k. The
    inverse-variance combination has variance 1 / sum(1 / s_k^2), which is never above the
    smallest single variance. The independence of the errors is an assumption the
    institution must defend: sources that copy one another do not add precision.
    """
    if not sds or any(s <= 0 for s in sds):
        raise ValueError("need at least one positive standard deviation")
    return float(1.0 / math.sqrt(sum(1.0 / (s * s) for s in sds)))


def rounding_sd(step: float, on_log_scale_at: float | None = None) -> float:
    """Standard deviation of a rounding error to the nearest `step` (uniform, step / sqrt 12).

    If `on_log_scale_at` is a level, the error is converted to the log scale by dividing by
    that level, which is a first-order approximation valid for a step small against it.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    sd = step / math.sqrt(12.0)
    return sd / on_log_scale_at if on_log_scale_at else sd


def floor_by_precision(register, precisions, floors, tolerance: float = 0.12,
                       alpha: float = 0.05, replicates: int = 2000, seed: int = 0,
                       month: int = 0) -> list[dict]:
    """For each reading precision, the fixed-sequence floor on a credit register.

    Floors are tried from the most protective down; the sequence stops at the first whose
    exact upper bound exceeds the tolerance, and the answer is the last floor that passed
    (None when even the most protective failed). Every precision uses its own attack
    stream from a fixed seed, so the table reproduces.
    """
    from .credit import borrower_identification, released_loans
    ordered = sorted(floors, reverse=True)
    rows = []
    for j, sd in enumerate(precisions):
        chosen, detail = None, []
        for floor in ordered:
            released = released_loans(register, floor, month)
            rng = np.random.default_rng(seed + 1000 * j + floor)
            out = borrower_identification(register, released, sd, replicates, rng)
            k = int(round(out["rate"] * out["trials"]))
            bound = binomial_upper(k, out["trials"], alpha)
            detail.append({"floor": floor, "successes": k, "trials": out["trials"],
                           "bound": bound, "released": int(released.size)})
            if bound > tolerance:
                break
            chosen = floor
        rows.append({"precision": float(sd), "floor": chosen, "detail": detail})
    return rows


# ---- B.2 Setting the error budget -------------------------------------------------------

def reserve_share(i: int) -> float:
    """w_i = 6 / (pi^2 i^2), the open-ended schedule; sums to one over i >= 1."""
    if i < 1:
        raise ValueError("coordinates are numbered from 1")
    return 6.0 / (math.pi ** 2 * i * i)


def allocate_error_budget(total_alpha: float, costs: list[float], drifts: list[float],
                          reserve_fraction: float = 0.0) -> dict:
    """Divide the total error among known coordinates, holding a reserve for later ones.

    The known part gets (1 - reserve_fraction) * total_alpha, divided in proportion to
    c_i / mu_i (cost per draw over the planned drift). The reserve is spread over later
    coordinates by the 6 / (pi^2 j^2) schedule, so the shares never sum above total_alpha.
    """
    if not 0 < total_alpha < 1 or not 0 <= reserve_fraction < 1:
        raise ValueError("need 0 < total_alpha < 1 and 0 <= reserve_fraction < 1")
    if len(costs) != len(drifts) or not costs:
        raise ValueError("one cost and one drift per known coordinate")
    if any(c <= 0 for c in costs) or any(m <= 0 for m in drifts):
        raise ValueError("costs and drifts must be positive")
    weights = np.array(costs, float) / np.array(drifts, float)
    known_total = total_alpha * (1 - reserve_fraction)
    shares = known_total * weights / weights.sum()
    reserve_total = total_alpha * reserve_fraction
    later = [reserve_total * reserve_share(j) for j in range(1, 6)]
    return {"known_shares": shares.tolist(), "reserve_total": reserve_total,
            "reserve_next_five": later, "spent_on_known": float(shares.sum()),
            "log_threshold_known": [math.log(1.0 / s) for s in shares.tolist()]}


def reallocate_for_new_product(remaining: float, spent: float, total_alpha: float,
                               new_share: float) -> dict:
    """A new product may draw on the reserve only. The total and past spend never change."""
    if min(remaining, spent, total_alpha, new_share) < 0:
        raise ValueError("amounts must be non-negative")
    if abs(remaining + spent - total_alpha) > 1e-9:
        raise ValueError("spent plus remaining must equal the total")
    ok = new_share <= remaining + 1e-12
    return {"permitted": ok, "remaining_after": remaining - new_share if ok else remaining,
            "reason": "paid from the reserve" if ok else "the reserve is too small; the "
            "total cannot be raised after outcomes have been seen"}


# ---- B.3 Choosing a remedy for the composition gap --------------------------------------

def composition_remedy(pooled_upper: float, tolerance: float,
                       remedy_pooled_upper: dict[str, float]) -> dict:
    """The decision tree for the composition gap.

    `remedy_pooled_upper` maps each remedy to the pooled upper bound re-simulated with it
    applied, listed in order of increasing cost. The tree returns the first remedy that
    restores the tolerance, or the advice to change the product when none does. No remedy is
    credited without a re-run of the pooled attack: the gap is not a function of the
    per-release risks.
    """
    if pooled_upper <= tolerance:
        return {"branch": "small", "action": "release as planned", "remedy": None}
    for name, bound in remedy_pooled_upper.items():
        if bound <= tolerance:
            return {"branch": "moderate", "action": "apply the cheapest remedy that restores "
                    "the tolerance", "remedy": name, "pooled_upper": bound}
    return {"branch": "large", "action": "change the product, for example to aggregates",
            "remedy": None}


# ---- B.4 Defending the tolerance ----------------------------------------------------------

DEFENCE_ITEMS = ("legal_reading", "reasons", "harm_class", "protected_parties",
                 "access_route", "date", "review_trigger", "owner")


@dataclass
class DefenceRecord:
    """What an institution can put in front of an authority that asks why the tolerance is
    what it is. A completeness check, not a finding of compliance."""
    legal_reading: str = ""
    reasons: str = ""
    harm_class: str = ""
    protected_parties: str = ""
    access_route: str = ""
    date: str = ""
    review_trigger: str = ""
    owner: str = ""
    notes: dict = field(default_factory=dict)

    def missing(self) -> list[str]:
        return [n for n in DEFENCE_ITEMS if not str(getattr(self, n)).strip()]

    def ready(self) -> bool:
        return not self.missing()
