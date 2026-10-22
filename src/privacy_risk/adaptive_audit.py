"""Anytime-valid, alpha-spent release audit for adaptively registered scenarios.

Each coordinate owns a fresh stream of binary attack outcomes. The user supplies
the sampling protocol; this module cannot establish conditional trial validity.
The allocation is irreversible, including when a scenario is later withdrawn.
One release programme must use one gate instance: a second instance starts a fresh
error budget and voids the familywise guarantee.

A coordinate is certified the first time its e-process crosses its threshold, and
it accepts no further outcomes after that, so the certificate cannot be lost or
regained by continued sampling. Each outcome is logged with the alternative that was
bet on it, in order, so that the betting sequence can be audited.
"""
from __future__ import annotations

import math
import hashlib
import hmac
import json as _json
from dataclasses import dataclass


@dataclass(frozen=True)
class CoordinateSpec:
    """The registered, immutable part of a coordinate."""
    name: str
    tolerance: float
    weight: float
    alpha: float


class AuditCoordinate:
    """One registered coordinate: a frozen specification and its running e-process.

    The running state is held in private fields and exposed read-only, so that code
    outside this module cannot mark a coordinate as certified or rewrite its log.
    """

    __slots__ = ("_spec", "_log_e", "_trials", "_successes", "_crossed", "_log",
                 "_threshold")

    def __init__(self, name: str, tolerance: float, weight: float, alpha: float,
                 share: float | None = None):
        self._spec = CoordinateSpec(name, float(tolerance), float(weight), float(alpha))
        # the error share that sets the threshold; for a chain member this is the
        # chain's share, not the nominal weight recorded against the budget
        self._threshold = -math.log(float(alpha)
                                    * float(share if share is not None else weight))
        self._log_e = 0.0
        self._trials = 0
        self._successes = 0
        self._crossed = False
        self._log: list[tuple[int, float]] = []

    @property
    def name(self) -> str:
        return self._spec.name

    @property
    def tolerance(self) -> float:
        return self._spec.tolerance

    @property
    def weight(self) -> float:
        return self._spec.weight

    @property
    def alpha(self) -> float:
        return self._spec.alpha

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def log_e(self) -> float:
        return self._log_e

    @property
    def trials(self) -> int:
        return self._trials

    @property
    def successes(self) -> int:
        return self._successes

    @property
    def log(self) -> list[tuple[int, float]]:
        return list(self._log)

    @property
    def certified(self) -> bool:
        return self._crossed

    def observe(self, success: int, alternative: float) -> bool:
        """Advance the likelihood-ratio e-process; alternative is predictable.

        The caller must choose alternative from information available before
        observing this attack, and keep the target/sampling law fixed.
        """
        if self._crossed:
            raise RuntimeError("coordinate already certified; it accepts no more outcomes")
        if isinstance(success, bool) or not isinstance(success, (int, np_integer())) \
                or int(success) not in (0, 1):
            raise ValueError("success must be the integer 0 or 1")
        if isinstance(alternative, bool) or not isinstance(alternative, (int, float,
                                                                         np_floating())):
            raise ValueError("alternative must be a real number")
        alternative = float(alternative)
        if not math.isfinite(alternative):
            raise ValueError("alternative must be finite")
        if not 0 <= alternative < self.tolerance:
            raise ValueError("alternative must be in [0, tolerance)")
        success = int(success)
        self._log_e += (math.log(alternative / self.tolerance) if success and alternative
                        else -math.inf if success else
                        math.log1p(-alternative) - math.log1p(-self.tolerance))
        self._trials += 1
        self._successes += success
        self._log.append((success, alternative))
        self._crossed = self._log_e >= self._threshold
        return self._crossed


def np_integer():
    try:
        import numpy as np
        return np.integer
    except ImportError:  # pragma: no cover
        return int


def np_floating():
    try:
        import numpy as np
        return np.floating
    except ImportError:  # pragma: no cover
        return float


class AuditChain:
    """Coordinates audited in an order fixed at registration, each after the last.

    All members share the chain's weight. The next member can be opened only after
    the current one has certified, and the order cannot change. Any false
    certification in the chain then requires the first unsafe member in that order
    to be falsely certified, so the chain's familywise error is at most alpha times
    its weight (the chain proposition of the article; Maurer, Hothorn and Lehmacher 1995).
    """

    def __init__(self, gate: "AdaptiveAudit", prefix: str, tolerance: float,
                 weight: float, labels: list[str]):
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("a chain needs distinct labels")
        self._gate = gate
        self._prefix = prefix
        self._tolerance = float(tolerance)
        self._weight = float(weight)
        self._labels = tuple(str(label) for label in labels)
        self._opened: list[AuditCoordinate] = []

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def members(self) -> list[AuditCoordinate]:
        return list(self._opened)

    def open_next(self) -> AuditCoordinate | None:
        """Open the next member, or return None when the chain is finished."""
        if self._opened and not self._opened[-1].certified:
            raise RuntimeError("the current member has not certified; the chain stops")
        if len(self._opened) == len(self._labels):
            return None
        label = self._labels[len(self._opened)]
        coord = AuditCoordinate(f"{self._prefix}:{label}", self._tolerance,
                                self._weight, self._gate.alpha, share=self._weight)
        self._gate._add(coord)
        self._opened.append(coord)
        return coord

    def floor(self) -> str | None:
        """The last certified label, the chain's answer."""
        certified = [c for c in self._opened if c.certified]
        return certified[-1].name.split(":", 1)[1] if certified else None


class ReleaseChain:
    """Ordered release settings, each certified only when every declared attack is.

    The members of the chain are release settings in an order fixed at registration.
    At each setting one coordinate is opened for every declared component attack, and
    the setting passes only when all of them certify. The chain moves to the next
    setting only after the current one passes. Choosing the last passing setting is a
    fixed sequence of intersection-union tests: an unsafe setting can be chosen only
    if the first unsafe component met in the fixed order is falsely certified, so the
    probability of choosing an unsafe setting is at most alpha times the chain's
    weight, with no split over components or settings.
    """

    def __init__(self, gate: "AdaptiveAudit", prefix: str, tolerance: float,
                 weight: float, labels: list[str], components: list[str]):
        if not labels or len(labels) != len(set(labels)):
            raise ValueError("a chain needs distinct labels")
        if not components or len(components) != len(set(components)):
            raise ValueError("a chain needs distinct components")
        self._gate = gate
        self._prefix = prefix
        self._tolerance = float(tolerance)
        self._weight = float(weight)
        self._labels = tuple(str(x) for x in labels)
        self._components = tuple(str(x) for x in components)
        self._opened: list[dict[str, AuditCoordinate]] = []

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    @property
    def components(self) -> tuple[str, ...]:
        return self._components

    def _passed(self, level: dict[str, AuditCoordinate]) -> bool:
        return all(c.certified for c in level.values())

    def open_next(self) -> dict[str, AuditCoordinate] | None:
        """Open the next setting's coordinates, or return None when finished."""
        if self._opened and not self._passed(self._opened[-1]):
            raise RuntimeError("the current setting has not passed; the chain stops")
        if len(self._opened) == len(self._labels):
            return None
        label = self._labels[len(self._opened)]
        level = {}
        for component in self._components:
            coord = AuditCoordinate(f"{self._prefix}:{label}:{component}",
                                    self._tolerance, self._weight, self._gate.alpha,
                                    share=self._weight)
            self._gate._add(coord)
            level[component] = coord
        self._opened.append(level)
        return dict(level)

    def floor(self) -> str | None:
        passed = [i for i, level in enumerate(self._opened) if self._passed(level)]
        return self._labels[passed[-1]] if passed else None


class AdaptiveAudit:
    """Familywise risk gate with registration-time alpha allocation.

    New coordinates may be registered after earlier audit results. For the
    guarantee, allocations must be predictable and their sum must be <= 1 on
    every possible path. All coordinates of a release must be registered and
    certified before the release may be called eligible.
    """

    def __init__(self, alpha: float = .05):
        if not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self._alpha = float(alpha)
        self._coordinates: dict[str, AuditCoordinate] = {}
        self._spent = 0.0

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def coordinates(self) -> dict[str, AuditCoordinate]:
        return dict(self._coordinates)

    def _spend(self, weight: float) -> None:
        if (not isinstance(weight, (int, float)) or not math.isfinite(weight)
                or not 0 < weight <= 1 - self._spent + 1e-15):
            raise ValueError("weights must be positive and fit the remaining budget")
        if self._spent + weight > 1 + 1e-15:
            raise ValueError("weight sum exceeds one")
        self._spent += weight

    def _add(self, coord: AuditCoordinate) -> None:
        if not coord.name or coord.name in self._coordinates:
            raise ValueError("coordinate name must be new and nonempty")
        self._coordinates[coord.name] = coord

    def register(self, name: str, tolerance: float, weight: float) -> AuditCoordinate:
        if not name or name in self._coordinates:
            raise ValueError("coordinate name must be new and nonempty")
        if (not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance)
                or not 0 < tolerance < 1):
            raise ValueError("tolerance must be in (0, 1)")
        self._spend(weight)
        coord = AuditCoordinate(name, float(tolerance), float(weight), self._alpha)
        self._add(coord)
        return coord

    def register_chain(self, prefix: str, tolerance: float, weight: float,
                       labels: list[str]) -> AuditChain:
        """Spend one weight for a chain of coordinates in a fixed order."""
        if not prefix or ":" in prefix:
            raise ValueError("chain prefix must be nonempty and contain no colon")
        if (not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance)
                or not 0 < tolerance < 1):
            raise ValueError("tolerance must be in (0, 1)")
        self._spend(weight)
        return AuditChain(self, prefix, tolerance, weight, labels)

    def register_release_chain(self, prefix: str, tolerance: float, weight: float,
                               labels: list[str], components: list[str]) -> ReleaseChain:
        """Spend one weight for ordered settings tested against all components."""
        if not prefix or ":" in prefix:
            raise ValueError("chain prefix must be nonempty and contain no colon")
        if (not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance)
                or not 0 < tolerance < 1):
            raise ValueError("tolerance must be in (0, 1)")
        self._spend(weight)
        return ReleaseChain(self, prefix, tolerance, weight, labels, components)

    def eligible(self, names: list[str]) -> bool:
        if not names or len(names) != len(set(names)):
            raise ValueError("supply distinct declared coordinates")
        return all(name in self._coordinates and self._coordinates[name].certified
                   for name in names)


def zero_success_budget(tolerance: float, alpha: float, weight: float) -> int:
    """Least trials needed if every observed attack fails, with alternative 0.

    This assumes a bet of zero on every draw; any positive alternative needs more
    trials. The closed form is checked against the accumulation that ``observe``
    performs, so the count returned is the trial at which the coordinate crosses.
    """
    gate = AdaptiveAudit(alpha)
    coordinate = gate.register("budget", tolerance, weight)
    step = -math.log1p(-coordinate.tolerance)
    n = max(1, math.ceil(coordinate.threshold / step))

    def crosses(trials: int) -> bool:
        total = 0.0
        for _ in range(trials):
            total += step
        return total >= coordinate.threshold

    while n > 1 and crosses(n - 1):
        n -= 1
    while not crosses(n):
        n += 1
    return n


def planned_weights(tolerances: list[float], trial_costs: list[float]) -> list[float]:
    """Continuous zero-success cost optimum for a frozen set of coordinates.

    Rounding trial counts, nonzero successes, and unknown future scenarios can
    alter the best actual allocation. Freeze this plan before its audit begins.
    """
    if len(tolerances) != len(trial_costs) or not tolerances:
        raise ValueError("provide matching nonempty coordinate arrays")
    importance = []
    for tau, cost in zip(tolerances, trial_costs):
        if (not isinstance(tau, (int, float)) or not math.isfinite(tau)
                or not 0 < tau < 1 or not isinstance(cost, (int, float))
                or not math.isfinite(cost) or cost <= 0):
            raise ValueError("require finite tolerance in (0,1) and positive cost")
        importance.append(cost / -math.log1p(-tau))
    total = math.fsum(importance)
    return [v / total for v in importance]


def drift(risk: float, bet: float, tolerance: float) -> float:
    """Mean log-growth per draw of the e-process at true risk ``risk`` and fixed bet."""
    if not 0 <= bet < tolerance < 1:
        raise ValueError("require 0 <= bet < tolerance < 1")
    growth = (1 - risk) * (math.log1p(-bet) - math.log1p(-tolerance))
    if risk > 0:
        growth += risk * (math.log(bet / tolerance) if bet > 0 else -math.inf)
    return growth


def expected_length_bound(risk: float, bet: float, tolerance: float, alpha: float,
                          weight: float) -> float:
    """The expected-length proposition: an upper bound on the expected draws to certify, or inf."""
    mu = drift(risk, bet, tolerance)
    if not mu > 0:
        return math.inf
    return (-math.log(alpha * weight) + math.log1p(-bet) - math.log1p(-tolerance)) / mu


def drift_weights(tolerances: list[float], trial_costs: list[float],
                  planning_risks: list[float], bets: list[float]) -> list[float]:
    """The allocation proposition: weights proportional to cost over planned drift.

    Planning risks and bets must be fixed before any outcome. With zero planning
    risks and zero bets this reduces to ``planned_weights``.
    """
    if not (len(tolerances) == len(trial_costs) == len(planning_risks) == len(bets)) \
            or not tolerances:
        raise ValueError("provide matching nonempty coordinate arrays")
    importance = []
    for tau, cost, p, q in zip(tolerances, trial_costs, planning_risks, bets):
        mu = drift(p, q, tau)
        if not mu > 0 or cost <= 0:
            raise ValueError("planned drift and cost must be positive")
        importance.append(cost / mu)
    total = math.fsum(importance)
    return [v / total for v in importance]


# --------------------------------------------------------------------------------
# Controls on the audit process itself.
#
# The gate's error bound assumes the operator cannot predict attack targets when a
# coordinate is registered and cannot choose a bet after seeing the outcome. The
# helpers below make those two conditions checkable after the fact. They are not a
# security boundary on their own: code in the same Python process can still change
# private state. A deployment that needs protection against the operator must run
# the registry as a separate service with its own storage, keys and time source.
# --------------------------------------------------------------------------------



def commit(secret: bytes) -> str:
    """Commitment to an operator secret, published at registration."""
    return hashlib.sha256(b"privacy-risk/commit/" + secret).hexdigest()


def target_seed(secret: bytes, beacon: str, coordinate: str, draw: int) -> int:
    """Seed for one attack target, fixed by a committed secret and a later beacon.

    The operator commits to ``secret`` when it registers the coordinate and names a
    future round of a public randomness beacon. Neither party alone can choose the
    targets: the operator cannot grind seeds without grinding the beacon, and a
    verifier who later sees the secret can recompute every target.
    """
    message = f"{beacon}|{coordinate}|{int(draw)}".encode()
    digest = hmac.new(secret, message, hashlib.sha256).digest()
    return int.from_bytes(digest[:8], "big")


class AuditLedger:
    """An append-only, hash-chained record of registrations, bets and outcomes.

    Each entry carries the hash of the one before it, so an entry that is changed,
    dropped or moved breaks the chain. The chain head should be time-stamped or
    published outside the operator's control; without that, an operator who can
    rewrite the whole file can rewrite the chain.
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []

    @property
    def head(self) -> str:
        return self._entries[-1]["hash"] if self._entries else "0" * 64

    @property
    def entries(self) -> list[dict]:
        return [dict(e) for e in self._entries]

    def append(self, kind: str, payload: dict) -> str:
        body = {"index": len(self._entries), "kind": str(kind), "previous": self.head,
                "payload": payload}
        encoded = _json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        entry = dict(body, hash=hashlib.sha256(encoded).hexdigest())
        self._entries.append(entry)
        return entry["hash"]

    @staticmethod
    def verify(entries: list[dict]) -> bool:
        previous = "0" * 64
        for index, entry in enumerate(entries):
            body = {k: entry[k] for k in ("index", "kind", "previous", "payload")}
            if entry["index"] != index or entry["previous"] != previous:
                return False
            encoded = _json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
            if hashlib.sha256(encoded).hexdigest() != entry["hash"]:
                return False
            previous = entry["hash"]
        return True


class CommittedCoordinate:
    """A coordinate whose bet is recorded before the outcome exists.

    ``commit_bet`` logs the alternative for the next draw. Only then does
    ``run`` derive the target from the committed secret and the beacon, call the
    registered attack on it, and log the outcome. The analyst never supplies an
    outcome, so a bet cannot be chosen after the result is known.
    """

    def __init__(self, coordinate: AuditCoordinate, ledger: AuditLedger,
                 secret_commitment: str, beacon: str):
        self._coord = coordinate
        self._ledger = ledger
        self._commitment = secret_commitment
        self._beacon = beacon
        self._pending: float | None = None
        ledger.append("register", {"coordinate": coordinate.name,
                                   "tolerance": coordinate.tolerance,
                                   "weight": coordinate.weight,
                                   "threshold": coordinate.threshold,
                                   "commitment": secret_commitment, "beacon": beacon})

    @property
    def coordinate(self) -> AuditCoordinate:
        return self._coord

    def commit_bet(self, alternative: float) -> None:
        if self._pending is not None:
            raise RuntimeError("a bet is already committed for the next draw")
        if not 0 <= float(alternative) < self._coord.tolerance:
            raise ValueError("alternative must be in [0, tolerance)")
        self._pending = float(alternative)
        self._ledger.append("bet", {"coordinate": self._coord.name,
                                    "draw": self._coord.trials,
                                    "alternative": self._pending})

    def run(self, secret: bytes, attack) -> int:
        """Draw the committed target, run ``attack(seed) -> 0/1`` and record it."""
        if self._pending is None:
            raise RuntimeError("commit a bet before the draw")
        if commit(secret) != self._commitment:
            raise ValueError("secret does not match the registered commitment")
        seed = target_seed(secret, self._beacon, self._coord.name, self._coord.trials)
        outcome = int(attack(seed))
        self._ledger.append("outcome", {"coordinate": self._coord.name,
                                        "draw": self._coord.trials, "outcome": outcome})
        alternative, self._pending = self._pending, None
        self._coord.observe(outcome, alternative)
        return outcome
