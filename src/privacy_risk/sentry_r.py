"""SENTRY-R: scenario-enriched neural triage for release review.

LLM text is an untrusted proposal. Only measured development outcomes fit the
model. The shortlist freeze fixes alpha, planned audit trials, the tolerance
owner, an expiry date and the development evidence IDs. The final audit reads
those values from the freeze and refuses counts that depart from them. An
optional append-only ledger refuses a second audit of the same freeze.

The network is closed. Its only input is a feature vector that passed the
schema, a named review and a measured attack; an LLM injects scenarios into
that closed system and nothing else. With a :class:`ProvenanceLedger`, every
stage is written to one keyed, hash-chained file, so the order proposal, review,
measurement, training, freeze, audit can be shown afterwards, each audit spends
a programme error budget, and a model may not mix an audited freeze's evidence
with evidence measured after that audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from .neural_trust import ReleaseEvidence, certify
from .provenance import ProvenanceLedger, load_key

SCHEMA_VERSION = "sentry-r-1"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical(row).decode("utf-8") + "\n" for row in rows),
                    encoding="utf-8")


@dataclass(frozen=True)
class ScenarioSchema:
    features: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    coordinates: tuple[str, ...]
    tolerances: tuple[float, ...]

    @classmethod
    def load(cls, path: Path) -> ScenarioSchema:
        obj = json.loads(path.read_text(encoding="utf-8"))
        features = obj["features"]
        names = tuple(f["name"] for f in features)
        lower = tuple(float(f["min"]) for f in features)
        upper = tuple(float(f["max"]) for f in features)
        coordinates = tuple(obj["coordinates"])
        tolerances = tuple(float(x) for x in obj["tolerances"])
        if (not names or not coordinates or len(set(names)) != len(names)
                or len(set(coordinates)) != len(coordinates)
                or not all(isinstance(v, str) and v for v in names + coordinates)
                or len(tolerances) != len(coordinates)
                or not np.isfinite(lower + upper + tolerances).all()
                or any(a >= b for a, b in zip(lower, upper))
                or any(not 0 < t < 1 for t in tolerances)):
            raise ValueError("invalid feature bounds, coordinates or tolerances")
        return cls(names, lower, upper, coordinates, tolerances)

    def as_dict(self) -> dict:
        return {"features": [{"name": n, "min": lo, "max": hi}
                             for n, lo, hi in zip(self.features, self.lower, self.upper)],
                "coordinates": list(self.coordinates),
                "tolerances": list(self.tolerances)}

    def vector(self, values: dict) -> np.ndarray:
        if not isinstance(values, dict) or set(values) != set(self.features):
            raise ValueError("feature names must match the declared schema exactly")
        raw = [values[name] for name in self.features]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in raw):
            raise ValueError("features must be numeric, not strings or booleans")
        vector = np.array(raw, dtype=float)
        if (not np.isfinite(vector).all() or np.any(vector < self.lower)
                or np.any(vector > self.upper)):
            raise ValueError("nonfinite or out-of-domain feature")
        return vector


MAX_PROPOSALS = 32
MAX_RESPONSE_BYTES = 256_000


def stage_llm(rows: list[dict], schema: ScenarioSchema, *,
              provenance: ProvenanceLedger | None = None) -> list[dict]:
    """Validate LLM proposals; preserve text as data, never as an instruction."""
    if not rows:
        raise ValueError("no proposals to stage")
    if len(rows) > MAX_PROPOSALS:
        raise ValueError(f"at most {MAX_PROPOSALS} proposals per intake; split the batch")
    output, seen = [], set()
    for row in rows:
        if set(row) != {"proposal_id", "scenario", "features", "llm_model", "prompt_hash"}:
            raise ValueError("LLM proposal fields must match the intake contract")
        key = row["proposal_id"]
        if not isinstance(key, str) or not 0 < len(key) <= 120 or key in seen:
            raise ValueError("duplicate or invalid proposal_id")
        seen.add(key)
        if (not isinstance(row["scenario"], str) or not 0 < len(row["scenario"]) <= 4000
                or not isinstance(row["llm_model"], str) or not row["llm_model"]
                or not isinstance(row["prompt_hash"], str)
                or not row["prompt_hash"]):
            raise ValueError("missing proposal text or model provenance")
        schema.vector(row["features"])
        output.append({**row, "proposal_digest": digest(row),
                       "status": "unverified_llm_proposal"})
    if provenance is not None:
        provenance.record_proposals(output)
    return output


LLM_RESPONSE_FIELDS = frozenset({"proposal_id", "scenario", "features"})


def propose_from_llm(call: Callable[[str], str], schema: ScenarioSchema,
                     question: str, *, model_name: str,
                     max_bytes: int = MAX_RESPONSE_BYTES,
                     provenance: ProvenanceLedger | None = None) -> list[dict]:
    """Provider-neutral adapter: inject a callable, parse JSON, then stage.

    The caller decides whether any information is allowed to leave its system.
    The response must be a JSON array of objects with exactly ``proposal_id``,
    ``scenario`` and ``features``. Provenance is set here, not by the model:
    ``llm_model`` is the caller's ``model_name`` and ``prompt_hash`` is the
    SHA-256 of the exact prompt sent. No response text is executed and no
    success labels are accepted. A response longer than ``max_bytes`` or
    holding more than ``MAX_PROPOSALS`` objects is refused before parsing, so
    the model cannot flood the intake.
    """
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model_name must be a nonempty string set by the caller")
    prompt = ("Propose attack scenarios as a JSON array only. Required fields "
              "for each object: proposal_id, scenario, features. "
              "Numeric feature bounds and risk coordinates are: "
              + canonical(schema.as_dict()).decode("utf-8") + "\nQuestion: " + question)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    text = call(prompt)
    if not isinstance(text, str):
        raise ValueError("LLM adapter must return the response text")
    if len(text.encode("utf-8")) > max_bytes:
        raise ValueError(f"LLM response exceeds {max_bytes} bytes; refused unparsed")
    result = json.loads(text)
    if not isinstance(result, list):
        raise ValueError("LLM response must be a JSON array")
    if len(result) > MAX_PROPOSALS:
        raise ValueError(f"LLM returned more than {MAX_PROPOSALS} proposals; refused")
    rows = []
    for item in result:
        if not isinstance(item, dict) or set(item) != LLM_RESPONSE_FIELDS:
            raise ValueError("LLM response objects must hold exactly proposal_id, "
                             "scenario and features")
        rows.append({**item, "llm_model": model_name, "prompt_hash": prompt_hash})
    return stage_llm(rows, schema, provenance=provenance)


def review_proposals(staged: list[dict], decisions: list[dict], *,
                     provenance: ProvenanceLedger | None = None) -> list[dict]:
    """Record explicit human scenario review before measuring attacks."""
    proposals = {row["proposal_id"]: row for row in staged}
    if len(proposals) != len(staged):
        raise ValueError("duplicate staged proposals")
    approved, seen = [], set()
    for decision in decisions:
        if set(decision) != {"proposal_id", "proposal_digest", "approved", "reviewer",
                             "reason"} or decision["proposal_id"] not in proposals:
            raise ValueError("review decision has invalid or unknown proposal")
        key = decision["proposal_id"]
        row = proposals[key]
        if key in seen or row["proposal_digest"] != decision["proposal_digest"]:
            raise ValueError("duplicate review or proposal digest mismatch")
        seen.add(key)
        if (type(decision["approved"]) is not bool
                or not isinstance(decision["reviewer"], str) or not decision["reviewer"]
                or not isinstance(decision["reason"], str) or not decision["reason"]):
            raise ValueError("review needs boolean decision, reviewer and reason")
        if decision["approved"]:
            approved.append({**row, "status": "approved_for_development",
                             "review": {"reviewer": decision["reviewer"],
                                        "reason": decision["reason"],
                                        "decision_digest": digest(decision)}})
    if provenance is not None and approved:
        provenance.record_review(approved)
    return approved


def join_evidence(staged: list[dict], outcomes: list[dict],
                  schema: ScenarioSchema, *,
                  provenance: ProvenanceLedger | None = None) -> list[dict]:
    """Measured trials turn a proposed scenario into *development* data only."""
    proposals = {row["proposal_id"]: row for row in staged}
    if len(proposals) != len(staged):
        raise ValueError("duplicate staged proposals")
    output, seen, evidence_ids = [], set(), set()
    for row in outcomes:
        required = {"proposal_id", "proposal_digest", "group", "evidence_id",
                    "successes", "trials", "evidence_kind"}
        if set(row) != required or row["proposal_id"] not in proposals:
            raise ValueError("unmatched or invalid outcome")
        proposal = proposals[row["proposal_id"]]
        origin = {key: proposal[key] for key in
                  ("proposal_id", "scenario", "features", "llm_model", "prompt_hash")}
        review = proposal.get("review", {})
        decision = {"proposal_id": proposal["proposal_id"],
                    "proposal_digest": proposal.get("proposal_digest"),
                    "approved": True, "reviewer": review.get("reviewer"),
                    "reason": review.get("reason")}
        if (proposal.get("status") != "approved_for_development"
                or not review.get("reviewer")
                or review.get("decision_digest") != digest(decision)
                or digest(origin) != proposal.get("proposal_digest")):
            raise ValueError("unreviewed or changed proposal")
        if row["proposal_id"] in seen or row["proposal_digest"] != proposal["proposal_digest"]:
            raise ValueError("duplicate or changed proposal")
        seen.add(row["proposal_id"])
        if (row["evidence_kind"] != "observed_development_attack"
                or not isinstance(row["group"], str) or not row["group"]
                or not isinstance(row["evidence_id"], str) or not row["evidence_id"]
                or row["evidence_id"] in evidence_ids):
            raise ValueError("development evidence and unique group/id required")
        evidence_ids.add(row["evidence_id"])
        successes, trials = validate_counts(row, len(schema.coordinates))
        output.append({"record_id": row["proposal_id"], "features": proposal["features"],
                       "group": row["group"], "evidence_id": row["evidence_id"],
                       "evidence_kind": row["evidence_kind"],
                       "successes": successes, "trials": trials,
                       "proposal_digest": row["proposal_digest"]})
    if provenance is not None:
        provenance.record_measurements(output)
    return output


def validate_counts(row: dict, count: int) -> tuple[list[int], list[int]]:
    successes, trials = row["successes"], row["trials"]
    if (not isinstance(successes, list) or not isinstance(trials, list)
            or len(successes) != count or len(trials) != count
            or any(type(k) is not int or type(b) is not int or b < 1 or k < 0 or k > b
                   for k, b in zip(successes, trials))):
        raise ValueError("one integer success/trial count required per coordinate")
    return successes, trials


def matrix(rows: list[dict], schema: ScenarioSchema) -> tuple[np.ndarray, np.ndarray,
                                                                  np.ndarray]:
    if (not rows or any(row.get("evidence_kind") != "observed_development_attack"
                        for row in rows)):
        raise ValueError("only observed development attack outcomes can train")
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate development records")
    x = np.stack([schema.vector(row["features"]) for row in rows])
    counts = [validate_counts(row, len(schema.coordinates)) for row in rows]
    k = np.array([a for a, _ in counts], dtype=float)
    b = np.array([a for _, a in counts], dtype=float)
    return x, k, b


def augment_development(base: list[dict], measured: list[dict],
                        schema: ScenarioSchema) -> list[dict]:
    """Create a new development version; never append in place or accept audits."""
    combined = [*base, *measured]
    matrix(combined, schema)
    ids = [row.get("evidence_id") for row in combined]
    if any(not isinstance(v, str) or not v for v in ids) or len(set(ids)) != len(ids):
        raise ValueError("duplicate or missing development evidence ID")
    if not measured:
        raise ValueError("no measured proposals to append")
    return combined


class DeepRiskNet:
    """Dense ReLU network with explicit weighted-count backpropagation and Adam."""

    def __init__(self, dimension: int, coordinates: int, width: int = 32,
                 depth: int = 3, seed: int = 19):
        if min(dimension, coordinates, width) < 1 or depth < 0 or depth > 12:
            raise ValueError("invalid network dimensions")
        rng = np.random.default_rng(seed)
        sizes = [dimension] + [width] * depth + [coordinates]
        self.weights = [rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i + 1]))
                        for i in range(len(sizes) - 1)]
        self.biases = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]
        self.dimension, self.coordinates, self.depth = dimension, coordinates, depth

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray],
                                               list[np.ndarray]]:
        activations = [x]
        preacts = []
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = activations[-1] @ weight + bias
            preacts.append(z)
            activations.append(np.maximum(z, 0))
        logits = activations[-1] @ self.weights[-1] + self.biases[-1]
        p = 1 / (1 + np.exp(-np.clip(logits, -35, 35)))
        return p, activations, preacts

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != self.dimension or not np.isfinite(x).all():
            raise ValueError("invalid prediction features")
        return self.forward(x)[0]

    def loss_grads(self, x: np.ndarray, k: np.ndarray, b: np.ndarray,
                   l2: float = 0.) -> tuple[float, list[np.ndarray], list[np.ndarray]]:
        if (x.ndim != 2 or x.shape[1] != self.dimension
                or k.shape != (len(x), self.coordinates) or b.shape != k.shape
                or np.any(b < 1) or np.any(k < 0) or np.any(k > b)):
            raise ValueError("invalid counted attack training shapes")
        p, activations, preacts = self.forward(x)
        total = float(np.sum(b))
        # Binomial negative log likelihood without the parameter-free combinatorial term.
        loss = float(-np.sum(k * np.log(p) + (b - k) * np.log1p(-p)) / total)
        loss += .5 * l2 * sum(float(np.sum(w*w)) for w in self.weights)
        delta = (b*p-k) / total
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(v) for v in self.biases]
        for layer in range(len(self.weights) - 1, -1, -1):
            dw[layer] = activations[layer].T @ delta + l2*self.weights[layer]
            db[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights[layer].T) * (preacts[layer-1] > 0)
        return loss, dw, db

    def fit(self, train: tuple[np.ndarray, np.ndarray, np.ndarray],
            valid: tuple[np.ndarray, np.ndarray, np.ndarray], *,
            epochs: int = 250, batch_size: int = 64, lr: float = .005,
            patience: int = 25, l2: float = 1e-4, seed: int = 19) -> dict:
        if (min(epochs, batch_size, patience) < 1 or lr <= 0 or l2 < 0):
            raise ValueError("invalid optimiser settings")
        x, k, b = train
        xv, kv, bv = valid
        if len(x) < 2 or not len(xv):
            raise ValueError("nonempty development and validation partitions needed")
        self.loss_grads(x[:1], k[:1], b[:1])
        self.loss_grads(xv[:1], kv[:1], bv[:1])
        rng = np.random.default_rng(seed)
        parameters = [*self.weights, *self.biases]
        first = [np.zeros_like(p) for p in parameters]
        second = [np.zeros_like(p) for p in parameters]
        best_loss, best, best_epoch, no_improvement, step = float("inf"), None, 0, 0, 0
        for epoch in range(1, epochs + 1):
            for batch in np.array_split(rng.permutation(len(x)),
                                        max(1, (len(x) + batch_size - 1)//batch_size)):
                _, gw, gb = self.loss_grads(x[batch], k[batch], b[batch], l2)
                grads = [*gw, *gb]
                norm = np.sqrt(sum(float(np.sum(g*g)) for g in grads))
                scale = min(1., 10. / max(norm, 1e-12))
                step += 1
                for i, (param, grad) in enumerate(zip(parameters, grads)):
                    grad = grad * scale
                    first[i] = .9*first[i] + .1*grad
                    second[i] = .999*second[i] + .001*grad*grad
                    param -= lr*(first[i]/(1-.9**step)) / (
                        np.sqrt(second[i]/(1-.999**step)) + 1e-8)
            validation_loss = self.loss_grads(xv, kv, bv)[0]
            if validation_loss < best_loss - 1e-7:
                best_loss, best_epoch, no_improvement = validation_loss, epoch, 0
                best = ([w.copy() for w in self.weights], [v.copy() for v in self.biases])
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break
        self.weights, self.biases = best
        return {"best_epoch": best_epoch, "validation_log_loss": best_loss,
                "training_log_loss": self.loss_grads(x, k, b)[0]}

    def to_dict(self) -> dict:
        return {"weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "dimension": self.dimension, "coordinates": self.coordinates,
                "depth": self.depth}

    @classmethod
    def from_dict(cls, payload: dict) -> DeepRiskNet:
        sizes = [payload["dimension"]] + [len(b) for b in payload["biases"]]
        net = cls(sizes[0], payload["coordinates"],
                  width=sizes[1] if payload["depth"] else 1,
                  depth=payload["depth"])
        weights = [np.asarray(w, dtype=float) for w in payload["weights"]]
        biases = [np.asarray(b, dtype=float) for b in payload["biases"]]
        if (len(weights) != len(sizes)-1 or len(biases) != len(weights)
                or sizes[-1] != payload["coordinates"]
                or any(w.shape != (sizes[i], sizes[i+1]) or b.shape != (sizes[i+1],)
                       for i, (w, b) in enumerate(zip(weights, biases)))
                or not all(np.isfinite(w).all() and np.isfinite(b).all()
                           for w, b in zip(weights, biases))):
            raise ValueError("invalid model weight shapes or values")
        net.weights, net.biases = weights, biases
        return net


def train_model(train_rows: list[dict], valid_rows: list[dict], schema: ScenarioSchema,
                *, width: int = 32, depth: int = 3, epochs: int = 250,
                patience: int = 25, seed: int = 19,
                provenance: ProvenanceLedger | None = None) -> dict:
    train_groups = {row["group"] for row in train_rows}
    valid_groups = {row["group"] for row in valid_rows}
    train_ids = {row["record_id"] for row in train_rows}
    valid_ids = {row["record_id"] for row in valid_rows}
    if (train_groups & valid_groups or train_ids & valid_ids
            or {row["evidence_id"] for row in train_rows}
               & {row["evidence_id"] for row in valid_rows}):
        raise ValueError("train and validation groups/records/evidence must be disjoint")
    x, k, b = matrix(train_rows, schema)
    xv, kv, bv = matrix(valid_rows, schema)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) < 1e-6, 1., x.std(axis=0))
    net = DeepRiskNet(x.shape[1], k.shape[1], width, depth, seed)
    metrics = net.fit(((x-mean)/scale, k, b), ((xv-mean)/scale, kv, bv),
                      epochs=epochs, patience=patience, seed=seed)
    evidence_ids = [row["evidence_id"] for row in [*train_rows, *valid_rows]]
    if any(not isinstance(v, str) or not v for v in evidence_ids):
        raise ValueError("every development row needs a nonempty evidence_id")
    bundle = {"method": "SENTRY-R", "schema_version": SCHEMA_VERSION,
            "schema": schema.as_dict(), "schema_sha256": digest(schema.as_dict()),
            "mean": mean.tolist(), "scale": scale.tolist(), "network": net.to_dict(),
            "metrics": metrics, "training_sha256": digest(train_rows),
            "validation_sha256": digest(valid_rows),
            "training_groups": sorted(train_groups), "validation_groups": sorted(valid_groups),
            "development_evidence_ids": sorted(set(evidence_ids)),
            "training_settings": {"width": width, "depth": depth, "epoch_cap": epochs,
                                  "patience": patience, "seed": seed},
            "development_only": True,
            "warning": "Predictions rank proposals; independent attack audits certify releases."}
    if provenance is not None:
        provenance.record_training(bundle, train_rows, valid_rows)
    return bundle


def planned_trials_by_candidate(trials: int | list[int] | dict, names: list[str],
                                coordinates: int) -> dict[str, list[int]]:
    """Expand a trial plan to one integer count per candidate and coordinate."""
    if isinstance(trials, dict):
        plan = trials
    else:
        plan = {name: trials for name in names}
    if set(plan) != set(names):
        raise ValueError("planned trials must cover exactly the shortlisted candidates")
    output = {}
    for name in names:
        value = plan[name]
        row = [value] * coordinates if type(value) is int else value
        if (not isinstance(row, list) or len(row) != coordinates
                or any(type(v) is not int or v < 1 for v in row)):
            raise ValueError("planned trials must be positive integers, one per coordinate")
        output[name] = list(row)
    return output


def shortlist(bundle: dict, candidates: list[dict], size: int, *,
              alpha: float, trials: int | list[int] | dict, owner: str,
              expiry: str, programme: str | None = None,
              provenance: ProvenanceLedger | None = None) -> dict:
    """Rank candidates and freeze everything the final audit may not change.

    The freeze records the family-wise ``alpha``, the planned final-audit
    trials for every shortlisted candidate and coordinate, the named tolerance
    owner, an ISO expiry date and every development evidence ID. Its SHA-256
    digest binds these fields; ``final_audit`` reads them from here.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("a named tolerance owner is required")
    parse_expiry(expiry)
    development_ids = bundle.get("development_evidence_ids")
    if not isinstance(development_ids, list) or not development_ids:
        raise ValueError("model lacks development evidence IDs; retrain with this version")
    schema = schema_from_bundle(bundle)
    net = DeepRiskNet.from_dict(bundle["network"])
    if bundle["schema_sha256"] != digest(schema.as_dict()):
        raise ValueError("model and scenario schema mismatch")
    if not 1 <= size <= len(candidates):
        raise ValueError("invalid shortlist size")
    names = [row["candidate_id"] for row in candidates]
    if len(set(names)) != len(names) or not all(isinstance(v, str) and v for v in names):
        raise ValueError("candidate IDs must be unique")
    features = np.stack([schema.vector(row["features"]) for row in candidates])
    utility = np.array([row["utility"] for row in candidates], dtype=float)
    if not np.isfinite(utility).all():
        raise ValueError("candidate utilities must be finite")
    mean, scale = np.asarray(bundle["mean"]), np.asarray(bundle["scale"])
    risks = net.predict((features-mean)/scale)
    rank = np.argsort(np.max(risks/np.asarray(schema.tolerances), axis=1), kind="stable")
    selected = [{"candidate_id": names[i], "features": candidates[i]["features"],
                 "utility": float(utility[i]), "predicted_risks": risks[i].tolist()}
                for i in rank[:size]]
    plan = planned_trials_by_candidate(trials, [row["candidate_id"] for row in selected],
                                       len(schema.coordinates))
    freeze = {"method": "SENTRY-R", "schema_sha256": bundle["schema_sha256"],
              "model_sha256": digest(bundle), "coordinates": list(schema.coordinates),
              "tolerances": list(schema.tolerances), "candidates": selected,
              "alpha_family": float(alpha), "planned_trials": plan,
              "tolerance_owner": owner.strip(), "expiry": expiry,
              "development_evidence_ids": sorted(development_ids),
              "ranking_rule": "ascending largest predicted risk-to-tolerance ratio",
              "status": "fixed_shortlist_before_audit"}
    freeze["freeze_sha256"] = digest(freeze)
    if provenance is not None:
        if not isinstance(programme, str) or not programme:
            raise ValueError("a programme name is required to record a freeze on the ledger")
        provenance.record_freeze(freeze, programme)
    return freeze


def schema_from_bundle(bundle: dict) -> ScenarioSchema:
    obj = bundle["schema"]
    return ScenarioSchema(tuple(f["name"] for f in obj["features"]),
                          tuple(f["min"] for f in obj["features"]),
                          tuple(f["max"] for f in obj["features"]),
                          tuple(obj["coordinates"]), tuple(obj["tolerances"]))


def parse_expiry(expiry: str) -> date:
    if not isinstance(expiry, str):
        raise ValueError("expiry must be an ISO date string YYYY-MM-DD")
    try:
        return date.fromisoformat(expiry)
    except ValueError as error:
        raise ValueError("expiry must be an ISO date string YYYY-MM-DD") from error


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def final_audit(freeze: dict, outcomes: list[dict], *, ledger: Path | None = None,
                as_of: date | None = None,
                provenance: ProvenanceLedger | None = None) -> dict:
    """Certify a frozen shortlist from fresh audit counts.

    Alpha, planned trials, owner and expiry come from the freeze. The audit is
    refused if the freeze changed, if the date is past expiry, if any count
    departs from the planned trials, if an evidence ID was used in development,
    or if ``ledger`` already records an audit of this freeze. With ``ledger``
    set, the decision is appended to that JSON-lines file. With ``provenance``
    set, the audit is also refused unless its freeze is on the chain, unaudited,
    and within the programme's error budget; the decision is then chained.
    """
    expected = freeze.copy()
    actual_hash = expected.pop("freeze_sha256")
    if digest(expected) != actual_hash or freeze["status"] != "fixed_shortlist_before_audit":
        raise ValueError("shortlist was changed after freezing")
    required = {"alpha_family", "planned_trials", "tolerance_owner", "expiry",
                "development_evidence_ids"}
    if not required <= set(freeze):
        raise ValueError("freeze lacks alpha, trial plan, owner, expiry or evidence IDs; "
                         "re-freeze with this version")
    alpha = freeze["alpha_family"]
    expiry = parse_expiry(freeze["expiry"])
    if (as_of or date.today()) > expiry:
        raise ValueError("frozen shortlist has expired; the owner must re-freeze")
    if ledger is not None and any(entry.get("freeze_sha256") == actual_hash
                                  for entry in read_ledger(ledger)):
        raise ValueError("ledger already holds a final audit of this freeze")
    development_ids = set(freeze["development_evidence_ids"])
    candidates = {row["candidate_id"]: row for row in freeze["candidates"]}
    names = [row["candidate_id"] for row in outcomes]
    if len(set(names)) != len(names) or set(names) != set(candidates):
        raise ValueError("every frozen candidate must have one final audit")
    evidence_ids = [row.get("evidence_id") for row in outcomes]
    if len(set(evidence_ids)) != len(evidence_ids) or any(not v for v in evidence_ids):
        raise ValueError("one distinct audit evidence ID per candidate required")
    n = len(freeze["coordinates"])
    reports = []
    for row in outcomes:
        if row.get("evidence_kind") != "independent_final_audit":
            raise ValueError("final outcomes must be independent audit evidence")
        if row.get("freeze_sha256") != actual_hash:
            raise ValueError("audit outcome is not bound to this frozen shortlist")
        k, b = validate_counts(row, n)
        if b != freeze["planned_trials"][row["candidate_id"]]:
            raise ValueError("audit trials differ from the frozen plan")
        if row["evidence_id"] in development_ids:
            raise ValueError("final audit reuses development evidence")
        candidate = candidates[row["candidate_id"]]
        evidence = ReleaseEvidence(row["candidate_id"], np.array(k), np.array(b),
                                   np.array(freeze["tolerances"]), candidate["utility"])
        report = certify(evidence, alpha, len(candidates)*n)
        report["evidence_id"] = row["evidence_id"]
        reports.append(report)
    eligible = [row for row in reports if row["certified"]]
    decision = {"method": "SENTRY-R", "freeze_sha256": actual_hash,
                "alpha_family": alpha, "audited_coordinates": len(candidates)*n,
                "tolerance_owner": freeze["tolerance_owner"], "expiry": freeze["expiry"],
                "selected": max(eligible, key=lambda row: row["utility"])["name"]
                if eligible else None, "reports": reports,
                "scope": "conditional on declared scenarios and independent final attack draws"}
    if provenance is not None:
        provenance.record_audit(decision)
    if ledger is not None:
        entry = {"freeze_sha256": actual_hash, "schema_sha256": freeze["schema_sha256"],
                 "decision_sha256": digest(decision), "selected": decision["selected"],
                 "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(canonical(entry).decode("utf-8") + "\n")
    return decision


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path,
                        help="keyed, hash-chained ledger across every stage (JSON lines)")
    parser.add_argument("--key-file", type=Path,
                        help="file holding the ledger key; else the SENTRY_R_LEDGER_KEY variable")
    parser.add_argument("--unkeyed", action="store_true",
                        help="use the provenance ledger without signatures (chain only)")
    subs = parser.add_subparsers(dest="command", required=True)
    s = subs.add_parser("budget", help="Declare a programme's error budget on the ledger")
    s.add_argument("--programme", required=True)
    s.add_argument("--alpha-budget", type=float, required=True)
    s = subs.add_parser("verify-ledger", help="Check the chain and signatures, print a summary")
    s = subs.add_parser("stage-llm", help="Validate untrusted LLM scenario proposals")
    s.add_argument("--schema", type=Path, required=True)
    s.add_argument("--input", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = subs.add_parser("join-evidence", help="Attach observed development attacks")
    s.add_argument("--schema", type=Path, required=True)
    s.add_argument("--staged", type=Path, required=True)
    s.add_argument("--outcomes", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = subs.add_parser("review", help="Apply documented human proposal decisions")
    s.add_argument("--staged", type=Path, required=True)
    s.add_argument("--decisions", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = subs.add_parser("augment", help="Version measured proposals into development data")
    s.add_argument("--schema", type=Path, required=True)
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--measured", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s = subs.add_parser("train", help="Fit a deep model from observed attack counts")
    s.add_argument("--schema", type=Path, required=True)
    s.add_argument("--train", type=Path, required=True)
    s.add_argument("--validation", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--depth", type=int, default=3)
    s.add_argument("--width", type=int, default=32)
    s.add_argument("--epochs", type=int, default=250, help="epoch cap")
    s.add_argument("--patience", type=int, default=25,
                   help="stop after this many epochs without validation improvement")
    s.add_argument("--seed", type=int, default=19)
    s = subs.add_parser("shortlist", help="Freeze neural-ranked candidates")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--candidates", type=Path, required=True)
    s.add_argument("--size", type=int, required=True)
    s.add_argument("--alpha", type=float, default=.05,
                   help="family-wise error; frozen and read back by the audit")
    s.add_argument("--trials", type=int, required=True,
                   help="planned final-audit trials per candidate and coordinate")
    s.add_argument("--owner", required=True, help="named owner of the tolerances")
    s.add_argument("--expiry", required=True, help="ISO date after which the freeze lapses")
    s.add_argument("--programme", help="programme whose error budget this freeze draws on "
                                       "(required with --provenance)")
    s.add_argument("--output", type=Path, required=True)
    s = subs.add_parser("audit", help="Certify frozen candidates with fresh attacks")
    s.add_argument("--shortlist", type=Path, required=True)
    s.add_argument("--outcomes", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--ledger", type=Path,
                   help="append-only audit ledger (default: audit_ledger.jsonl "
                        "next to --output)")
    s.add_argument("--no-ledger", action="store_true",
                   help="do not read or write a ledger")
    args = parser.parse_args()
    provenance = None
    if args.command in ("budget", "verify-ledger") and args.provenance is None:
        parser.error(f"{args.command} needs --provenance")
    if args.provenance is not None:
        key = None if args.unkeyed else load_key(args.key_file)
        provenance = ProvenanceLedger(args.provenance, key)
    if args.command == "budget":
        provenance.set_budget(args.programme, args.alpha_budget)
        print(f"SENTRY-R budget: {args.programme} alpha budget {args.alpha_budget} recorded")
        return
    if args.command == "verify-ledger":
        provenance.verify()
        print(json.dumps(provenance.summary(), indent=2))
        return
    if args.command == "stage-llm":
        result = stage_llm(read_jsonl(args.input), ScenarioSchema.load(args.schema),
                           provenance=provenance)
        write_jsonl(args.output, result)
    elif args.command == "join-evidence":
        result = join_evidence(read_jsonl(args.staged), read_jsonl(args.outcomes),
                               ScenarioSchema.load(args.schema), provenance=provenance)
        write_jsonl(args.output, result)
    elif args.command == "review":
        result = review_proposals(read_jsonl(args.staged), read_jsonl(args.decisions),
                                  provenance=provenance)
        if not result:
            raise ValueError("no proposals approved for development")
        write_jsonl(args.output, result)
    elif args.command == "augment":
        result = augment_development(read_jsonl(args.base), read_jsonl(args.measured),
                                     ScenarioSchema.load(args.schema))
        write_jsonl(args.output, result)
    elif args.command == "train":
        result = train_model(read_jsonl(args.train), read_jsonl(args.validation),
                             ScenarioSchema.load(args.schema), width=args.width,
                             depth=args.depth, epochs=args.epochs,
                             patience=args.patience, seed=args.seed, provenance=provenance)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
    elif args.command == "shortlist":
        result = shortlist(json.loads(args.model.read_text()), read_jsonl(args.candidates),
                           args.size, alpha=args.alpha, trials=args.trials,
                           owner=args.owner, expiry=args.expiry, programme=args.programme,
                           provenance=provenance)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    else:
        ledger = None if args.no_ledger else (
            args.ledger or args.output.parent / "audit_ledger.jsonl")
        result = final_audit(json.loads(args.shortlist.read_text()),
                             read_jsonl(args.outcomes), ledger=ledger, provenance=provenance)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"SENTRY-R {args.command}: wrote {args.output}")


if __name__ == "__main__":
    cli()
