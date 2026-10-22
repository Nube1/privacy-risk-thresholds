"""Generate synthetic SENTRY-R inputs. No real-person or live-release evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from privacy_risk.sentry_r import ScenarioSchema, digest, review_proposals, stage_llm, write_jsonl

FEATURES = ("key_sparsity", "sample_fraction", "auxiliary_strength",
            "release_overlap", "access_strength", "mechanism_noise")
LEDGER = "audit_ledger.jsonl"


def toy_probabilities(x: np.ndarray) -> np.ndarray:
    """The toy attack-success law. It is synthetic and has no empirical basis."""
    z = (-3 + .7*x[:, 0] + .4*x[:, 1] + .75*x[:, 2] +
         .6*x[:, 3] - .4*x[:, 4] - .5*x[:, 5] + .45*x[:, 0]*x[:, 2])
    base = 1/(1+np.exp(-z))
    return np.column_stack([base, np.minimum(.99, base*1.3)])


def toy_development(rng: np.random.Generator, count: int, group: str,
                    trials: int = 50) -> list[dict]:
    """Draw counted development attacks from the toy law for one programme."""
    x = rng.uniform(-1.5, 1.5, size=(count, len(FEATURES)))
    b = np.full((count, 2), trials)
    k = rng.binomial(b, toy_probabilities(x))
    return [{"record_id": f"{group}-{i}",
             "features": {name: float(v) for name, v in zip(FEATURES, x[i])},
             "group": group, "evidence_id": f"toy-{group}-{i}",
             "evidence_kind": "observed_development_attack",
             "successes": k[i].tolist(), "trials": b[i].tolist()}
            for i in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/sentry_r"))
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--audit-shortlist", type=Path,
                        help="after training, draw fresh toy audits for this frozen shortlist")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    schema_object = {
        "features": [{"name": name, "min": -2, "max": 2} for name in FEATURES],
        "coordinates": ["target_linkage", "pooled_linkage"],
        "tolerances": [.12, .18],
    }
    (root / "schema.json").write_text(json.dumps(schema_object, indent=2) + "\n")
    schema = ScenarioSchema.load(root / "schema.json")

    probabilities = toy_probabilities

    def features(row: np.ndarray) -> dict:
        return {name: float(value) for name, value in zip(schema.features, row)}

    write_jsonl(root / "train.jsonl", toy_development(rng, 320, "synthetic-programme-train"))
    write_jsonl(root / "validation.jsonl",
                toy_development(rng, 100, "synthetic-programme-heldout"))
    candidate_x = rng.uniform(-1.3, 1.3, size=(8, len(schema.features)))
    candidates = [{"candidate_id": f"setting-{i}", "features": features(candidate_x[i]),
                   "utility": float(.5 + .1*candidate_x[i, 1] - .05*candidate_x[i, 5])}
                  for i in range(len(candidate_x))]
    write_jsonl(root / "candidates.jsonl", candidates)
    # This is an LLM-shaped *proposal*; it contains no measured labels. The
    # provenance fields are written by this script (the caller), not by a model.
    proposed = [{"proposal_id": "proposed-pooled-1",
                 "scenario": "Test joining a historical vintage with the current release.",
                 "features": features(candidate_x[0]),
                 "llm_model": "illustrative-json-proposal", "prompt_hash": "toy-prompt-v1"}]
    write_jsonl(root / "llm_proposals.jsonl", proposed)
    staged = stage_llm(proposed, schema)
    write_jsonl(root / "staged_llm.jsonl", staged)
    decisions = [{"proposal_id": staged[0]["proposal_id"],
                  "proposal_digest": staged[0]["proposal_digest"],
                  "approved": True, "reviewer": "synthetic-demo-review",
                  "reason": "toy attack has defined sampling unit and synthetic source"}]
    write_jsonl(root / "review_decisions.jsonl", decisions)
    write_jsonl(root / "approved_llm.jsonl", review_proposals(staged, decisions))
    observed = [{"proposal_id": staged[0]["proposal_id"],
                 "proposal_digest": staged[0]["proposal_digest"],
                 "group": "synthetic-programme-future-development",
                 "evidence_id": "toy-independent-development-attack-1",
                 "evidence_kind": "observed_development_attack",
                 "successes": rng.binomial(100, probabilities(candidate_x[:1])[0]).tolist(),
                 "trials": [100, 100]}]
    write_jsonl(root / "development_outcomes.jsonl", observed)
    if args.audit_shortlist:
        freeze = json.loads(args.audit_shortlist.read_text())
        chosen = freeze["candidates"]
        if freeze["schema_sha256"] != digest(schema.as_dict()):
            raise ValueError("audit shortlist schema differs")
        # Separate stream from development, drawn only for the frozen shortlist.
        audit_rng = np.random.default_rng(args.seed + 1000003)
        x_audit = np.stack([schema.vector(row["features"]) for row in chosen])
        # Draw exactly the trial counts frozen in the shortlist plan.
        b = np.array([freeze["planned_trials"][row["candidate_id"]] for row in chosen])
        k = audit_rng.binomial(b, probabilities(x_audit))
        write_jsonl(root / "toy_audit_shortlist.jsonl", [
            {"candidate_id": row["candidate_id"],
             "evidence_kind": "independent_final_audit",
             "freeze_sha256": freeze["freeze_sha256"],
             "evidence_id": f"toy-final-{i}", "successes": k[i].tolist(),
             "trials": b[i].tolist()} for i, row in enumerate(chosen)])
    else:
        # Regenerating the toy inputs starts a new toy history, so the toy
        # ledger is cleared. A real ledger must never be cleared this way.
        (root / LEDGER).unlink(missing_ok=True)
    (root / "SYNTHETIC_ONLY.txt").write_text(
        "Synthetic demonstration only. Attack counts were sampled from a toy law, "
        "not measured on a real release. The demo is not a release certificate.\n")
    print(json.dumps({"output": str(root), "seed": args.seed,
                      "schema_sha256": digest(schema.as_dict())}))


if __name__ == "__main__":
    main()
