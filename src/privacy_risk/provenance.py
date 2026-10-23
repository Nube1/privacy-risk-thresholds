"""A keyed, hash-chained provenance ledger for the SENTRY-R stages.

The network in SENTRY-R is closed: the only thing that reaches it is a feature
vector that passed the schema, a named human review and a measured attack. The
LLM is an injector of scenarios into that closed system, nothing more. The
earlier ledger recorded only the final audit, so the order of the stages,
proposal, review, measurement, training, freeze and audit, could be asserted but
not shown. This ledger records every stage, chains the entries by hash, signs
each entry with an HMAC key the operator holds, and refuses an entry whose
prerequisites are not already on the chain.

Three rules are enforced:

1. Order. A review needs its proposal, a measurement needs an approving review,
   a training run needs every measured proposal it uses, a freeze needs its
   model, and an audit needs its freeze.
2. Budget. Each audit spends the family-wise alpha of its freeze against the
   programme's declared error budget. The budget is set once, on the chain, and
   an audit that would overspend it is refused. A fresh shortlist after a failed
   audit therefore costs error budget rather than starting from zero.
3. Closure. Once a freeze has been audited, no training run may use that
   freeze's development evidence together with evidence recorded after the
   audit; otherwise the audit outcome could steer the model that is audited next.

What the ledger cannot do is also stated: an operator who holds the key and can
rewrite the file can rewrite the chain. The chain head should be time-stamped or
published outside the operator's control.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGES = ("budget", "proposal", "review", "measurement", "training", "freeze", "audit")
GENESIS = "0" * 64
KEY_ENV = "SENTRY_R_LEDGER_KEY"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def load_key(path: Path | None = None) -> bytes:
    """Read the ledger key from a file, else from the environment.

    The key is raw bytes from ``path`` or the UTF-8 bytes of ``SENTRY_R_LEDGER_KEY``.
    It must be at least 16 bytes. No default key exists: an unkeyed ledger is a
    choice the caller makes explicitly with ``key=None``.
    """
    if path is not None:
        key = Path(path).read_bytes().strip()
    else:
        key = os.environ.get(KEY_ENV, "").encode("utf-8")
    if len(key) < 16:
        raise ValueError("ledger key must be at least 16 bytes, from a key file or "
                         f"the {KEY_ENV} environment variable")
    return key


class LedgerError(ValueError):
    """The ledger refused an entry or failed verification."""


class ProvenanceLedger:
    """Append-only JSON-lines ledger across the SENTRY-R stages.

    Each line holds ``index``, ``stage``, ``previous`` (hash of the line
    before), ``payload``, ``recorded_utc`` and ``hash``; with a key it also
    holds ``mac``, the HMAC-SHA256 of the hash. A payload is a small dict of
    digests and identifiers, never a proposal's text and never a count.
    """

    def __init__(self, path: Path, key: bytes | None):
        self.path = Path(path)
        self.key = key
        self._entries: list[dict] = []
        if self.path.exists():
            self._entries = [json.loads(line) for line in
                             self.path.read_text(encoding="utf-8").splitlines()
                             if line.strip()]
            self.verify()

    # -- reading -----------------------------------------------------------------

    @property
    def entries(self) -> list[dict]:
        return [dict(e) for e in self._entries]

    @property
    def head(self) -> str:
        return self._entries[-1]["hash"] if self._entries else GENESIS

    def _of(self, stage: str) -> list[dict]:
        return [e for e in self._entries if e["stage"] == stage]

    def verify(self) -> bool:
        """Check the chain and, with a key, every signature. Raise on failure."""
        previous = GENESIS
        for index, entry in enumerate(self._entries):
            body = {k: entry.get(k) for k in
                    ("index", "stage", "previous", "payload", "recorded_utc")}
            if entry.get("index") != index or entry.get("previous") != previous:
                raise LedgerError(f"ledger chain broken at entry {index}")
            if entry.get("stage") not in STAGES:
                raise LedgerError(f"unknown stage at entry {index}")
            if _sha256(body) != entry.get("hash"):
                raise LedgerError(f"ledger entry {index} was altered")
            if self.key is not None:
                expected = hmac.new(self.key, entry["hash"].encode(), hashlib.sha256)
                if not hmac.compare_digest(expected.hexdigest(), str(entry.get("mac"))):
                    raise LedgerError(f"ledger entry {index} fails its signature")
            previous = entry["hash"]
        return True

    # -- writing -----------------------------------------------------------------

    def _append(self, stage: str, payload: dict) -> dict:
        body = {"index": len(self._entries), "stage": stage, "previous": self.head,
                "payload": payload,
                "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        entry = dict(body, hash=_sha256(body))
        if self.key is not None:
            entry["mac"] = hmac.new(self.key, entry["hash"].encode(),
                                    hashlib.sha256).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(entry).decode("utf-8") + "\n")
        self._entries.append(entry)
        return entry

    def set_budget(self, programme: str, alpha_budget: float) -> dict:
        """Declare a programme's total family-wise error budget, once."""
        if not isinstance(programme, str) or not programme:
            raise LedgerError("programme name required")
        if not 0 < float(alpha_budget) <= 1:
            raise LedgerError("alpha budget must lie in (0, 1]")
        if any(e["payload"]["programme"] == programme for e in self._of("budget")):
            raise LedgerError(f"budget for programme {programme!r} is already on the chain")
        return self._append("budget", {"programme": programme,
                                       "alpha_budget": float(alpha_budget)})

    def record_proposals(self, staged: list[dict]) -> dict:
        digests = sorted(row["proposal_digest"] for row in staged)
        if len(set(digests)) != len(digests):
            raise LedgerError("duplicate proposal digests")
        known = self._proposal_digests()
        repeated = [d for d in digests if d in known]
        if repeated:
            raise LedgerError("proposal already on the chain: " + repeated[0][:12])
        return self._append("proposal", {"proposal_digests": digests,
                                         "llm_models": sorted({r["llm_model"] for r in staged}),
                                         "prompt_hashes": sorted({r["prompt_hash"]
                                                                  for r in staged})})

    def record_review(self, approved: list[dict]) -> dict:
        known = self._proposal_digests()
        for row in approved:
            if row["proposal_digest"] not in known:
                raise LedgerError("review of a proposal that is not on the chain")
        return self._append("review", {
            "approved": sorted((row["proposal_digest"], row["review"]["decision_digest"])
                               for row in approved)})

    def record_measurements(self, measured: list[dict]) -> dict:
        approved = self._approved_digests()
        for row in measured:
            if row["proposal_digest"] not in approved:
                raise LedgerError("measurement of a proposal without an approving review")
        return self._append("measurement", {
            "evidence": sorted((row["proposal_digest"], row["evidence_id"])
                               for row in measured),
            "content_sha256": sorted((row["proposal_digest"], row["evidence_id"],
                                      _sha256(row)) for row in measured)})

    def record_training(self, bundle: dict, train_rows: list[dict],
                        valid_rows: list[dict]) -> dict:
        measured = self._measured()
        content = {tuple(item) for e in self._of("measurement")
                   for item in e["payload"].get("content_sha256", [])}
        development = [*train_rows, *valid_rows]
        for row in development:
            d = row.get("proposal_digest")
            if d is not None and (d, row["evidence_id"]) not in measured:
                raise LedgerError("training uses a measured proposal that is not on the chain")
            if d is not None and (d, row["evidence_id"], _sha256(row)) not in content:
                raise LedgerError("measured proposal content differs from the ledger commitment; "
                                  "legacy measurements need a new content-bound measurement")
        evidence_ids = sorted(bundle["development_evidence_ids"])
        self._check_closure(set(evidence_ids))
        return self._append("training", {"model_sha256": _sha256(bundle),
                                         "schema_sha256": bundle["schema_sha256"],
                                         "development_evidence_ids": evidence_ids})

    def record_freeze(self, freeze: dict, programme: str) -> dict:
        models = {e["payload"]["model_sha256"] for e in self._of("training")}
        if freeze["model_sha256"] not in models:
            raise LedgerError("freeze refers to a model whose training is not on the chain")
        if not any(e["payload"]["programme"] == programme for e in self._of("budget")):
            raise LedgerError(f"no error budget on the chain for programme {programme!r}")
        return self._append("freeze", {"freeze_sha256": freeze["freeze_sha256"],
                                       "model_sha256": freeze["model_sha256"],
                                       "programme": programme,
                                       "alpha_family": float(freeze["alpha_family"]),
                                       "development_evidence_ids":
                                           sorted(freeze["development_evidence_ids"])})

    def record_audit(self, decision: dict) -> dict:
        freeze_hash = decision["freeze_sha256"]
        freezes = [e for e in self._of("freeze")
                   if e["payload"]["freeze_sha256"] == freeze_hash]
        if not freezes:
            raise LedgerError("audit of a freeze that is not on the chain")
        if any(e["payload"]["freeze_sha256"] == freeze_hash for e in self._of("audit")):
            raise LedgerError("this freeze has already been audited")
        programme = freezes[0]["payload"]["programme"]
        alpha = float(decision["alpha_family"])
        if alpha != freezes[0]["payload"]["alpha_family"]:
            raise LedgerError("audit alpha differs from the frozen alpha on the chain")
        spent = self.alpha_spent(programme) + alpha
        budget = self.alpha_budget(programme)
        if spent > budget + 1e-12:
            raise LedgerError(f"audit would spend {spent:.4f} of a {budget:.4f} error "
                              f"budget for programme {programme!r}")
        return self._append("audit", {"freeze_sha256": freeze_hash, "programme": programme,
                                      "alpha_family": alpha,
                                      "decision_sha256": _sha256(decision),
                                      "selected": decision.get("selected"),
                                      "alpha_spent": spent})

    # -- queries -----------------------------------------------------------------

    def alpha_budget(self, programme: str) -> float:
        for e in self._of("budget"):
            if e["payload"]["programme"] == programme:
                return float(e["payload"]["alpha_budget"])
        raise LedgerError(f"no error budget on the chain for programme {programme!r}")

    def alpha_spent(self, programme: str) -> float:
        return float(sum(e["payload"]["alpha_family"] for e in self._of("audit")
                         if e["payload"]["programme"] == programme))

    def _proposal_digests(self) -> set[str]:
        return {d for e in self._of("proposal") for d in e["payload"]["proposal_digests"]}

    def _approved_digests(self) -> set[str]:
        return {d for e in self._of("review") for d, _ in e["payload"]["approved"]}

    def _measured(self) -> set[tuple[str, str]]:
        return {tuple(pair) for e in self._of("measurement")
                for pair in e["payload"]["evidence"]}

    def _check_closure(self, evidence_ids: set[str]) -> None:
        """No model may mix an audited freeze's evidence with post-audit evidence."""
        for audit in self._of("audit"):
            freeze = next(e for e in self._of("freeze")
                          if e["payload"]["freeze_sha256"] == audit["payload"]["freeze_sha256"])
            audited = set(freeze["payload"]["development_evidence_ids"])
            later = {pair[1] for e in self._of("measurement")
                     if e["index"] > audit["index"] for pair in e["payload"]["evidence"]}
            if evidence_ids & audited and evidence_ids & later:
                raise LedgerError("training mixes evidence of an audited freeze with "
                                  "evidence measured after that audit; start a new "
                                  "development set or spend a new freeze")

    def summary(self) -> dict:
        return {"entries": len(self._entries), "head": self.head, "keyed": self.key is not None,
                "stages": {s: len(self._of(s)) for s in STAGES},
                "programmes": {e["payload"]["programme"]: {
                    "alpha_budget": e["payload"]["alpha_budget"],
                    "alpha_spent": self.alpha_spent(e["payload"]["programme"])}
                    for e in self._of("budget")}}
