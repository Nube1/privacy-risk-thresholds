"""The keyed provenance ledger: order, budget, closure, signatures and injector limits."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from privacy_risk.provenance import KEY_ENV, LedgerError, ProvenanceLedger, load_key
from privacy_risk.sentry_r import (
    MAX_PROPOSALS, ScenarioSchema, cli, final_audit, join_evidence, propose_from_llm,
    review_proposals, shortlist, stage_llm, train_model, write_jsonl,
)

KEY = b"0123456789abcdef0123456789abcdef"


def development(group, count, start=0):
    return [{"record_id": f"{group}-{i}", "group": group,
             "features": {"a": i/10, "b": -.1}, "evidence_id": f"e-{group}-{i}",
             "evidence_kind": "observed_development_attack",
             "successes": [i % 2, 1], "trials": [20, 20]}
            for i in range(start, start+count)]


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        path = self.root/"schema.json"
        path.write_text(json.dumps({"features": [
            {"name": "a", "min": -2, "max": 2}, {"name": "b", "min": -2, "max": 2}],
            "coordinates": ["link", "pool"], "tolerances": [.2, .3]}))
        self.schema = ScenarioSchema.load(path)
        self.ledger = ProvenanceLedger(self.root/"chain.jsonl", KEY)

    def tearDown(self):
        self.tmp.cleanup()

    def proposal(self, key="P"):
        return {"proposal_id": key, "scenario": "join vintages", "llm_model": "m",
                "features": {"a": .2, "b": .1}, "prompt_hash": "h"}

    def run_pipeline(self, ledger, programme="credit", alpha=.05, suffix=""):
        """LLM proposal -> review -> measurement -> training -> freeze -> audit."""
        staged = stage_llm([self.proposal("P"+suffix)], self.schema, provenance=ledger)
        approved = review_proposals(staged, [{
            "proposal_id": "P"+suffix, "proposal_digest": staged[0]["proposal_digest"],
            "approved": True, "reviewer": "analyst", "reason": "declared frame"}],
            provenance=ledger)
        measured = join_evidence(approved, [{
            "proposal_id": "P"+suffix, "proposal_digest": staged[0]["proposal_digest"],
            "group": "dev"+suffix, "evidence_id": "measured"+suffix,
            "evidence_kind": "observed_development_attack",
            "successes": [0, 1], "trials": [20, 20]}], self.schema, provenance=ledger)
        train = [*development("train"+suffix, 3), *measured]
        valid = development("valid"+suffix, 2)
        bundle = train_model(train, valid, self.schema, width=4, depth=2, epochs=4,
                             provenance=ledger)
        fixed = shortlist(bundle, [{"candidate_id": "A", "features": {"a": 0., "b": 0.},
                                    "utility": 1.}], 1, alpha=alpha, trials=60,
                          owner="committee", expiry="2099-12-31", programme=programme,
                          provenance=ledger)
        outcome = [{"candidate_id": "A", "evidence_kind": "independent_final_audit",
                    "freeze_sha256": fixed["freeze_sha256"], "evidence_id": "final"+suffix,
                    "successes": [0, 0], "trials": [60, 60]}]
        return staged, approved, measured, bundle, fixed, outcome

    def test_full_chain_is_recorded_in_order_and_verifies(self):
        self.ledger.set_budget("credit", .10)
        *_, fixed, outcome = self.run_pipeline(self.ledger)
        decision = final_audit(fixed, outcome, provenance=self.ledger)
        self.assertEqual(decision["selected"], "A")
        stages = [e["stage"] for e in self.ledger.entries]
        self.assertEqual(stages, ["budget", "proposal", "review", "measurement",
                                  "training", "freeze", "audit"])
        reopened = ProvenanceLedger(self.root/"chain.jsonl", KEY)
        self.assertTrue(reopened.verify())
        self.assertAlmostEqual(reopened.alpha_spent("credit"), .05)
        self.assertEqual(reopened.summary()["stages"]["audit"], 1)
        # No proposal text and no count is ever written to the chain.
        text = (self.root/"chain.jsonl").read_text()
        self.assertNotIn("join vintages", text)
        self.assertNotIn('"successes"', text)

    def test_stage_without_its_prerequisite_is_refused(self):
        staged = stage_llm([self.proposal()], self.schema)  # not on the chain
        approved = review_proposals(staged, [{
            "proposal_id": "P", "proposal_digest": staged[0]["proposal_digest"],
            "approved": True, "reviewer": "analyst", "reason": "ok"}])
        with self.assertRaisesRegex(LedgerError, "not on the chain"):
            self.ledger.record_review(approved)
        measured = join_evidence(approved, [{
            "proposal_id": "P", "proposal_digest": staged[0]["proposal_digest"],
            "group": "dev", "evidence_id": "measured",
            "evidence_kind": "observed_development_attack",
            "successes": [0, 1], "trials": [20, 20]}], self.schema)
        with self.assertRaisesRegex(LedgerError, "approving review"):
            self.ledger.record_measurements(measured)
        bundle = train_model([*development("t", 3), *measured], development("v", 2),
                             self.schema, width=4, depth=2, epochs=4)
        with self.assertRaisesRegex(LedgerError, "not on the chain"):
            self.ledger.record_training(bundle, [*development("t", 3), *measured],
                                        development("v", 2))
        fixed = shortlist(bundle, [{"candidate_id": "A", "features": {"a": 0., "b": 0.},
                                    "utility": 1.}], 1, alpha=.05, trials=60,
                          owner="committee", expiry="2099-12-31")
        with self.assertRaisesRegex(LedgerError, "training is not on the chain"):
            self.ledger.record_freeze(fixed, "credit")
        with self.assertRaisesRegex(ValueError, "programme name"):
            shortlist(bundle, [{"candidate_id": "A", "features": {"a": 0., "b": 0.},
                                "utility": 1.}], 1, alpha=.05, trials=60,
                      owner="committee", expiry="2099-12-31", provenance=self.ledger)
        with self.assertRaisesRegex(LedgerError, "not on the chain"):
            self.ledger.record_audit({"freeze_sha256": fixed["freeze_sha256"],
                                      "alpha_family": .05})

    def test_each_audit_spends_the_programme_budget(self):
        self.ledger.set_budget("credit", .08)
        with self.assertRaisesRegex(LedgerError, "already on the chain"):
            self.ledger.set_budget("credit", .5)
        *_, fixed, outcome = self.run_pipeline(self.ledger, suffix="1")
        final_audit(fixed, outcome, provenance=self.ledger)
        with self.assertRaisesRegex(LedgerError, "already been audited"):
            final_audit(fixed, outcome, provenance=self.ledger)
        # A second shortlist after the first audit must fit in what is left: 0.03.
        *_, fixed2, outcome2 = self.run_pipeline(self.ledger, suffix="2")
        with self.assertRaisesRegex(LedgerError, "error budget"):
            final_audit(fixed2, outcome2, provenance=self.ledger)
        *_, fixed3, outcome3 = self.run_pipeline(self.ledger, alpha=.03, suffix="3")
        final_audit(fixed3, outcome3, provenance=self.ledger)
        self.assertAlmostEqual(self.ledger.alpha_spent("credit"), .08)
        with self.assertRaisesRegex(LedgerError, "no error budget"):
            self.run_pipeline(self.ledger, programme="unknown", suffix="4")

    def test_audit_outcome_cannot_feed_the_next_model_on_the_same_evidence(self):
        self.ledger.set_budget("credit", 1.)
        staged, approved, measured, bundle, fixed, outcome = self.run_pipeline(self.ledger)
        final_audit(fixed, outcome, provenance=self.ledger)
        # New evidence measured after the audit.
        staged2 = stage_llm([self.proposal("Q")], self.schema, provenance=self.ledger)
        approved2 = review_proposals(staged2, [{
            "proposal_id": "Q", "proposal_digest": staged2[0]["proposal_digest"],
            "approved": True, "reviewer": "analyst", "reason": "ok"}], provenance=self.ledger)
        measured2 = join_evidence(approved2, [{
            "proposal_id": "Q", "proposal_digest": staged2[0]["proposal_digest"],
            "group": "dev2", "evidence_id": "measured2",
            "evidence_kind": "observed_development_attack",
            "successes": [1, 1], "trials": [20, 20]}], self.schema, provenance=self.ledger)
        old_train = [*development("train", 3), *measured]
        with self.assertRaisesRegex(LedgerError, "measured after that audit"):
            train_model([*old_train, *measured2], development("valid", 2), self.schema,
                        width=4, depth=2, epochs=4, provenance=self.ledger)
        # A fresh development set is allowed.
        train_model([*development("fresh", 3), *measured2], development("fresh-v", 2),
                    self.schema, width=4, depth=2, epochs=4, provenance=self.ledger)

    def test_tampering_and_wrong_key_are_detected(self):
        self.ledger.set_budget("credit", .1)
        self.run_pipeline(self.ledger)
        path = self.root/"chain.jsonl"
        lines = path.read_text().splitlines()
        entry = json.loads(lines[3])
        entry["payload"]["evidence"][0][1] = "forged"
        path.write_text("\n".join([*lines[:3], json.dumps(entry), *lines[4:]]) + "\n")
        with self.assertRaisesRegex(LedgerError, "altered"):
            ProvenanceLedger(path, KEY)
        path.write_text("\n".join(lines[:3] + lines[4:]) + "\n")
        with self.assertRaisesRegex(LedgerError, "chain broken"):
            ProvenanceLedger(path, KEY)
        path.write_text("\n".join(lines) + "\n")
        with self.assertRaisesRegex(LedgerError, "signature"):
            ProvenanceLedger(path, b"another-key-of-sixteen-bytes")
        # An unkeyed reader still checks the chain but cannot check who wrote it.
        self.assertTrue(ProvenanceLedger(path, None).verify())
        with self.assertRaisesRegex(ValueError, "16 bytes"):
            load_key(None)
        key_file = self.root/"key"
        key_file.write_bytes(KEY)
        self.assertEqual(load_key(key_file), KEY)
        with patch.dict(os.environ, {KEY_ENV: KEY.decode()}):
            self.assertEqual(load_key(), KEY)

    def test_injector_is_bounded(self):
        many = [{"proposal_id": f"P{i}", "scenario": "x", "features": {"a": 0, "b": 0}}
                for i in range(MAX_PROPOSALS+1)]
        with self.assertRaisesRegex(ValueError, "more than"):
            propose_from_llm(lambda _: json.dumps(many), self.schema, "q", model_name="m")
        with self.assertRaisesRegex(ValueError, "exceeds"):
            propose_from_llm(lambda _: "[" + " "*300_000 + "]", self.schema, "q",
                             model_name="m")
        with self.assertRaisesRegex(ValueError, "response text"):
            propose_from_llm(lambda _: ["not", "text"], self.schema, "q", model_name="m")
        with self.assertRaisesRegex(ValueError, "no proposals"):
            stage_llm([], self.schema)
        # A proposal already on the chain cannot be staged twice.
        stage_llm([self.proposal()], self.schema, provenance=self.ledger)
        with self.assertRaisesRegex(LedgerError, "already on the chain"):
            stage_llm([self.proposal()], self.schema, provenance=self.ledger)

    def test_cli_runs_every_stage_against_one_keyed_ledger(self):
        root = self.root
        key_file = root/"key"
        key_file.write_bytes(KEY)
        chain = root/"cli-chain.jsonl"

        def invoke(*args):
            with patch("sys.argv", ["sentry-r", "--provenance", str(chain),
                                    "--key-file", str(key_file), *map(str, args)]):
                cli()
        with self.assertRaises(SystemExit):
            with patch("sys.argv", ["sentry-r", "budget", "--programme", "credit",
                                    "--alpha-budget", "0.1"]):
                cli()
        invoke("budget", "--programme", "credit", "--alpha-budget", "0.1")
        write_jsonl(root/"llm.jsonl", [self.proposal()])
        invoke("stage-llm", "--schema", root/"schema.json", "--input", root/"llm.jsonl",
               "--output", root/"staged.jsonl")
        staged = json.loads((root/"staged.jsonl").read_text().splitlines()[0])
        write_jsonl(root/"decisions.jsonl", [{"proposal_id": "P",
                    "proposal_digest": staged["proposal_digest"], "approved": True,
                    "reviewer": "analyst", "reason": "declared frame"}])
        invoke("review", "--staged", root/"staged.jsonl", "--decisions", root/"decisions.jsonl",
               "--output", root/"approved.jsonl")
        write_jsonl(root/"measures.jsonl", [{"proposal_id": "P",
                    "proposal_digest": staged["proposal_digest"], "group": "dev",
                    "evidence_id": "measured", "evidence_kind": "observed_development_attack",
                    "successes": [0, 1], "trials": [20, 20]}])
        invoke("join-evidence", "--schema", root/"schema.json", "--staged", root/"approved.jsonl",
               "--outcomes", root/"measures.jsonl", "--output", root/"joined.jsonl")
        write_jsonl(root/"base.jsonl", development("base", 3))
        write_jsonl(root/"valid.jsonl", development("valid", 2))
        invoke("augment", "--schema", root/"schema.json", "--base", root/"base.jsonl",
               "--measured", root/"joined.jsonl", "--output", root/"v2.jsonl")
        invoke("train", "--schema", root/"schema.json", "--train", root/"v2.jsonl",
               "--validation", root/"valid.jsonl", "--depth", "2", "--width", "4",
               "--epochs", "4", "--output", root/"model.json")
        write_jsonl(root/"candidates.jsonl", [{"candidate_id": "A",
                    "features": {"a": 0, "b": 0}, "utility": 2}])
        invoke("shortlist", "--model", root/"model.json", "--candidates", root/"candidates.jsonl",
               "--size", "1", "--trials", "60", "--owner", "analyst", "--expiry", "2099-12-31",
               "--programme", "credit", "--output", root/"fixed.json")
        fixed = json.loads((root/"fixed.json").read_text())
        write_jsonl(root/"audits.jsonl", [{"candidate_id": "A",
                    "evidence_kind": "independent_final_audit",
                    "freeze_sha256": fixed["freeze_sha256"], "evidence_id": "fresh",
                    "successes": [0, 0], "trials": [60, 60]}])
        invoke("audit", "--shortlist", root/"fixed.json", "--outcomes", root/"audits.jsonl",
               "--output", root/"decision.json", "--no-ledger")
        invoke("verify-ledger")
        chain_entries = ProvenanceLedger(chain, KEY).entries
        self.assertEqual([e["stage"] for e in chain_entries],
                         ["budget", "proposal", "review", "measurement", "training",
                          "freeze", "audit"])
        self.assertTrue(all("mac" in e for e in chain_entries))


if __name__ == "__main__":
    unittest.main()
