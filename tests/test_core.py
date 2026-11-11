import copy
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from harness_ocr.core import (AuthorizationError, DocumentStore, EvidenceError,
                              RevisionConflict, StateError, TraceRecorder, replay)

SHA = "a" * 64


def observation(oid="o1", text="Сумма 80800", **extra):
    return {"id": oid, "source_sha256": SHA, "text": text,
            "evidence_kind": "image_region", "page": 1, "page_id": "page1",
            "bbox": [0, 0, 100, 20], "source_size": [200, 100], "completed": True, **extra}


def claim(cid="c1", field="payment.amount", value="80800", evidence="o1", depends=(), role="payment"):
    return {"id": cid, "field_key": field, "raw_value": value, "evidence_ids": [evidence],
            "depends_on": list(depends), "party_role": role, "origin": "model"}


class DocumentStateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = DocumentStore(Path(self.temp.name) / "documents.sqlite")
        self.store.create_document("doc1", SHA, "receipt_v1")

    def tearDown(self):
        self.temp.cleanup()

    def test_conflict_withdraws_dependent_not_unrelated_field(self):
        self.store.revise("doc1", observations=[observation(text="Сумма 80800 Итог 90000"), observation("date", "01.11.2026")],
                          claims=[claim(), claim("derived", "derived.total", "90000", depends=["c1"]),
                                  claim("datec", "payment.date", "01.11.2026", evidence="date")])
        previous = self.store.get("doc1")
        now = self.store.revise("doc1", observations=[observation("o2", "Сумма 808.00")],
                                claims=[claim("c2", value="808.00", evidence="o2")], expected_revision=previous["revision"])
        self.assertEqual(now["claims"]["c1"]["status"], "conflict")
        self.assertEqual(now["claims"]["derived"]["status"], "retracted")
        self.assertIsNone(now["fields"]["payment.amount"]["value"])
        self.assertEqual(now["fields"]["payment.date"]["value"], "01.11.2026")
        self.assertEqual(previous["claims"]["c1"]["status"], "supported")
        self.assertEqual(self.store.get("doc1", revision=previous["revision"]), previous)

    def test_authorized_human_value_survives_model_disagreement(self):
        state = self.store.revise("doc1", observations=[observation()], claims=[claim()])
        state = self.store.human_correct("doc1", "payment.amount", "808.00", ["o1"],
                                         actor="reviewer7", expected_revision=state["revision"], authorized=True, party_role="payment")
        human_id = state["fields"]["payment.amount"]["claim_ids"][0]
        new = self.store.revise("doc1", observations=[observation("wrong", "Сумма 999.00")],
                               claims=[claim("challenger", value="999.00", evidence="wrong")])
        self.assertEqual(new["fields"]["payment.amount"]["value"], "808.00")
        self.assertEqual(new["claims"][human_id]["status"], "human_supported")
        self.assertTrue(new["fields"]["payment.amount"]["needs_review"])
        resolved = self.store.human_correct("doc1", "payment.amount", "808.00", ["o1"], actor="reviewer7",
                                            expected_revision=new["revision"], authorized=True, party_role="payment")
        self.assertEqual(resolved["conflicts"], [])
        self.assertFalse(resolved["fields"]["payment.amount"]["needs_review"])

    def test_unauthorized_human_and_stale_update_do_not_partially_commit(self):
        initial = self.store.get("doc1")
        with self.assertRaises(AuthorizationError):
            self.store.revise("doc1", observations=[observation()], claims=[{**claim(), "origin": "human"}], expected_revision=0)
        self.assertEqual(self.store.get("doc1"), initial)
        self.store.add_observation("doc1", observation(), expected_revision=0)
        with self.assertRaises(RevisionConflict):
            self.store.revise("doc1", observations=[observation("o2")], expected_revision=0)
        self.assertNotIn("o2", self.store.get("doc1")["observations"])

    def test_observations_immutable_and_partial_evidence_ineligible(self):
        self.store.add_observation("doc1", observation())
        initial = self.store.get("doc1")
        with self.assertRaises(EvidenceError):
            self.store.add_observation("doc1", observation(text="different"))
        self.assertEqual(self.store.get("doc1"), initial)
        self.store.add_observation("doc1", observation("partial", completed=False))
        with self.assertRaises(EvidenceError):
            self.store.add_claim("doc1", claim("partialclaim", evidence="partial"))
        self.assertFalse(self.store.get("doc1")["observations"]["partial"]["completed"])

    def test_foreign_source_coordinates_and_generated_values_rejected(self):
        for bad in (observation(source_sha256="b" * 64), observation(bbox=[0, 0, 300, 20]),
                    observation(bbox=None), observation(oracle=True)):
            with self.assertRaises(EvidenceError):
                self.store.add_observation("doc1", bad)
        self.store.add_observation("doc1", observation())
        with self.assertRaises(EvidenceError):
            self.store.add_claim("doc1", claim(value="808.00"))
        with self.assertRaises(EvidenceError):
            self.store.add_claim("doc1", {**claim(), "can_auto_accept": True})

    def test_invalidating_source_retracts_transitively(self):
        self.store.revise("doc1", observations=[observation(text="80800 90000")],
                          claims=[claim(), claim("child", "derived", "90000", depends=["c1"])])
        result = self.store.revise("doc1", invalidate=["o1"])
        self.assertFalse(result["observations"]["o1"]["active"])
        self.assertEqual(result["claims"]["c1"]["status"], "retracted")
        self.assertEqual(result["claims"]["child"]["status"], "retracted")
        with self.assertRaises(EvidenceError):
            self.store.add_claim("doc1", claim("third"))

    def test_native_xml_has_locator_instead_of_fake_bbox(self):
        native = {"id": "xml1", "source_sha256": SHA, "text": "Иванов Иван", "completed": True,
                  "evidence_kind": "structured_xml", "xpath": "/doc/name", "namespace_map": {},
                  "schema_version": "1", "structural_validation_status": "pass"}
        result = self.store.revise("doc1", observations=[native], claims=[claim("namec", "holder.name", "Иванов Иван", "xml1")])
        self.assertEqual(result["fields"]["holder.name"]["value"], "Иванов Иван")
        with self.assertRaises(EvidenceError):
            self.store.add_observation("doc1", {**native, "id": "xml2", "bbox": [0, 0, 1, 1]})

    def test_multispan_requires_exact_declared_source_fragments(self):
        self.store.revise("doc1", observations=[observation("n1", "ООО Ромашка"), observation("n2", "Сервис")])
        spans = [{"observation_id": "n1", "start": 0, "end": 11, "separator_after": " "},
                 {"observation_id": "n2", "start": 0, "end": 6}]
        good = {**claim("name", "seller.name", "ООО Ромашка Сервис", "n1"), "evidence_ids": ["n1", "n2"], "spans": spans}
        result = self.store.add_claim("doc1", good)
        self.assertEqual(result["fields"]["seller.name"]["value"], good["raw_value"])
        with self.assertRaises(EvidenceError):
            self.store.add_claim("doc1", {**good, "id": "invented", "raw_value": "ООО Ромашка-Сервис"})

    def test_concurrent_compare_and_swap_one_winner(self):
        barrier = threading.Barrier(2)
        def run(n):
            barrier.wait()
            try:
                self.store.add_observation("doc1", observation("parallel" + str(n)), expected_revision=0)
                return "ok"
            except RevisionConflict:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, [1, 2]))
        self.assertCountEqual(results, ["ok", "conflict"])
        self.assertEqual(len(self.store.get("doc1")["observations"]), 1)

    def test_role_evidence_references_and_retraction(self):
        self.store.revise("doc1", observations=[observation(), observation("label", "Сумма")])
        value = {**claim(), "role_evidence_ids": ["missing"]}
        with self.assertRaises(EvidenceError):
            self.store.revise("doc1", claims=[value])
        value["role_evidence_ids"] = ["label"]
        value["role_spans"] = [{"observation_id": "label", "start": 0, "end": 50}]
        with self.assertRaises(EvidenceError):
            self.store.revise("doc1", claims=[value])
        value["role_spans"][0]["end"] = 5
        self.store.revise("doc1", claims=[value])
        result = self.store.revise("doc1", invalidate=["label"])
        self.assertEqual(result["claims"]["c1"]["status"], "retracted")
        self.assertTrue(result["observations"]["o1"]["active"])


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trace.sqlite"
        self.trace = TraceRecorder(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_pure_replay_preserves_source_and_branches(self):
        record = self.trace.record("resolve", {"value": "001"}, {"result": "001"}, {"policy": "v1"},
                                   stage_kind="pure", deterministic=True, document_id="d", trial_id="t0")
        result = self.trace.replay(record["id"], lambda inputs, config: {"result": inputs["value"]}, trial_id="t1")
        self.assertTrue(result["matches_recorded_output"])
        self.assertEqual(self.trace.get(record["id"])["outputs"], {"result": "001"})
        with self.assertRaises(StateError):
            self.trace.replay(record, lambda i, c: {}, config={"policy": "v2"})
        changed = self.trace.replay(record, lambda i, c: {"result": "different"}, config={"policy": "v2"}, allow_changed_config=True)
        self.assertTrue(changed["config_changed"])
        self.assertFalse(changed["matches_recorded_output"])
        self.assertEqual(self.trace.audit()["records"], 3)

    def test_models_incomplete_and_hidden_reasoning_cannot_replay(self):
        record = self.trace.record("qwen", {}, {"text": "answer"}, {}, stage_kind="model")
        with self.assertRaises(StateError):
            replay(record, lambda i, c: {})
        with self.assertRaises(StateError):
            self.trace.record("qwen", {}, {}, {}, stage_kind="model", deterministic=True)
        with self.assertRaises(StateError):
            self.trace.record("qwen", {}, {"reasoning_content": "not retained"}, {})

    def test_hash_chain_detects_tampering(self):
        self.trace.record("one", {}, {"value": "1"}, {})
        self.trace.record("two", {}, {"value": "2"}, {})
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT payload FROM stage_trace WHERE seq=1").fetchone()
            payload = json.loads(row[0]);payload["outputs"]["value"] = "999"
            db.execute("UPDATE stage_trace SET payload=? WHERE seq=1", (json.dumps(payload),))
        db.close()
        with self.assertRaises(StateError):
            self.trace.audit()


if __name__ == "__main__":
    unittest.main()
