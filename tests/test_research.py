"""Behavioral research-tool tests using controlled data, no neural inference."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from harness_ocr.research import minimize_graph, run_blind_exam, run_oracle_study


def case(identity, source, value, evidence, *, correct, text=True, vision=True, origin="injected"):
    return {"id": identity, "original_id": source,
            "input": {"candidate": {"value": value}, "evidence": {"text": evidence}},
            "gold": {"candidate_correct": correct, "observable": {"text": text, "vision": vision}},
            "origin": origin}


class BlindVerifierTests(unittest.TestCase):
    def test_gold_and_case_identifiers_do_not_reach_verifier(self):
        cases = [case("PRIVATE-good", "source-a", "123", "123", correct=True),
                 case("PRIVATE-bad", "source-a", "456", "123", correct=False),
                 case("PRIVATE-unobservable", "source-b", "777", "777", correct=False,
                      text=False, vision=True, origin="natural")]
        seen = []
        def verifier(payload):
            self.assertEqual(set(payload), {"trial_id", "modality", "input"})
            self.assertNotIn("PRIVATE", json.dumps(payload))
            self.assertNotIn("gold", json.dumps(payload))
            seen.append(payload)
            material = payload["input"]
            return {"verdict": "correct" if material["candidate"]["value"] == material["evidence"]["text"]
                    else "incorrect"}
        result = run_blind_exam(cases, verifier, seed=9)
        self.assertEqual(len(seen), 3)
        self.assertEqual(result["metrics"]["observable_trials"], 2)
        self.assertEqual(result["metrics"]["unobservable_trials"], 1)
        self.assertEqual(result["metrics"]["error_detection_rate"], 1)
        self.assertEqual(result["metrics"]["correct_retention_rate"], 1)
        self.assertEqual(result["metrics"]["false_alarm_rate"], 0)
        self.assertEqual(result["original_groups"], 2)
        self.assertIsNone(result["per_original"]["source-b"]["error_detection_rate"])
        self.assertEqual(result["per_origin"]["natural"]["unobservable_trials"], 1)

    def test_always_abstaining_is_not_success(self):
        data = [case("a", "one", "a", "a", correct=True),
                case("b", "one", "b", "a", correct=False)]
        result = run_blind_exam(data, lambda _: "abstain")
        self.assertEqual(result["metrics"]["error_detection_rate"], 0)
        self.assertEqual(result["metrics"]["correct_retention_rate"], 0)
        self.assertEqual(result["metrics"]["observable_abstention_rate"], 1)
        self.assertEqual(result["metrics"]["wrong_accept_rate"], 0)

    def test_group_macro_does_not_treat_variants_as_new_sources(self):
        data = [case("a1", "A", "1", "0", correct=False),
                case("a2", "A", "1", "0", correct=False),
                case("b", "B", "2", "0", correct=False)]
        def verifier(payload):
            return "incorrect" if payload["input"]["candidate"]["value"] == "1" else "correct"
        result = run_blind_exam(data, verifier)
        self.assertAlmostEqual(result["metrics"]["error_detection_rate"], 2 / 3)
        self.assertAlmostEqual(result["macro_by_original"]["error_detection_rate"], .5)

    def test_bad_verdict_and_exception_are_explicit_errors(self):
        data = [case("a", "one", "a", "a", correct=True),
                case("b", "two", "b", "a", correct=False)]
        def verifier(payload):
            if payload["input"]["candidate"]["value"] == "a":
                return {"verdict": True}
            raise RuntimeError("model timed out")
        result = run_blind_exam(data, verifier)
        self.assertEqual(result["metrics"]["verifier_errors"], 2)
        self.assertEqual(result["metrics"]["error_detection_rate"], 0)
        self.assertTrue(all(r["verdict"] is None for r in result["trials"]))

    def test_annotation_leak_and_wrong_bool_are_rejected(self):
        data = [case("a", "one", "a", "a", correct=True)]
        data[0]["input"]["evidence"]["gold"] = "secret"
        with self.assertRaisesRegex(ValueError, "Private annotation"):
            run_blind_exam(data, lambda _: "correct")
        data = [case("a", "one", "a", "a", correct=True)]
        data[0]["gold"]["candidate_correct"] = 1
        with self.assertRaises(ValueError):
            run_blind_exam(data, lambda _: "correct")

    def test_seed_payload_copy_and_runtime_manifest(self):
        data = [case(str(i), "group", str(i), str(i), correct=True) for i in range(8)]
        before = copy.deepcopy(data)
        def mutating(payload):
            payload["input"]["candidate"]["value"] = "mutated"
            return "correct"
        first = run_blind_exam(data, mutating, seed=22, verifier_metadata={"model": "frozen-model"})
        second = run_blind_exam(data, mutating, seed=22, verifier_metadata={"model": "frozen-model"})
        self.assertEqual(data, before)
        self.assertEqual(first["trials"], second["trials"])
        changed = run_blind_exam(data, mutating, seed=22, verifier_metadata={"model": "another"})
        self.assertNotEqual(first["study_sha256"], changed["study_sha256"])

    def test_create_only_journal_refuses_silent_rerun(self):
        data = [case("a", "one", "a", "a", correct=True)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trials.jsonl"
            run_blind_exam(data, lambda _: "correct", trial_path=output)
            content = output.read_bytes()
            calls = []
            with self.assertRaises(FileExistsError):
                run_blind_exam(data, lambda payload: calls.append(payload) or "correct", trial_path=output)
            self.assertEqual(calls, [])
            self.assertEqual(content, output.read_bytes())
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["record_type"] for r in rows], ["manifest", "trial"])

    def test_non_object_runtime_manifest_is_rejected_before_callback(self):
        for metadata in ([], False, ""):
            with self.assertRaises(ValueError):
                run_blind_exam([case("a", "one", "a", "a", correct=True)],
                               lambda _: self.fail("callback must not run"), verifier_metadata=metadata)


def graph_fixture():
    return {"task_contract": {"field": "payer.inn", "role": "payer", "source_sha": "fixed"},
            "nodes": [{"id": "source", "kind": "provenance", "sha256": "actual-image"},
                      {"id": "line", "kind": "observation", "text": "ИНН 123", "source_ids": ["source"]},
                      {"id": "target", "kind": "claim", "evidence_ids": ["line"]},
                      {"id": "confuser", "kind": "candidate", "text": "other-party"},
                      {"id": "noise1", "kind": "candidate"}, {"id": "noise2", "kind": "candidate"}],
            "edges": [{"source": "line", "target": "target", "type": "supports"},
                      {"source": "noise1", "target": "noise2", "type": "layout"}]}


class MinimizerTests(unittest.TestCase):
    def test_shrinks_same_failure_and_preserves_evidence_and_contract(self):
        data = graph_fixture()
        before = copy.deepcopy(data)
        def predicate(g):
            ids = {n["id"] for n in g["nodes"]}
            return {"failed": {"target", "confuser"} <= ids, "failure_id": "payer-payee-swap"}
        result = minimize_graph(data, predicate, required_ids=["target"])
        self.assertEqual(data, before)
        self.assertTrue(result["changed"])
        self.assertEqual({n["id"] for n in result["graph"]["nodes"]},
                         {"source", "line", "target", "confuser"})
        self.assertEqual(result["graph"]["task_contract"], before["task_contract"])
        self.assertEqual(result["graph"]["edges"], [before["edges"][0]])
        self.assertEqual(result["target_failure_id"], "payer-payee-swap")
        self.assertEqual(result["stop_reason"], "locally_minimal")

    def test_other_error_does_not_count_as_same_failure(self):
        def predicate(g):
            ids = {n["id"] for n in g["nodes"]}
            return {"failed": True, "failure_id": "original" if "confuser" in ids else "different"}
        result = minimize_graph(graph_fixture(), predicate, required_ids=["target"])
        self.assertIn("confuser", {n["id"] for n in result["graph"]["nodes"]})
        self.assertEqual(result["target_failure_id"], "original")

    def test_edge_reduction_keeps_provenance_edges(self):
        data = graph_fixture()
        result = minimize_graph(data, lambda _: True, required_ids=[n["id"] for n in data["nodes"]])
        self.assertEqual(result["graph"]["nodes"], data["nodes"])
        self.assertEqual(result["graph"]["edges"], [data["edges"][0]])

    def test_call_budget_and_between_call_time_guard(self):
        data = graph_fixture()
        result = minimize_graph(data, lambda _: True, required_ids=["target"], max_calls=1)
        self.assertEqual(result["calls"], 1)
        self.assertEqual(result["graph"], data)
        self.assertEqual(result["stop_reason"], "call_budget_exhausted")
        def slow(_):
            time.sleep(.01)
            return True
        result = minimize_graph(data, slow, required_ids=["target"], time_budget_seconds=.001)
        self.assertEqual(result["calls"], 1)
        self.assertEqual(result["stop_reason"], "time_budget_exhausted")

    def test_no_failure_and_baseline_error_are_not_shrinking_success(self):
        data = graph_fixture()
        result = minimize_graph(data, lambda _: False)
        self.assertFalse(result["changed"])
        self.assertEqual(result["stop_reason"], "baseline_does_not_fail")
        def broken(_):
            raise RuntimeError("not this defect")
        result = minimize_graph(data, broken)
        self.assertFalse(result["changed"])
        self.assertEqual(result["stop_reason"], "baseline_predicate_error")

    def test_candidate_exception_is_not_a_reproduced_failure(self):
        data = graph_fixture()
        def predicate(g):
            if len(g["nodes"]) < len(data["nodes"]):
                raise RuntimeError("fixture destroyed")
            return True
        result = minimize_graph(data, predicate, required_ids=["target"])
        self.assertEqual(len(result["graph"]["nodes"]), len(data["nodes"]))
        self.assertGreater(result["predicate_errors"], 0)
        self.assertEqual(result["stop_reason"], "search_complete_with_predicate_errors")

    def test_invalid_graph_or_unknown_required_is_rejected(self):
        data = graph_fixture()
        with self.assertRaises(ValueError):
            minimize_graph(data, lambda _: True, required_ids=["nonexistent"])
        data["edges"].append({"source": "missing", "target": "target"})
        with self.assertRaises(ValueError):
            minimize_graph(data, lambda _: True)

    def test_absent_edge_key_is_preserved(self):
        data = {"nodes": [{"id": "keep"}, {"id": "noise"}], "task_contract": {"role": "payer"}}
        result = minimize_graph(data, lambda g: any(n["id"] == "keep" for n in g["nodes"]))
        self.assertEqual(result["graph"], {"nodes": [{"id": "keep"}], "task_contract": data["task_contract"]})


def oracle_document(identity="doc", original="source"):
    return {"id": identity, "original_id": original, "split": "dev",
            "input": {"text": "OCR", "roles": {"value": "wrong"}, "geometry": {"cell": 1},
                      "source_sha256": "fixed-original"},
            "corrections": {"text": "human", "roles": {"value": "correct"}, "geometry": {"cell": 2}}}


class OracleTests(unittest.TestCase):
    def test_factorial_effects_and_marked_inputs(self):
        documents = [oracle_document()]
        before = copy.deepcopy(documents)
        observed = []
        values = {(0, 0): .1, (1, 0): .5, (0, 1): .4, (1, 1): .9}
        def evaluator(payload):
            self.assertNotIn("corrections", payload)
            self.assertNotIn("gold", payload)
            self.assertEqual(payload["research_mode"], "dev")
            material, marker = payload["input"], payload["oracle"]
            text = int(marker["text_factor"] == "corrected")
            roles = int(marker["roles_factor"] == "corrected")
            self.assertEqual(material["text"], "human" if text else "OCR")
            self.assertEqual(material["geometry"]["cell"], 2 if roles else 1)
            self.assertEqual(marker["not_model_capture"], bool(text or roles))
            observed.append((text, roles))
            material["source_sha256"] = "callback-mutation"
            return {"score": values[(text, roles)]}
        result = run_oracle_study(documents, evaluator, seed=13)
        self.assertEqual(documents, before)
        self.assertEqual(set(observed), set(values))
        self.assertEqual(result["complete_documents"], 1)
        self.assertAlmostEqual(result["macro_by_original"]["text_effect_at_original_roles"], .4)
        self.assertAlmostEqual(result["macro_by_original"]["roles_effect_at_original_text"], .3)
        self.assertAlmostEqual(result["macro_by_original"]["joint_effect"], .8)
        self.assertAlmostEqual(result["macro_by_original"]["interaction"], .1)

    def test_production_and_non_dev_sources_refused_before_callback(self):
        calls = []
        with self.assertRaises(ValueError):
            run_oracle_study([oracle_document()], lambda x: calls.append(x), mode="production")
        bad = oracle_document()
        bad["split"] = "blind"
        with self.assertRaises(ValueError):
            run_oracle_study([bad], lambda x: calls.append(x))
        del bad["split"]
        with self.assertRaises(ValueError):
            run_oracle_study([bad], lambda x: calls.append(x))
        self.assertEqual(calls, [])

    def test_missing_condition_has_no_imputed_zero_effect(self):
        def evaluator(payload):
            marker = payload["oracle"]
            if marker["text_factor"] == "corrected" and marker["roles_factor"] == "original":
                raise RuntimeError("model budget exhausted")
            return .5
        result = run_oracle_study([oracle_document()], evaluator)
        self.assertEqual(result["trial_errors"], 1)
        self.assertEqual(result["complete_documents"], 0)
        self.assertEqual(result["effects"], [])
        self.assertIsNone(result["macro_by_original"]["joint_effect"])
        self.assertEqual(len(result["incomplete_documents"]), 1)

    def test_group_average_not_variant_average(self):
        documents = [oracle_document("a1", "A"), oracle_document("a2", "A"), oracle_document("b", "B")]
        for d in documents:
            d["input"]["group_marker"] = d["original_id"]
        def evaluator(payload):
            return float(payload["input"]["group_marker"] == "A" and payload["oracle"]["enabled"])
        result = run_oracle_study(documents, evaluator)
        self.assertEqual(result["complete_original_groups"], 2)
        self.assertAlmostEqual(result["macro_by_original"]["joint_effect"], .5)

    def test_nonfinite_or_boolean_score_remains_error(self):
        for value in (float("nan"), True):
            result = run_oracle_study([oracle_document()], lambda _, v=value: {"score": v})
            self.assertEqual(result["trial_errors"], 4)
            self.assertEqual(result["complete_documents"], 0)

    def test_create_only_oracle_journal_and_reproducible_seed(self):
        documents = [oracle_document()]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oracle.jsonl"
            result = run_oracle_study(documents, lambda _: .5, seed=4, trial_path=output,
                                      evaluator_metadata={"model_sha": "fixed"})
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 5)
            with self.assertRaises(FileExistsError):
                run_oracle_study(documents, lambda _: .5, seed=4, trial_path=output)
            repeat = run_oracle_study(documents, lambda _: .5, seed=4,
                                     evaluator_metadata={"model_sha": "fixed"})
            self.assertEqual(result["trials"], repeat["trials"])
            changed = run_oracle_study(documents, lambda _: .5, seed=4,
                                      evaluator_metadata={"model_sha": "new"})
            self.assertNotEqual(result["study_sha256"], changed["study_sha256"])

    def test_non_object_evaluator_manifest_is_rejected_before_callback(self):
        for metadata in ([], False, ""):
            with self.assertRaises(ValueError):
                run_oracle_study([oracle_document()], lambda _: self.fail("callback must not run"),
                                 evaluator_metadata=metadata)


if __name__ == "__main__":
    unittest.main()
