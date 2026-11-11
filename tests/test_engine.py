"""Runtime grounding integration with deterministic fake backends only."""
from __future__ import annotations
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from harness_ocr.engine import Engine
from harness_ocr.grounding import build_candidates, resolve_response, validate_literal
from harness_ocr.inference.supervisor import ModelCanceled, ModelDeadline
from harness_ocr.profiles import ProfileRegistry
from harness_ocr.settings import Settings
from harness_ocr.control import BindingPolicyGateway,DocumentRiskCalibrator
from harness_ocr.control.runtime import SCORE_CONTRACT_ID
from harness_ocr.util import digest,write_json


SHA = "a" * 64


def observation(text="Номер паспорта: 123456", ident="o1", completed=True):
    return {"id": ident, "source_sha256": SHA, "text": text, "evidence_kind": "image_region",
            "page": 1, "page_id": "p1", "source_size": [800, 1200], "bbox": [0, 0, 800, 1200],
            "completed": completed, "capture_id": ident, "model_id": "deterministic-test"}


def optics(text="Номер паспорта: 123456", regions=None):
    return {"source_sha256": SHA, "observations": [observation(text)],
            "regions": regions or [], "completed": True}


def reply(prompt, *, reread=False):
    request = json.loads(prompt)
    choices = [c for c in request["candidates"] if c["text"] == "123456" and c["role_anchor_observed"]]
    proposals = [{"field_key": c["field_key"], "candidate_ids": [c["id"]], "party_role": c["party_role"]}
                 for c in choices[:1]]
    requests = [{"field_key": "passport.number", "region_ids": ["r1"], "reason_code": "critical_recheck"}] if reread else []
    return json.dumps({"proposals": proposals, "reread_requests": requests})


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.contract = ProfileRegistry().compile("D01", subset=["passport.number"])

    def verified(self, text):
        return [c for c in build_candidates([observation(text)], self.contract).values() if c["role_verified"]]

    def test_digits_before_label_and_after_other_field_do_not_borrow_role(self):
        candidates = self.verified("Телефон: 654321 Номер паспорта: 123456; другой номер: 999999")
        self.assertTrue(candidates)
        self.assertEqual({c["raw_value"] for c in candidates}, {"123456"})

    def test_two_values_in_one_labelled_area_remain_ambiguous(self):
        self.assertEqual(self.verified("Номер паспорта: 123456 или 654321"), [])

    def test_subset_does_not_hide_longer_opposite_role_label(self):
        contract = {"field_definitions": {"seller.inn": {"label": "ИНН продавца", "role": "seller", "value_type": "inn"}},
                    "requested_fields": ["seller.inn"]}
        values = build_candidates([observation("ИНН плательщика: 7707083893")], contract)
        self.assertTrue(values)
        self.assertFalse(any(c["role_verified"] for c in values.values()))

    def test_adjacent_known_labels_define_value_boundaries(self):
        contract = ProfileRegistry().compile("D01", subset=["holder.birth_date", "passport.issue_date"])
        values = build_candidates([observation("Дата рождения: 12.03.2000 Дата выдачи: 14.05.2020")], contract)
        verified = {(c["field_key"], c["raw_value"]) for c in values.values() if c["role_verified"]}
        self.assertEqual(verified, {("holder.birth_date", "12.03.2000"), ("passport.issue_date", "14.05.2020")})

    def test_incomplete_or_inactive_observations_never_supply_candidates(self):
        self.assertEqual(build_candidates([observation(completed=False)], self.contract), {})
        inactive = observation(); inactive["active"] = False
        self.assertEqual(build_candidates([inactive], self.contract), {})

    def test_cannot_forge_candidate_role_field_value_or_json_keys(self):
        values = build_candidates([observation()], self.contract)
        valid = next(c for c in values.values() if c["raw_value"] == "123456")
        proposal = {"field_key": "passport.number", "candidate_ids": [valid["id"]], "party_role": "document"}
        variants = []
        p = copy.deepcopy(proposal); p["candidate_ids"] = ["made-up"]; variants.append(p)
        p = copy.deepcopy(proposal); p["party_role"] = "seller"; variants.append(p)
        p = copy.deepcopy(proposal); p["field_key"] = "seller.inn"; variants.append(p)
        p = copy.deepcopy(proposal); p["value"] = "654321"; variants.append(p)
        p = copy.deepcopy(proposal); p["candidate_ids"] = [{"id": valid["id"]}]; variants.append(p)
        for p in variants:
            with self.subTest(p=p), self.assertRaises(ValueError):
                resolve_response(json.dumps({"proposals": [p], "reread_requests": []}), values, [], self.contract)
        with self.assertRaises(ValueError):
            resolve_response('{"proposals":[],"proposals":[],"reread_requests":[]}', values, [], self.contract)

    def test_unsupported_role_is_rejected_even_when_value_exists(self):
        values = build_candidates([observation("123456")], self.contract)
        c = next(c for c in values.values() if c["raw_value"] == "123456")
        resolved = resolve_response(json.dumps({"proposals": [{"field_key": "passport.number",
            "candidate_ids": [c["id"]], "party_role": "document"}], "reread_requests": []}), values, [], self.contract)
        self.assertFalse(resolved["claims"])
        self.assertEqual(resolved["rejected"][0]["reason"], "role_anchor_not_observed")

    def test_role_anchor_spans_survive_in_claim_provenance(self):
        values = build_candidates([observation()], self.contract)
        response = reply(json.dumps({"candidates": [{"id": c["id"], "field_key": c["field_key"],
            "text": c["raw_value"], "role_anchor_observed": c["role_verified"], "party_role": c["party_role"]}
            for c in values.values()]}))
        resolved = resolve_response(response, values, [], self.contract)
        claim = resolved["claims"][0]
        self.assertEqual(claim["raw_value"], "123456")
        span = claim["role_spans"][0]
        self.assertEqual(observation()["text"][span["start"]:span["end"]], "Номер паспорта")

    def test_numeric_and_calendar_validation_preserve_raw_without_repair(self):
        numeric = {"value_type": "digits_6", "normalization_transform_ids": ["numeric-separators-v1"]}
        self.assertEqual(validate_literal("12 3456", numeric)["canonical_value"], "123456")
        self.assertEqual(validate_literal("12 3456", numeric)["transform_ids"], ["numeric-separators-v1"])
        self.assertEqual(validate_literal("12 3456", numeric)["validation_status"], "pass")
        self.assertEqual(validate_literal("١٢٣٤٥٦", numeric)["validation_status"], "fail")
        self.assertEqual(validate_literal("31.02.2020", {"value_type": "date"})["validation_status"], "fail")
        self.assertEqual(validate_literal("1.2.2020", {"value_type": "date"})["validation_status"], "fail")

    def test_undeclared_numeric_normalization_cannot_change_canonical_value(self):
        for spec in [{'value_type': 'digits_6'}, {'value_type': 'digits_6', 'normalization_transform_ids': []},
                     {'value_type': 'digits_6', 'normalization_transform_ids': 'numeric-separators-v1'}]:
            value = validate_literal('12 3456', spec)
            self.assertEqual(value['canonical_value'], '12 3456')
            self.assertEqual(value['validation_status'], 'fail')
            self.assertEqual(value['transform_ids'], [])
        self.assertEqual(validate_literal('123456', {'value_type': 'digits_6'})['validation_status'], 'pass')


class FakeBackend:
    def __init__(self, callback=None):
        self.callback, self.calls = callback, []

    def call(self, op, args, **kwargs):
        self.calls.append(op)
        if self.callback:
            value = self.callback(op, args)
            if value is not None:
                return value
        if op == "select":
            return {"text": reply(args["prompt"]), "completed": True}
        if op == "cheap":
            return {"observations": [observation()], "regions": [], "completed": True}
        if op == "ocr":
            return {"text": "Номер паспорта: 123456", "completed": True}
        raise AssertionError(op)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.settings = Settings(self.path / "data", self.path / "models", device="cpu", dtype="float32")

    def run_engine(self, backend=None, *, shared=None, **kwargs):
        engine = Engine(self.settings, backend or FakeBackend())
        output = engine.run("d1", self.path / "not-read-when-frozen.png", SHA, "D01",
                            subset=["passport.number"], shared_optics=shared or optics(), **kwargs)
        return engine, output

    def test_real_engine_contract_projection_matches_generated_json_schema(self):
        from jsonschema import Draft202012Validator
        engine, output = self.run_engine()
        contract = engine.profiles.compile('D01', subset=['passport.number'])
        Draft202012Validator(contract['json_schema']).validate(output['contract_result'])
        self.assertEqual(set(output['contract_result']), {'profile_id', 'profile_version', 'subset_contract_id', 'fields', 'needs_review'})
        self.assertEqual(output['contract_result']['fields']['passport.number']['raw_value'], '123456')
        self.assertNotIn('claim_ids', output['contract_result']['fields']['passport.number'])
        self.assertIn('observations', output)
        self.assertIn('audit', output)

    def configured_engine(self, *, strategy_action=None):
        engine=Engine(self.settings,FakeBackend())
        contract=engine.profiles.compile("D01",subset=["passport.number"])
        if strategy_action:
            payload={"card_id":"passport_test_card","version":"1",
                "scope":{"profile_id":contract["profile_id"],"profile_version":contract["profile_version"],
                    "service_ids":[],"field_keys":["passport.number"]},
                "directives":[{"action":strategy_action,"field_keys":["passport.number"],"parameters":{}}],
                "evidence_sha256":["b"*64],"negative_findings":[]}
            store=engine.strategies
            store.propose(payload,actor_id="a",actor_scopes=["strategy:propose"])
            store.approve(payload["card_id"],"1",actor_id="b",actor_scopes=["strategy:approve"])
            store.activate(payload["card_id"],"1",actor_id="c",actor_scopes=["strategy:activate"])
        bound=engine.binding(contract,"qwen-0.8b")
        cohort=[{"document_id":split+str(i),"group_id":split+"-original-"+str(i),
            "split":split,"critical_error":False,"risk_score":0.0}
            for split,n in (("calibration",10),("qualification",30)) for i in range(n)]
        calibrator=DocumentRiskCalibrator(target_error_rate=0.2,min_calibration_groups=10,
            min_qualification_groups=20,required_fields=["passport.number"],score_contract_id=SCORE_CONTRACT_ID)
        policy=calibrator.fit(cohort,bound)
        engine.policies.save_candidate(policy,actor_id="a",actor_scopes=["policy:propose"])
        engine.policies.approve(policy["policy_id"],actor_id="b",actor_scopes=["policy:approve"])
        engine.policies.activate(policy["policy_id"],actor_id="c",actor_scopes=["policy:activate"])
        key=digest(bound)
        config={"qualified_bindings":[key],"policy_ids":{key:policy["policy_id"]},"score_contract_id":SCORE_CONTRACT_ID}
        write_json(engine.root/"runtime_policy.json",config)
        engine.gateway=BindingPolicyGateway.from_file(engine.root/"runtime_policy.json",engine.policies)
        return engine,contract,bound

    def test_full_grounded_result_is_review_without_real_qualification(self):
        engine, output = self.run_engine()
        self.assertEqual(output["terminal_status"], "completed", output["errors"])
        self.assertTrue(output["completed"])
        self.assertEqual(output["fields"]["passport.number"]["raw_value"], "123456")
        self.assertEqual(output["fields"]["passport.number"]["validation_status"], "pass")
        self.assertEqual(output["decision"], "review")
        self.assertIn("profile_not_qualified", output["reason_codes"])
        self.assertIsNone(output["estimated_individual_error_probability"])
        self.assertFalse(output["audit"]["selected"])
        self.assertIn("o1", engine.documents.get("d1")["observations"])

    def test_rules_baseline_uses_same_grounding_and_validators(self):
        backend = FakeBackend()
        _, output = self.run_engine(backend, mode="rules")
        self.assertEqual(output["fields"]["passport.number"]["value"], "123456")
        self.assertEqual(backend.calls, [])
        self.assertEqual(output["decision"], "review")

    def test_partial_selector_cannot_be_parsed_as_completed_claim(self):
        backend = FakeBackend(lambda op, args: {"text": reply(args["prompt"]), "completed": False} if op == "select" else None)
        engine, output = self.run_engine(backend)
        self.assertFalse(output["completed"])
        self.assertIn("selector_token_limit", output["errors"])
        self.assertIsNone(output["fields"]["passport.number"]["value"])
        self.assertEqual(len(engine.documents.get("d1")["observations"]), 1)
        self.assertEqual(output["decision"], "review")

    def test_invalid_selector_fails_without_losing_prior_evidence(self):
        backend = FakeBackend(lambda op, args: {"text": '{"proposals":[{"field_key":"passport.number","candidate_ids":["forged"],"party_role":"document"}],"reread_requests":[]}',
                                                   "completed": True} if op == "select" else None)
        engine, output = self.run_engine(backend)
        self.assertEqual(output["terminal_status"], "failed")
        self.assertFalse(output["completed"])
        self.assertIn("o1", engine.documents.get("d1")["observations"])
        self.assertIsNone(output["fields"]["passport.number"]["value"])

    def test_deadline_after_first_claim_preserves_claim_but_never_accepts(self):
        def backend(op, args):
            if op == "select":return {"text": reply(args["prompt"], reread=True), "completed": True}
            if op == "ocr":raise ModelDeadline("forced-budget-exhausted")
        region = {"id": "r1", "path": "fixture-unused.png", "page": 1, "page_id": "p1",
                  "source_size": [800, 1200], "quad": [[0,0],[800,0],[800,100],[0,100]]}
        _, output = self.run_engine(FakeBackend(backend), shared=optics(regions=[region]))
        self.assertEqual(output["terminal_status"], "budget_exhausted")
        self.assertFalse(output["completed"])
        self.assertEqual(output["fields"]["passport.number"]["value"], "123456")
        self.assertEqual(output["decision"], "review")

    def test_incomplete_reread_remains_incomplete_in_frozen_optics(self):
        def backend(op, args):
            if op == "select":return {"text": reply(args["prompt"], reread=True), "completed": True}
            if op == "ocr":return {"text": "partial", "completed": False}
        region = {"id": "r1", "path": "fixture-unused.png", "page": 1, "page_id": "p1",
                  "source_size": [800, 1200], "quad": [[0,0],[800,0],[800,100],[0,100]]}
        _, output = self.run_engine(FakeBackend(backend), shared=optics(regions=[region]))
        self.assertFalse(output["completed"])
        self.assertFalse(output["optics"]["completed"])
        self.assertTrue(any(not o["completed"] for o in output["optics"]["observations"]))

    def test_cancellation_after_cheap_ocr_preserves_partial_observations(self):
        ingestion = {"source_sha256": SHA, "observations": [], "pages": [{"path": "fixture.png", "page": 1,
            "page_id": "p1", "source_size": [800,1200], "transform": {}}]}
        def backend(op, args):
            if op == "ingest":return ingestion
            if op == "ocr":raise ModelCanceled("forced-cancel")
        engine = Engine(self.settings, FakeBackend(backend))
        output = engine.run("d1", "fixture.png", SHA, "D01", subset=["passport.number"])
        self.assertEqual(output["terminal_status"], "canceled")
        self.assertFalse(output["completed"])
        self.assertIn("o1", output["observations"])
        self.assertEqual(output["decision"], "review")

    def test_mismatched_input_hash_rejected_before_any_model_call(self):
        backend = FakeBackend(lambda op,args: {"source_sha256": "b" * 64} if op == "ingest" else None)
        engine = Engine(self.settings, backend)
        output = engine.run("d1", "fixture.png", SHA, "D01", subset=["passport.number"])
        self.assertEqual(output["terminal_status"], "failed")
        self.assertIn("input_source_sha_mismatch", " ".join(output["errors"]))
        self.assertEqual(backend.calls, ["ingest"])

    def test_runtime_qualified_policy_bridge_is_live_and_fields_agree(self):
        engine,_,_=self.configured_engine()
        output=engine.run("d1","not-read.png",SHA,"D01",subset=["passport.number"],shared_optics=optics())
        self.assertEqual(output["decision"],"accept_for_autofill",output)
        self.assertEqual(output["fields"]["passport.number"]["decision"],"accept_for_autofill")
        self.assertTrue(output["profile_qualified"])
        self.assertEqual(output["risk_score"],0.0)
        self.assertFalse(output["service_approval"])
        self.assertIsNone(output["estimated_individual_error_probability"])
        self.assertFalse(output["drift"]["model_or_policy_promoted"])
        self.assertEqual(output["audit"]["group_id"],SHA)

    def test_vetted_but_unimplemented_strategy_forces_explicit_review(self):
        engine,_,_=self.configured_engine(strategy_action="prefer_same_cell")
        output=engine.run("d1","not-read.png",SHA,"D01",subset=["passport.number"],shared_optics=optics())
        self.assertEqual(output["decision"],"review")
        self.assertIn("unsupported_strategy_action:prefer_same_cell",output["reason_codes"])
        self.assertIn("unsupported_strategy_action:prefer_same_cell",output["errors"])
        self.assertEqual(output["strategy_runtime"]["unsupported_actions"],["prefer_same_cell"])
        self.assertFalse(output["audit"]["selected"])

    def test_runtime_binding_includes_execution_budget_mode_threads_and_cards(self):
        engine=Engine(self.settings,FakeBackend())
        contract=engine.profiles.compile("D01",subset=["passport.number"])
        original=engine.binding(contract,"qwen-0.8b")
        modes=[engine.binding(contract,"qwen-0.8b",mode="rules"),
               engine.binding(contract,"qwen-0.8b",allow_reread=False)]
        for settings in (replace(self.settings,max_new_tokens=2048),replace(self.settings,document_timeout_seconds=700),
                         replace(self.settings,cpu_threads=2),replace(self.settings,max_rereads=1)):
            altered=Engine(settings,FakeBackend()).binding(contract,"qwen-0.8b")
            modes.append(altered)
        for changed in modes:
            self.assertNotEqual(original["model_bundle_sha256"],changed["model_bundle_sha256"])
        engine,contract,with_card=self.configured_engine(strategy_action="review_on_conflict")
        self.assertNotEqual(original["model_bundle_sha256"],with_card["model_bundle_sha256"])
        engine.strategies.revoke("passport_test_card","1",actor_id="op",actor_scopes=["strategy:revoke"])
        self.assertEqual(original,engine.binding(contract,"qwen-0.8b"))


if __name__ == "__main__":
    unittest.main()
