"""F06/F07/F08 failure-oriented tests; no neural model or external service."""
from __future__ import annotations

import copy
from contextlib import closing
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from harness_ocr.control import (
    AuditController, AuditLog, AuditSampler, DriftMonitor, DocumentRiskCalibrator,
    RiskPolicyStore, StrategyStore, binomial_error_upper_bound,
)
from harness_ocr.control.common import canonical, digest
from harness_ocr.control.strategies import validate_card
from harness_ocr.control.runtime import BindingPolicyGateway, SCORE_CONTRACT_ID, hard_gate_rank, unsupported_strategy_actions


def binding(**changes):
    value = {"profile_id": "receipt", "profile_version": "1.0", "subset_contract_id": "receipt-critical-v1",
             "model_bundle_sha256": "a" * 64, "policy_version": "1", "input_scope_version": "scan-v1"}
    return {**value, **changes}


def records(prefix, split, n, *, error=False, score=0.1):
    return [{"document_id": f"{prefix}{i}", "group_id": f"{prefix}-original-{i}",
             "split": split, "risk_score": score, "critical_error": error} for i in range(n)]


def qualified_policy(*, conditional_fields=(), score_contract_id="server-document-risk-v1"):
    # This deliberately permissive TEST target is not a production quality claim.
    calibrator = DocumentRiskCalibrator(target_error_rate=0.2, confidence=0.95,
        min_calibration_groups=10, min_qualification_groups=20,
        minimum_coverage=0.8, required_fields=["amount"], conditional_fields=conditional_fields,
        score_contract_id=score_contract_id)
    policy = calibrator.fit(records("cal", "calibration", 20) + records("qual", "qualification", 30), binding())
    return calibrator, policy


def result():
    return {"document_id": "doc", "revision": 4, "completed": True, "terminal_status": "completed",
            "conflicts": [], "required_fields": ["amount"], "triggered_conditional_fields": [],
            "risk_score": 0.1, "score_contract_id": "server-document-risk-v1",
            "fields": {"amount": {"value": "125.00", "raw_value": "125,00", "critical": True,
                "extraction_status": "observed", "validation_status": "pass",
                "evidence_ids": ["obs1"], "party_role": "payment_total", "decision": "eligible"}},
            "observations": {"obs1": {"id": "obs1", "source_sha256": "b" * 64,
                "completed": True, "active": True, "text": "125,00"}}}


def card(version="1"):
    return {"card_id": "receipt_seller", "version": version,
            "scope": {"profile_id": "receipt", "profile_version": "1.0",
                      "service_ids": ["payment"], "field_keys": ["seller", "seller_inn"]},
            "directives": [{"action": "require_role_label", "field_keys": ["seller_inn"], "parameters": {}},
                {"action": "reread", "field_keys": ["seller"], "parameters": {
                    "max_rounds": 1, "max_regions": 2, "scale_policy": "native", "reason": "role_ambiguity"}}],
            "evidence_sha256": ["c" * 64],
            "negative_findings": [{"experiment_sha256": "d" * 64, "action": "reread", "outcome": "no_gain"}]}


class DurableTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "controls.sqlite3"


class AuditTests(DurableTest):
    def test_private_sampling_seed_persists_without_entering_events(self):
        log=AuditLog(self.db)
        seed=log.server_sampling_seed()
        self.assertEqual(seed,AuditLog(self.db).server_sampling_seed())
        self.assertNotEqual(seed,log.server_sampling_seed("other-sampler"))
        self.assertEqual(log.events(),[])
        self.assertEqual(len(seed),64)

    def test_same_revision_can_have_distinct_logged_server_policy_decisions(self):
        control=AuditController(AuditSampler(1.0,"secret"),AuditLog(self.db))
        initial=control.plan("d",1,binding(),accepted=False,acceptance_context="no-policy")
        later=control.plan("d",1,binding(),accepted=True,acceptance_context="qualified-policy")
        self.assertNotEqual(initial["task_id"],later["task_id"])
        self.assertFalse(initial["selected"])
        self.assertTrue(later["selected"])

    def test_hash_sample_has_known_probability_and_no_reroll(self):
        sampler = AuditSampler({"default": 0.3, "rare": 1.0, "excluded": 0.0}, "private-test-seed")
        same = sampler.decide("original-1")
        self.assertEqual(same, sampler.decide("original-1"))
        self.assertEqual(same["inclusion_probability"], 0.3)
        self.assertNotIn("private-test-seed", json.dumps(same))
        self.assertTrue(sampler.decide("original-1", "rare")["selected"])
        self.assertFalse(sampler.decide("original-1", "excluded")["selected"])
        # Finite hash samples are deterministic. This checks gross sampling bugs,
        # not a scientific significance claim about this particular seed.
        selected = sum(sampler.decide(f"original-{i}")["selected"] for i in range(2000))
        self.assertTrue(450 < selected < 750)
        with self.assertRaises(ValueError):
            AuditSampler({"rare": 0.5}, "s").decide("x", "missing")

    def test_durable_selection_idempotency_and_accepted_only(self):
        control = AuditController(AuditSampler(1.0, "s"), AuditLog(self.db))
        first = control.plan("d", 1, binding(), group_id="original")
        self.assertTrue(first["selected"])
        self.assertEqual(first, control.plan("d", 1, binding(), group_id="original"))
        self.assertEqual(len(AuditLog(self.db).events("audit_selection")), 1)
        with self.assertRaises(ValueError):
            control.plan("d", 1, binding(), group_id="different")
        skipped = control.plan("d2", 1, binding(), accepted=False)
        self.assertFalse(skipped["selected"])
        with self.assertRaises(ValueError):
            control.record_outcome(skipped["task_id"], critical_error=False, actor_id="reviewer",
                                   actor_scopes=["audit:write"], evidence_ids=["obs"])

    def test_outcomes_require_auth_evidence_and_followup_is_not_confidence(self):
        control = AuditController(AuditSampler(1.0, "s"), AuditLog(self.db))
        a = control.plan("a", 1, binding(), group_id="same-original")
        control.plan("b", 1, binding(), group_id="same-original")
        with self.assertRaises(PermissionError):
            control.record_outcome(a["task_id"], critical_error=True, actor_id="llm",
                                   actor_scopes=[], evidence_ids=["obs"])
        with self.assertRaises(ValueError):
            control.record_outcome(a["task_id"], critical_error=True, actor_id="reviewer",
                                   actor_scopes=["audit:write"], evidence_ids=[])
        outcome = control.record_outcome(a["task_id"], critical_error=True, actor_id="reviewer",
                                         actor_scopes=["audit:write"], evidence_ids=["obs"])
        self.assertFalse(outcome["becomes_training_gold"])
        summary = control.summary()
        self.assertEqual(summary["missing_followup"], 1)
        self.assertEqual(summary["independent_original_groups_reviewed"], 1)
        self.assertEqual(summary["population_estimate_status"], "incomplete_audit_followup")
        self.assertIsNone(summary["confidence_interval"])
        with self.assertRaises(ValueError):
            control.record_outcome(a["task_id"], critical_error=False, actor_id="reviewer",
                                   actor_scopes=["audit:write"], evidence_ids=["obs"])

    def test_log_detects_corruption_and_different_idempotent_payload(self):
        log = AuditLog(self.db)
        log.append("same", {"a": 1})
        with self.assertRaises(ValueError):
            log.append("same", {"a": 2})
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("UPDATE control_events SET payload_json=? WHERE event_id='same'", ('{"a":2}',))
        with self.assertRaises(ValueError):
            AuditLog(self.db).events()

    def test_drift_quarantines_and_never_promotes_or_auto_clears(self):
        log = AuditLog(self.db)
        drift = DriftMonitor(log, binding(), baseline_acceptance_rate=1.0, max_acceptance_drop=0.2,
                             window_size=2, min_observations=2)
        self.assertFalse(drift.observe("a", binding(), accepted=False)["quarantined"])
        alert = drift.observe("b", binding(), accepted=False)
        self.assertTrue(alert["quarantined"])
        self.assertEqual(alert["action"], "review_only")
        self.assertFalse(alert["model_or_policy_promoted"])
        drift.observe("c", binding(), accepted=True)
        self.assertTrue(drift.observe("d", binding(), accepted=True)["quarantined"])
        self.assertTrue(DriftMonitor(log, binding(), baseline_acceptance_rate=1.0).is_quarantined(binding()))
        with self.assertRaises(PermissionError):
            drift.clear_quarantine(binding(), actor_id="llm", actor_scopes=[], reason="ok")
        drift.clear_quarantine(binding(), actor_id="operator", actor_scopes=["policy:quarantine:clear"], reason="review completed")
        self.assertFalse(drift.is_quarantined(binding()))

    def test_unknown_binding_and_confirmed_error_fail_closed(self):
        drift = DriftMonitor(AuditLog(self.db), binding(), baseline_acceptance_rate=0.8)
        changed = binding(model_bundle_sha256="e" * 64)
        alert = drift.observe("a", changed, accepted=True)
        self.assertIn("unqualified_binding", alert["reason_codes"])
        self.assertFalse(drift.is_quarantined(binding()))
        alert = drift.observe("b", binding(), accepted=True, reviewed_critical_error=True)
        self.assertIn("confirmed_critical_error", alert["reason_codes"])

    def test_repeated_originals_not_independent_and_latest_order_updates(self):
        drift = DriftMonitor(AuditLog(self.db), binding(), baseline_acceptance_rate=0.5,
                             window_size=2, min_observations=2)
        drift.observe("a", binding(), accepted=False, event_id="a0")
        drift.observe("b", binding(), accepted=True)
        drift.observe("a", binding(), accepted=True, event_id="a1")
        observed = drift.observe("c", binding(), accepted=False)
        self.assertEqual(observed["observed_original_documents"], 2)
        self.assertEqual(observed["observed_acceptance_rate"], 0.5)


class StrategyTests(DurableTest):
    def activate(self, store, payload):
        store.propose(payload, actor_id="researcher", actor_scopes=["strategy:propose"])
        store.approve(payload["card_id"], payload["version"], actor_id="reviewer", actor_scopes=["strategy:approve"])
        return store.activate(payload["card_id"], payload["version"], actor_id="operator", actor_scopes=["strategy:activate"])

    def test_cards_reject_values_prompts_unbounded_actions_and_scope_escape(self):
        malformed = []
        a = card(); a["personal_values"] = {"inn": "1234567890"}; malformed.append(a)
        a = card(); a["directives"][0]["parameters"] = {"value": "1234567890"}; malformed.append(a)
        a = card(); a["directives"][0]["action"] = "execute_python"; malformed.append(a)
        a = card(); a["directives"][0]["field_keys"] = ["passport_number"]; malformed.append(a)
        a = card(); a["directives"][1]["parameters"]["max_rounds"] = 999; malformed.append(a)
        a = card(); a["evidence_sha256"] = []; malformed.append(a)
        a = card(); a["scope"]["field_keys"] = ["Иванов Иван"]; malformed.append(a)
        a = card(); a["negative_findings"][0]["value"] = "private"; malformed.append(a)
        for payload in malformed:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    validate_card(payload)
        self.assertEqual(validate_card(card())["negative_findings"][0]["outcome"], "no_gain")

    def test_approval_activation_scope_and_revocation_are_durable(self):
        store = StrategyStore(self.db)
        with self.assertRaises(PermissionError):
            store.propose(card(), actor_id="llm", actor_scopes=[])
        store.propose(card(), actor_id="llm", actor_scopes=["strategy:propose"])
        with self.assertRaises(ValueError):
            store.activate("receipt_seller", "1", actor_id="op", actor_scopes=["strategy:activate"])
        self.assertEqual(store.retrieve("receipt", "1.0", service_id="payment", actor_scopes=["strategy:read"]), [])
        self.activate(store, card())
        query = {"service_id": "payment", "actor_scopes": ["strategy:read"]}
        self.assertEqual(len(store.retrieve("receipt", "1.0", **query)), 1)
        self.assertEqual(store.retrieve("receipt", "2.0", **query), [])
        self.assertEqual(store.retrieve("receipt", "1.0", service_id="other", actor_scopes=["strategy:read"]), [])
        filtered = store.retrieve("receipt", "1.0", field_keys=["seller_inn"], **query)[0]
        self.assertEqual(len(filtered["directives"]), 1)
        self.assertEqual(filtered["directives"][0]["action"], "require_role_label")
        store.revoke("receipt_seller", "1", actor_id="op", actor_scopes=["strategy:revoke"])
        reloaded = StrategyStore(self.db)
        self.assertEqual(reloaded.retrieve("receipt", "1.0", **query), [])
        with self.assertRaises(ValueError):
            reloaded.activate("receipt_seller", "1", actor_id="op", actor_scopes=["strategy:activate"])

    def test_card_versions_immutable_and_rollback_does_not_lose_history(self):
        store = StrategyStore(self.db)
        self.activate(store, card("1"))
        changed = card("1"); changed["directives"] = changed["directives"][:1]
        with self.assertRaises(ValueError):
            store.propose(changed, actor_id="llm", actor_scopes=["strategy:propose"])
        self.activate(store, card("2"))
        self.assertEqual(store.get("receipt_seller", "1", actor_scopes=["strategy:read"])["state"], "superseded")
        store.activate("receipt_seller", "1", actor_id="rollback-op", actor_scopes=["strategy:activate"])
        store.activate("receipt_seller", "2", actor_id="new-op", actor_scopes=["strategy:activate"])
        history = AuditLog(self.db).events("strategy_lifecycle")
        self.assertEqual(sum(e["payload"]["action"] == "activate" for e in history), 4)


class RiskTests(DurableTest):
    def test_exact_one_sided_error_bounds(self):
        self.assertIsNone(binomial_error_upper_bound(0, 0))
        self.assertAlmostEqual(binomial_error_upper_bound(0, 100), 1 - 0.05 ** (1 / 100), places=14)
        self.assertAlmostEqual(binomial_error_upper_bound(1, 100), 0.04655981145, places=9)
        self.assertLess(binomial_error_upper_bound(0, 3000), 0.001)
        self.assertGreater(binomial_error_upper_bound(0, 2995), 0.00099)
        self.assertEqual(binomial_error_upper_bound(7, 7), 1.0)
        for args in ((True, 100), (-1, 10), (11, 10), (0, -1), (0, 10, 1.0)):
            with self.assertRaises(ValueError):
                binomial_error_upper_bound(*args)

    def test_empty_or_small_data_reports_insufficient_without_fake_probability(self):
        calibrator = DocumentRiskCalibrator(required_fields=["amount"])
        empty = calibrator.fit([], binding())
        self.assertEqual(empty["qualification_status"], "insufficient_calibration_data")
        self.assertFalse(empty["qualified_for_autoaccept"])
        applied = calibrator.apply(empty, result(), binding())
        self.assertEqual(applied["decision"], "review")
        self.assertIsNone(applied["estimated_individual_error_probability"])
        self.assertIsNone(applied["qualified_group_error_upper_bound"])
        small = calibrator.fit(records("c", "calibration", 30) + records("q", "qualification", 100), binding())
        self.assertFalse(small["qualified_for_autoaccept"])
        self.assertIn("qualification_error_bound_exceeded", small["reason_codes"])
        self.assertAlmostEqual(small["qualification"]["group_error_upper_bound"], 0.0295130496, places=9)

    def test_group_leakage_duplicates_mixed_bindings_and_test_fit_rejected(self):
        calibrator, _ = qualified_policy()
        base = records("a", "calibration", 10)
        leaked = copy.deepcopy(base[0]); leaked.update(document_id="other", split="qualification")
        for cohort in (base + [leaked], base + [copy.deepcopy(base[0])],
                       base + records("x", "test", 1)):
            with self.assertRaises(ValueError):
                calibrator.fit(cohort, binding())
        base[0]["binding"] = binding(model_bundle_sha256="e" * 64)
        with self.assertRaises(ValueError):
            calibrator.fit(base, binding())

    def test_thousands_of_correlated_variants_are_one_group(self):
        calibrator = DocumentRiskCalibrator(target_error_rate=0.2, min_calibration_groups=1,
            min_qualification_groups=20, required_fields=["amount"])
        repeats = records("q", "qualification", 100)
        for row in repeats:
            row["group_id"] = "one-original"
        repeats[-1]["critical_error"] = True
        policy = calibrator.fit(records("c", "calibration", 2) + repeats, binding())
        self.assertEqual(policy["qualification"]["accepted_groups"], 1)
        self.assertEqual(policy["qualification"]["critical_error_groups"], 1)
        self.assertEqual(policy["qualification"]["group_error_upper_bound"], 1.0)
        self.assertFalse(policy["qualified_for_autoaccept"])

    def test_qualification_labels_do_not_choose_or_retry_threshold(self):
        calibrator, _ = qualified_policy()
        cal = records("cal", "calibration", 20)
        cal += records("high", "calibration", 1, error=True, score=0.9)
        good = calibrator.fit(cal + records("q", "qualification", 30), binding())
        bad = calibrator.fit(cal + records("q", "qualification", 30, error=True), binding())
        self.assertEqual(good["risk_score_threshold"], bad["risk_score_threshold"])
        self.assertEqual(good["threshold_commit_sha256"], bad["threshold_commit_sha256"])
        self.assertTrue(good["qualified_for_autoaccept"])
        self.assertFalse(bad["qualified_for_autoaccept"])
        self.assertIn("qualification_error_bound_exceeded", bad["reason_codes"])

    def test_heldout_evaluation_cannot_reuse_development_or_qualification(self):
        calibrator, policy = qualified_policy()
        tested = calibrator.evaluate(policy, records("test", "test", 3))
        self.assertTrue(tested["independent_from_calibration_and_qualification"])
        self.assertFalse(tested["promotes_policy"])
        for prefix in ("cal", "qual"):
            with self.assertRaises(ValueError):
                calibrator.evaluate(policy, records(prefix, "test", 2))
        with self.assertRaises(ValueError):
            calibrator.evaluate(policy, records("new", "qualification", 3))
        diagnostic = calibrator.evaluate(policy, records("cal", "calibration", 2), allow_known_groups=True)
        self.assertEqual(diagnostic["evaluation_status"], "descriptive_known_groups")

    def test_trusted_apply_accepts_qualified_complete_evidence_not_field_probability_product(self):
        calibrator, policy = qualified_policy()
        applied = calibrator.apply(policy, result(), binding())
        self.assertEqual(applied["decision"], "accept_for_autofill")
        self.assertIsNone(applied["estimated_individual_error_probability"])
        self.assertFalse(applied["service_approval"])
        self.assertEqual(applied["qualification_accepted_groups"], 30)

    def test_hard_failures_cannot_be_overridden_by_client_validated_flag(self):
        calibrator, policy = qualified_policy()
        mutations = [
            lambda r: r.update(completed=False),
            lambda r: r.update(terminal_status="canceled"),
            lambda r: r.update(terminal_status="budget_exhausted"),
            lambda r: r.update(conflicts=["amount"]),
            lambda r: r.pop("conflicts"),
            lambda r: r["fields"]["amount"].update(value=None),
            lambda r: r["fields"]["amount"].update(validation_status="fail"),
            lambda r: r["fields"]["amount"].update(extraction_status="inferred"),
            lambda r: r["fields"]["amount"].update(party_role=None),
            lambda r: r["fields"]["amount"].update(evidence_ids=[]),
            lambda r: r["fields"]["amount"].update(needs_review=True),
            lambda r: r["observations"]["obs1"].update(active=False),
            lambda r: r["observations"]["obs1"].update(completed=False),
            lambda r: r["observations"]["obs1"].update(source_sha256="z" * 64),
            lambda r: r.update(score_contract_id="client-made-up"),
            lambda r: r.update(risk_score=float("nan")),
            lambda r: r.update(risk_score=0.9),
            lambda r: r["fields"].update(extra={"critical": True, "value": "unsupported"}),
        ]
        for mutate in mutations:
            candidate = result(); candidate["validated"] = True; mutate(candidate)
            with self.subTest(mutation=mutate):
                answer = calibrator.apply(policy, candidate, binding())
                self.assertEqual(answer["decision"], "review", answer)
                self.assertTrue(answer["reason_codes"])

    def test_conditional_applicability_must_be_explicit(self):
        calibrator, policy = qualified_policy(conditional_fields=["representative_id"])
        candidate = result()
        self.assertEqual(calibrator.apply(policy, candidate, binding())["decision"], "review")
        candidate["conditional_applicability"] = {"representative_id": False}
        self.assertEqual(calibrator.apply(policy, candidate, binding())["decision"], "accept_for_autofill")
        candidate["conditional_applicability"]["representative_id"] = True
        self.assertIn("missing_required_field:representative_id", calibrator.apply(policy, candidate, binding())["reason_codes"])

    def test_model_profile_config_and_subset_changes_invalidate_binding(self):
        calibrator, policy = qualified_policy()
        for key, value in (("profile_id", "passport"), ("profile_version", "2.0"),
            ("subset_contract_id", "another-subset"), ("model_bundle_sha256", "f" * 64),
            ("policy_version", "2"), ("input_scope_version", "photo-v2")):
            with self.subTest(key=key):
                self.assertIn("policy_binding_mismatch", calibrator.apply(policy, result(), binding(**{key: value}))["reason_codes"])

    def test_hash_tampering_and_no_required_server_contract_fail_closed(self):
        calibrator, policy = qualified_policy()
        changed = copy.deepcopy(policy); changed["risk_score_threshold"] = 1.0
        self.assertIn("invalid_policy_integrity", calibrator.apply(changed, result(), binding())["reason_codes"])
        missing = DocumentRiskCalibrator(target_error_rate=0.2, min_calibration_groups=1, min_qualification_groups=10)
        candidate = missing.fit(records("c", "calibration", 20) + records("q", "qualification", 30), binding())
        self.assertFalse(candidate["qualified_for_autoaccept"])
        self.assertIn("missing_server_required_fields", candidate["reason_codes"])

    def test_trusted_store_requires_approval_activation_and_revocation_persists(self):
        _, policy = qualified_policy()
        store = RiskPolicyStore(self.db)
        ident = policy["policy_id"]
        with self.assertRaises(PermissionError):
            store.save_candidate(policy, actor_id="client", actor_scopes=[])
        store.save_candidate(policy, actor_id="analyst", actor_scopes=["policy:propose"])
        self.assertEqual(store.apply(ident, result(), binding())["decision"], "review")
        with self.assertRaises(ValueError):
            store.activate(ident, actor_id="op", actor_scopes=["policy:activate"])
        store.approve(ident, actor_id="reviewer", actor_scopes=["policy:approve"])
        self.assertEqual(store.apply(ident, result(), binding())["decision"], "review")
        store.activate(ident, actor_id="op", actor_scopes=["policy:activate"])
        self.assertEqual(RiskPolicyStore(self.db).apply(ident, result(), binding())["decision"], "accept_for_autofill")
        store.revoke(ident, actor_id="op", actor_scopes=["policy:revoke"])
        self.assertEqual(RiskPolicyStore(self.db).apply(ident, result(), binding())["decision"], "review")
        with self.assertRaises(ValueError):
            store.activate(ident, actor_id="op", actor_scopes=["policy:activate"])

    def test_sticky_quarantine_overrides_active_qualified_policy(self):
        _, policy = qualified_policy()
        store = RiskPolicyStore(self.db)
        ident = policy["policy_id"]
        store.save_candidate(policy, actor_id="analyst", actor_scopes=["policy:propose"])
        store.approve(ident, actor_id="reviewer", actor_scopes=["policy:approve"])
        store.activate(ident, actor_id="op", actor_scopes=["policy:activate"])
        monitor = DriftMonitor(AuditLog(self.db), binding(), baseline_acceptance_rate=0.9)
        monitor.observe("bad", binding(), accepted=True, reviewed_critical_error=True)
        applied = RiskPolicyStore(self.db).apply(ident, result(), binding())
        self.assertEqual(applied["decision"], "review")
        self.assertIn("binding_quarantined", applied["reason_codes"])

    def test_insufficient_policy_cannot_be_approved_even_with_auth(self):
        policy = DocumentRiskCalibrator(required_fields=["amount"]).fit([], binding())
        store = RiskPolicyStore(self.db)
        store.save_candidate(policy, actor_id="a", actor_scopes=["policy:propose"])
        with self.assertRaises(ValueError):
            store.approve(policy["policy_id"], actor_id="b", actor_scopes=["policy:approve"])

    def test_stored_policy_integrity_checked_again_at_inference(self):
        _, policy = qualified_policy()
        store = RiskPolicyStore(self.db)
        store.save_candidate(policy, actor_id="a", actor_scopes=["policy:propose"])
        store.approve(policy["policy_id"], actor_id="b", actor_scopes=["policy:approve"])
        store.activate(policy["policy_id"], actor_id="c", actor_scopes=["policy:activate"])
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("UPDATE risk_policies SET policy_sha256=?", ("0" * 64,))
        answer = store.apply(policy["policy_id"], result(), binding())
        self.assertEqual(answer["decision"], "review")
        self.assertIn("invalid_policy_integrity", answer["reason_codes"])


class RuntimeGatewayTests(DurableTest):
    def policy_store(self, *, active=True, score_contract=SCORE_CONTRACT_ID):
        _, policy = qualified_policy(score_contract_id=score_contract)
        store = RiskPolicyStore(self.db)
        store.save_candidate(policy, actor_id="a", actor_scopes=["policy:propose"])
        if active:
            store.approve(policy["policy_id"], actor_id="b", actor_scopes=["policy:approve"])
            store.activate(policy["policy_id"], actor_id="c", actor_scopes=["policy:activate"])
        key = digest(binding())
        config = {"qualified_bindings": [key], "policy_ids": {key: policy["policy_id"]},
                  "score_contract_id": SCORE_CONTRACT_ID}
        return store, config

    def test_default_empty_configuration_ignores_caller_qualification_and_score(self):
        candidate = result()
        candidate.update(validated=True, profile_qualified=True, risk_score=0, confidence=1)
        applied = BindingPolicyGateway(None, RiskPolicyStore(self.db)).apply(candidate, binding())
        self.assertEqual(applied["decision"], "review")
        self.assertFalse(applied["profile_qualified"])
        self.assertIsNone(applied["risk_score"])
        self.assertIn("profile_not_qualified", applied["reason_codes"])

    def test_server_config_plus_active_matching_policy_recomputes_rank(self):
        store, config = self.policy_store()
        candidate = result(); candidate["risk_score"] = 0.999; candidate["score_contract_id"] = "malicious"
        applied = BindingPolicyGateway(config, store).apply(candidate, binding())
        self.assertEqual(applied["decision"], "accept_for_autofill")
        self.assertEqual(applied["risk_score"], 0.0)
        self.assertEqual(applied["risk_score_features"]["hard_guard_failure_count"], 0)
        self.assertFalse(applied["score_is_probability"])
        self.assertIsNone(applied["estimated_individual_error_probability"])

    def test_config_does_not_activate_candidate_policy(self):
        store, config = self.policy_store(active=False)
        applied = BindingPolicyGateway(config, store).apply(result(), binding())
        self.assertEqual(applied["decision"], "review")
        self.assertIn("policy_not_active_or_unknown", applied["reason_codes"])
        self.assertEqual(store.get(next(iter(config["policy_ids"].values())))["state"], "candidate")

    def test_unknown_score_contract_or_binding_cannot_reuse_qualification(self):
        store, config = self.policy_store(score_contract="other-ranking-v1")
        applied = BindingPolicyGateway(config, store).apply(result(), binding())
        self.assertIn("unsupported_runtime_score_contract", applied["reason_codes"])
        self.assertEqual(applied["decision"], "review")
        changed = BindingPolicyGateway(config, store).apply(result(), binding(input_scope_version="new-photo"))
        self.assertIn("profile_not_qualified", changed["reason_codes"])

    def test_gate_rank_describes_rule_failures_not_correctness_probability(self):
        contract = {"required_fields": ["amount"], "conditional_fields": []}
        good = hard_gate_rank(result(), contract)
        bad = result(); bad["fields"]["amount"]["validation_status"] = "fail"
        failed = hard_gate_rank(bad, contract)
        self.assertEqual(good["risk_score"], 0)
        self.assertGreater(failed["risk_score"], 0)
        self.assertFalse(failed["score_is_probability"])
        self.assertIn("unverified_required_field:amount", failed["risk_score_features"]["hard_guard_reason_codes"])

    def test_runtime_quarantine_and_terminal_state_still_override_server_config(self):
        store, config = self.policy_store()
        gateway = BindingPolicyGateway(config, store)
        failed = result(); failed["completed"] = False
        self.assertEqual(gateway.apply(failed, binding())["decision"], "review")
        DriftMonitor(AuditLog(self.db), binding(), baseline_acceptance_rate=0.9).observe(
            "bad", binding(), accepted=True, reviewed_critical_error=True)
        applied = gateway.apply(result(), binding())
        self.assertIn("binding_quarantined", applied["reason_codes"])
        self.assertEqual(applied["decision"], "review")

    def test_bad_server_configuration_fails_at_load(self):
        store = RiskPolicyStore(self.db)
        variants = [{"qualified_bindings": ["profile_name"]}, {"score_contract_id": "make_up_confidence"},
                    {"validated": True}, {"policy_ids": {digest(binding()): "x"}}, {"qualified_bindings": "all"}]
        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                BindingPolicyGateway(config, store)
        self.assertEqual(BindingPolicyGateway.from_file(Path(self.temp.name) / "missing.json", store).qualified, frozenset())

    def test_unimplemented_vetted_directives_are_reported(self):
        self.assertEqual(unsupported_strategy_actions([card()]), ["require_role_label"])
        self.assertEqual(unsupported_strategy_actions([card()], {"reread", "require_role_label"}), [])


if __name__ == "__main__":
    unittest.main()
