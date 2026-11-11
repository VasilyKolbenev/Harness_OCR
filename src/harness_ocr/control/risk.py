"""F08: independent document-level risk qualification and trusted policy store.

A risk score is a ranking feature, not an individual probability. A policy is
fitted on calibration groups and qualified on different original-document groups.
The reported one-sided binomial bound concerns the event "at least one accepted
critical value/role is wrong" per selected group; field confidences are never
multiplied. No data or a weak sample results in review, not fabricated confidence.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from typing import Iterable
import uuid

from .common import (
    append_event, canonical, connect, digest, init_events, probability,
    require_id, require_scope, transaction, utc_now, validate_binding,
)


def binomial_error_upper_bound(errors: int, count: int, confidence: float = 0.95) -> float | None:
    """Exact one-sided Clopper–Pearson upper bound, using stdlib arithmetic."""
    if type(errors) is not int or type(count) is not int or not 0 <= errors <= count:
        raise ValueError("Require integer0 <= errors <= count")
    confidence = probability(confidence, "confidence")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between0 and1")
    if count == 0:
        return None
    if count > 1_000_000 and errors not in (0, count):
        raise ValueError("Nonzero-error exact calculation is bounded to one million independent groups")
    alpha = 1.0 - confidence
    if errors == count:
        return 1.0
    if errors == 0:
        return -math.expm1(math.log(alpha) / count)
    # CDF in log space avoids underflow for large groups/rare errors.
    coefficients = [math.lgamma(count + 1) - math.lgamma(i + 1) - math.lgamma(count - i + 1)
                    for i in range(errors + 1)]
    log_alpha = math.log(alpha)
    low, high = errors / count, 1.0
    for _ in range(70):
        p = (low + high) / 2
        if p >= 1:
            high = p
            continue
        terms = [coefficients[i] + i * math.log(p) + (count - i) * math.log1p(-p)
                 for i in range(errors + 1)]
        largest = max(terms)
        log_cdf = largest + math.log(sum(math.exp(t - largest) for t in terms))
        if log_cdf > log_alpha:
            low = p
        else:
            high = p
    return high


def _clean_records(records: list[dict], binding: dict) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    result, ids, group_split = [], set(), {}
    for r in records:
        if not isinstance(r, dict):
            raise ValueError("Each annotated record must be a dictionary")
        ident = require_id(r.get("document_id"), "document_id")
        if ident in ids:
            raise ValueError("Duplicate document_id would inflate the sample")
        ids.add(ident)
        group = require_id(r.get("group_id"), "independent original group_id")
        split = r.get("split")
        if split not in {"calibration", "qualification", "test"}:
            raise ValueError("Records require calibration/qualification/test split")
        if group in group_split and group_split[group] != split:
            raise ValueError("One original group cannot cross data partitions")
        group_split[group] = split
        if "binding" in r and validate_binding(r["binding"]) != binding:
            raise ValueError("Mixed model/profile/subset bindings are not a calibration cohort")
        score = probability(r.get("risk_score"), "risk_score ranking feature")
        if type(r.get("critical_error")) is not bool:
            raise ValueError("Every calibration label must be an independently reviewed bool")
        eligible = r.get("eligible", True)
        if type(eligible) is not bool:
            raise ValueError("eligible must be bool")
        result.append({"document_id": ident, "group_id": group, "split": split,
                       "risk_score": score, "critical_error": r["critical_error"], "eligible": eligible})
    return result


def _stats(records: list[dict], threshold: float | None, confidence: float) -> dict:
    selected = [r for r in records if threshold is not None and r["eligible"] and r["risk_score"] <= threshold]
    groups = {}
    for r in selected:
        groups[r["group_id"]] = bool(groups.get(r["group_id"], False) or r["critical_error"])
    errors = sum(groups.values())
    total_groups = len({r["group_id"] for r in records})
    return {"original_groups": total_groups, "documents": len(records), "accepted_documents": len(selected),
            "accepted_groups": len(groups), "critical_error_groups": errors,
            "critical_error_documents": sum(r["critical_error"] for r in selected),
            "document_coverage": len(selected) / len(records) if records else 0.0,
            "group_coverage": len(groups) / total_groups if total_groups else 0.0,
            "observed_group_error_rate": errors / len(groups) if groups else None,
            "critical_correct_accepted_document_rate": sum(not r["critical_error"] for r in selected) / len(records) if records else None,
            "one_sided_confidence": confidence,
            "group_error_upper_bound": binomial_error_upper_bound(errors, len(groups), confidence),
            "unit": "independent original group; any selected critical error in its accepted variants",
            "not_an_individual_document_probability": True}


def _seal(policy: dict) -> dict:
    sealed = copy.deepcopy(policy)
    sealed.pop("policy_id", None)
    sealed.pop("policy_sha256", None)
    sha = digest(sealed)
    sealed["policy_id"] = "risk_" + sha[:32]
    sealed["policy_sha256"] = sha
    return sealed


def verify_policy(policy: dict) -> dict:
    if not isinstance(policy, dict) or policy.get("schema_version") != "harness-document-risk/1.0":
        raise ValueError("Unsupported risk policy schema")
    sealed = _seal(policy)
    if sealed["policy_id"] != policy.get("policy_id") or sealed["policy_sha256"] != policy.get("policy_sha256"):
        raise ValueError("Risk policy integrity failure")
    validate_binding(policy.get("binding"))
    probability(policy.get("target_error_rate"), "target_error_rate")
    confidence = probability(policy.get("confidence"), "confidence")
    if not 0 < confidence < 1:
        raise ValueError("Invalid qualification confidence")
    probability(policy.get("minimum_coverage"), "minimum_coverage")
    require_id(policy.get("score_contract_id"), "score_contract_id")
    if policy.get("risk_score_threshold") is not None:
        probability(policy["risk_score_threshold"], "risk_score_threshold")
    contract = policy.get("hard_contract")
    if not isinstance(contract, dict) or set(contract) != {"required_fields", "conditional_fields"}:
        raise ValueError("Invalid policy field contract")
    for keys in contract.values():
        if not isinstance(keys, list) or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("Invalid policy field keys")
    for key in ("min_calibration_groups", "min_qualification_groups"):
        if type(policy.get(key)) is not int or policy[key] < 1:
            raise ValueError("Invalid policy group count")
    for key in ("calibration_group_digests", "qualification_group_digests", "reason_codes"):
        if not isinstance(policy.get(key), list) or any(not isinstance(x, str) for x in policy[key]):
            raise ValueError("Invalid policy partitions or reasons")
    if type(policy.get("qualified_for_autoaccept")) is not bool or policy.get("qualification_status") not in {
            "qualified", "not_qualified", "insufficient_calibration_data"}:
        raise ValueError("Invalid qualification state")
    if policy["qualified_for_autoaccept"]:
        q = policy.get("qualification")
        if (policy["qualification_status"] != "qualified" or policy["reason_codes"] or not contract["required_fields"]
                or policy["risk_score_threshold"] is None or not isinstance(q, dict)):
            raise ValueError("Invalid qualified policy")
        errors, count = q.get("critical_error_groups"), q.get("accepted_groups")
        bound = binomial_error_upper_bound(errors, count, confidence)
        if (count < policy["min_qualification_groups"] or bound is None or bound > policy["target_error_rate"]
                or q.get("group_error_upper_bound") != bound
                or not isinstance(q.get("document_coverage"), (int, float))
                or q["document_coverage"] < policy["minimum_coverage"]):
            raise ValueError("Qualified policy contradicts its independent error bound")
    return copy.deepcopy(policy)


def _stored_policy(row) -> dict:
    policy = verify_policy(json.loads(row["policy_json"]))
    if (policy["policy_id"] != row["policy_id"] or policy["policy_sha256"] != row["policy_sha256"]
            or digest(policy["binding"]) != row["binding_sha256"]):
        raise ValueError("Stored risk policy identity or binding mismatch")
    return policy


class DocumentRiskCalibrator:
    def __init__(self, *, target_error_rate: float = 0.001, confidence: float = 0.95,
                 min_calibration_groups: int = 30, min_qualification_groups: int = 100,
                 minimum_coverage: float = 0.8, required_fields: Iterable[str] = (),
                 conditional_fields: Iterable[str] = (), score_contract_id: str = "server-document-risk-v1"):
        self.target_error_rate = probability(target_error_rate, "target_error_rate")
        self.confidence = probability(confidence, "confidence")
        self.minimum_coverage = probability(minimum_coverage, "minimum_coverage")
        if not 0 < self.confidence < 1 or not 0 < self.target_error_rate < 1:
            raise ValueError("confidence and target_error_rate must be strictly between0 and1")
        if any(type(n) is not int or n < 1 for n in (min_calibration_groups, min_qualification_groups)):
            raise ValueError("Minimum group counts must be positive integers")
        self.min_calibration_groups, self.min_qualification_groups = min_calibration_groups, min_qualification_groups
        if isinstance(required_fields, str) or isinstance(conditional_fields, str):
            raise ValueError("Field sets must be lists, not strings")
        self.required_fields = sorted({require_id(k, "required field") for k in required_fields})
        self.conditional_fields = sorted({require_id(k, "conditional field") for k in conditional_fields})
        self.score_contract_id = require_id(score_contract_id, "score_contract_id")

    def fit(self, records: list[dict], binding: dict) -> dict:
        binding = validate_binding(binding)
        records = _clean_records(records, binding)
        if any(r["split"] == "test" for r in records):
            raise ValueError("Test data cannot be used in fit; use evaluate after policy freeze")
        tuning = [r for r in records if r["split"] == "calibration"]
        qualification = [r for r in records if r["split"] == "qualification"]
        threshold, best = None, None
        tuning_group_count = len({r["group_id"] for r in tuning})
        if tuning_group_count >= self.min_calibration_groups:
            ordered = sorted((r for r in tuning if r["eligible"]), key=lambda r: r["risk_score"])
            selected_groups, accepted, errors = {}, 0, 0
            for i, record in enumerate(ordered):
                accepted += 1
                old_error = selected_groups.get(record["group_id"], False)
                selected_groups[record["group_id"]] = old_error or record["critical_error"]
                errors += int(record["critical_error"] and not old_error)
                candidate = record["risk_score"]
                if i + 1 < len(ordered) and ordered[i + 1]["risk_score"] == candidate:
                    continue
                if (accepted / len(tuning) >= self.minimum_coverage
                        and len(selected_groups) >= self.min_calibration_groups
                        and errors / len(selected_groups) <= self.target_error_rate):
                    rank = (accepted, -errors, -candidate)
                    if best is None or rank > best:
                        best, threshold = rank, candidate
        policy = {
            "schema_version": "harness-document-risk/1.0", "binding": binding,
            "score_contract_id": self.score_contract_id, "score_is_probability": False,
            "risk_score_threshold": threshold, "target_error_rate": self.target_error_rate,
            "confidence": self.confidence, "minimum_coverage": self.minimum_coverage,
            "min_calibration_groups": self.min_calibration_groups,
            "min_qualification_groups": self.min_qualification_groups,
            "hard_contract": {"required_fields": self.required_fields, "conditional_fields": self.conditional_fields},
            "calibration": _stats(tuning, threshold, self.confidence),
            "calibration_group_digests": sorted(digest(r["group_id"]) for r in {x["group_id"]: x for x in tuning}.values()),
            "source_records_sha256": digest(records),
            "qualification": None, "qualification_group_digests": [],
            "qualification_status": "insufficient_calibration_data", "qualified_for_autoaccept": False,
            "reason_codes": [],
            "method": "threshold selected on calibration only; independent grouped binomial qualification",
        }
        # Commit the selected threshold without using qualification labels. A
        # failed qualification never triggers a search on those labels.
        policy["threshold_commit_sha256"] = digest({"binding": binding, "threshold": threshold,
            "score_contract_id": self.score_contract_id, "calibration": policy["calibration"],
            "calibration_group_digests": policy["calibration_group_digests"]})
        if tuning_group_count < self.min_calibration_groups:
            policy["reason_codes"].append("insufficient_calibration_groups")
        elif threshold is None:
            policy["qualification_status"] = "not_qualified"
            policy["reason_codes"].append("no_safe_calibration_threshold")
        if not self.required_fields:
            policy["reason_codes"].append("missing_server_required_fields")
        candidate = _seal(policy)
        if qualification:
            return self.qualify(candidate, qualification, binding)
        if not policy["reason_codes"]:
            policy["reason_codes"].append("independent_qualification_data_missing")
        return _seal(policy)

    def qualify(self, candidate_policy: dict, records: list[dict], binding: dict) -> dict:
        policy = verify_policy(candidate_policy)
        binding = validate_binding(binding)
        if policy["binding"] != binding:
            raise ValueError("Qualification binding differs from fitted policy")
        cleaned = _clean_records(records, binding)
        if any(r["split"] != "qualification" for r in cleaned):
            raise ValueError("Qualification requires its own declared partition")
        known = set(policy["calibration_group_digests"])
        if any(digest(r["group_id"]) in known for r in cleaned):
            raise ValueError("Qualification leaks a calibration original")
        stats = _stats(cleaned, policy["risk_score_threshold"], policy["confidence"])
        policy["qualification"] = stats
        policy["qualification_group_digests"] = sorted({digest(r["group_id"]) for r in cleaned})
        policy["qualification_records_sha256"] = digest(cleaned)
        reasons = [x for x in policy["reason_codes"] if x not in {"independent_qualification_data_missing", "insufficient_qualification_groups", "qualification_error_bound_exceeded", "qualification_coverage_below_minimum"}]
        if stats["accepted_groups"] < policy["min_qualification_groups"]:
            reasons.append("insufficient_qualification_groups")
        if stats["document_coverage"] < policy["minimum_coverage"]:
            reasons.append("qualification_coverage_below_minimum")
        bound = stats["group_error_upper_bound"]
        if bound is None or bound > policy["target_error_rate"]:
            reasons.append("qualification_error_bound_exceeded")
        if policy["risk_score_threshold"] is None:
            reasons.append("no_fitted_threshold")
        policy["reason_codes"] = sorted(set(reasons))
        policy["qualified_for_autoaccept"] = not reasons
        policy["qualification_status"] = "qualified" if not reasons else (
            "insufficient_calibration_data" if any(x.startswith("insufficient_") or x == "missing_server_required_fields" for x in reasons) else "not_qualified")
        return _seal(policy)

    def evaluate(self, policy: dict, records: list[dict], *, allow_known_groups: bool = False) -> dict:
        policy = verify_policy(policy)
        cleaned = _clean_records(records, policy["binding"])
        if not allow_known_groups and any(r["split"] != "test" for r in cleaned):
            raise ValueError("Independent evaluation requires the held-out test partition")
        known = set(policy["calibration_group_digests"] + policy["qualification_group_digests"])
        overlap = {digest(r["group_id"]) for r in cleaned} & known
        if overlap and not allow_known_groups:
            raise ValueError("Independent evaluation overlaps development/qualification originals")
        return {"policy_id": policy["policy_id"], "binding": policy["binding"],
                "stats": _stats(cleaned, policy["risk_score_threshold"], policy["confidence"]),
                "independent_from_calibration_and_qualification": not overlap,
                "evaluation_status": "descriptive_known_groups" if overlap else "held_out_evaluation",
                "promotes_policy": False}

    def apply(self, policy: dict | None, result: dict, binding: dict) -> dict:
        """Internal application to an already trusted policy; public service uses the store below."""
        return apply_document_policy(policy, result, binding)


def _hard_guard_reasons(policy: dict, result: dict) -> list[str]:
    reasons = []
    if not isinstance(result, dict):
        return ["invalid_document_result"]
    if result.get("completed") is not True or result.get("terminal_status") != "completed":
        reasons.append("document_not_completed")
    if not isinstance(result.get("conflicts"), list) or result.get("conflicts"):
        reasons.append("document_conflict_or_missing_conflict_check")
    if result.get("unresolved_conditions"):
        reasons.append("unresolved_conditional_applicability")
    fields = result.get("fields")
    if not isinstance(fields, dict):
        return reasons + ["missing_fields"]
    required = set(policy["hard_contract"]["required_fields"])
    if not required:
        reasons.append("missing_server_required_fields")
    declared_required = result.get("required_fields", [])
    triggered = result.get("triggered_conditional_fields", [])
    if not isinstance(declared_required, list) or not isinstance(triggered, list) or any(not isinstance(k, str) for k in declared_required + triggered):
        return reasons + ["invalid_required_field_contract"]
    required.update(declared_required)
    required.update(triggered)
    conditions = result.get("conditional_applicability", {})
    if not isinstance(conditions, dict):
        return reasons + ["invalid_conditional_applicability"]
    # Static required fields remain required. A critical conditional field may
    # be excluded only by the server's explicit applicability contract, never
    # because it was missing from OCR.
    required.update(key for key, value in fields.items() if isinstance(key, str) and isinstance(value, dict)
                    and value.get("critical") is True and conditions.get(key) is not False)
    for key in policy["hard_contract"]["conditional_fields"]:
        applicable = conditions.get(key)
        if applicable is True:
            required.add(key)
        elif applicable is not False:
            reasons.append("conditional_applicability_unknown:" + key)
    registry = result.get("observations", result.get("evidence"))
    if isinstance(registry, list):
        registry = {o.get("id"): o for o in registry if isinstance(o, dict)}
    if not isinstance(registry, dict):
        registry = {}
        reasons.append("missing_evidence_registry")
    for key in sorted(required):
        value = fields.get(key)
        if not isinstance(value, dict) or value.get("value") in (None, "", [], {}):
            reasons.append("missing_required_field:" + key)
            continue
        if value.get("extraction_status") != "observed" or value.get("validation_status") != "pass":
            reasons.append("unverified_required_field:" + key)
        if value.get("needs_review") is True or value.get("decision") in {"review", "reject_input"}:
            reasons.append("field_requests_review:" + key)
        if not isinstance(value.get("party_role"), str) or not value["party_role"]:
            reasons.append("missing_field_role:" + key)
        refs = value.get("evidence_ids")
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
            reasons.append("missing_field_evidence:" + key)
            continue
        for ref in refs:
            observation = registry.get(ref)
            if (not isinstance(observation, dict) or observation.get("completed") is not True
                    or observation.get("active") is not True or not isinstance(observation.get("source_sha256"), str)
                    or not re.fullmatch(r"[0-9a-fA-F]{64}", observation["source_sha256"])):
                reasons.append("ineligible_evidence:" + key)
                break
    return sorted(set(reasons))


def apply_document_policy(policy: dict | None, result: dict, binding: dict) -> dict:
    binding = validate_binding(binding)
    if policy is None:
        return {"decision": "review", "reason_codes": ["insufficient_calibration_data"],
                "policy_id": None, "estimated_individual_error_probability": None}
    try:
        policy = verify_policy(policy)
    except ValueError:
        return {"decision": "review", "reason_codes": ["invalid_policy_integrity"],
                "policy_id": None, "estimated_individual_error_probability": None}
    reasons = []
    if policy["binding"] != binding:
        reasons.append("policy_binding_mismatch")
    if policy.get("qualified_for_autoaccept") is not True or policy.get("qualification_status") != "qualified":
        reasons.append("insufficient_calibration_data" if policy.get("qualification_status") == "insufficient_calibration_data" else "policy_not_qualified")
    reasons += _hard_guard_reasons(policy, result)
    score = result.get("risk_score") if isinstance(result, dict) else None
    try:
        score = probability(score, "server risk ranking score")
    except ValueError:
        reasons.append("missing_or_invalid_risk_score")
        score = None
    if isinstance(result, dict) and result.get("score_contract_id") != policy["score_contract_id"]:
        reasons.append("risk_score_contract_mismatch")
    if score is not None and (policy["risk_score_threshold"] is None or score > policy["risk_score_threshold"]):
        reasons.append("risk_score_above_qualified_threshold")
    stats = policy.get("qualification") or {}
    return {"decision": "review" if reasons else "accept_for_autofill", "reason_codes": sorted(set(reasons)),
            "policy_id": policy["policy_id"], "policy_sha256": policy["policy_sha256"],
            "binding": binding, "risk_ranking_score": score,
            "qualified_group_error_upper_bound": stats.get("group_error_upper_bound"),
            "qualification_accepted_groups": stats.get("accepted_groups", 0),
            "estimated_individual_error_probability": None,
            "risk_unit": "at least one wrong critical accepted value/role per original group",
            "service_approval": False}


class RiskPolicyStore:
    """Server-owned policy lifecycle. Clients pass policy IDs, never policy approval booleans."""
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        with connect(self.db_path) as con:
            init_events(con)
            con.execute("""CREATE TABLE IF NOT EXISTS risk_policies (
                policy_id TEXT PRIMARY KEY, policy_json TEXT NOT NULL,
                policy_sha256 TEXT NOT NULL, binding_sha256 TEXT NOT NULL,
                state TEXT NOT NULL, created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL
            )""")

    def save_candidate(self, policy: dict, *, actor_id: str, actor_scopes: Iterable[str]) -> dict:
        require_scope(actor_scopes, "policy:propose")
        require_id(actor_id, "actor_id")
        policy = verify_policy(policy)
        with connect(self.db_path) as con, transaction(con):
            old = con.execute("SELECT * FROM risk_policies WHERE policy_id=?", (policy["policy_id"],)).fetchone()
            if old:
                if old["policy_sha256"] != policy["policy_sha256"]:
                    raise ValueError("Policy ID collision")
                return {"policy_id": policy["policy_id"], "state": old["state"]}
            now = utc_now()
            con.execute("INSERT INTO risk_policies VALUES(?,?,?,?,?,?,?)", (policy["policy_id"], canonical(policy), policy["policy_sha256"], digest(policy["binding"]), "candidate", now, now))
            append_event(con, "policy_propose_" + policy["policy_sha256"], "policy_lifecycle", {"policy_id": policy["policy_id"], "action": "propose", "actor_id": actor_id})
        return {"policy_id": policy["policy_id"], "state": "candidate"}

    def _transition(self, policy_id, action, actor_id, actor_scopes):
        require_scope(actor_scopes, "policy:" + action)
        require_id(actor_id, "actor_id")
        with connect(self.db_path) as con, transaction(con):
            row = con.execute("SELECT * FROM risk_policies WHERE policy_id=?", (policy_id,)).fetchone()
            if not row:
                raise KeyError("Unknown server policy")
            policy = _stored_policy(row)
            desired = {"approve": "approved", "activate": "active", "revoke": "revoked"}[action]
            if row["state"] == desired:
                return {"policy_id": policy_id, "state": desired}
            allowed = {"approve": {"candidate"}, "activate": {"approved", "superseded"}, "revoke": {"candidate", "approved", "active", "superseded"}}
            if row["state"] not in allowed[action]:
                raise ValueError("Forbidden policy lifecycle transition")
            if action in {"approve", "activate"}:
                q = policy.get("qualification") or {}
                if (policy.get("qualified_for_autoaccept") is not True or policy.get("qualification_status") != "qualified"
                        or not policy["hard_contract"]["required_fields"] or q.get("accepted_groups", 0) < policy["min_qualification_groups"]
                        or q.get("group_error_upper_bound") is None or q["group_error_upper_bound"] > policy["target_error_rate"]):
                    raise ValueError("Insufficient independently qualified evidence for activation")
                quarantine = con.execute("SELECT active FROM control_quarantines WHERE binding_sha256=?", (row["binding_sha256"],)).fetchone()
                if quarantine and quarantine["active"]:
                    raise ValueError("Binding is quarantined")
            now = utc_now()
            if action == "activate":
                con.execute("UPDATE risk_policies SET state='superseded',updated_utc=? WHERE binding_sha256=? AND state='active'", (now, row["binding_sha256"]))
            con.execute("UPDATE risk_policies SET state=?,updated_utc=? WHERE policy_id=?", (desired, now, policy_id))
            append_event(con, "policy_" + action + "_" + uuid.uuid4().hex, "policy_lifecycle", {"policy_id": policy_id, "action": action, "actor_id": actor_id})
        return {"policy_id": policy_id, "state": desired}

    def approve(self, policy_id, *, actor_id, actor_scopes):
        return self._transition(policy_id, "approve", actor_id, actor_scopes)

    def activate(self, policy_id, *, actor_id, actor_scopes):
        return self._transition(policy_id, "activate", actor_id, actor_scopes)

    def revoke(self, policy_id, *, actor_id, actor_scopes):
        return self._transition(policy_id, "revoke", actor_id, actor_scopes)

    def get(self, policy_id: str) -> dict:
        with connect(self.db_path) as con:
            row = con.execute("SELECT * FROM risk_policies WHERE policy_id=?", (policy_id,)).fetchone()
        if not row:
            raise KeyError("Unknown server policy")
        policy = _stored_policy(row)
        return {"policy": policy, "state": row["state"]}

    def apply(self, policy_id: str | None, result: dict, binding: dict) -> dict:
        binding = validate_binding(binding)
        reason = None
        policy = None
        with connect(self.db_path) as con:
            if policy_id is not None:
                row = con.execute("SELECT * FROM risk_policies WHERE policy_id=?", (policy_id,)).fetchone()
                if row and row["state"] == "active":
                    try:
                        policy = _stored_policy(row)
                    except ValueError:
                        reason = "invalid_policy_integrity"
                else:
                    reason = "policy_not_active_or_unknown"
            quarantine = con.execute("SELECT active FROM control_quarantines WHERE binding_sha256=?", (digest(binding),)).fetchone()
        result_policy = apply_document_policy(policy, result, binding)
        if reason:
            result_policy["reason_codes"].append(reason)
        if quarantine and quarantine["active"]:
            result_policy["reason_codes"].append("binding_quarantined")
        if reason or (quarantine and quarantine["active"]):
            result_policy["decision"] = "review"
            result_policy["reason_codes"] = sorted(set(result_policy["reason_codes"]))
        return result_policy
