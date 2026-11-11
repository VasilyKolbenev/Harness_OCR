"""F06: durable probability sampling, audit outcomes and fail-safe drift alerts.

Sampling is keyed by the original document/group. Repeated crops/revisions do not
become independent random draws. Monitoring may restrict automation; it never
promotes a profile, strategy or acceptance policy.
"""
from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
import secrets
from typing import Iterable

from .common import (
    append_event, canonical, connect, digest, init_events, probability,
    read_events, require_id, require_scope, transaction, utc_now, validate_binding,
)


class AuditSampler:
    def __init__(self, probabilities: dict[str, float] | float, seed: str, policy_id: str = "audit-v1"):
        require_id(seed, "private sampling seed")
        self._seed = seed.encode("utf-8")
        self.policy_id = require_id(policy_id, "policy_id")
        if isinstance(probabilities, dict):
            if not probabilities:
                raise ValueError("At least one stratum probability is required")
            self.probabilities = {require_id(k, "stratum"): probability(v) for k, v in probabilities.items()}
        else:
            self.probabilities = {"default": probability(probabilities)}
        self.seed_id = hashlib.sha256(self._seed).hexdigest()[:16]
        self.config_sha256 = digest({"policy_id": self.policy_id, "probabilities": self.probabilities, "seed_id": self.seed_id})

    def decide(self, unit_id: str, stratum: str = "default") -> dict:
        require_id(unit_id, "sampling unit_id")
        require_id(stratum, "stratum")
        if stratum not in self.probabilities and "default" not in self.probabilities:
            raise ValueError("Unconfigured stratum")
        p = self.probabilities.get(stratum, self.probabilities.get("default"))
        # The draw does not change when a caller changes a revision, crop or
        # stratum. The server is responsible for the predeclared stratum itself.
        message = canonical({"policy_id": self.policy_id, "unit_id": unit_id}).encode("utf-8")
        integer = int.from_bytes(hmac.new(self._seed, message, hashlib.sha256).digest(), "big")
        selected = integer < int(p * (1 << 256))
        return {"sampling_unit_id": unit_id, "sampling_unit": "original_document_group", "stratum": stratum,
                "selected": selected, "inclusion_probability": p, "policy_id": self.policy_id,
                "config_sha256": self.config_sha256, "seed_id": self.seed_id,
                "independent_crops_assumed": False}


class AuditLog:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        with connect(self.db_path) as con:
            init_events(con)

    def append(self, event_id: str, event: dict, event_type: str = "audit_event") -> dict:
        if not isinstance(event, dict):
            raise ValueError("event must be a dictionary")
        with connect(self.db_path) as con, transaction(con):
            return append_event(con, event_id, event_type, event)

    def events(self, event_type: str | None = None) -> list[dict]:
        with connect(self.db_path) as con:
            return read_events(con, event_type)

    def server_sampling_seed(self, name: str = "default") -> str:
        """Server-only secret, durable across restarts and never copied to events."""
        require_id(name, "sampler seed name")
        with connect(self.db_path) as con, transaction(con):
            con.execute("CREATE TABLE IF NOT EXISTS audit_sampling_seeds(name TEXT PRIMARY KEY,seed TEXT NOT NULL)")
            row = con.execute("SELECT seed FROM audit_sampling_seeds WHERE name=?", (name,)).fetchone()
            if not row:
                seed = secrets.token_hex(32)
                con.execute("INSERT INTO audit_sampling_seeds VALUES(?,?)", (name, seed))
            else:
                seed = row["seed"]
        return seed


class AuditController:
    def __init__(self, sampler: AuditSampler, log: AuditLog):
        self.sampler, self.log = sampler, log

    def plan(self, document_id: str, revision: int, binding: dict, *, group_id: str | None = None,
             stratum: str = "default", accepted: bool = True, acceptance_context: str | None = None) -> dict:
        require_id(document_id, "document_id")
        if type(revision) is not int or revision < 0:
            raise ValueError("revision must be a nonnegative integer")
        if type(accepted) is not bool:
            raise ValueError("accepted must be bool")
        binding = validate_binding(binding)
        if acceptance_context is not None:
            require_id(acceptance_context, "server acceptance_context")
        unit = require_id(group_id or document_id, "group_id")
        decision = self.sampler.decide(unit, stratum)
        task_id = "audit_" + digest({"document_id": document_id, "revision": revision,
                                     "binding": binding, "sampler": self.sampler.config_sha256,
                                     "acceptance_context": acceptance_context})[:32]
        payload = {**decision, "task_id": task_id, "document_id": document_id, "revision": revision,
                   "group_id": unit, "binding": binding, "acceptance_context": acceptance_context,
                   "eligible_accepted_result": accepted,
                   "selected": bool(accepted and decision["selected"])}
        self.log.append(task_id, payload, "audit_selection")
        return payload

    def record_outcome(self, task_id: str, *, critical_error: bool, actor_id: str,
                       actor_scopes: Iterable[str], evidence_ids: list[str],
                       correction_revision: int | None = None) -> dict:
        require_scope(actor_scopes, "audit:write")
        require_id(actor_id, "actor_id")
        if type(critical_error) is not bool:
            raise ValueError("critical_error must be an independently reviewed bool")
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise ValueError("Reviewed audit outcomes require evidence references")
        for ref in evidence_ids:
            require_id(ref, "audit evidence ID")
        if correction_revision is not None and (type(correction_revision) is not int or correction_revision < 0):
            raise ValueError("correction_revision must be a nonnegative integer")
        with connect(self.log.db_path) as con, transaction(con):
            selections = [e["payload"] for e in read_events(con, "audit_selection") if e["payload"]["task_id"] == task_id]
            if len(selections) != 1 or not selections[0]["selected"]:
                raise ValueError("Outcome must reference one selected audit task")
            selection = selections[0]
            outcome = {"task_id": task_id, "document_id": selection["document_id"],
                       "group_id": selection["group_id"], "inclusion_probability": selection["inclusion_probability"],
                       "critical_error": critical_error, "actor_id": actor_id,
                       "evidence_ids": list(evidence_ids), "correction_revision": correction_revision,
                       "becomes_training_gold": False}
            append_event(con, "outcome_" + task_id, "audit_outcome", outcome)
            if critical_error:
                # Confirmation and the safety action commit together. A crash
                # cannot leave a logged critical error with automation enabled.
                # The per-task event also prevents a replay from undoing a
                # later, explicitly authorized quarantine clearance.
                event_id="audit_quarantine_"+task_id
                prior=con.execute("SELECT * FROM control_events WHERE event_id=?",(event_id,)).fetchone()
                if prior:
                    if prior['event_type']!='audit_quarantine' or digest(json.loads(prior['payload_json']))!=prior['payload_sha256']:
                        raise ValueError('Audit quarantine event integrity check failed')
                else:
                    actual=validate_binding(selection['binding']);key=digest(actual)
                    previous=con.execute("SELECT active,reason_json FROM control_quarantines WHERE binding_sha256=?",(key,)).fetchone()
                    old_reasons=json.loads(previous['reason_json']) if previous and previous['active'] else []
                    reasons=sorted(set(old_reasons+['confirmed_critical_error']))
                    con.execute("INSERT INTO control_quarantines VALUES(?,?,?,?,?) ON CONFLICT(binding_sha256) DO UPDATE SET active=1,reason_json=excluded.reason_json,updated_utc=excluded.updated_utc",
                                (key,canonical(actual),1,canonical(reasons),utc_now()))
                    append_event(con,event_id,'audit_quarantine',{
                        'task_id':task_id,'binding':actual,'binding_sha256':key,
                        'reason_codes':reasons,'action':'review_only','actor_id':actor_id,
                        'outcome_event_id':'outcome_'+task_id,'model_or_policy_promoted':False})
        return outcome

    def summary(self) -> dict:
        selected = [e["payload"] for e in self.log.events("audit_selection") if e["payload"]["selected"]]
        outcomes = [e["payload"] for e in self.log.events("audit_outcome")]
        selected_ids = {p["task_id"] for p in selected}
        outcomes = [p for p in outcomes if p["task_id"] in selected_ids]
        missing = selected_ids - {p["task_id"] for p in outcomes}
        weight = sum(1 / p["inclusion_probability"] for p in outcomes)
        error_weight = sum(1 / p["inclusion_probability"] for p in outcomes if p["critical_error"])
        return {"selected_tasks": len(selected), "reviewed_tasks": len(outcomes), "missing_followup": len(missing),
                "independent_original_groups_reviewed": len({p["group_id"] for p in outcomes}),
                "weighted_error_rate_on_reviewed_tasks": error_weight / weight if weight else None,
                "population_estimate_status": "incomplete_audit_followup" if missing else ("descriptive_only" if outcomes else "not_measured"),
                "confidence_interval": None, "correlated_revisions_counted_as_independent": False}


class DriftMonitor:
    """Version-specific rolling checks. Quarantine can only reduce automation."""
    def __init__(self, log: AuditLog, baseline_binding: dict, *, baseline_acceptance_rate: float,
                 max_acceptance_drop: float = 0.2, window_size: int = 50, min_observations: int = 20):
        self.log = log
        self.binding = validate_binding(baseline_binding)
        self.baseline_acceptance_rate = probability(baseline_acceptance_rate)
        self.max_acceptance_drop = probability(max_acceptance_drop)
        if type(window_size) is not int or type(min_observations) is not int or not 1 <= min_observations <= window_size:
            raise ValueError("Require1 <= min_observations <= window_size")
        self.window_size, self.min_observations = window_size, min_observations
        self.monitor_sha256 = digest({"binding": self.binding, "baseline_acceptance_rate": self.baseline_acceptance_rate,
                                      "max_acceptance_drop": self.max_acceptance_drop, "window_size": window_size,
                                      "min_observations": min_observations})

    def observe(self, document_id: str, binding: dict, *, accepted: bool,
                reviewed_critical_error: bool | None = None, event_id: str | None = None) -> dict:
        require_id(document_id, "document_id")
        actual = validate_binding(binding)
        if type(accepted) is not bool or (reviewed_critical_error is not None and type(reviewed_critical_error) is not bool):
            raise ValueError("accepted/error must be explicit booleans")
        payload = {"document_id": document_id, "binding": actual, "accepted": accepted,
                   "reviewed_critical_error": reviewed_critical_error, "monitor_sha256": self.monitor_sha256}
        ident = event_id or "drift_" + digest({"document_id": document_id, "binding": actual, "monitor": self.monitor_sha256})
        with connect(self.log.db_path) as con, transaction(con):
            append_event(con, ident, "drift_observation", payload)
            relevant = [e for e in read_events(con, "drift_observation")
                        if e["payload"]["binding"] == actual and e["payload"]["monitor_sha256"] == self.monitor_sha256]
            # One latest observation per original ID, not a confidence boost from
            # repeated requests or revisions of the same input.
            latest = {}
            for event in relevant:
                ident = event["payload"]["document_id"]
                latest.pop(ident, None)
                latest[ident] = event["payload"]
            window = list(latest.values())[-self.window_size:]
            reasons = []
            if actual != self.binding:
                reasons.append("unqualified_binding")
            if any(item["reviewed_critical_error"] is True for item in window):
                reasons.append("confirmed_critical_error")
            rate = sum(item["accepted"] for item in window) / len(window) if window else None
            if len(window) >= self.min_observations and rate < self.baseline_acceptance_rate - self.max_acceptance_drop:
                reasons.append("acceptance_rate_drop")
            key = digest(actual)
            previous = con.execute("SELECT active,reason_json FROM control_quarantines WHERE binding_sha256=?", (key,)).fetchone()
            if reasons:
                old_reasons = json.loads(previous["reason_json"]) if previous and previous["active"] else []
                reasons = sorted(set(old_reasons + reasons))
                con.execute("INSERT INTO control_quarantines VALUES(?,?,?,?,?) ON CONFLICT(binding_sha256) DO UPDATE SET active=1,reason_json=excluded.reason_json,updated_utc=excluded.updated_utc",
                            (key, canonical(actual), 1, canonical(reasons), utc_now()))
            active = bool(reasons or (previous and previous["active"]))
            if not reasons and active:
                reasons = json.loads(previous["reason_json"])
        return {"quarantined": active, "action": "review_only" if active else "no_change", "reason_codes": reasons,
                "observed_original_documents": len(window), "observed_acceptance_rate": rate,
                "error_rate": None, "model_or_policy_promoted": False}

    def is_quarantined(self, binding: dict) -> bool:
        actual = validate_binding(binding)
        with connect(self.log.db_path) as con:
            row = con.execute("SELECT active FROM control_quarantines WHERE binding_sha256=?", (digest(actual),)).fetchone()
        return bool(row and row["active"])

    def clear_quarantine(self, binding: dict, *, actor_id: str, actor_scopes: Iterable[str], reason: str) -> None:
        require_scope(actor_scopes, "policy:quarantine:clear")
        require_id(actor_id, "actor_id")
        require_id(reason, "clearance reason")
        actual = validate_binding(binding)
        payload = {"binding": actual, "actor_id": actor_id, "reason": reason, "requires_new_qualification_if_binding_changed": True}
        with connect(self.log.db_path) as con, transaction(con):
            append_event(con, "clear_" + digest(payload), "quarantine_clearance", payload)
            con.execute("UPDATE control_quarantines SET active=0,updated_utc=? WHERE binding_sha256=?", (utc_now(), digest(actual)))
