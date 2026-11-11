"""F01: durable, compare-and-swap document revisions and dependent claims.

Evidence is immutable; conclusions may be disputed or retracted. A supported
claim only has literal evidence. It does not establish optical correctness,
document authenticity, service eligibility or calibrated automatic acceptance.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

from .common import (AuthorizationError, EvidenceError, RevisionConflict, StateError,
                     canonical, clone, digest, identifier, sha256, utc_now)
from .evidence import literal_supported, normalize_observation
from ..typed_fields import typed_supported, value_identity,all_evidence_ids


def _value_key(raw):
    return digest(value_identity(raw))


def _claims_disagree(items):
    return (len({(_value_key(x['raw_value']),x.get('party_role')) for x in items})>1
        or len({x['format_context_identity'] for x in items if x.get('format_context_identity')})>1)

AUTOMATED = {"model", "rule"}
LIVE = {"supported", "human_supported"}


def _fields(state: dict) -> tuple[dict, list[dict]]:
    groups = defaultdict(list)
    for item in state["claims"].values():
        groups[item["field_key"]].append(item)
    fields, conflicts = {}, []
    for key, items in groups.items():
        live = [x for x in items if x["status"] in LIVE]
        humans = [x for x in live if x["origin"] == "human"]
        contradictory = [x for x in items if x["status"] == "conflict"]
        if humans:
            chosen = sorted(humans, key=lambda x: x["created_revision"])[-1]
            fields[key] = {"raw_value": chosen["raw_value"], "value": chosen["raw_value"],
                           "status": "human_corrected", "origin": "human", "claim_ids": [chosen["id"]],
                           "evidence_ids": chosen["evidence_ids"], "party_role": chosen.get("party_role"),
                           "needs_review": bool(contradictory or chosen.get("evidence_conflict"))}
        elif live:
            values = {_value_key(x["raw_value"]) for x in live}
            if len(values) != 1:
                raise StateError("internal_unresolved_live_claim_conflict")
            chosen = live[-1]
            fields[key] = {"raw_value": chosen["raw_value"], "value": chosen["raw_value"],
                           "status": "supported", "origin": chosen["origin"],
                           "claim_ids": [x["id"] for x in live],
                           "evidence_ids": sorted({i for x in live for i in x["evidence_ids"]}),
                           "party_role": chosen.get("party_role"), "needs_review": bool(contradictory)}
        else:
            fields[key] = {"raw_value": None, "value": None, "status": "conflict" if contradictory else "withheld",
                           "origin": None, "claim_ids": [x["id"] for x in items],
                           "evidence_ids": sorted({i for x in items for i in x["evidence_ids"]}),
                           "party_role": None, "needs_review": True}
        if contradictory or any(x.get("evidence_conflict") and x["status"] in LIVE for x in items):
            conflicts.append({"field_key": key, "claim_ids": [x["id"] for x in items if x["status"] == "conflict" or (x.get("evidence_conflict") and x["status"] in LIVE)],
                              "reason": "contradictory_format_context" if len({x['format_context_identity'] for x in items
                                  if x.get('format_context_identity') and x['status'] in LIVE|{'conflict'}})>1
                                  else "contradictory_evidence_or_role"})
        active_checks=[c for c in state.get('visual_checks',{}).values()
            if c['field_key']==key and c['verdict']!='correct'
            and any(state['claims'].get(cid,{}).get('status') in LIVE for cid in c['claim_ids'])]
        if active_checks:
            fields[key]['needs_review']=True
            conflicts.append({'field_key':key,'claim_ids':sorted({cid for c in active_checks for cid in c['claim_ids']}),
                'check_ids':[c['id'] for c in active_checks],'reason':'blind_visual_disagreement_or_abstention'})
    return fields, conflicts


class DocumentStore:
    def __init__(self, path: str | Path, *, mode: str = "production"):
        if mode not in {"production", "research"}:
            raise StateError("unsupported_store_mode")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents(
                    document_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, state_json TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS document_revisions(
                    document_id TEXT NOT NULL, revision INTEGER NOT NULL, state_json TEXT NOT NULL,
                    state_sha256 TEXT NOT NULL, event_json TEXT NOT NULL, created_utc TEXT NOT NULL,
                    PRIMARY KEY(document_id, revision));
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def create_document(self, document_id: str, source_sha256: str, profile_id: str, metadata: dict | None = None) -> dict:
        identifier(document_id, "document_id")
        sha256(source_sha256)
        identifier(profile_id, "profile_id")
        metadata = clone(metadata or {})
        allowed = metadata.get("allowed_source_sha256", [])
        if not isinstance(allowed, list):
            raise EvidenceError("allowed_sources_must_be_list")
        for item in allowed:
            sha256(item)
        state = {"document_id": document_id, "source_sha256": source_sha256, "profile_id": profile_id,
                 "revision": 0, "metadata": metadata, "observations": {}, "claims": {},
                 "fields": {}, "conflicts": [], "needs_review": True,
                 "automatic_acceptance_authorized": False, "created_utc": utc_now()}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT state_json FROM documents WHERE document_id=?", (document_id,)).fetchone()
            if old:
                previous = json.loads(old[0])
                if any(previous[k] != state[k] for k in ("source_sha256", "profile_id", "metadata")):
                    raise StateError("document_id_already_bound_to_different_input")
                return previous
            self._persist(db, state, {"type": "document_created", "actor": "service"})
        return clone(state)

    create = create_document

    def get(self, document_id: str, revision: int | None = None) -> dict:
        identifier(document_id, "document_id")
        with self._connect() as db:
            if revision is None:
                row = db.execute("SELECT state_json FROM documents WHERE document_id=?", (document_id,)).fetchone()
            else:
                row = db.execute("SELECT state_json FROM document_revisions WHERE document_id=? AND revision=?", (document_id, revision)).fetchone()
        if row is None:
            raise StateError("document_or_revision_not_found")
        return json.loads(row[0])

    load = get

    def revisions(self, document_id: str) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT revision,state_sha256,event_json,created_utc FROM document_revisions WHERE document_id=? ORDER BY revision", (document_id,)).fetchall()
        return [{"revision": n, "state_sha256": h, "event": json.loads(e), "created_utc": t} for n, h, e, t in rows]

    @staticmethod
    def _persist(db, state, event):
        raw = canonical(state)
        db.execute("INSERT INTO documents VALUES(?,?,?) ON CONFLICT(document_id) DO UPDATE SET revision=excluded.revision,state_json=excluded.state_json",
                   (state["document_id"], state["revision"], raw))
        db.execute("INSERT INTO document_revisions VALUES(?,?,?,?,?,?)",
                   (state["document_id"], state["revision"], raw, digest(state), canonical(event), utc_now()))

    def revise(self, document_id: str, *, observations=(), claims=(), visual_checks=(), native_result=None,
               invalidate=(), expected_revision: int | None = None,
               actor: str = "engine", reason: str = "evidence_update", allow_human: bool = False) -> dict:
        identifier(document_id, "document_id")
        if not isinstance(actor, str) or not actor:
            raise AuthorizationError("actor_required")
        incoming_observations, incoming_claims, invalidated = clone(list(observations)), clone(list(claims)), list(invalidate)
        native_result=clone(native_result)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state_json FROM documents WHERE document_id=?", (document_id,)).fetchone()
            if row is None:
                raise StateError("document_not_found")
            state = json.loads(row[0])
            if any(x.get("origin") == "human" for x in incoming_claims) and type(expected_revision) is not int:
                raise RevisionConflict("human_correction_requires_expected_revision")
            if expected_revision is not None and (type(expected_revision) is not int or expected_revision != state["revision"]):
                raise RevisionConflict("stale_document_revision")
            before = canonical(state)
            revision = state["revision"] + 1
            allowed = {state["source_sha256"], *state["metadata"].get("allowed_source_sha256", [])}
            native_verified_registry=None
            if native_result is not None:
                from ..native.mchd import verify_native_result,PROFILE_ID
                if (state['profile_id']!=PROFILE_ID or native_result.get('source_sha256')!=state['source_sha256']
                    or not verify_native_result(native_result,state['metadata'].get('source_path'))):
                    raise EvidenceError('native_result_source_replay_failed')
                summary={k:clone(v) for k,v in native_result.items()
                    if k not in {'observations','field_candidates','native_claims','pages'}}
                if state.get('native_ingestion') not in (None,summary):raise EvidenceError('immutable_native_ingestion_changed')
                state['native_ingestion']=summary
                native_verified_registry={o['id']:o for o in native_result['observations']}
            for obs in incoming_observations:
                normalized = normalize_observation(obs, allowed, mode=self.mode,
                    native_source=state["metadata"].get("source_path"),native_verified_registry=native_verified_registry)
                oid = normalized["id"]
                if oid in state["claims"]:
                    raise EvidenceError("id_collision_between_claim_and_observation")
                old = state["observations"].get(oid)
                if old:
                    if {k:v for k,v in old.items() if k not in {"active","created_revision"}} != normalized:
                        raise EvidenceError("immutable_observation_changed")
                else:
                    state["observations"][oid] = {**normalized, "active": True, "created_revision": revision}

            for raw in incoming_claims:
                claim = self._normalize_claim(raw, state, allow_human=allow_human)
                cid = claim["id"]
                if cid in state["observations"]:
                    raise EvidenceError("id_collision_between_claim_and_observation")
                old = state["claims"].get(cid)
                if old:
                    original = {k: old[k] for k in claim}
                    if original != claim:
                        raise EvidenceError("immutable_claim_changed")
                    continue
                claim.update(status="human_supported" if claim["origin"] == "human" else "supported",
                             created_revision=revision, status_reason="literal_evidence" if claim["origin"] != "human" else "authorized_human_correction")
                if claim["origin"] == "human":
                    # Explicit authorized correction resolves older candidates of this field.
                    for previous in state["claims"].values():
                        if previous["field_key"] == claim["field_key"] and previous["status"] not in {"retracted", "superseded"}:
                            previous.update(status="superseded", status_reason="authorized_human_correction", changed_revision=revision)
                state["claims"][cid] = claim

            self._assert_acyclic(state["claims"])
            for item in visual_checks:
                checked=self._visual_check(item,state)
                old=state.setdefault('visual_checks',{}).get(checked['id'])
                if old is not None and old!=checked:raise EvidenceError('immutable_visual_check_changed')
                state['visual_checks'][checked['id']]=checked
            for node_id in invalidated:
                identifier(node_id, "invalidation_id")
                if node_id in state["observations"]:
                    state["observations"][node_id]["active"] = False
                elif node_id in state["claims"]:
                    item = state["claims"][node_id]
                    if item["origin"] == "human":
                        item["evidence_conflict"] = True
                    else:
                        item.update(status="retracted", status_reason="explicit_invalidation", changed_revision=revision)
                else:
                    raise EvidenceError("unknown_invalidation_id")
            self._resolve_conflicts(state, revision)
            self._propagate(state, revision)
            self._retire_conflicts_without_active_opposition(state, revision)
            self._propagate(state, revision)
            state["fields"], state["conflicts"] = _fields(state)
            # Core is not an acceptance authority. F08/server policy decides above it.
            state["needs_review"] = True
            if canonical(state) == before:
                return clone(state)
            state["revision"] = revision
            state["updated_utc"] = utc_now()
            self._persist(db, state, {"type": "human_correction" if any(x.get("origin") == "human" for x in incoming_claims) else "document_revised",
                                      "actor": actor, "reason": reason, "observation_ids": [x["id"] for x in incoming_observations],
                                      "claim_ids": [x.get("id") for x in incoming_claims], "invalidated_ids": invalidated})
        return clone(state)

    @staticmethod
    def _visual_check(item,state):
        from ..inference.vision import compare_visual_capture
        if not isinstance(item,dict) or set(item)!={'id','field_key','claim_ids','capture_id','expected_observation_id'}:
            raise EvidenceError('visual_check_requires_closed_source_references')
        check=clone(item);identifier(check['id'],'visual_check_id')
        refs=check['claim_ids']
        if not isinstance(refs,list) or not refs or len(set(refs))!=len(refs):raise EvidenceError('visual_check_requires_claims')
        original=state['observations'].get(check['expected_observation_id'],{})
        captured=state['observations'].get(check['capture_id'],{})
        if not all(o.get('active') is True and o.get('completed') is True for o in (original,captured)):
            raise EvidenceError('visual_check_evidence_not_active_complete')
        if captured.get('purpose')!='blind_optical_capture':raise EvidenceError('visual_check_not_blind_capture')
        for key in ('source_sha256','page','page_id','source_size'):
            if original.get(key)!=captured.get(key):raise EvidenceError('visual_check_source_geometry_mismatch')
        geometry='quad' if original.get('quad') is not None else 'bbox'
        if original.get(geometry) is None or original.get(geometry)!=captured.get(geometry):
            raise EvidenceError('visual_check_requires_identical_roi')
        for cid in refs:
            claim=state['claims'].get(cid,{})
            if claim.get('field_key')!=check['field_key'] or check['expected_observation_id'] not in claim.get('evidence_ids',[]):
                raise EvidenceError('visual_check_not_bound_to_claim')
        sealed={k:v for k,v in captured.items() if k not in {'active','created_revision'}}
        check.update(compare_visual_capture(sealed,original['text']))
        check['comparison_scope']='complete_original_observation_literal'
        return check

    @staticmethod
    def _normalize_claim(raw, state, *, allow_human):
        if not isinstance(raw, dict):
            raise EvidenceError("claim_must_be_object")
        claim = clone(raw)
        if 'format_context_identity' in claim:raise EvidenceError('caller_cannot_set_format_context_identity')
        claim.setdefault("origin", "model")
        if claim["origin"] not in AUTOMATED | {"human"}:
            raise EvidenceError("unsupported_claim_origin")
        if claim["origin"] == "human" and allow_human is not True:
            raise AuthorizationError("human_transition_not_authorized")
        identifier(claim.get("field_key"), "field_key")
        if state["metadata"].get("field_keys") and claim["field_key"] not in state["metadata"]["field_keys"]:
            raise EvidenceError("field_not_in_profile")
        raw_value=claim.get("raw_value")
        if not isinstance(raw_value, (str,dict)) or not raw_value:
            raise EvidenceError("raw_value_must_be_nonempty_literal_or_typed_value")
        if len(canonical(raw_value))>250000:
            raise EvidenceError("raw_value_size_limit")
        evidence_ids = claim.get("evidence_ids")
        if not isinstance(evidence_ids, list) or not evidence_ids or len(evidence_ids) != len(set(evidence_ids)):
            raise EvidenceError("claim_requires_unique_evidence_ids")
        evidence = []
        for oid in evidence_ids:
            item = state["observations"].get(oid)
            if item is None or not item["active"]:
                raise EvidenceError("claim_evidence_missing_or_inactive")
            if item["completed"] is not True:
                raise EvidenceError("incomplete_observation_cannot_support_claim")
            evidence.append(item)
        role_ids = claim.get("role_evidence_ids", [])
        if not isinstance(role_ids, list) or any(not isinstance(x, str) for x in role_ids) or len(role_ids) != len(set(role_ids)):
            raise EvidenceError("role_evidence_must_be_unique_ids")
        for oid in role_ids:
            item = state["observations"].get(oid)
            if item is None or not item["active"] or item["completed"] is not True:
                raise EvidenceError("role_evidence_missing_inactive_or_incomplete")
        role_spans = claim.get("role_spans", [])
        if not isinstance(role_spans, list):
            raise EvidenceError("role_spans_must_be_list")
        for span in role_spans:
            if not isinstance(span, dict) or span.get("observation_id") not in role_ids:
                raise EvidenceError("role_span_not_bound_to_role_evidence")
            start, end = span.get("start"), span.get("end")
            text = state["observations"][span["observation_id"]]["text"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise EvidenceError("invalid_role_span_offsets")
        if isinstance(raw_value,dict) and not typed_supported(raw_value,evidence):
            raise EvidenceError("typed_value_not_supported_in_evidence")
        if isinstance(raw_value,dict) and any(i['party_role']!=claim.get('party_role') for i in raw_value['items']):
            raise EvidenceError('typed_claim_party_role_mismatch')
        if isinstance(raw_value,dict):
            if (all(i['association']['kind']!='validated_native_xml' for i in raw_value['items'])
                and set(evidence_ids)!=set(all_evidence_ids(raw_value))):
                raise EvidenceError('typed_optical_evidence_contains_unbound_sources')
            if raw_value['contract_id']=='machine-identifiers-v1':
                from ..machine_ids import derive_machine_context
                context=derive_machine_context(raw_value,evidence)
                if context is not None:claim['format_context_identity']=digest(context)
        if claim["origin"] != "human" and isinstance(raw_value,str) and not literal_supported(raw_value, evidence, claim.get("spans")):
            raise EvidenceError("raw_value_not_literal_in_evidence")
        depends = claim.setdefault("depends_on", [])
        if not isinstance(depends, list) or any(not isinstance(x, str) for x in depends):
            raise EvidenceError("depends_on_must_be_ids")
        if len(depends) != len(set(depends)):
            raise EvidenceError("duplicate_dependencies")
        for item in depends:
            if item not in state["claims"]:
                raise EvidenceError("unknown_claim_dependency")
        claim.setdefault("party_role", None)
        claim.setdefault("id", "claim:" + digest({k: v for k, v in claim.items() if k != "id"})[:32])
        identifier(claim["id"], "claim_id")
        # Caller-supplied acceptance/state flags are not an authority.
        for forbidden in ("status", "created_revision", "changed_revision", "evidence_conflict", "status_reason", "validated", "can_auto_accept"):
            if forbidden in claim:
                raise EvidenceError("caller_cannot_set_" + forbidden)
        return claim

    @staticmethod
    def _assert_acyclic(claims):
        visiting, done = set(), set()
        def visit(cid):
            if cid in visiting:
                raise EvidenceError("claim_dependency_cycle")
            if cid in done:
                return
            visiting.add(cid)
            for dep in claims[cid]["depends_on"]:
                visit(dep)
            visiting.remove(cid)
            done.add(cid)
        for cid in claims:
            visit(cid)

    @staticmethod
    def _resolve_conflicts(state, revision):
        groups = defaultdict(list)
        for claim in state["claims"].values():
            if claim["status"] in LIVE:
                groups[claim["field_key"]].append(claim)
        for items in groups.values():
            humans = [x for x in items if x["origin"] == "human"]
            if _claims_disagree(items):
                for item in items:
                    if item["origin"] == "human":
                        item["evidence_conflict"] = True
                    else:
                        item.update(status="conflict", status_reason="different_literal_or_party_role", changed_revision=revision)

    @staticmethod
    def _propagate(state, revision):
        changed = True
        while changed:
            changed = False
            for item in state["claims"].values():
                bad_evidence = any(not state["observations"][oid]["active"] for oid in [*item["evidence_ids"], *item.get("role_evidence_ids", [])])
                bad_dependency = any(state["claims"][cid]["status"] not in LIVE for cid in item["depends_on"])
                if bad_evidence or bad_dependency:
                    if item["origin"] == "human":
                        item["evidence_conflict"] = True
                    elif item["status"] in LIVE | {"conflict"}:
                        item.update(status="retracted", status_reason="inactive_evidence_or_dependency", changed_revision=revision)
                        changed = True

    @staticmethod
    def _retire_conflicts_without_active_opposition(state, revision):
        """Withdraw obsolete conflicts without silently resurrecting a claim.

        When opposing evidence is explicitly withdrawn, its conflicted claim is
        retracted by propagation. The remaining historic conflicted claim is
        retired too: resolving the field requires a new source-backed claim or
        an authorized human correction. Historic conflicts must not permanently
        poison a later independent reading of the field.
        """
        groups = defaultdict(list)
        for item in state['claims'].values():
            if item['status'] in LIVE | {'conflict'}:
                groups[item['field_key']].append(item)
        for items in groups.values():
            if _claims_disagree(items):continue
            for item in items:
                if item['status']=='conflict' and item['origin']!='human':
                    item.update(status='retracted',status_reason='resolved_conflict_requires_new_claim_or_proof',
                                changed_revision=revision)

    def add_observation(self, document_id, observation, *, expected_revision=None):
        return self.revise(document_id, observations=[observation], expected_revision=expected_revision)

    def add_claim(self, document_id, claim, *, expected_revision=None):
        return self.revise(document_id, claims=[claim], expected_revision=expected_revision)

    def human_correct(self, document_id: str, field_key: str, raw_value: str | dict, evidence_ids: list[str], *, actor: str,
                      expected_revision: int, authorized: bool = False, party_role: str | None = None) -> dict:
        if authorized is not True:
            raise AuthorizationError("human_correction_not_authorized")
        if type(expected_revision) is not int:
            raise RevisionConflict("human_correction_requires_expected_revision")
        claim = {"field_key": field_key, "raw_value": raw_value, "evidence_ids": evidence_ids,
                 "depends_on": [], "party_role": party_role, "origin": "human"}
        claim["id"] = "human:" + digest({"claim": claim, "actor": actor, "revision": expected_revision})[:32]
        return self.revise(document_id, claims=[claim], expected_revision=expected_revision,
                           actor=actor, reason="authorized_human_correction", allow_human=True)
