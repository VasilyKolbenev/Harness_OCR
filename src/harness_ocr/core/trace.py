"""F02: durable stage records and explicit deterministic replay.

This stores pipeline inputs/outputs, not hidden model reasoning. Model stages
cannot be replayed by the pure replay helper; a new inference is a new trial.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .common import StateError, canonical, clone, digest, identifier, utc_now

FORBIDDEN_TRACE_KEYS = {"chain_of_thought", "hidden_reasoning", "reasoning_content"}


def _check_keys(value):
    if isinstance(value, dict):
        if FORBIDDEN_TRACE_KEYS.intersection(value):
            raise StateError("hidden_reasoning_not_part_of_trace_contract")
        for item in value.values():
            _check_keys(item)
    elif isinstance(value, list):
        for item in value:
            _check_keys(item)


def replay(record: dict, function, *, config: dict | None = None, allow_changed_config: bool = False) -> dict:
    if record.get("stage_kind") != "pure" or record.get("deterministic") is not True:
        raise StateError("only_explicit_pure_deterministic_stages_can_replay")
    if record.get("status") != "completed":
        raise StateError("cannot_replay_incomplete_stage_as_complete")
    if any(digest(record[name]) != record[name + "_sha256"] for name in ("inputs", "outputs", "config")):
        raise StateError("trace_payload_hash_mismatch")
    chosen = clone(record["config"] if config is None else config)
    config_changed = digest(chosen) != record["config_sha256"]
    if config_changed and not allow_changed_config:
        raise StateError("changed_config_requires_explicit_new_branch")
    output = function(clone(record["inputs"]), chosen)
    return {"replay_of": record["id"], "stage": record["stage"], "outputs": clone(output),
            "outputs_sha256": digest(output), "matches_recorded_output": digest(output) == record["outputs_sha256"],
            "config_changed": config_changed, "model_calls": 0,
            "note": "Caller must supply a pure function; no backend/model is imported or invoked by this helper."}


class TraceRecorder:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS stage_trace(seq INTEGER PRIMARY KEY,id TEXT UNIQUE NOT NULL,payload TEXT NOT NULL,record_sha256 TEXT NOT NULL)")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def record(self, stage: str, inputs, outputs, config: dict, *, document_id: str | None = None,
               trial_id: str | None = None, status: str = "completed", stop_reason: str | None = None,
               deterministic: bool = False, stage_kind: str = "external", parent_record_id: str | None = None,
               record_id: str | None = None, cost: dict | None = None) -> dict:
        identifier(stage, "stage")
        if stage_kind not in {"pure", "model", "external"}:
            raise StateError("unsupported_stage_kind")
        if deterministic and stage_kind != "pure":
            raise StateError("only_pure_stage_may_claim_deterministic_replay")
        if status not in {"completed", "failed", "canceled", "incomplete", "budget_exhausted", "needs_review"}:
            raise StateError("unsupported_trace_status")
        for value in (inputs, outputs, config, cost):
            _check_keys(value)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT seq,record_sha256 FROM stage_trace ORDER BY seq DESC LIMIT 1").fetchone()
            seq = 1 if previous is None else previous[0] + 1
            payload = {"id": record_id or "trace:" + digest({"stage": stage, "seq": seq, "created": utc_now()})[:32],
                       "seq": seq, "stage": stage, "stage_kind": stage_kind, "deterministic": bool(deterministic),
                       "document_id": document_id, "trial_id": trial_id, "status": status, "stop_reason": stop_reason,
                       "inputs": clone(inputs), "outputs": clone(outputs), "config": clone(config),
                       "inputs_sha256": digest(inputs), "outputs_sha256": digest(outputs), "config_sha256": digest(config),
                       "cost": clone(cost or {}), "parent_record_id": parent_record_id,
                       "previous_record_sha256": previous[1] if previous else None, "created_utc": utc_now()}
            identifier(payload["id"], "record_id")
            db.execute("INSERT INTO stage_trace VALUES(?,?,?,?)", (seq, payload["id"], canonical(payload), digest(payload)))
        return {**payload, "record_sha256": digest(payload)}

    def records(self, *, document_id: str | None = None) -> list[dict]:
        with self._connect() as db:
            rows = db.execute("SELECT payload,record_sha256 FROM stage_trace ORDER BY seq").fetchall()
        result = [{**json.loads(p), "record_sha256": h} for p, h in rows]
        return result if document_id is None else [x for x in result if x["document_id"] == document_id]

    def get(self, record_id: str) -> dict:
        with self._connect() as db:
            row = db.execute("SELECT payload,record_sha256 FROM stage_trace WHERE id=?", (record_id,)).fetchone()
        if row is None:
            raise StateError("trace_record_not_found")
        return {**json.loads(row[0]), "record_sha256": row[1]}

    def audit(self) -> dict:
        previous = None
        for expected_seq, record in enumerate(self.records(), 1):
            claimed = record.pop("record_sha256")
            if record["seq"] != expected_seq or record["previous_record_sha256"] != previous or digest(record) != claimed:
                raise StateError("trace_hash_chain_corrupted")
            for name in ("inputs", "outputs", "config"):
                if digest(record[name]) != record[name + "_sha256"]:
                    raise StateError("trace_payload_hash_mismatch")
            previous = claimed
        return {"status": "PASS", "records": len(self.records()), "last_record_sha256": previous}

    def replay(self, record_or_id, function, *, config=None, allow_changed_config=False, trial_id=None):
        record = self.get(record_or_id) if isinstance(record_or_id, str) else clone(record_or_id)
        result = replay(record, function, config=config, allow_changed_config=allow_changed_config)
        chosen = record["config"] if config is None else config
        branch = self.record(record["stage"], record["inputs"], result["outputs"], chosen,
                             document_id=record["document_id"], trial_id=trial_id,
                             stage_kind="pure", deterministic=True, parent_record_id=record["id"])
        return {**result, "branch_record_id": branch["id"]}

    def export_jsonl(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text("".join(canonical(r) + "\n" for r in self.records()), encoding="utf-8")
        temporary.replace(target)
