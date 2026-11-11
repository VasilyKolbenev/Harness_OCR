"""F07: reviewed strategy cards, not a memory of citizens' document values.

The executable payload is a deliberately small declarative DSL. It contains field
keys and relative reading actions, never free-text answers, arbitrary URLs, model
prompts, Python, document IDs or personal identifier literals. Authorization is
supplied by the trusted server, not by an LLM's proposal.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable
import uuid

from .common import append_event, canonical, connect, digest, init_events, require_id, require_scope, transaction, utc_now


TOKEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]{0,127}$")
VERSION = re.compile(r"^(?:v)?[0-9]+(?:\.[0-9]+){0,3}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ACTIONS = {
    "prefer_same_cell": {},
    "require_role_label": {},
    "review_on_conflict": {},
    "preserve_source_punctuation": {},
    "avoid_identical_reread": {},
    "join_adjacent_spans": {"max_spans", "order"},
    "reread": {"max_rounds", "max_regions", "scale_policy", "reason"},
}
FINDINGS = {"no_gain", "regression", "higher_cost", "ambiguous", "not_observable"}


def _token(value, name):
    if not isinstance(value, str) or not TOKEN.fullmatch(value):
        raise ValueError(f"{name} must be a controlled ASCII identifier, not literal document text")
    return value


def _version(value, name):
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise ValueError(f"{name} must be an explicit numeric version")
    return value


def _tokens(value, name, *, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty) or len(value) > 100:
        raise ValueError(f"{name} must be a bounded identifier list")
    items = [_token(item, name) for item in value]
    if len(items) != len(set(items)):
        raise ValueError(f"{name} must not contain duplicates")
    return sorted(items)


def validate_card(card: dict) -> dict:
    allowed = {"card_id", "version", "scope", "directives", "evidence_sha256", "negative_findings"}
    if not isinstance(card, dict) or set(card) != allowed:
        raise ValueError("Strategy cards accept only the declarative schema; free-text/personal-value fields are forbidden")
    result = {"card_id": _token(card["card_id"], "card_id"), "version": _version(card["version"], "version")}
    scope = card["scope"]
    if not isinstance(scope, dict) or set(scope) != {"profile_id", "profile_version", "service_ids", "field_keys"}:
        raise ValueError("An exact versioned scope is required")
    result["scope"] = {"profile_id": _token(scope["profile_id"], "profile_id"),
                       "profile_version": _version(scope["profile_version"], "profile_version"),
                       "service_ids": _tokens(scope["service_ids"], "service_ids", allow_empty=True),
                       "field_keys": _tokens(scope["field_keys"], "field_keys")}
    refs = card["evidence_sha256"]
    if not isinstance(refs, list) or not refs or len(refs) > 100 or any(not isinstance(x, str) or not SHA256.fullmatch(x) for x in refs):
        raise ValueError("Strategy proposals require SHA256 references to reviewed experiments")
    result["evidence_sha256"] = sorted(set(refs))
    directives = card["directives"]
    if not isinstance(directives, list) or not 1 <= len(directives) <= 20:
        raise ValueError("A strategy needs1–20 bounded directives")
    result["directives"] = []
    for directive in directives:
        if not isinstance(directive, dict) or set(directive) != {"action", "field_keys", "parameters"}:
            raise ValueError("Directive must contain exactly action/field_keys/parameters")
        action = directive["action"]
        if action not in ACTIONS:
            raise ValueError("Unknown strategy action; arbitrary tools/prompts are not executable")
        fields = _tokens(directive["field_keys"], "directive field_keys")
        if not set(fields) <= set(result["scope"]["field_keys"]):
            raise ValueError("Directive escapes the card's field scope")
        params = directive["parameters"]
        if not isinstance(params, dict) or set(params) != set(ACTIONS[action]):
            raise ValueError("Unexpected strategy parameters")
        if action == "join_adjacent_spans":
            if type(params["max_spans"]) is not int or not 1 <= params["max_spans"] <= 12 or params["order"] not in {"reading_order", "same_cell"}:
                raise ValueError("Invalid bounded multi-span directive")
        if action == "reread":
            if (type(params["max_rounds"]) is not int or not 1 <= params["max_rounds"] <= 3
                    or type(params["max_regions"]) is not int or not 1 <= params["max_regions"] <= 8
                    or params["scale_policy"] not in {"native", "fixed_patch_grid"}
                    or params["reason"] not in {"optical_ambiguity", "role_ambiguity", "missing_field", "separator_ambiguity"}):
                raise ValueError("Invalid bounded reread directive")
        result["directives"].append({"action": action, "field_keys": fields, "parameters": dict(params)})
    negative = card["negative_findings"]
    if not isinstance(negative, list) or len(negative) > 100:
        raise ValueError("negative_findings must be a bounded list")
    result["negative_findings"] = []
    for finding in negative:
        if (not isinstance(finding, dict) or set(finding) != {"experiment_sha256", "action", "outcome"}
                or not isinstance(finding["experiment_sha256"], str) or not SHA256.fullmatch(finding["experiment_sha256"])
                or finding["action"] not in ACTIONS or finding["outcome"] not in FINDINGS):
            raise ValueError("Negative findings contain only approved experiment hashes and enumerated outcomes")
        result["negative_findings"].append(dict(finding))
    # This schema excludes document-value slots. It does not replace expert
    # review of whether a strategy actually generalizes or the references exist.
    return result


class StrategyStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        with connect(self.db_path) as con:
            init_events(con)
            con.execute("""CREATE TABLE IF NOT EXISTS strategy_cards (
                card_id TEXT NOT NULL, version TEXT NOT NULL,
                scope_sha256 TEXT NOT NULL, card_json TEXT NOT NULL,
                card_sha256 TEXT NOT NULL, state TEXT NOT NULL,
                created_utc TEXT NOT NULL, updated_utc TEXT NOT NULL,
                PRIMARY KEY(card_id,version)
            )""")

    def propose(self, card: dict, *, actor_id: str, actor_scopes: Iterable[str]) -> dict:
        require_scope(actor_scopes, "strategy:propose")
        require_id(actor_id, "actor_id")
        card = validate_card(card)
        encoded, sha = canonical(card), digest(card)
        with connect(self.db_path) as con, transaction(con):
            old = con.execute("SELECT * FROM strategy_cards WHERE card_id=? AND version=?", (card["card_id"], card["version"])).fetchone()
            if old:
                if old["card_sha256"] != sha:
                    raise ValueError("Strategy revisions are immutable; use a new version")
                return {"card_id": card["card_id"], "version": card["version"], "card_sha256": sha, "state": old["state"]}
            now = utc_now()
            con.execute("INSERT INTO strategy_cards VALUES(?,?,?,?,?,?,?,?)", (card["card_id"], card["version"], digest(card["scope"]), encoded, sha, "proposed", now, now))
            append_event(con, "strategy_propose_" + sha, "strategy_lifecycle", {"card_id": card["card_id"], "version": card["version"], "card_sha256": sha, "action": "propose", "actor_id": actor_id})
        return {"card_id": card["card_id"], "version": card["version"], "card_sha256": sha, "state": "proposed"}

    def _transition(self, card_id, version, action, actor_id, actor_scopes):
        require_scope(actor_scopes, "strategy:" + action)
        _token(card_id, "card_id")
        _version(version, "version")
        require_id(actor_id, "actor_id")
        with connect(self.db_path) as con, transaction(con):
            row = con.execute("SELECT * FROM strategy_cards WHERE card_id=? AND version=?", (card_id, version)).fetchone()
            if not row:
                raise KeyError("Unknown strategy revision")
            card = json.loads(row["card_json"])
            if digest(validate_card(card)) != row["card_sha256"]:
                raise ValueError("Strategy integrity failure")
            desired = {"approve": "approved", "activate": "active", "revoke": "revoked"}[action]
            if row["state"] == desired:
                return {"card_id": card_id, "version": version, "state": desired, "card_sha256": row["card_sha256"]}
            allowed = {"approve": {"proposed"}, "activate": {"approved", "superseded"}, "revoke": {"proposed", "approved", "active", "superseded"}}
            if row["state"] not in allowed[action]:
                raise ValueError(f"Cannot {action} strategy in state {row['state']}")
            now = utc_now()
            if action == "activate":
                con.execute("UPDATE strategy_cards SET state='superseded',updated_utc=? WHERE card_id=? AND scope_sha256=? AND state='active'", (now, card_id, row["scope_sha256"]))
            con.execute("UPDATE strategy_cards SET state=?,updated_utc=? WHERE card_id=? AND version=?", (desired, now, card_id, version))
            append_event(con, "strategy_" + action + "_" + uuid.uuid4().hex, "strategy_lifecycle", {"card_id": card_id, "version": version, "card_sha256": row["card_sha256"], "action": action, "actor_id": actor_id})
        return {"card_id": card_id, "version": version, "state": desired, "card_sha256": row["card_sha256"]}

    def approve(self, card_id, version, *, actor_id, actor_scopes):
        return self._transition(card_id, version, "approve", actor_id, actor_scopes)

    def activate(self, card_id, version, *, actor_id, actor_scopes):
        return self._transition(card_id, version, "activate", actor_id, actor_scopes)

    def revoke(self, card_id, version, *, actor_id, actor_scopes):
        return self._transition(card_id, version, "revoke", actor_id, actor_scopes)

    def retrieve(self, profile_id: str, profile_version: str, *, service_id: str | None = None,
                 field_keys: list[str] | None = None, actor_scopes: Iterable[str]) -> list[dict]:
        require_scope(actor_scopes, "strategy:read")
        _token(profile_id, "profile_id")
        _version(profile_version, "profile_version")
        if service_id is not None:
            _token(service_id, "service_id")
        requested = set(_tokens(field_keys, "requested field_keys")) if field_keys is not None else None
        with connect(self.db_path) as con:
            rows = con.execute("SELECT * FROM strategy_cards WHERE state='active' ORDER BY card_id,version").fetchall()
        result = []
        for row in rows:
            card = validate_card(json.loads(row["card_json"]))
            if digest(card) != row["card_sha256"]:
                raise ValueError("Strategy integrity failure")
            scope = card["scope"]
            if scope["profile_id"] != profile_id or scope["profile_version"] != profile_version:
                continue
            if scope["service_ids"] and service_id not in scope["service_ids"]:
                continue
            directives = card["directives"]
            if requested is not None:
                directives = [d for d in directives if set(d["field_keys"]) <= requested]
            if not directives:
                continue
            result.append({"card_id": card["card_id"], "version": card["version"], "card_sha256": row["card_sha256"],
                           "scope": scope, "directives": directives, "negative_findings": card["negative_findings"],
                           "evidence_sha256": card["evidence_sha256"], "state": "active"})
        return result

    def get(self, card_id: str, version: str, *, actor_scopes: Iterable[str]) -> dict:
        require_scope(actor_scopes, "strategy:read")
        with connect(self.db_path) as con:
            row = con.execute("SELECT * FROM strategy_cards WHERE card_id=? AND version=?", (card_id, version)).fetchone()
        if not row:
            raise KeyError("Unknown strategy revision")
        card = validate_card(json.loads(row["card_json"]))
        if digest(card) != row["card_sha256"]:
            raise ValueError("Strategy integrity failure")
        return {"card": card, "card_sha256": row["card_sha256"], "state": row["state"]}
