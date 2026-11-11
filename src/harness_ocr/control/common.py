"""Small durable primitives shared by the control modules; no model imports."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterable


BINDING_KEYS = (
    "profile_id", "profile_version", "subset_contract_id",
    "model_bundle_sha256", "policy_version", "input_scope_version",
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_scope(scopes: Iterable[str] | None, required: str) -> None:
    if isinstance(scopes, str) or required not in set(scopes or ()):
        raise PermissionError(f"Required server-authorized scope: {required}")


def require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{label} must be a nonempty string of at most512 characters")
    return value


def probability(value: Any, label: str = "probability") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label} must be finite and in [0,1]")
    return result


def validate_binding(binding: dict) -> dict:
    if not isinstance(binding, dict):
        raise ValueError("binding must be a dictionary")
    if set(binding) != set(BINDING_KEYS):
        raise ValueError("binding must contain exactly the declared qualification keys")
    result = {key: require_id(binding[key], key) for key in BINDING_KEYS}
    if not re.fullmatch(r"[0-9a-fA-F]{64}", result["model_bundle_sha256"]):
        raise ValueError("model_bundle_sha256 must be a SHA256 hex digest")
    result["model_bundle_sha256"] = result["model_bundle_sha256"].lower()
    return result


@contextmanager
def connect(db_path: str | Path):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=FULL")
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=30000")
    try:
        yield con
    finally:
        con.close()


@contextmanager
def transaction(con: sqlite3.Connection):
    con.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        con.execute("ROLLBACK")
        raise
    else:
        con.execute("COMMIT")


def init_events(con: sqlite3.Connection) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS control_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id TEXT UNIQUE NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        created_utc TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS control_quarantines (
        binding_sha256 TEXT PRIMARY KEY,
        binding_json TEXT NOT NULL,
        active INTEGER NOT NULL,
        reason_json TEXT NOT NULL,
        updated_utc TEXT NOT NULL
    )""")


def append_event(con: sqlite3.Connection, event_id: str, event_type: str, payload: dict) -> dict:
    """Append once; an idempotency collision with different content fails closed."""
    require_id(event_id, "event_id")
    require_id(event_type, "event_type")
    encoded = canonical(payload)
    payload_sha = digest(payload)
    old = con.execute("SELECT * FROM control_events WHERE event_id=?", (event_id,)).fetchone()
    if old:
        if old["event_type"] != event_type or old["payload_sha256"] != payload_sha:
            raise ValueError("Idempotency collision: event_id already has different content")
        return dict(old)
    con.execute(
        "INSERT INTO control_events(event_id,event_type,payload_json,payload_sha256,created_utc) VALUES(?,?,?,?,?)",
        (event_id, event_type, encoded, payload_sha, utc_now()),
    )
    return dict(con.execute("SELECT * FROM control_events WHERE event_id=?", (event_id,)).fetchone())


def read_events(con: sqlite3.Connection, event_type: str | None = None) -> list[dict]:
    if event_type is None:
        rows = con.execute("SELECT * FROM control_events ORDER BY sequence").fetchall()
    else:
        rows = con.execute("SELECT * FROM control_events WHERE event_type=? ORDER BY sequence", (event_type,)).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        payload = json.loads(item.pop("payload_json"))
        if digest(payload) != item["payload_sha256"]:
            raise ValueError("Control event integrity check failed")
        item["payload"] = payload
        result.append(item)
    return result
