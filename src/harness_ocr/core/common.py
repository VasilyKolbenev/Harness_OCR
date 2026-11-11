"""Portable serialization and identifiers used by the evidence core."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")


class StateError(ValueError):
    """A document mutation does not satisfy the state contract."""


class EvidenceError(StateError):
    """Evidence is missing, foreign, incomplete or inconsistent."""


class RevisionConflict(StateError):
    """A compare-and-swap update used an obsolete document revision."""


class AuthorizationError(StateError):
    """A privileged transition has not been authorized by the service."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def clone(value: Any) -> Any:
    canonical(value)  # Refuse NaN/non-JSON state before copying it.
    return copy.deepcopy(value)


def identifier(value: Any, label: str = "id") -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise StateError(f"invalid_{label}")
    return value


def sha256(value: Any, label: str = "source_sha256") -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise EvidenceError(f"invalid_{label}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
