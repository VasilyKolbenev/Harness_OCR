"""Strict JSON, immutable per-run trial files, and source-group summaries."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    """Reject non-JSON and non-finite values instead of silently stringifying."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def clone_json(value: Any) -> Any:
    canonical(value)
    return copy.deepcopy(value)


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


class TrialJournal:
    """Create-only JSONL journal, fsynced after every record.

    A path is never reused or appended by another experiment. This prevents a
    second invocation from silently replacing failed trials or their manifest.
    Individual process crashes can still leave an interrupted run; the caller
    must keep its partial journal and start a separately identified retry.
    """

    def __init__(self, path: str | Path | None, manifest: dict):
        self.path = Path(path) if path is not None else None
        self._handle = None
        self.manifest = clone_json(manifest)

    def __enter__(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("x", encoding="utf-8", newline="\n")
            self.write({"record_type": "manifest", "created_utc":
                        datetime.now(timezone.utc).isoformat(), **self.manifest})
        return self

    def write(self, record: dict) -> None:
        serialized = canonical(record)
        if self._handle is not None:
            self._handle.write(serialized + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def __exit__(self, exc_type, exc, traceback):
        if self._handle is not None:
            self._handle.close()
        return False
