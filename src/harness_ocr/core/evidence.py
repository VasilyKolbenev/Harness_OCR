"""Validation of literal observations, never optical truth or authenticity."""
from __future__ import annotations

import math
from typing import Any

from .common import EvidenceError, clone, identifier, sha256

KINDS = {"image_region", "native_pdf_text", "structured_xml", "trusted_record"}


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _box(box: Any, size: Any) -> None:
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(_number(x) and x > 0 for x in size):
        raise EvidenceError("invalid_source_size")
    if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_number(x) for x in box):
        raise EvidenceError("invalid_bbox")
    x0, y0, x1, y1 = box
    w, h = size
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise EvidenceError("bbox_outside_source")


def _quad(quad: Any, size: Any) -> None:
    if not isinstance(quad, (list, tuple)) or len(quad) != 4:
        raise EvidenceError("invalid_quad")
    if not isinstance(size, (list, tuple)) or len(size) != 2 or not all(_number(x) and x > 0 for x in size):
        raise EvidenceError("invalid_source_size")
    for point in quad:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or not all(_number(x) for x in point):
            raise EvidenceError("invalid_quad")
        if not (0 <= point[0] <= size[0] and 0 <= point[1] <= size[1]):
            raise EvidenceError("quad_outside_source")
    area = abs(sum(quad[i][0] * quad[(i + 1) % 4][1] - quad[(i + 1) % 4][0] * quad[i][1] for i in range(4))) / 2
    if area <= 0:
        raise EvidenceError("degenerate_quad")


def normalize_observation(value: dict, allowed_sources: set[str], *, mode: str = "production") -> dict:
    if not isinstance(value, dict):
        raise EvidenceError("observation_must_be_object")
    obs = clone(value)
    if any(key in obs for key in ("active", "created_revision", "can_auto_accept", "validated")):
        raise EvidenceError("caller_cannot_set_observation_state")
    identifier(obs.get("id"), "observation_id")
    source = sha256(obs.get("source_sha256"))
    if source not in allowed_sources:
        raise EvidenceError("cross_document_source")
    kind = obs.get("evidence_kind")
    if kind not in KINDS:
        raise EvidenceError("unsupported_evidence_kind")
    if not isinstance(obs.get("text"), str):
        raise EvidenceError("observation_text_must_be_string")
    if not isinstance(obs.get("completed", False), bool):
        raise EvidenceError("completed_must_be_boolean")
    obs.setdefault("completed", False)
    if obs.get("oracle") or obs.get("oracle_modified"):
        if mode != "research":
            raise EvidenceError("oracle_forbidden_in_production")
    if kind in {"image_region", "native_pdf_text"}:
        if not isinstance(obs.get("page"), int) or isinstance(obs["page"], bool) or obs["page"] < 1:
            raise EvidenceError("invalid_page")
        if not isinstance(obs.get("page_id"), str) or not obs["page_id"]:
            raise EvidenceError("missing_page_id")
        if obs.get("bbox") is not None:
            _box(obs["bbox"], obs.get("source_size"))
        elif obs.get("quad") is not None:
            _quad(obs["quad"], obs.get("source_size"))
        else:
            raise EvidenceError("visual_evidence_requires_coordinates")
        if kind == "native_pdf_text" and obs.get("text_layer_checks") is not True:
            raise EvidenceError("native_pdf_text_layer_not_verified")
    elif kind == "structured_xml":
        if any(obs.get(k) is not None for k in ("bbox", "quad", "page")):
            raise EvidenceError("xml_must_not_have_invented_page_coordinates")
        if not isinstance(obs.get("xpath") or obs.get("field_path"), str):
            raise EvidenceError("xml_requires_locator")
        if not obs.get("schema_version") or not isinstance(obs.get("namespace_map", {}), dict):
            raise EvidenceError("xml_requires_schema_and_namespaces")
        if obs.get("structural_validation_status") not in {"pass", "not_checked"}:
            raise EvidenceError("invalid_xml_structural_status")
    elif kind == "trusted_record":
        if any(obs.get(k) is not None for k in ("bbox", "quad", "page")):
            raise EvidenceError("record_must_not_have_invented_coordinates")
        for key in ("authorized_source_id", "record_or_request_id", "retrieval_time", "field_path"):
            if not isinstance(obs.get(key), str) or not obs[key]:
                raise EvidenceError("trusted_record_requires_" + key)
        sha256(obs.get("response_sha256"), "response_sha256")
    return obs


def literal_supported(raw_value: str, observations: list[dict], spans: list[dict] | None = None) -> bool:
    """A value is a literal substring or an explicitly ordered, lossless span join."""
    if not isinstance(raw_value, str):
        return False
    if any(raw_value in obs["text"] for obs in observations):
        return True
    if not spans:
        return False
    by_id = {x["id"]: x for x in observations}
    parts = []
    for item in spans:
        if not isinstance(item, dict) or item.get("observation_id") not in by_id:
            return False
        text = by_id[item["observation_id"]]["text"]
        start, end = item.get("start"), item.get("end")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
            return False
        separator = item.get("separator_after", "")
        if not isinstance(separator, str) or separator.strip():
            return False  # Joins may add whitespace, never letters or punctuation.
        parts.append(text[start:end] + separator)
    return "".join(parts) == raw_value
