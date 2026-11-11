"""Run operator-supplied documents through the real, already-running HTTP service.

No models, gold, repairs, or review operations run here. Values and traces are
saved only to the operator's new local output directory. Exit 0 means technical
execution passed; it is deliberately independent of extraction coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

from acceptance import ApiClient, TERMINAL, job_state, wait_job
from doctor import write_report

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from harness_ocr.profiles import ProfileRegistry

SCHEMA = "harness-own-documents/1"
MAX_BYTES = 30 * 1024 * 1024
ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
OPAQUE_ID = re.compile(r"[0-9a-f]{32}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class RunFailure(ValueError):
    """An allowlisted error code, never a remote response or document value."""


def _require(condition, code):
    if not condition:
        raise RunFailure(code)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_bytes(path):
    with Path(path).open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    _require(0 < len(raw) <= MAX_BYTES, "source_size")
    return raw


def _positive_budget(value, maximum, code):
    _require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= maximum, code)


def load_manifest(path):
    """Validate every local source and contract before any HTTP call/submission."""
    path = Path(path).resolve()
    _require(path.is_file() and path.stat().st_size <= 1024 * 1024, "manifest_size_or_path")
    with path.open("rb") as source:
        raw = source.read(1024 * 1024 + 1)
    _require(len(raw) <= 1024 * 1024, "manifest_size_or_path")
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError):
        raise RunFailure("manifest_json") from None
    _require(isinstance(value, dict) and set(value) == {"schema_version", "documents"}
             and value["schema_version"] == SCHEMA, "manifest_schema")
    documents = value["documents"]
    _require(isinstance(documents, list) and 1 <= len(documents) <= 100, "manifest_document_count")
    registry = ProfileRegistry()
    checked, ids = [], set()
    for doc in documents:
        _require(isinstance(doc, dict) and {"id", "path", "profile_id"} <= set(doc)
                 and set(doc) <= {"id", "path", "profile_id", "subset"}, "manifest_document_shape")
        identity = doc["id"]
        _require(isinstance(identity, str) and ID.fullmatch(identity) is not None
                 and identity.lower() not in ids, "manifest_document_id")
        ids.add(identity.lower())
        name = doc["path"]
        _require(isinstance(name, str) and 1 <= len(name) <= 1024 and not Path(name).is_absolute(), "source_relative_path_required")
        source = (path.parent / name).resolve()
        _require(source.is_relative_to(path.parent) and source.is_file(), "source_outside_manifest_or_missing")
        _require(0 < source.stat().st_size <= MAX_BYTES, "source_size")
        _require(isinstance(doc["profile_id"], str) and 1 <= len(doc["profile_id"]) <= 128, "profile_id_required")
        if "subset" in doc:
            _require(isinstance(doc["subset"], list) and 1 <= len(doc["subset"]) <= 100, "subset_list_required")
        try:
            contract = registry.compile(doc["profile_id"], subset=doc.get("subset"))
        except ValueError:
            raise RunFailure("profile_or_subset_invalid") from None
        checked.append({"id": identity, "path": source, "subset": doc.get("subset"),
                        "contract": contract, "source_sha256": _sha(_source_bytes(source))})
    return raw, checked


def check_result(result, doc, job_id):
    """Check binding and selected structural properties, not optical correctness.

    This is not advertised as complete JSON Schema/evidence/trace verification.
    Server-supplied conditional applicability affects the compiled subset hash;
    it is checked for consistency here, not independently authorized by a client.
    """
    base = doc["contract"]
    _require(isinstance(result, dict), "result_object")
    _require(result.get("document_id") == job_id and result.get("source_sha256") == doc["source_sha256"], "result_source_or_job_binding")
    _require(result.get("profile_id") == base["profile_id"]
             and result.get("profile_version") == base["profile_version"], "result_profile_binding")
    _require(result.get("completed") is True and result.get("terminal_status") == "completed", "result_not_completed")
    _require(result.get("runtime_binding_changed", False) is False, "runtime_binding_changed")
    context = result.get("conditional_applicability")
    _require(isinstance(context, dict) and set(context) == set(base["conditionals"])
             and all(type(v) is bool or v == "unknown" for v in context.values()), "result_condition_context")
    contract = ProfileRegistry().compile(base["profile_id"], subset=doc["subset"], condition_context=context)
    _require(result.get("subset_contract_id") == contract["subset_contract_id"], "result_subset_binding")
    _require(type(result.get("needs_review")) is bool
             and result.get("decision") in {"review", "accept_for_autofill", "reject_input"}, "result_decision_shape")
    projection = result.get("contract_result")
    _require(isinstance(projection, dict) and all(projection.get(k) == result.get(k)
             for k in ("profile_id", "profile_version", "subset_contract_id", "needs_review")), "contract_projection_binding")
    fields = result.get("fields")
    _require(isinstance(fields, dict) and set(fields) == set(contract["requested_fields"]), "result_requested_fields")
    _require(isinstance(projection.get("fields"), dict) and set(projection["fields"]) == set(fields), "projection_requested_fields")
    for key, field in fields.items():
        _require(isinstance(field, dict) and {"raw_value", "canonical_value", "party_role", "extraction_status",
                 "validation_status", "decision", "evidence_ids"} <= set(field), "field_shape")
        _require(field["raw_value"] is None or isinstance(field["raw_value"], (str, dict)), "field_literal_type")
        _require(field["party_role"] in (None, contract["field_definitions"][key]["role"]), "field_role")
        _require(field["extraction_status"] in {"observed", "ambiguous", "unreadable", "not_observed", "not_applicable"}
                 and field["validation_status"] in {"pass", "fail", "not_checked"}
                 and field["decision"] in {"review", "accept_for_autofill", "reject_input"}, "field_status")
        _require(isinstance(field["evidence_ids"], list) and all(isinstance(e, str) for e in field["evidence_ids"]), "field_evidence_list")
        # Projection may omit rich runtime properties, but may not rewrite any
        # property it publishes, especially values, roles or evidence links.
        p = projection["fields"][key]
        _require(isinstance(p, dict) and all(k in field and field[k] == v for k, v in p.items())
                 and {"raw_value", "canonical_value", "party_role", "extraction_status", "validation_status",
                      "decision", "evidence_ids"} <= set(p), "projection_field_mismatch")
    _require(isinstance(result.get("model_bundle_sha256"), str)
             and SHA.fullmatch(result["model_bundle_sha256"]) is not None, "model_binding_missing")
    return contract


def coverage(result):
    fields = result["fields"]
    states = Counter(f["extraction_status"] for f in fields.values())
    observed = states["observed"]
    nonempty = sum(f["raw_value"] is not None and f["raw_value"] != "" and f["raw_value"] != {} for f in fields.values())
    all_missing = nonempty == 0
    return {"requested_fields": len(fields), "observed_fields": observed, "nonempty_fields": nonempty,
            "extraction_counts": dict(states), "validation_counts": dict(Counter(f["validation_status"] for f in fields.values())),
            "all_fields_missing": all_missing, "needs_review": result["needs_review"],
            "needs_attention": all_missing or observed < len(fields) or result["needs_review"],
            "correctness_measured": False, "quality_qualified": False}


def _cancel(client, job_id, seconds):
    try:
        code, _ = client.request("POST", f"/v1/jobs/{job_id}/cancel", payload={})
        _require(code in {200, 202}, "cancel_request_failed")
        final = wait_job(client, job_id, deadline=time.perf_counter() + seconds)
        return {"requested": True, "terminal_confirmed": job_state(final) in TERMINAL
                and final.get("id", final.get("job_id")) == job_id,
                "final_status": job_state(final)}, final
    except Exception as exc:
        return {"requested": True, "terminal_confirmed": False, "error_type": type(exc).__name__}, None


def run_documents(manifest, output, client, *, qwen="qwen-0.8b", seconds=3600,
                  document_seconds=600, cancel_seconds=30):
    _positive_budget(seconds, 86400, "batch_budget")
    _positive_budget(document_seconds, 3600, "document_budget")
    _positive_budget(cancel_seconds, 120, "cancel_budget")
    _require(qwen in {"qwen-0.8b", "qwen-4b"}, "qwen_choice")
    raw_manifest, documents = load_manifest(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "input-manifest.json").write_bytes(raw_manifest)
    report = {"schema_version": "harness-own-documents-report/1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": _sha(raw_manifest), "planned_documents": len(documents), "qwen": qwen,
              "models_run_locally": False, "jobs_parallel": False, "quality_qualified": False,
              "correctness_measured": False, "documents": [], "status": "running",
              "checks_scope": "HTTP readiness; upload/source/job/profile/subset binding; requested field shapes; projection agreement; trace availability and document IDs. No full JSON Schema, cryptographic trace audit, optical accuracy or GPU qualification claim."}
    deadline = time.perf_counter() + seconds
    try:
        code, ready = client.request("GET", "/health/ready")
        _require(code == 200 and isinstance(ready, dict) and ready.get("ready") is True
                 and ready.get("fixture") is False, "service_not_ready_or_fixture")
        code, profiles = client.request("GET", "/v1/profiles")
        _require(code == 200 and isinstance(profiles, list), "service_profiles_unavailable")
        supported = {(p.get("profile_id"), p.get("profile_version")) for p in profiles if isinstance(p, dict)}
        _require(all((d["contract"]["profile_id"], d["contract"]["profile_version"]) in supported
                     for d in documents), "service_profile_version_mismatch")
        blocked = False
        for doc in documents:
            entry = {"id": doc["id"], "source_sha256": doc["source_sha256"],
                     "profile_id": doc["contract"]["profile_id"], "status": "not_started"}
            target = output / doc["id"]
            target.mkdir()
            job_id, state, submitted = None, None, False
            try:
                _require(not blocked, "previous_job_unresolved_or_batch_aborted")
                _require(time.perf_counter() < deadline, "batch_budget_exhausted")
                raw = _source_bytes(doc["path"])
                _require(0 < len(raw) <= MAX_BYTES and _sha(raw) == doc["source_sha256"], "source_changed_after_preflight")
                code, artifact = client.request("POST", "/v1/artifacts", raw=raw)
                _require(code in {200, 201} and isinstance(artifact, dict)
                         and artifact.get("source_sha256") == doc["source_sha256"]
                         and isinstance(artifact.get("artifact_id"), str)
                         and OPAQUE_ID.fullmatch(artifact["artifact_id"]) is not None, "upload_integrity_failed")
                payload = {"artifact_id": artifact["artifact_id"], "profile_id": doc["contract"]["profile_id"], "qwen": qwen}
                if doc["subset"] is not None:
                    payload["subset"] = doc["subset"]
                submitted = True  # Connection failure after POST can leave an unknown accepted job.
                code, created = client.request("POST", "/v1/jobs", payload=payload)
                _require(code in {200, 201, 202} and isinstance(created, dict)
                         and isinstance(created.get("id", created.get("job_id")), str), "job_submit_failed")
                candidate_id = created.get("id", created.get("job_id"))
                _require(OPAQUE_ID.fullmatch(candidate_id) is not None, "job_id_invalid")
                job_id = candidate_id
                entry["job_id"] = job_id
                state = wait_job(client, job_id, deadline=min(deadline, time.perf_counter() + document_seconds))
                _require(state.get("id", state.get("job_id")) == job_id, "job_status_binding")
                actual_request = state.get("request")
                _require(isinstance(actual_request, dict)
                         and all(actual_request.get(k) == payload.get(k) for k in ("artifact_id", "profile_id", "subset", "qwen"))
                         and actual_request.get("mode") == "hybrid", "job_request_binding")
                _require(job_state(state) == "completed", "job_not_completed")
                code, result = client.request("GET", f"/v1/jobs/{job_id}/result")
                _require(code == 200 and isinstance(result, dict), "result_unavailable")
                write_report(target / "result.json", result)
                check_result(result, doc, job_id)
                code, trace = client.request("GET", f"/v1/jobs/{job_id}/trace")
                if code == 200 and isinstance(trace, list):
                    write_report(target / "trace.json", trace)
                _require(code == 200 and isinstance(trace, list) and trace
                         and all(isinstance(row, dict) and row.get("document_id") == job_id for row in trace), "trace_unavailable_or_wrong_document")
                entry.update(status="pass", execution_passed=True, **coverage(result))
            except Exception as exc:
                entry.update(status="fail", execution_passed=False,
                             error_code=str(exc) if isinstance(exc, RunFailure) else "runtime_exception",
                             error_type=type(exc).__name__, quality_qualified=False, needs_attention=True)
                terminal_known = (isinstance(state, dict) and state.get("id", state.get("job_id")) == job_id
                                  and job_state(state) in TERMINAL)
                if job_id and not terminal_known:
                    cancellation, final = _cancel(client, job_id, cancel_seconds)
                    entry["cancellation"] = cancellation
                    if final is not None:
                        state = final
                    blocked = not cancellation["terminal_confirmed"]
                elif submitted and job_id is None:
                    entry["possible_unknown_submitted_job"] = True
                    blocked = True
                if isinstance(exc, TimeoutError) or time.perf_counter() >= deadline:
                    blocked = True
                if job_id and isinstance(state, dict) and state.get("id", state.get("job_id")) == job_id and job_state(state) in TERMINAL:
                    # Preserve available partial responses as diagnostics only;
                    # a terminal failure never becomes a successful document.
                    for endpoint in ("result", "trace"):
                        destination = target / (endpoint + ".json")
                        if destination.exists():
                            continue
                        try:
                            code, body = client.request("GET", f"/v1/jobs/{job_id}/{endpoint}")
                            if code == 200 and isinstance(body, (dict, list)):
                                write_report(destination, body)
                        except Exception:
                            pass  # Original execution failure remains authoritative.
            write_report(target / "status.json", {"outcome": entry, "job_status": state})
            report["documents"].append(entry)
    except Exception as exc:
        report.update(error_code=str(exc) if isinstance(exc, RunFailure) else "runtime_exception", error_type=type(exc).__name__)
    report["completed_documents"] = sum(row["status"] == "pass" for row in report["documents"])
    report["needs_attention"] = any(row.get("needs_attention", True) for row in report["documents"]) or report["completed_documents"] != len(documents)
    report["status"] = "pass" if report["completed_documents"] == len(documents) and "error_code" not in report else "fail"
    write_report(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New protected output directory; contains document values")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--qwen", choices=["qwen-0.8b", "qwen-4b"], default="qwen-0.8b")
    parser.add_argument("--seconds", type=float, default=3600, help="Total batch wait budget")
    parser.add_argument("--document-seconds", type=float, default=600)
    parser.add_argument("--cancel-seconds", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        client = ApiClient(args.base_url, os.getenv("HARNESS_API_TOKEN"), timeout=30)
        report = run_documents(args.manifest, args.output, client, qwen=args.qwen, seconds=args.seconds,
                               document_seconds=args.document_seconds, cancel_seconds=args.cancel_seconds)
        print(json.dumps({"status": report["status"], "manifest_sha256": report["manifest_sha256"],
                          "completed_documents": report["completed_documents"], "planned_documents": report["planned_documents"],
                          "needs_attention": report["needs_attention"], "quality_qualified": False}))
        return 0 if report["status"] == "pass" else 1
    except Exception as exc:
        print(json.dumps({"status": "fail", "error_code": str(exc) if isinstance(exc, RunFailure) else "invalid_input_or_output",
                          "error_type": type(exc).__name__, "quality_qualified": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
