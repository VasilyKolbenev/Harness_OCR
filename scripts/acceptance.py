"""Real HTTP lifecycle acceptance against a running Harness OCR service.

Uses a document supplied by the operator, never synthetic quality claims. The
report contains hashes and statuses, not document text or authentication tokens.
Exit codes: 0 pass, 1 fail, 2 pending. Output is create-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from doctor import write_report

TERMINAL = {"completed", "failed", "canceled", "cancelled", "error", "budget_exhausted", "interrupted"}


class ApiClient:
    def __init__(self, base_url, token=None, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def request(self, method, route, *, payload=None, raw=None, authenticated=True, token=None):
        if payload is not None and raw is not None:
            raise ValueError("payload and raw are mutually exclusive")
        body = json.dumps(payload, allow_nan=False).encode("utf-8") if payload is not None else raw
        headers = {"Content-Type": "application/json" if payload is not None else "application/octet-stream"}
        secret = self.token if token is None else token
        if authenticated and secret:
            headers["Authorization"] = "Bearer " + secret
        request = urllib.request.Request(self.base_url + route, data=body, method=method, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            data = response.read(16 * 1024 * 1024 + 1)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError("API response exceeded acceptance limit")
            try:
                value = json.loads(data)
            except (ValueError, UnicodeDecodeError):
                value = None
            return response.code, value


def job_state(value):
    if not isinstance(value, dict):
        return None
    return value.get("status", value.get("state"))


def wait_job(client, identity, *, deadline, poll_seconds=.5):
    while time.perf_counter() < deadline:
        status, body = client.request("GET", "/v1/jobs/" + identity)
        if status != 200 or not isinstance(body, dict):
            raise RuntimeError("job_poll_failed")
        if job_state(body) in TERMINAL:
            return body
        time.sleep(min(poll_seconds, max(0, deadline - time.perf_counter())))
    raise TimeoutError("job did not finish within acceptance budget")


def run_acceptance(client, *, document=None, profile_id=None, qwen="qwen-0.8b",
                   seconds=900, review_payload=None, review_token=None, review_observed_field=None):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if review_payload is not None and review_observed_field is not None:
        raise ValueError("Choose review_payload or review_observed_field")
    checks = []
    def add(name, status, **details):
        checks.append({"check": name, "status": status, **details})
    deadline = time.perf_counter() + seconds
    try:
        status, live = client.request("GET", "/health/live", authenticated=False)
        add("liveness", "pass" if status == 200 else "fail", http_status=status)
        status, ready = client.request("GET", "/health/ready")
        ready_flag = (ready.get("ready", ready.get("ok", False)) is True and ready.get("fixture") is False
                      if isinstance(ready, dict) else False)
        add("readiness", "pass" if status == 200 and ready_flag is True else "fail", http_status=status)
        status, _ = client.request("GET", "/v1/jobs/" + "0" * 32, authenticated=False)
        add("unauthenticated_job_access_denied", "pass" if status in {401, 403} else "fail", http_status=status)
        if not client.token:
            add("authenticated_lifecycle", "pending", reason="HARNESS_API_TOKEN is missing")
        elif document is None or profile_id is None:
            add("authenticated_lifecycle", "pending", reason="Supply --document and --profile")
        elif any(item["status"] == "fail" for item in checks):
            add("authenticated_lifecycle", "pending", reason="Preconditions failed; no job submitted")
        else:
            raw = Path(document).read_bytes()
            local_sha = hashlib.sha256(raw).hexdigest()
            status, uploaded = client.request("POST", "/v1/artifacts", raw=raw)
            artifact_id = uploaded.get("artifact_id") if isinstance(uploaded, dict) else None
            upload_ok = (status in {200, 201} and isinstance(artifact_id, str) and
                         uploaded.get("source_sha256") == local_sha)
            add("upload_integrity", "pass" if upload_ok else "fail", http_status=status,
                source_sha256=local_sha)
            if upload_ok:
                payload = {"artifact_id": artifact_id, "profile_id": profile_id, "qwen": qwen}
                status, created = client.request("POST", "/v1/jobs", payload=payload)
                identity = created.get("job_id", created.get("id")) if isinstance(created, dict) else None
                created_ok = status in {200, 201, 202} and isinstance(identity, str)
                add("submit", "pass" if created_ok else "fail", http_status=status)
                if created_ok:
                    finished = wait_job(client, identity, deadline=deadline)
                    completed = job_state(finished) == "completed"
                    add("inference_lifecycle", "pass" if completed else "fail", job_id=identity,
                        final_state=job_state(finished))
                    if completed:
                        status, result = client.request("GET", f"/v1/jobs/{identity}/result")
                        add("result_available", "pass" if status == 200 and isinstance(result, dict) else "fail",
                            http_status=status)
                        status, trace = client.request("GET", f"/v1/jobs/{identity}/trace")
                        add("trace_available", "pass" if status == 200 and isinstance(trace, (dict, list)) else "fail",
                            http_status=status)
                        if review_observed_field is not None:
                            field = result.get("fields", {}).get(review_observed_field, {}) if isinstance(result, dict) else {}
                            raw_value, evidence_ids = field.get("raw_value"), field.get("evidence_ids")
                            revision = result.get("revision") if isinstance(result, dict) else None
                            if (isinstance(raw_value, str) and raw_value and isinstance(evidence_ids, list)
                                and evidence_ids and type(revision) is int):
                                review_payload = {"field_key": review_observed_field, "raw_value": raw_value,
                                                  "evidence_ids": evidence_ids, "expected_revision": revision}
                                add("review_observed_literal", "pass", interpretation="Exact current raw/evidence re-review; no human or quality annotation claim")
                            else:
                                add("review_observed_literal", "pending", reason="Requested field has no observed literal/evidence/revision")
                        if review_payload is None or not review_token:
                            add("review_revision_lifecycle", "pending",
                                reason="Supply a real review JSON payload and separate HARNESS_REVIEW_TOKEN")
                        else:
                            # Analysis credentials must never acquire review authority.
                            route = f"/v1/jobs/{identity}/review"
                            status, _ = client.request("POST", route, payload=review_payload)
                            add("analysis_token_review_denied", "pass" if status in {401, 403} else "fail",
                                http_status=status)
                            status, reviewed = client.request("POST", route, payload=review_payload, token=review_token)
                            add("review_revision_lifecycle", "pass" if status == 200 else "fail", http_status=status)
                            if status == 200:
                                stale, _ = client.request("POST", route, payload=review_payload, token=review_token)
                                add("stale_review_revision_denied", "pass" if stale in {409, 412} else "fail",
                                    http_status=stale)
                    # New job is canceled before it can consume a full inference budget.
                    if time.perf_counter() >= deadline:
                        add("cancel_lifecycle", "pending", reason="Acceptance time budget exhausted")
                    else:
                        status, queued = client.request("POST", "/v1/jobs", payload=payload)
                        cancel_id = queued.get("job_id", queued.get("id")) if isinstance(queued, dict) else None
                        if status not in {200, 201, 202} or not isinstance(cancel_id, str):
                            add("cancel_lifecycle", "fail", http_status=status)
                        else:
                            status, _ = client.request("POST", f"/v1/jobs/{cancel_id}/cancel", payload={})
                            final = wait_job(client, cancel_id, deadline=deadline)
                            state = job_state(final)
                            outcome = "pass" if status in {200, 202} and state in {"canceled", "cancelled"} else (
                                "pending" if state == "completed" else "fail")
                            add("cancel_lifecycle", outcome, http_status=status, final_state=state,
                                reason="A completed-before-cancel race is not cancellation proof" if outcome == "pending" else None)
    except TimeoutError:
        add("acceptance_execution", "pending", reason="Wall-clock budget exhausted; preserve service logs")
    except Exception as exc:
        add("acceptance_execution", "fail", error_type=type(exc).__name__)
    result_status = "fail" if any(c["status"] == "fail" for c in checks) else (
        "pending" if any(c["status"] == "pending" for c in checks) else "pass")
    return {"schema_version": "harness-api-acceptance/1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": result_status, "checks": checks, "quality_qualified": False,
            "note": "HTTP and process lifecycle only; a completed review result is not automatic quality acceptance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--document")
    parser.add_argument("--profile")
    parser.add_argument("--qwen", choices=["qwen-0.8b", "qwen-4b"], default="qwen-0.8b")
    parser.add_argument("--seconds", type=float, default=900)
    parser.add_argument("--review-json", help="Authorized manual correction with evidence IDs and expected_revision")
    parser.add_argument("--review-observed-field", help="Re-review exactly the current raw value/evidence; lifecycle only")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if args.review_json and args.review_observed_field:
        parser.error("Choose --review-json or --review-observed-field")
    review = json.loads(Path(args.review_json).read_text(encoding="utf-8")) if args.review_json else None
    client = ApiClient(args.base_url, os.getenv("HARNESS_API_TOKEN"), timeout=min(180, args.seconds))
    report = run_acceptance(client, document=args.document, profile_id=args.profile, qwen=args.qwen,
                            seconds=args.seconds, review_payload=review, review_token=os.getenv("HARNESS_REVIEW_TOKEN"),
                            review_observed_field=args.review_observed_field)
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return {"pass": 0, "fail": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
