"""Own-document handoff regressions. Fake HTTP only; no inference or quality claim."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_documents as runner
from harness_ocr.profiles import ProfileRegistry, project_contract_result


class ControlledApi:
    def __init__(self, *, missing=False, bad_upload=False, mutate_result=None, bad_trace=False,
                 fixture=False, version_mismatch=False, unknown_submission=False, mutate_job=None):
        self.calls, self.jobs, self.artifacts = [], {}, {}
        self.missing, self.bad_upload, self.mutate_result = missing, bad_upload, mutate_result
        self.bad_trace, self.fixture, self.version_mismatch = bad_trace, fixture, version_mismatch
        self.unknown_submission = unknown_submission
        self.mutate_job = mutate_job
        self.token = "AUTHENTICATION-SECRET-MUST-NEVER-APPEAR"

    def request(self, method, route, *, payload=None, raw=None, **kwargs):
        self.calls.append((method, route))
        if route == "/health/ready":
            return 200, {"ready": True, "fixture": self.fixture}
        if route == "/v1/profiles":
            profiles = ProfileRegistry().list()
            if self.version_mismatch:
                for p in profiles:
                    p["profile_version"] = "unsupported-server-version"
            return 200, profiles
        if route == "/v1/artifacts":
            aid = f"{len(self.artifacts) + 100:032x}"
            sha = hashlib.sha256(raw).hexdigest()
            self.artifacts[aid] = sha
            return 200, {"artifact_id": aid, "source_sha256": "f" * 64 if self.bad_upload else sha}
        if route == "/v1/jobs":
            if self.unknown_submission:
                raise OSError(self.token)
            jid = f"{len(self.jobs) + 1:032x}"
            self.jobs[jid] = {"mode": "hybrid", **deepcopy(payload)}
            return 202, {"id": jid, "status": "queued"}
        jid = route.split("/")[3]
        if route.endswith("/cancel"):
            return 200, {"id": jid, "status": "canceled"}
        if route.endswith("/trace"):
            return 200, [{"document_id": "a" * 32 if self.bad_trace else jid, "stage": "test-protocol-only"}]
        if route.endswith("/result"):
            request = self.jobs[jid]
            contract = ProfileRegistry().compile(request["profile_id"], subset=request.get("subset"))
            fields = {key: {"raw_value": None if self.missing else "123456",
                "canonical_value": None if self.missing else "123456", "party_role": spec["role"],
                "evidence_ids": [] if self.missing else ["test-observation"],
                "extraction_status": "not_observed" if self.missing else "observed",
                "validation_status": "not_checked" if self.missing else "pass", "decision": "review"}
                for key, spec in contract["field_definitions"].items()}
            result = {"document_id": jid, "source_sha256": self.artifacts[request["artifact_id"]],
                "profile_id": contract["profile_id"], "profile_version": contract["profile_version"],
                "subset_contract_id": contract["subset_contract_id"], "conditional_applicability":
                    {key: value["applies"] for key, value in contract["conditionals"].items()},
                "model_bundle_sha256": "c" * 64, "completed": True, "terminal_status": "completed",
                "fields": fields, "decision": "review", "needs_review": True}
            result["contract_result"] = project_contract_result(contract, result)
            if self.mutate_result:
                self.mutate_result(result)
            return 200, result
        value = {"id": jid, "status": "completed", "request": deepcopy(self.jobs[jid])}
        if self.mutate_job:
            self.mutate_job(value)
        return 200, value


class OwnDocumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "scan.png").write_bytes(b"fake API input, never neural inference")
        self.manifest = self.root / "manifest.json"
        self.output = self.root / "output"
        self.write_manifest()

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self, documents=None):
        self.manifest.write_text(json.dumps({"schema_version": runner.SCHEMA,
            "documents": documents or [{"id": "first", "path": "scan.png", "profile_id": "D01", "subset": ["passport.number"]}]}), encoding="utf-8")

    def two_documents(self):
        self.write_manifest([{"id": name, "path": "scan.png", "profile_id": "D01", "subset": ["passport.number"]}
                             for name in ("first", "second")])

    def test_sequential_batch_saves_actual_values_trace_status_and_honest_summary(self):
        self.two_documents()
        api = ControlledApi()
        report = runner.run_documents(self.manifest, self.output, api, qwen="qwen-4b")
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["completed_documents"], 2)
        self.assertFalse(report["quality_qualified"])
        self.assertTrue(report["needs_attention"])  # review != failed execution
        submitted = [i for i, call in enumerate(api.calls) if call == ("POST", "/v1/jobs")]
        first_trace = next(i for i, call in enumerate(api.calls) if call[1].endswith("/trace"))
        self.assertLess(first_trace, submitted[1])
        self.assertTrue(all(j["qwen"] == "qwen-4b" for j in api.jobs.values()))
        for name in ("first", "second"):
            result = json.loads((self.output / name / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["fields"]["passport.number"]["raw_value"], "123456")
            self.assertTrue((self.output / name / "trace.json").is_file())
            self.assertTrue((self.output / name / "status.json").is_file())
        report_text = (self.output / "summary.json").read_text(encoding="utf-8")
        self.assertNotIn("123456", report_text)
        self.assertNotIn(api.token, report_text)
        self.assertFalse(any("review" in route or "audits" in route for _, route in api.calls))

    def test_all_missing_is_conspicuous_but_not_a_runtime_failure(self):
        report = runner.run_documents(self.manifest, self.output, ControlledApi(missing=True))
        self.assertEqual(report["status"], "pass")
        row = report["documents"][0]
        self.assertTrue(row["all_fields_missing"])
        self.assertEqual(row["observed_fields"], 0)
        self.assertTrue(row["needs_attention"])
        self.assertFalse(row["correctness_measured"])

    def test_accepts_actual_engine_projection_without_loading_models(self):
        from PIL import Image
        from harness_ocr.engine import Engine
        from harness_ocr.settings import Settings
        from harness_ocr.util import file_sha
        from test_benchmark import FakeBackend
        source = self.root / "scan.png"
        Image.new("RGB", (200, 100), "white").save(source)
        _, documents = runner.load_manifest(self.manifest)
        settings = Settings(self.root / "state", self.root / "models", device="cpu", dtype="float32")
        engine = Engine(settings, FakeBackend(settings))
        jid = "a" * 32
        result = engine.run(jid, source, file_sha(source), "D01", subset=["passport.number"])
        checked = runner.check_result(result, documents[0], jid)
        self.assertEqual(checked["requested_fields"], ["passport.number"])
        self.assertEqual(result["fields"]["passport.number"]["raw_value"], "123456")

    def test_prevalidates_every_document_before_any_http_side_effect(self):
        cases = [{"path": "missing.png"}, {"profile_id": "made_up"}, {"subset": ["made.up"]},
                 {"subset": ["passport.number", "passport.number"]}, {"subset": None},
                 {"id": "../../escape"}, {"path": "../outside.png"}]
        for replacement in cases:
            with self.subTest(replacement=replacement):
                valid = {"id": "first", "path": "scan.png", "profile_id": "D01", "subset": ["passport.number"]}
                self.write_manifest([valid, {**valid, "id": "second", **replacement}])
                api = ControlledApi()
                with self.assertRaises(runner.RunFailure):
                    runner.run_documents(self.manifest, self.output, api)
                self.assertEqual(api.calls, [])
                self.assertFalse(self.output.exists())

    def test_duplicate_ids_case_insensitive_and_gold_not_allowed(self):
        for extra in ({"id": "FIRST"}, {"id": "second", "gold": {"passport.number": "123456"}}):
            self.write_manifest([{"id": "first", "path": "scan.png", "profile_id": "D01"},
                                 {"id": "second", "path": "scan.png", "profile_id": "D01", **extra}])
            with self.assertRaises(runner.RunFailure):
                runner.load_manifest(self.manifest)

    def test_empty_and_oversized_source_rejected_before_upload(self):
        for raw in (b"", b"oversized"):
            (self.root / "scan.png").write_bytes(raw)
            with patch.object(runner, "MAX_BYTES", 3), self.assertRaises(runner.RunFailure):
                runner.load_manifest(self.manifest)

    def test_fixture_and_profile_version_mismatch_submit_nothing(self):
        for kwargs in ({"fixture": True}, {"version_mismatch": True}):
            with self.subTest(kwargs=kwargs):
                api = ControlledApi(**kwargs)
                output = self.root / str(len(kwargs)) / str(next(iter(kwargs)))
                report = runner.run_documents(self.manifest, output, api)
                self.assertEqual(report["status"], "fail")
                self.assertFalse(any(method == "POST" for method, _ in api.calls))

    def test_upload_mismatch_never_submits_job(self):
        api = ControlledApi(bad_upload=True)
        report = runner.run_documents(self.manifest, self.output, api)
        self.assertEqual(report["documents"][0]["error_code"], "upload_integrity_failed")
        self.assertEqual(api.jobs, {})

    def test_binding_or_projection_mismatch_is_runtime_failure(self):
        mutations = {
            "source": lambda r: r.update(source_sha256="f" * 64),
            "job": lambda r: r.update(document_id="f" * 32),
            "profile": lambda r: r.update(profile_id="other"),
            "subset": lambda r: r.update(subset_contract_id="wrong"),
            "fields": lambda r: r["fields"].update({"unexpected": r["fields"]["passport.number"]}),
            "projection": lambda r: r["contract_result"]["fields"]["passport.number"].update(raw_value="invented"),
            "truncated": lambda r: r.update(completed=False),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                report = runner.run_documents(self.manifest, self.root / name, ControlledApi(mutate_result=mutate))
                self.assertEqual(report["status"], "fail")
                self.assertTrue((self.root / name / "first" / "result.json").is_file())  # preserve diagnostic response

    def test_trace_must_be_available_and_bound_to_same_job(self):
        report = runner.run_documents(self.manifest, self.output, ControlledApi(bad_trace=True))
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["documents"][0]["error_code"], "trace_unavailable_or_wrong_document")

    def test_wrong_requested_model_or_document_subset_cannot_pass(self):
        for key, value in (("qwen", "qwen-4b"), ("profile_id", "D02"), ("artifact_id", "f" * 32),
                           ("subset", ["passport.series"]), ("mode", "rules")):
            with self.subTest(key=key):
                api = ControlledApi(mutate_job=lambda job: job["request"].update({key: value}))
                report = runner.run_documents(self.manifest, self.root / key, api)
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["documents"][0]["error_code"], "job_request_binding")

    def test_known_failed_job_preserves_partial_responses_without_claiming_success(self):
        api = ControlledApi(mutate_job=lambda job: job.update(status="failed"))
        report = runner.run_documents(self.manifest, self.output, api)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["completed_documents"], 0)
        self.assertTrue((self.output / "first" / "result.json").is_file())
        self.assertTrue((self.output / "first" / "trace.json").is_file())

    def test_source_change_between_preflight_and_upload_is_rejected(self):
        api = ControlledApi()
        request = api.request
        def change_after_preflight(method, route, **kwargs):
            if route == "/v1/profiles":
                (self.root / "scan.png").write_bytes(b"changed source")
            return request(method, route, **kwargs)
        api.request = change_after_preflight
        report = runner.run_documents(self.manifest, self.output, api)
        self.assertEqual(report["documents"][0]["error_code"], "source_changed_after_preflight")
        self.assertFalse(any(method == "POST" for method, _ in api.calls))

    def test_returned_trusted_conditional_context_has_consistent_subset_hash(self):
        _, docs = runner.load_manifest(self.manifest)
        doc = docs[0]
        # Full passport includes conditional fields, unlike number-only smoke.
        doc["subset"] = None
        doc["contract"] = ProfileRegistry().compile("D01")
        api = ControlledApi()
        aid = "b" * 32
        jid = "a" * 32
        api.artifacts[aid] = doc["source_sha256"]
        api.jobs[jid] = {"artifact_id": aid, "profile_id": doc["contract"]["profile_id"], "qwen": "qwen-0.8b", "mode": "hybrid"}
        _, result = api.request("GET", "/v1/jobs/" + jid + "/result")
        context = {key: False for key in doc["contract"]["conditionals"]}
        contract = ProfileRegistry().compile("D01", condition_context=context)
        result.update(conditional_applicability=context, subset_contract_id=contract["subset_contract_id"])
        result["contract_result"] = project_contract_result(contract, result)
        self.assertEqual(runner.check_result(result, doc, jid)["subset_contract_id"], contract["subset_contract_id"])

    def test_timeout_cancel_unresolved_blocks_next_document(self):
        self.two_documents()
        api = ControlledApi()
        with patch.object(runner, "wait_job", side_effect=TimeoutError("private diagnostic")):
            report = runner.run_documents(self.manifest, self.output, api)
        self.assertEqual(len(api.jobs), 1)
        self.assertIn(("POST", "/v1/jobs/" + "1".zfill(32) + "/cancel"), api.calls)
        self.assertFalse(report["documents"][0]["cancellation"]["terminal_confirmed"])
        self.assertEqual(report["documents"][1]["error_code"], "previous_job_unresolved_or_batch_aborted")

    def test_unknown_submission_connection_failure_does_not_start_next(self):
        self.two_documents()
        api = ControlledApi(unknown_submission=True)
        report = runner.run_documents(self.manifest, self.output, api)
        self.assertEqual(sum(call == ("POST", "/v1/jobs") for call in api.calls), 1)
        self.assertTrue(report["documents"][0]["possible_unknown_submitted_job"])
        self.assertNotIn(api.token, json.dumps(report))

    def test_output_is_create_only(self):
        api = ControlledApi()
        runner.run_documents(self.manifest, self.output, api)
        before = (self.output / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            runner.run_documents(self.manifest, self.output, api)
        self.assertEqual((self.output / "summary.json").read_bytes(), before)

    def test_cli_keeps_field_values_and_authentication_out_of_console(self):
        api = ControlledApi()
        stream = io.StringIO()
        with patch.object(runner, "ApiClient", return_value=api), redirect_stdout(stream):
            status = runner.main(["--manifest", str(self.manifest), "--output", str(self.output)])
        self.assertEqual(status, 0)
        self.assertNotIn("123456", stream.getvalue())
        self.assertNotIn(api.token, stream.getvalue())
        self.assertEqual(json.loads(stream.getvalue())["status"], "pass")


if __name__ == "__main__":
    unittest.main()
