"""Handoff gates must refuse incomplete evidence, secret files and false passes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acceptance
import build_release
import doctor
import gpu_acceptance


class ControlledApi:
    """Protocol fixture only; it is not model inference or a quality benchmark."""
    def __init__(self, *, bad_sha=False, allow_analysis_review=False, cancel_race=False):
        self.token = "analysis-fixture"
        self.jobs = 0
        self.reviewed = False
        self.bad_sha, self.allow_analysis_review, self.cancel_race = bad_sha, allow_analysis_review, cancel_race

    def request(self, method, route, *, payload=None, raw=None, authenticated=True, token=None):
        if route == "/health/live":
            return 200, {"ok": True}
        if route == "/health/ready":
            return 200, {"ready": True, "fixture": False}
        if not authenticated:
            return 401, {}
        if route == "/v1/artifacts":
            return 201, {"artifact_id": "artifact", "source_sha256": "wrong" if self.bad_sha else hashlib.sha256(raw).hexdigest()}
        if route == "/v1/jobs":
            self.jobs += 1
            return 202, {"id": str(self.jobs)}
        if route.endswith("/review"):
            if token is None and not self.allow_analysis_review:
                return 403, {}
            if self.reviewed:
                return 409, {}
            self.reviewed = True
            return 200, {"revision": 2}
        if route.endswith("/result"):
            return 200, {"revision": 3, "fields": {"seller": {"raw_value": "Fictional", "evidence_ids": ["r1"]}}}
        if route.endswith(("/trace", "/cancel")):
            return 200, {}
        state = "completed" if route.endswith("/1") or self.cancel_race else "canceled"
        return 200, {"status": state}


class HandoffTests(unittest.TestCase):
    def test_nonfinite_budgets_are_rejected_before_work(self):
        for value in (float("nan"), float("inf"), -1, True):
            with self.assertRaises(ValueError):
                acceptance.run_acceptance(ControlledApi(), seconds=value)
            with self.assertRaises(ValueError):
                gpu_acceptance.run_gpu_benchmark("does-not-exist", "not-created", "unused", seconds=value)

    def test_hard_stop_reaps_a_real_non_model_child(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 start_new_session=(os.name == "posix"))
        try:
            gpu_acceptance.stop_process_group(child)
            self.assertIsNotNone(child.poll())
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)

    def run_api(self, client, *, review=True):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "fictional.png"
            document.write_bytes(b"controlled-fixture-not-real-OCR")
            return acceptance.run_acceptance(client, document=document, profile_id="test-profile",
                review_payload={"field_key": "seller", "raw_value": "Fictional", "evidence_ids": ["r1"],
                                "expected_revision": 1} if review else None,
                review_token="review-fixture" if review else None)

    def test_full_protocol_can_pass_without_claiming_quality(self):
        report = self.run_api(ControlledApi())
        self.assertEqual(report["status"], "pass")
        self.assertFalse(report["quality_qualified"])
        serialized = json.dumps(report)
        self.assertNotIn("analysis-fixture", serialized)
        self.assertNotIn("review-fixture", serialized)
        self.assertNotIn("Fictional", serialized)

    def test_upload_corruption_prevents_inference(self):
        client = ControlledApi(bad_sha=True)
        report = self.run_api(client)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(client.jobs, 0)

    def test_review_privilege_violation_fails_and_missing_review_is_pending(self):
        self.assertEqual(self.run_api(ControlledApi(allow_analysis_review=True))["status"], "fail")
        self.assertEqual(self.run_api(ControlledApi(), review=False)["status"], "pending")

    def test_completed_before_cancel_race_is_pending_not_cancellation_proof(self):
        self.assertEqual(self.run_api(ControlledApi(cancel_race=True))["status"], "pending")

    def test_explicit_observed_field_review_uses_current_literal_and_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            document = Path(directory) / "fixture.png"
            document.write_bytes(b"not an OCR benchmark")
            client = ControlledApi()
            report = acceptance.run_acceptance(client, document=document, profile_id="profile",
                review_observed_field="seller", review_token="review-fixture")
            self.assertEqual(report["status"], "pass")
            self.assertTrue(client.reviewed)
            missing = acceptance.run_acceptance(ControlledApi(), document=document, profile_id="profile",
                review_observed_field="missing", review_token="review-fixture")
            self.assertEqual(missing["status"], "pending")

    def test_gpu_summary_rejects_truncation_missing_arms_and_false_quality_claims(self):
        manifest = {"documents": [{"id": "a", "original_id": "original-a"}]}
        summary = {"schema_version": "harness-bench-summary/1", "arms": ["A0", "A1", "A3", "A4"],
                   "paired_originals": 1, "planned_rows": 4, "completed_rows": 4,
                   "all_generations_completed": True, "quality_qualified": False,
                   "protocol": {"device": "cuda", "dtype": "bfloat16", "fixture": False},
                   "matched_optics_proofs": [{"document_id": "a", "match": True,
                       "saved_optics_sha256": "a" * 64,
                       "initial_optics_sha256": {arm: "a" * 64 for arm in ("A0", "A1", "A3", "A4")}}],
                   "metrics": {arm: {"documents": 1, "original_groups": 1} for arm in ("A0", "A1", "A3", "A4")}}
        self.assertTrue(all(c["status"] == "pass" for c in gpu_acceptance.validate_summary(summary, manifest)))
        for key, value in (("all_generations_completed", False), ("completed_rows", 3),
                           ("arms", ["A0", "A1"]), ("quality_qualified", True)):
            modified = {**summary, key: value}
            self.assertTrue(any(c["status"] == "fail" for c in gpu_acceptance.validate_summary(modified, manifest)))

    def test_doctor_checks_real_gate_inputs_and_unknown_readiness_is_not_pass(self):
        gpu = {"torch": "2.15.0+cu132", "torchvision": "0.29.2+cu132", "cuda_runtime": "13.2",
               "available": True, "devices": [{"capability": [12, 0], "bf16_supported": True,
               "bf16_matmul_ok": True}]}
        with patch.object(doctor.platform, "system", return_value="Linux"), \
             patch.object(doctor.platform, "machine", return_value="x86_64"), \
             patch.object(doctor, "probe_cuda", return_value=gpu), \
             patch.object(doctor, "probe_driver", return_value=[{"driver_version": "595.71.05"}]), \
             patch("harness_ocr.models.verify_models", return_value={"ok": True, "checked_files": 1}), \
             patch.object(doctor.subprocess, "run", return_value=types.SimpleNamespace(returncode=0)), \
             patch.object(doctor, "request_ready", return_value={"ready": True, "fixture": False}) as ready:
            self.assertEqual(doctor.run_doctor("unused", base_url="http://local")["status"], "pass")
            ready.return_value = {}
            self.assertEqual(doctor.run_doctor("unused", base_url="http://local")["status"], "fail")
            ready.return_value = {"ready": True, "fixture": True}
            self.assertEqual(doctor.run_doctor("unused", base_url="http://local")["status"], "fail")
            ready.return_value = {"ready": True, "fixture": False}
            gpu["devices"][0]["bf16_matmul_ok"] = False
            self.assertEqual(doctor.run_doctor("unused", base_url="http://local")["status"], "fail")

    def test_source_archive_excludes_runtime_data_and_verifies_hash_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            for name, data in {"src/harness_ocr/example.py": "print('source')", "src/harness_ocr/data/lock.json": "{}",
                               "models/weights.safetensors": "private", ".env": "SECRET=do-not-send",
                               "reports/passport.json": "private-document", "src/demo/__pycache__/cached.pyc": "cache",
                               "src/pkg.egg-info/requires.txt": "generated", "README_LINUX_GPU.txt": "handoff"}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(data, encoding="utf-8")
            output = Path(directory) / "source.zip"
            result = build_release.build_release(output, root)
            with zipfile.ZipFile(output) as archive:
                names = archive.namelist()
                self.assertEqual(len(names), 4)
                self.assertFalse(any("models/" in name or "reports/" in name or ".env" in name for name in names))
                manifest = json.loads(archive.read("harness-ocr/source-manifest.json"))
                for item in manifest["files"]:
                    self.assertEqual(hashlib.sha256(archive.read("harness-ocr/" + item["path"])).hexdigest(), item["sha256"])
            self.assertEqual(result["source_files"], 3)
            with self.assertRaises(FileExistsError):
                build_release.build_release(output, root)

    def test_embedded_credential_blocks_archive_before_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            (root / "README_SECRET.txt").write_text("-----BEGIN " + "PRIVATE KEY-----", encoding="utf-8")
            output = Path(directory) / "source.zip"
            with self.assertRaisesRegex(ValueError, "credential"):
                build_release.build_release(output, root)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
