"""Run one sequential four-arm benchmark on the target GPU, then check coverage.

Stop the HTTP service first: benchmark and service must never own models at once.
This records execution acceptance separately from still-unqualified OCR quality.
No dataset or gold enters production routing. Reports are create-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from doctor import write_report

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"A0", "A1", "A3", "A4"}


def stop_process_group(child):
    """Bounded cleanup of the benchmark and spawned model workers on Linux."""
    if child.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(child.pid, signal.SIGTERM)
        else:
            child.terminate()
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            return
        child.wait(timeout=10)


def validate_summary(summary, manifest):
    checks = []
    def check(name, valid):
        checks.append({"check": name, "status": "pass" if valid else "fail"})
    documents = manifest.get("documents", [])
    groups = {d.get("original_id", d.get("id")) for d in documents}
    check("summary_schema", summary.get("schema_version") == "harness-bench-summary/1")
    protocol = summary.get("protocol", {})
    check("real_cuda_bf16_run", protocol.get("device") == "cuda" and protocol.get("dtype") == "bfloat16"
          and protocol.get("fixture") is False)
    check("all_four_arms", set(summary.get("arms", [])) == ARMS)
    check("paired_originals", summary.get("paired_originals") == len(groups))
    planned = len(documents) * len(ARMS)
    check("full_coverage", summary.get("completed_rows") == planned == summary.get("planned_rows"))
    check("generation_completion", summary.get("all_generations_completed") is True)
    metrics = summary.get("metrics", {})
    check("arm_document_coverage", isinstance(metrics, dict) and all(
        isinstance(metrics.get(arm), dict) and metrics[arm].get("documents") == len(documents)
        and metrics[arm].get("original_groups") == len(groups) for arm in ARMS))
    check("quality_not_falsely_qualified", summary.get("quality_qualified") is False)
    proofs = summary.get("matched_optics_proofs", [])
    by_id = {proof.get("document_id"): proof for proof in proofs if isinstance(proof, dict)}
    check("matched_optical_evidence", len(proofs) == len(documents) and all(
        d["id"] in by_id and by_id[d["id"]].get("match") is True
        and set(by_id[d["id"]].get("initial_optics_sha256", {})) == ARMS
        and set(by_id[d["id"]]["initial_optics_sha256"].values()) == {by_id[d["id"]].get("saved_optics_sha256")}
        and isinstance(by_id[d["id"]].get("saved_optics_sha256"), str)
        and len(by_id[d["id"]]["saved_optics_sha256"]) == 64 for d in documents))
    return checks


def run_gpu_benchmark(manifest_path, output, models, *, seconds=14400):
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new benchmark output directory; old runs are immutable")
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    if (manifest.get("schema_version") != "harness-bench/1" or
        manifest.get("split") not in {"dev", "blind"} or
        not isinstance(manifest.get("documents"), list) or not manifest["documents"]):
        raise ValueError("Need a nonempty explicit dev/blind harness-bench/1 manifest")
    output.mkdir(parents=True)
    benchmark_output = output / "benchmark"
    frozen = output / "acceptance-input-manifest.json"
    # This protected benchmark directory may contain gold and local paths.
    frozen.write_bytes(manifest_raw)
    command = [sys.executable, "-m", "harness_ocr.cli", "bench", "--manifest", str(manifest_path),
               "--output", str(benchmark_output), "--device", "cuda", "--models", str(Path(models).resolve())]
    report = {"schema_version": "harness-gpu-acceptance/1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "status": "pending", "quality_qualified": False, "release_status": "pending",
              "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
              "expected_arms": sorted(ARMS), "command": command,
              "checks": [], "note": "Execution coverage does not qualify production accuracy or calibrate auto-accept risk."}
    start = time.perf_counter()
    environment = os.environ.copy()
    environment["HARNESS_DEVICE"] = "cuda"
    environment["HARNESS_MODELS"] = str(Path(models).resolve())
    environment["PYTHONPATH"] = str(ROOT / "src")
    # Service tokens are unnecessary for offline benchmarking and never inherited.
    environment.pop("HARNESS_API_TOKEN", None)
    environment.pop("HARNESS_REVIEW_TOKEN", None)
    with (output / "acceptance-stdout.log").open("x", encoding="utf-8") as stdout, \
         (output / "acceptance-stderr.log").open("x", encoding="utf-8") as stderr:
        child = None
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr,
                                     start_new_session=(os.name == "posix"))
            try:
                returncode = child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                stop_process_group(child)
                report["reason"] = "Time budget exhausted; partial observations preserved; no automatic rerun"
                report["checks"].append({"check": "benchmark_process", "status": "pending"})
                return report
            report["returncode"] = returncode
            report["checks"].append({"check": "benchmark_process", "status": "pass" if returncode == 0 else "fail"})
            summary_path = benchmark_output / "summary.json"
            if not summary_path.is_file():
                report["checks"].append({"check": "summary_present", "status": "fail"})
            else:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                report["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
                report["summary_path"] = str(summary_path)
                report["checks"].extend(validate_summary(summary, manifest))
                protocol = summary.get("protocol", {})
                report["checks"].append({"check": "frozen_manifest_matches",
                    "status": "pass" if protocol.get("manifest_sha256") == report["manifest_sha256"] else "fail"})
                # Recompute hashes from actual saved observations, not solely a
                # self-reported match boolean in the summary.
                saved_match = True
                for index, document in enumerate(manifest["documents"]):
                    optics = json.loads((benchmark_output / f"optics-{index}.json").read_text(encoding="utf-8"))
                    from harness_ocr.util import digest
                    optics_sha = digest(optics)
                    for arm in ARMS:
                        row = json.loads((benchmark_output / f"result-{index}-{arm}.json").read_text(encoding="utf-8"))
                        saved_match = saved_match and row.get("initial_optics_sha256") == optics_sha
                report["checks"].append({"check": "saved_optics_hashes_recomputed",
                                         "status": "pass" if saved_match else "fail"})
            report["status"] = "fail" if any(c["status"] == "fail" for c in report["checks"]) else "pass"
            return report
        except KeyboardInterrupt:
            report["status"] = "pending"
            report["reason"] = "Interrupted by operator; partial observations preserved; no automatic rerun"
            return report
        except Exception as exc:
            report["status"] = "fail"
            report["error_type"] = type(exc).__name__
            return report
        finally:
            if child is not None:
                stop_process_group(child)
            report["elapsed_seconds"] = time.perf_counter() - start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New protected directory for raw benchmark artifacts")
    parser.add_argument("--models", default=os.getenv("HARNESS_MODELS", "./models"))
    parser.add_argument("--seconds", type=float, default=14400)
    parser.add_argument("--report", required=True, help="New create-only acceptance JSON path")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be finite and positive")
    if Path(args.report).exists():
        raise FileExistsError("Acceptance report already exists")
    report = run_gpu_benchmark(args.manifest, args.output, args.models, seconds=args.seconds)
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return {"pass": 0, "fail": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":
    if os.name == "posix":
        def interrupted(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
