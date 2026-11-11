"""Target GPU preflight, model SHA verification and optional live API readiness.

No weights are loaded and no model is downloaded. The isolated CUDA child runs
only a tiny real BF16 matrix multiplication, so this is not model acceptance.
Exit codes: 0 pass, 1 fail, 2 pending. Reports are create-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GPU_PROBE = r'''
import importlib.metadata as md, json
import torch
out = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
       "available": torch.cuda.is_available(), "devices": [],
       "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else []}
for name in ("torchvision", "transformers", "nvidia-cuda-runtime", "nvidia-cuda-runtime-cu13", "nvidia-cublas", "nvidia-cublas-cu13"):
    try: out[name] = md.version(name)
    except md.PackageNotFoundError: out[name] = None
if out["available"]:
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            p = torch.cuda.get_device_properties(i)
            bf16 = torch.cuda.is_bf16_supported()
            # Known exact answer 64 tests an actual GPU BF16 kernel and transfer.
            x = torch.ones((64, 64), device=f"cuda:{i}", dtype=torch.bfloat16)
            y = x @ x
            torch.cuda.synchronize(i)
            out["devices"].append({"index": i, "name": p.name,
                "capability": list(torch.cuda.get_device_capability(i)),
                "total_memory_bytes": p.total_memory, "bf16_supported": bf16,
                "bf16_matmul_ok": bool(torch.all(y == 64).item())})
print(json.dumps(out))
'''


def write_report(path, value):
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())


def version_tuple(value):
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value or "")
    if not match:
        raise ValueError("invalid driver version")
    return tuple(int(x or 0) for x in match.groups())


def probe_cuda(timeout=90):
    child = subprocess.run([sys.executable, "-c", GPU_PROBE], capture_output=True,
                           text=True, timeout=timeout, encoding="utf-8")
    if child.returncode:
        # Don't copy arbitrary worker stdout or environment into a handoff report.
        raise RuntimeError("isolated_cuda_probe_failed (run doctor inside the GPU image)")
    return json.loads(child.stdout.strip().splitlines()[-1])


def probe_driver():
    child = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=15, encoding="utf-8")
    if child.returncode:
        raise RuntimeError("nvidia-smi failed")
    rows = []
    for line in child.stdout.strip().splitlines():
        index, name, version, memory = [item.strip() for item in line.split(",", 3)]
        rows.append({"index": int(index), "name": name, "driver_version": version,
                     "memory_mib": int(memory)})
    return rows


def request_ready(base_url, token=None):
    request = urllib.request.Request(base_url.rstrip("/") + "/health/ready")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read(4 * 1024 * 1024))
        if response.status != 200:
            raise RuntimeError("readiness status is not 200")
        return data


def run_doctor(models_dir, *, names=None, expected_capability=(12, 0),
               min_driver="595.71.05", base_url=None):
    names = names or ["xiaomi", "qwen-0.8b", "qwen-4b", "detector", "cyrillic"]
    checks = []
    def add(name, status, detail):
        checks.append({"check": name, "status": status, "detail": detail})

    add("python", "pass" if sys.version_info[:2] == (3, 12) else "fail", platform.python_version())
    add("platform", "pass" if platform.system() == "Linux" and platform.machine() in {"x86_64", "AMD64"}
        else "fail", {"system": platform.system(), "machine": platform.machine()})
    try:
        cuda = probe_cuda()
        valid_version = (str(cuda.get("torch", "")).split("+")[0] == "2.15.0" and
                         str(cuda.get("torchvision", "")).split("+")[0] == "0.29.2" and
                         cuda.get("cuda_runtime") == "13.2")
        add("candidate_runtime", "pass" if valid_version else "fail", cuda)
        devices = cuda.get("devices", [])
        valid_gpu = (cuda.get("available") is True and len(devices) > 0 and
                     all(d.get("bf16_supported") is True and d.get("bf16_matmul_ok") is True
                         and tuple(d.get("capability", [])) == tuple(expected_capability) for d in devices))
        add("cuda_bf16_target_kernel", "pass" if valid_gpu else "fail",
            {"expected_capability": list(expected_capability), "devices": devices})
    except (Exception, subprocess.TimeoutExpired) as exc:
        add("cuda_bf16_target_kernel", "fail", {"error_type": type(exc).__name__})
    try:
        drivers = probe_driver()
        valid_driver = bool(drivers) and all(version_tuple(d["driver_version"]) >= version_tuple(min_driver)
                                             for d in drivers)
        add("driver_release_gate", "pass" if valid_driver else "fail",
            {"minimum_for_this_candidate": min_driver, "devices": drivers,
             "scope": "Conservative candidate release gate; not CUDA13 generic compatibility minimum"})
    except (Exception, subprocess.TimeoutExpired) as exc:
        add("driver_release_gate", "fail", {"error_type": type(exc).__name__})
    try:
        from harness_ocr.models import verify_models
        result = verify_models(models_dir, names)
        add("model_sha256", "pass" if result["ok"] else "fail", result)
    except Exception as exc:
        add("model_sha256", "fail", {"error_type": type(exc).__name__})
    try:
        child = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True,
                               text=True, timeout=45, encoding="utf-8")
        add("dependency_consistency", "pass" if child.returncode == 0 else "fail",
            {"returncode": child.returncode})
    except (Exception, subprocess.TimeoutExpired) as exc:
        add("dependency_consistency", "fail", {"error_type": type(exc).__name__})
    if base_url:
        try:
            ready = request_ready(base_url, os.getenv("HARNESS_API_TOKEN"))
            # Readiness endpoint is responsible for actual weights/runtime status;
            # explicit false indicators cannot be converted to a preflight pass.
            valid = (isinstance(ready, dict) and ready.get("ready", ready.get("ok", False)) is True
                     and ready.get("fixture") is False)
            add("api_readiness", "pass" if valid else "fail", ready)
        except Exception as exc:
            add("api_readiness", "fail", {"error_type": type(exc).__name__})
    else:
        add("api_readiness", "pending", "Pass --base-url after starting the service")
    status = "fail" if any(c["status"] == "fail" for c in checks) else (
        "pending" if any(c["status"] == "pending" for c in checks) else "pass")
    return {"schema_version": "harness-doctor/1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": status, "checks": checks,
            "quality_qualified": False, "model_generation_performed": False,
            "image_digest_verified": False,
            "note": "Preflight only. Actual model smoke, paired benchmark and independent quality gates remain separate."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=os.getenv("HARNESS_MODELS", "./models"))
    parser.add_argument("--model", action="append", dest="names")
    parser.add_argument("--expected-capability", default="12.0")
    parser.add_argument("--min-driver", default="595.71.05")
    parser.add_argument("--base-url")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    capability = tuple(int(n) for n in args.expected_capability.split("."))
    if len(capability) != 2:
        parser.error("--expected-capability needs major.minor")
    report = run_doctor(args.models, names=args.names, expected_capability=capability,
                        min_driver=args.min_driver, base_url=args.base_url)
    write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return {"pass": 0, "fail": 1, "pending": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
