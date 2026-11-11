"""Build an allowlisted source ZIP with SHA256 inventory; no models or documents.

Only maintained source/config/test/documentation files enter the archive. Runtime
data, arbitrary fixtures, .env, credentials, caches and model weights are excluded.
This is source delivery, not a Docker image or proof of GPU/OCR acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ROOT_NAMES = {"Dockerfile", "compose.yaml", "pyproject.toml", ".dockerignore", ".gitignore", "LICENSE.txt", "THIRD_PARTY_NOTICES.txt"}
EXCLUDED = {"__pycache__", ".git", ".venv", "models", "var", "reports", "outputs", "dist"}
SECRET = re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|hf_[A-Za-z0-9]{24,}")


def allowed(path):
    parts = path.parts
    if any(part in EXCLUDED or part.endswith(".egg-info") or part.startswith(".env") for part in parts):
        return False
    if len(parts) == 1:
        return (path.name in ROOT_NAMES or
                path.name.startswith("requirements-") and path.suffix == ".txt" or
                path.name.startswith("README") and path.suffix in {".txt", ".md"})
    if parts[0] == "src":
        return path.suffix in {".py", ".json"} or path.name == "README.txt"
    if parts[0] == "deployment":
        return path.name in {"base-image-lock.json","actions-lock.json"}
    if parts[0] == "LICENSES":
        return path.name == "Apache-2.0.txt"
    if parts[0] == "examples":
        return path.name == "runtime_policy.disabled.json"
    if parts[0] == "docs":
        return path.suffix in {".md",".txt",".json",".mmd",".svg"}
    if len(parts)>2 and parts[0:2]==('.github','workflows'):
        return path.suffix in {'.yml','.yaml'}
    return parts[0] in {"tests", "scripts"} and path.suffix == ".py"


def build_release(output, root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Release archive already exists")
    files = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not allowed(relative) or not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Symlink/outside-workspace source is not eligible")
        data = path.read_bytes()
        if SECRET.search(data):
            raise ValueError(f"Possible embedded credential in {relative}; source archive not created")
        files.append((relative.as_posix(), data))
    if not files:
        raise ValueError("No eligible source files")
    inventory = {"schema_version": "harness-source-delivery/1", "contains_models": False,
                 "contains_runtime_documents": False, "gpu_qualified": False,
                 "files": [{"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                           for name, data in files]}
    files.append(("source-manifest.json", json.dumps(inventory, ensure_ascii=False, indent=2).encode("utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in files:
            info = zipfile.ZipInfo("harness-ocr/" + name, date_time=(2026, 11, 11, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return {"path": str(output), "sha256": digest, "source_files": len(files) - 1,
            "models_included": False, "runtime_documents_included": False, "gpu_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_release(args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
