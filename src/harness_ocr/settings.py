from __future__ import annotations
import os
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    device: str = 'cuda'
    dtype: str = 'bfloat16'
    max_upload_bytes: int = 30*1024*1024
    max_pages: int = 10
    max_page_pixels: int = 12_000_000
    max_input_tokens: int = 8192
    max_new_tokens: int = 1024
    call_timeout_seconds: float = 120
    document_timeout_seconds: float = 600
    max_rereads: int = 2
    qwen: str = 'qwen-0.8b'
    policy_version: str = 'review-default-v1'
    input_scope_version: str = 'raster-12mp-10pages-v1'
    model_bundle_sha256: str = ''
    allow_fixture: bool = False
    cpu_threads: int = 4
    runtime_lock_dir: Path | None = None

    @property
    def model_lock_dir(self):
        # All API/CLI/benchmark state directories share this admission namespace.
        # Containers must mount the SAME writable path; compose does this explicitly.
        configured=self.runtime_lock_dir or os.getenv('HARNESS_LOCK_DIR')
        if configured:return Path(configured).resolve()
        owner=str(os.getuid()) if hasattr(os,'getuid') else 'current-user'
        return Path(tempfile.gettempdir())/'harness-ocr-runtime-locks'/owner

    @classmethod
    def from_env(cls):
        return cls(Path(os.getenv('HARNESS_DATA', './var')).resolve(),
                   Path(os.getenv('HARNESS_MODELS', './models')).resolve(),
                   device=os.getenv('HARNESS_DEVICE','cuda'),
                   dtype=os.getenv('HARNESS_DTYPE','bfloat16'),
                   qwen=os.getenv('HARNESS_QWEN','qwen-0.8b'),
                   cpu_threads=int(os.getenv('HARNESS_CPU_THREADS','4')),
                   runtime_lock_dir=Path(os.environ['HARNESS_LOCK_DIR']).resolve() if os.getenv('HARNESS_LOCK_DIR') else None)

    def validate(self):
        if self.device not in {'cpu','cuda'} or self.dtype not in {'float32','bfloat16'}:
            raise ValueError('unsupported_device_or_dtype')
        if self.device == 'cpu' and self.dtype != 'float32': raise ValueError('cpu_requires_explicit_float32')
        if self.qwen not in {'qwen-0.8b','qwen-4b'}: raise ValueError('unknown_qwen')
        for n in ('max_upload_bytes','max_pages','max_page_pixels','max_input_tokens','max_new_tokens','max_rereads'):
            if type(getattr(self,n)) is not int or getattr(self,n) < (0 if n=='max_rereads' else 1):
                raise ValueError('invalid_limit:'+n)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
               for v in (self.call_timeout_seconds,self.document_timeout_seconds)):
            raise ValueError('invalid_timeout')
        if type(self.cpu_threads) is not int or not 1<=self.cpu_threads<=16:raise ValueError('invalid_cpu_thread_budget')
        if self.runtime_lock_dir is not None and not isinstance(self.runtime_lock_dir,Path):raise ValueError('invalid_runtime_lock_directory')
