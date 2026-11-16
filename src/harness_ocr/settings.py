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
    visual_verifier: bool = False
    max_visual_regions: int = 2
    text_detector: str = 'db'
    attention_backend: str = 'eager'
    reading_policy: str = 'adaptive'
    rescue_text_detector: str = 'none'
    max_detector_rescues: int = 1
    max_whole_page_reads: int = 1

    @property
    def rescue_enabled(self):
        return (self.reading_policy == 'adaptive' and self.rescue_text_detector != 'none'
                and self.max_detector_rescues > 0)

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
        if os.getenv('HARNESS_VISUAL_VERIFIER','0') not in {'0','1'}:raise ValueError('invalid_visual_verifier_environment')
        return cls(Path(os.getenv('HARNESS_DATA', './var')).resolve(),
                   Path(os.getenv('HARNESS_MODELS', './models')).resolve(),
                   device=os.getenv('HARNESS_DEVICE','cuda'),
                   dtype=os.getenv('HARNESS_DTYPE','bfloat16'),
                   qwen=os.getenv('HARNESS_QWEN','qwen-0.8b'),
                   text_detector=os.getenv('HARNESS_TEXT_DETECTOR','db'),
                   attention_backend=os.getenv('HARNESS_ATTENTION_BACKEND','eager'),
                   reading_policy=os.getenv('HARNESS_READING_POLICY','adaptive'),
                   rescue_text_detector=os.getenv('HARNESS_RESCUE_TEXT_DETECTOR','none'),
                   max_detector_rescues=int(os.getenv('HARNESS_MAX_DETECTOR_RESCUES','1')),
                   max_whole_page_reads=int(os.getenv('HARNESS_MAX_WHOLE_PAGE_READS','1')),
                   cpu_threads=int(os.getenv('HARNESS_CPU_THREADS','4')),
                   visual_verifier=os.getenv('HARNESS_VISUAL_VERIFIER','0')=='1',
                   runtime_lock_dir=Path(os.environ['HARNESS_LOCK_DIR']).resolve() if os.getenv('HARNESS_LOCK_DIR') else None)

    def validate(self):
        if self.device not in {'cpu','cuda'} or self.dtype not in {'float32','bfloat16'}:
            raise ValueError('unsupported_device_or_dtype')
        if self.device == 'cpu' and self.dtype != 'float32': raise ValueError('cpu_requires_explicit_float32')
        if self.qwen not in {'qwen-0.8b','qwen-4b'}: raise ValueError('unknown_qwen')
        if self.text_detector not in {'db','yolo26l','yolo26x'}:raise ValueError('unknown_text_detector')
        if self.attention_backend not in {'eager','sdpa'}:raise ValueError('unsupported_attention_backend')
        if not isinstance(self.reading_policy,str) or self.reading_policy not in {'adaptive','legacy'}:
            raise ValueError('unknown_reading_policy')
        if not isinstance(self.rescue_text_detector,str) or self.rescue_text_detector not in {'none','yolo26l','yolo26x'}:
            raise ValueError('unknown_rescue_text_detector')
        for n in ('max_upload_bytes','max_pages','max_page_pixels','max_input_tokens','max_new_tokens','max_rereads'):
            if type(getattr(self,n)) is not int or getattr(self,n) < (0 if n=='max_rereads' else 1):
                raise ValueError('invalid_limit:'+n)
        if type(self.max_detector_rescues) is not int or not 0<=self.max_detector_rescues<=2:
            raise ValueError('invalid_detector_rescue_budget')
        if type(self.max_whole_page_reads) is not int or not 0<=self.max_whole_page_reads<=self.max_pages:
            raise ValueError('invalid_whole_page_read_budget')
        if self.rescue_enabled and self.rescue_text_detector==self.text_detector:
            raise ValueError('rescue_detector_must_differ_from_primary')
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
               for v in (self.call_timeout_seconds,self.document_timeout_seconds)):
            raise ValueError('invalid_timeout')
        if type(self.cpu_threads) is not int or not 1<=self.cpu_threads<=16:raise ValueError('invalid_cpu_thread_budget')
        if type(self.visual_verifier) is not bool:raise ValueError('invalid_visual_verifier_flag')
        if type(self.max_visual_regions) is not int or not 1<=self.max_visual_regions<=4:raise ValueError('invalid_visual_region_budget')
        if self.runtime_lock_dir is not None and not isinstance(self.runtime_lock_dir,Path):raise ValueError('invalid_runtime_lock_directory')
