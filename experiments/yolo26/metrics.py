"""Scoring and sampled process measurements; contains no inference code."""
from __future__ import annotations
import os
import re
import threading
import time


def score_fields(result, gold):
    """Exact raw literals; validation-pass is distinct from automatic acceptance."""
    rows = []
    for key, expected in gold.items():
        field = result.get('fields', {}).get(key, {})
        observed = field.get('extraction_status') == 'observed' and field.get('raw_value') is not None
        same = observed and field['raw_value'] == expected
        accepted = (observed and result.get('decision') == 'accept_for_autofill'
                    and field.get('decision') == 'accept_for_autofill'
                    and field.get('validation_status') == 'pass' and not field.get('needs_review'))
        rows.append({'field_key': key, 'correct_literal': bool(same),
            'wrong_literal': bool(observed and not same), 'withheld': not observed,
            'validation_pass': observed and field.get('validation_status') == 'pass',
            'wrong_validation_pass': bool(observed and not same and field.get('validation_status') == 'pass'),
            'accepted_for_autofill': accepted, 'wrong_accepted': bool(accepted and not same)})
    return {'completed': result.get('completed') is True, 'fields': rows, 'total': len(rows),
            **{key: sum(bool(r[key]) for r in rows) for key in (
                'correct_literal', 'wrong_literal', 'withheld', 'validation_pass',
                'wrong_validation_pass', 'accepted_for_autofill', 'wrong_accepted')},
            'document_decision': result.get('decision'),
            'all_scored_fields_correct': bool(rows) and all(r['correct_literal'] for r in rows)}


def score_optical_literals(observations, literals):
    """Diagnostic of transcription recovery, NOT field extraction or full-page CER.

    Exact substring within one source observation, never concatenation of lines;
    a numeric literal must not be a suffix/prefix of a larger numeric run.
    """
    texts = [o.get('text', '') for o in observations if o.get('completed') is True]
    rows = []
    for literal in literals:
        pattern = re.escape(literal)
        if literal and literal[0].isdigit():
            pattern = r'(?<!\d)' + pattern
        if literal and literal[-1].isdigit():
            pattern += r'(?!\d)'
        rows.append({'literal': literal, 'recovered': any(re.search(pattern, t) for t in texts)})
    return {'scope': 'role_free_exact_literal_recovery_in_CTC_only_not_field_accuracy',
            'total': len(rows), 'recovered': sum(r['recovered'] for r in rows), 'literals': rows}


class ResourceSampler:
    """Sample runner + current supervisor worker RSS and attributable GPU memory.

    RSS sum may double-count shared pages. It is a sampled peak, not a guaranteed
    OS high-water mark. GPU memory is only for matching process IDs, never an
    unrelated whole-device memory delta. A missing GPU measurement stays null.
    """
    def __init__(self, owner, device, interval=.05):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError('benchmark_requires_psutil_before_compute') from exc
        self.psutil, self.owner, self.device = psutil, owner, device
        self.interval, self.stop_event = interval, threading.Event()
        self.samples, self.peak_rss, self.peak_worker_rss = 0, None, None
        self.peak_gpu, self.gpu_samples, self.pid_peaks = None, 0, {}
        self.errors, self.nvml, self.handles = [], None, []
        if device == 'cuda':
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                                for i in range(pynvml.nvmlDeviceGetCount())]
            except Exception as exc:
                self.errors.append('NVML_unavailable:' + type(exc).__name__)

    def _sample(self):
        pids = {os.getpid()}
        process = getattr(self.owner, 'process', None)
        if process is not None:
            try:
                if process.pid:
                    pids.add(process.pid)
            except (ValueError, AttributeError):
                pass
        worker_pids = pids - {os.getpid()}
        total, worker_total = 0, 0
        for pid in pids:
            try:
                rss = self.psutil.Process(pid).memory_info().rss
                total += rss
                if pid in worker_pids:
                    worker_total += rss
                self.pid_peaks[str(pid)] = max(self.pid_peaks.get(str(pid), 0), rss)
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue
        self.samples += 1
        self.peak_rss = max(self.peak_rss or 0, total)
        if worker_pids:
            self.peak_worker_rss = max(self.peak_worker_rss or 0, worker_total)
        if self.nvml:
            measured, total_gpu = False, 0
            for handle in self.handles:
                try:
                    processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in processes:
                        if p.pid in pids:
                            value = getattr(p, 'usedGpuMemory', None)
                            if isinstance(value, int) and 0 <= value < (1 << 63):
                                total_gpu += value
                                measured = True
                except Exception as exc:
                    name = 'NVML_sample_error:' + type(exc).__name__
                    if name not in self.errors:
                        self.errors.append(name)
            if measured:
                self.gpu_samples += 1
                self.peak_gpu = max(self.peak_gpu or 0, total_gpu)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.started = time.perf_counter()
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=2)
        self._sample()
        if self.nvml:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass
        self.finished = time.perf_counter()

    def result(self):
        return {'wall_seconds': getattr(self, 'finished', time.perf_counter()) - self.started,
            'sample_interval_seconds': self.interval, 'samples': self.samples,
            'runner_plus_worker_peak_sampled_rss_bytes': self.peak_rss,
            'worker_peak_sampled_rss_bytes': self.peak_worker_rss,
            'per_pid_peak_sampled_rss_bytes': self.pid_peaks,
            'peak_sampled_attributable_gpu_bytes': self.peak_gpu,
            'gpu_samples': self.gpu_samples, 'measurement_errors': self.errors,
            'scope': 'sampled_runner_plus_supervisor_worker_RSS; shared_pages_may_be_counted_twice; GPU_driver_compute_process_bytes_matching_PIDs_only',
            'not_measured': ['GPU_utilization', 'energy', 'true_unsampled_OS_high_water', 'service_concurrency_capacity']}
