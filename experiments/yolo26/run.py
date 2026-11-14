"""Serial paired detector comparison, with separate optical and E2E stages.

No model output is reused between detector arms. References are opened only by
the post-run report. All planned rows, including missing/censored rows, remain
visible. Run with: python -m experiments.yolo26.run --help
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from importlib import metadata
import json
import platform
from pathlib import Path
import time

from harness_ocr.engine import Engine
from harness_ocr.inference.supervisor import Supervisor
from harness_ocr.models import model_lock, verify_models
from harness_ocr.settings import Settings
from harness_ocr.util import digest, file_sha, utc, write_json
from .metrics import ResourceSampler


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def code_pins():
    root = Path(__file__).resolve().parents[2]
    paths = list((root / 'src/harness_ocr').rglob('*.py'))
    paths += list((root / 'experiments/yolo26').glob('*.py'))
    return {p.relative_to(root).as_posix(): file_sha(p) for p in sorted(paths)}


def load_dataset(data):
    data = Path(data).resolve()
    freeze = read(data / 'data-freeze.json')
    for filename, key in [('inference.json', 'inference_sha256'), ('references.json', 'references_sha256')]:
        if file_sha(data / filename) != freeze[key]:
            raise ValueError('data_freeze_changed:' + filename)
    whitelist = read(data / 'inference.json')
    if whitelist.get('schema_version') != 'yolo26-inference/1' or whitelist.get('contains_ground_truth') is not False:
        raise ValueError('inference_whitelist_invalid')
    docs = whitelist.get('documents', [])
    if not docs or len({d['id'] for d in docs}) != len(docs):
        raise ValueError('empty_or_duplicate_documents')
    forbidden = {'gold', 'fields', 'expected_fields', 'optical_literals', 'expected_text'}
    for doc in docs:
        if forbidden.intersection(doc):
            raise ValueError('ground_truth_in_inference_whitelist')
        path = Path(doc['path'])
        if not path.is_absolute():
            path = data / path
        if file_sha(path) != doc['sha256']:
            raise ValueError('source_changed:' + doc['id'])
        doc['path'] = str(path.resolve())
    return whitelist, freeze


def _run_optics(owner, settings, doc, output, global_deadline):
    deadline = min(global_deadline, time.perf_counter() + settings.document_timeout_seconds)
    ingested = owner.call('ingest', {'path': doc['path'], 'output': str(output / 'rasters'),
        'profile_id': doc['profile_id']}, timeout=min(settings.call_timeout_seconds,
                                                   max(.001, deadline-time.perf_counter())))
    if ingested['source_sha256'] != doc['sha256']:
        raise ValueError('ingest_source_mismatch')
    if ingested.get('observations') or not ingested.get('pages'):
        raise ValueError('optical_benchmark_requires_raster_pages')
    pages = []
    for page in ingested['pages']:
        left = deadline - time.perf_counter()
        if left <= 0:
            raise TimeoutError('optical_document_budget_exhausted')
        value = owner.call('cheap', {'page': page, 'source_sha256': doc['sha256'],
            'output_dir': str(output / 'regions' / page['page_id'])},
            timeout=min(left, settings.call_timeout_seconds))
        pages.append(value)
    return {'completed': all(p.get('completed') is True for p in pages),
            'observations': [o for p in pages for o in p['observations']], 'pages': pages,
            'source_sha256': doc['sha256'], 'scope': 'detector_plus_same_CTC_only_no_Xiaomi_or_Qwen_calls'}


def run(data, output, settings, *, stage='e2e', arms=('db', 'yolo26l', 'yolo26x'),
        ids=None, total_seconds=7200, repeats=1, owner_factory=Supervisor,
        sampler_factory=ResourceSampler, fixture=False):
    if stage not in {'optics', 'e2e'} or repeats not in {1, 2, 3}:
        raise ValueError('invalid_stage_or_repeats')
    if not arms or len(set(arms)) != len(arms) or not set(arms) <= {'db', 'yolo26l', 'yolo26x'}:
        raise ValueError('invalid_detector_arms')
    if not isinstance(total_seconds, (int, float)) or total_seconds <= 0:
        raise ValueError('invalid_total_compute_budget')
    whitelist, data_freeze = load_dataset(data)
    docs = whitelist['documents']
    if ids:
        if len(set(ids)) != len(ids) or not set(ids) <= {d['id'] for d in docs}:
            raise ValueError('unknown_or_duplicate_requested_document_ids')
        docs = [d for d in docs if d['id'] in ids]
    if any(stage not in d.get('allowed_stages', ['optics', 'e2e']) for d in docs):
        raise ValueError('document_not_supported_in_selected_stage')
    output = Path(output).resolve()
    if output.exists():
        raise ValueError('run_output_must_be_new_no_automatic_resume')
    if not fixture:
        for arm in arms:
            selected = ['detector' if arm == 'db' else arm + '-text', 'cyrillic']
            if stage == 'e2e':
                selected += ['xiaomi', settings.qwen]
            verified = verify_models(settings.models_dir, selected)
            if not verified['ok']:
                raise ValueError('unverified_models:' + arm + ':' + str(verified.get('errors')))
        # Fail before model compute if the required memory instrumentation is absent.
        import psutil  # noqa: F401
    settings.validate()
    pins = code_pins()
    versions = {}
    for name in ('torch', 'transformers', 'onnxruntime', 'ultralytics', 'ultralytics-opencv-headless', 'numpy', 'psutil', 'nvidia-ml-py'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    planned = [{'document_id': d['id'], 'original_id': d['original_id'], 'arm': arm,
                'repeat': repeat, 'cohort': d['cohort']}
               for repeat in range(repeats) for arm in arms for d in docs]
    protocol = {'schema_version': 'yolo26-comparison-protocol/1', 'created_utc': utc(),
        'stage': stage, 'fixture': fixture, 'data_freeze': data_freeze,
        'inference_sha256': file_sha(Path(data) / 'inference.json'),
        'references_sha256': file_sha(Path(data) / 'references.json'),
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(settings).items()},
        'model_lock': model_lock(), 'code_pins': pins, 'versions': versions,
        'hardware': {'platform': platform.platform(), 'processor': platform.processor()},
        'planned_rows': planned, 'arm_order': list(arms), 'repeats': repeats,
        'order_caveat': 'arm_major_blocks; thermal_and_order_effects_not_eliminated; repeat_order_fixed',
        'total_budget_seconds': total_seconds,
        'e2e': {'Xiaomi': 'unchanged', 'Qwen': settings.qwen, 'mode': 'hybrid',
                'shared_optics': False, 'allow_reread': True, 'strict_checks': 'unchanged'},
        'cold_load': 'separate_probe_warmup_e2e; optics_first_document_includes_detector_load',
        'first_timed_document': 'first_in_block_may_include_lazy_kernel_setup_not_steady_state',
        'metrics': 'exact_raw_scalar_fields_and_false_accept_on_scored_fields; optics_CTC_literal_recovery_separate',
        'sampling_unit': 'original_id; synthetic_and_physical_cohorts_separate',
        'independent_new_holdout': False, 'quality_qualified': False}
    protocol['protocol_sha256'] = digest(protocol)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)
    started = time.perf_counter()
    rows, loads = [], []
    budget_exhausted = False
    try:
        for repeat in range(repeats):
            for arm in arms:
                if time.perf_counter() - started >= total_seconds:
                    budget_exhausted = True
                    break
                arm_settings = replace(settings, text_detector=arm,
                    data_dir=output / 'state' / f'{arm}-r{repeat}')
                owner = owner_factory(arm_settings)
                try:
                    load_start = time.perf_counter()
                    with sampler_factory(owner, settings.device) as resources:
                        probe = owner.call('probe', {'warmup': stage == 'e2e'},
                            timeout=min(settings.call_timeout_seconds, total_seconds - (time.perf_counter() - started)))
                    loads.append({'arm': arm, 'repeat': repeat,
                        'wall_seconds': time.perf_counter() - load_start, 'probe': probe,
                        'resources': resources.result()})
                    write_json(output / 'loads.json', loads)
                    engine = Engine(arm_settings, owner) if stage == 'e2e' else None
                    for doc_index, doc in enumerate(docs):
                        if time.perf_counter() - started >= total_seconds:
                            budget_exhausted = True
                            break
                        if code_pins() != pins:
                            raise RuntimeError('implementation_changed_after_protocol_freeze')
                        if file_sha(Path(doc['path'])) != doc['sha256']:
                            raise RuntimeError('source_changed_after_protocol_freeze')
                        row_id = digest([protocol['protocol_sha256'], doc['id'], arm, repeat])[:24]
                        row_dir = output / 'rows' / row_id
                        row_dir.mkdir(parents=True)
                        row = {'document_id': doc['id'], 'original_id': doc['original_id'],
                            'arm': arm, 'repeat': repeat, 'cohort': doc['cohort'],
                            'source_sha256': doc['sha256'], 'first_timed_document_in_block': doc_index == 0,
                            'result_path': f'rows/{row_id}/result.json', 'status': 'running'}
                        write_json(output / 'status.json', {'status': 'running', 'current': row,
                            'completed_rows': len(rows), 'planned_rows': len(planned)})
                        try:
                            # Each arm executes its own optics. No reference values enter Engine.run.
                            with sampler_factory(owner, settings.device) as resources:
                                result = (_run_optics(owner, arm_settings, doc, row_dir, started + total_seconds)
                                    if stage == 'optics' else engine.run('yolo_' + row_id,
                                        doc['path'], doc['sha256'], doc['profile_id'],
                                        subset=doc['subset'], qwen=settings.qwen,
                                        mode='hybrid', allow_reread=True,
                                        cancel=lambda: time.perf_counter() >= started + total_seconds,
                                        emit_operational_events=False))
                            optics = result.pop('optics', None)
                            if optics is not None:
                                write_json(row_dir / 'optics.json', optics)
                                row['optics_path'] = f'rows/{row_id}/optics.json'
                            write_json(row_dir / 'result.json', result)
                            row.update(status='completed' if result.get('completed') is True else 'incomplete',
                                       resources=resources.result())
                        except Exception as exc:
                            row.update(status='failed', error=type(exc).__name__ + ':' + str(exc)[:500])
                            if 'resources' in locals():
                                row['resources'] = resources.result()
                        rows.append(row)
                        write_json(row_dir / 'measurement.json', row)
                        write_json(output / 'measurements.json', rows)
                finally:
                    owner.close()
                if budget_exhausted:
                    break
            if budget_exhausted:
                break
    except BaseException as exc:
        write_json(output / 'status.json', {'status': 'failed', 'error': type(exc).__name__ + ':' + str(exc)[:500],
            'completed_rows': len(rows), 'planned_rows': len(planned), 'automatic_restart': False})
        raise
    status = {'status': 'budget_exhausted' if budget_exhausted else 'completed',
              'completed_rows': len(rows), 'planned_rows': len(planned), 'finished_utc': utc(),
              'automatic_restart': False}
    write_json(output / 'status.json', status)
    from .report import report
    return report(output, Path(data) / 'references.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--stage', choices=['optics', 'e2e'], default='e2e')
    p.add_argument('--arms', nargs='+', choices=['db', 'yolo26l', 'yolo26x'], default=['db', 'yolo26l', 'yolo26x'])
    p.add_argument('--ids', nargs='+')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--qwen', choices=['qwen-0.8b', 'qwen-4b'], default='qwen-0.8b')
    p.add_argument('--total-seconds', type=float, default=7200)
    p.add_argument('--repeats', type=int, choices=[1, 2, 3], default=1)
    p.add_argument('--call-timeout', type=float, default=120)
    p.add_argument('--document-timeout', type=float, default=600)
    args = p.parse_args()
    settings = replace(Settings.from_env(), models_dir=args.models.resolve(), data_dir=args.output / 'state',
        device=args.device, dtype='float32' if args.device == 'cpu' else 'bfloat16',
        qwen=args.qwen, call_timeout_seconds=args.call_timeout,
        document_timeout_seconds=args.document_timeout)
    summary = run(args.data, args.output, settings, stage=args.stage, arms=tuple(args.arms),
                  ids=args.ids, total_seconds=args.total_seconds, repeats=args.repeats)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
