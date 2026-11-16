"""Export allowlisted counts/resources only; no inputs, literals, logs or secrets.

Host Python standard library only. The ZIP is a return packet, never a model,
document, trace or qualification archive. All outputs are create-only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

STATES = frozenset(('pass','fail','pending','prepared','running','completed','incomplete',
    'failed','missing','budget_exhausted','stopped','canceled','cancelled','interrupted',
    'skipped','not_run','unknown','administratively_censored'))
ARMS = frozenset(f'{q}_{r}' for q in ('qwen08b','qwen4b')
                 for r in ('db','db_yolo26l','db_yolo26x'))
COUNTS = ('requested_fields','observed_fields','validation_pass_fields','review_fields',
          'quality_review_fields','model_generation_calls')
RESOURCES = ('wall_seconds','samples','gpu_samples','sample_interval_seconds',
    'runner_plus_worker_peak_sampled_rss_bytes','worker_peak_sampled_rss_bytes',
    'peak_sampled_attributable_gpu_bytes')
SCORE_KEYS = frozenset(('correct','correct_literal','correct_validation_pass','wrong',
    'wrong_literal','withheld','unknown','unknown_fields','unknown_reference_fields',
    'planned_fields','total','scored_fields','unscored_fields','accepted','wrong_accepted',
    'wrong_validation_pass','has_result','completed','all_fields_correct','all_scored_fields_exact',
    'measured','missing_result'))


def read_optional(root, relative):
    path = root / relative
    if not path.exists():
        return None
    if path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError('report_source_outside_run')
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError('report_source_too_large')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,UnicodeError):
        return {'status':'unknown','export_read_error':'invalid_json'}
    if not isinstance(value,dict):
        raise ValueError('report_source_not_object')
    return value


def number(value):
    return value if type(value) in (int,float) and math.isfinite(value) and value >= 0 else None


def boolean(value):
    return value if type(value) is bool else None


def state(value):
    return value if isinstance(value,str) and value in STATES else 'unknown'


def fingerprint(value):
    return value if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) else None


def resource(value):
    if not isinstance(value,dict):
        return None
    result={k:number(value.get(k)) for k in RESOURCES}
    errors=value.get('measurement_errors')
    result['measurement_error_count']=len(errors) if isinstance(errors,list) else None
    return result


def score_counts(value):
    # Scores can contain field-level literals. Only bounded scalar counts leave
    # the private run, even when an upstream report adds new keys.
    if not isinstance(value,dict):
        return None
    result={k:(boolean(v) if type(v) is bool else number(v))
            for k,v in value.items() if k in SCORE_KEYS}
    if value.get('measured') is True and isinstance(value.get('fields'),list):
        result['correct_validation_pass']=sum(isinstance(f,dict) and f.get('correct') is True
            and f.get('validation_pass') is True for f in value['fields'])
    if value.get('measured') is False and number(value.get('planned_fields')) is not None:
        result['unknown_reference_fields']=number(value.get('planned_fields'))
    return result


def public_summary(raw):
    if not isinstance(raw,dict) or raw.get('schema_version')!='harness-gpu-delivery-summary/1':
        return {'status':'not_run' if raw is None else 'unknown','rows':[],
                'technical_execution_passed':False,'correctness_measured':False}
    result={'schema_version':raw['schema_version'],'status':state(raw.get('status')),
        'quality_qualified':False,'resources_qualified':False,
        'technical_execution_passed':boolean(raw.get('technical_execution_passed')) is True,
        'integrity_verification_passed':boolean(raw.get('integrity_verification_passed')),
        'resource_measurement_complete':boolean(raw.get('resource_measurement_complete')),
        'model_owner_parallelism':number(raw.get('model_owner_parallelism')),
        'automatic_retries':number(raw.get('automatic_retries')),
        'new_worker_per_row':boolean(raw.get('new_worker_per_row')),
        'shared_final_optics_or_claims':boolean(raw.get('shared_final_optics_or_claims'))}
    for k in ('planned_rows','recorded_rows','missing_rows','input_documents',
              'exact_duplicate_source_groups','elapsed_wall_seconds_including_preflight_and_close'):
        result[k]=number(raw.get(k))
    for k in ('protocol_sha256','source_code_sha256','model_registry_subset_sha256',
              'model_content_pin_sha256','manifest_sha256'):
        result[k]=fingerprint(raw.get(k))
    caps=raw.get('common_caps',{})
    result['common_caps']={k:number(caps.get(k)) for k in ('max_input_tokens','max_new_tokens',
        'call_timeout_seconds','document_timeout_seconds','max_rereads','max_visual_regions')}
    rows=[]
    incoming=raw.get('rows',[])
    if not isinstance(incoming,list) or len(incoming)>600:
        raise ValueError('summary_rows_outside_supported_scope')
    for row in incoming:
        if not isinstance(row,dict) or row.get('arm') not in ARMS:
            raise ValueError('unknown_summary_arm')
        doc=row.get('public_document_id')
        identity=doc if isinstance(doc,str) and re.fullmatch(r'doc_[0-9a-f]{20}',doc) else 'unknown'
        r={'document':identity,'arm':row['arm'],'status':state(row.get('status')),
           'technical_execution_passed':boolean(row.get('technical_execution_passed')) is True,
           'administrative_censor':boolean(row.get('administrative_censor')),
           'full_row_wall_seconds':number(row.get('full_row_wall_seconds')),
           'preflight_wall_seconds':number(row.get('preflight_wall_seconds')),
           'rescue_activated':boolean(row.get('rescue_activated')),
           'max_call_cuda_allocated_bytes':number(row.get('max_call_cuda_allocated_bytes')),
           'max_call_cuda_reserved_bytes':number(row.get('max_call_cuda_reserved_bytes'))}
        for k in COUNTS:r[k]=number(row.get(k))
        for k in ('warmup_resources','document_resources','last_stage_resources'):
            r[k]=resource(row.get(k))
        rows.append(r)
    result['rows']=rows
    q=raw.get('quality')
    result['references_provided']=isinstance(q,dict) and (q.get('references_provided') is True
        or q.get('measured') is True or fingerprint(q.get('gold_sha256')) is not None)
    result['reference_field_count_per_arm']=number(q.get('reference_field_count_per_arm')) if isinstance(q,dict) else None
    result['quality_scores']=[]
    if result['references_provided']:
        for row in q.get('rows',[]):
            if not isinstance(row,dict) or row.get('arm') not in ARMS:
                raise ValueError('unknown_quality_arm')
            doc=row.get('public_document_id')
            result['quality_scores'].append({'document':doc if isinstance(doc,str)
                and re.fullmatch(r'doc_[0-9a-f]{20}',doc) else 'unknown',
                'arm':row['arm'],'score':score_counts(row.get('score'))})
    result['correctness_measured']=any(isinstance(r.get('score'),dict)
        and r['score'].get('measured') is True for r in result['quality_scores'])
    if len(rows)!=result['planned_rows'] or result['missing_rows'] != sum(r['status']=='missing' for r in rows):
        raise ValueError('summary_coverage_mismatch')
    if result['technical_execution_passed'] and (not rows or any(not r['technical_execution_passed'] for r in rows)):
        raise ValueError('summary_false_technical_pass')
    return result


def sanitize_environment(value):
    if not isinstance(value,dict):return {'status':'not_run','checks':[]}
    checks=[]
    allowed={'python','platform','candidate_runtime','cuda_bf16_target_kernel','driver_release_gate',
             'model_sha256','dependency_consistency','api_readiness'}
    for c in value.get('checks',[]):
        if isinstance(c,dict) and c.get('check') in allowed:
            checks.append({'check':c['check'],'status':state(c.get('status'))})
    result={'status':state(value.get('status')),'model_generation_performed':False,
            'api_checked':boolean(value.get('api_checked')),'checks':checks,'versions':{},'devices':[]}
    version=lambda v:v if isinstance(v,str) and re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,3}(?:\+cu[0-9]+)?',v) else None
    for c in value.get('checks',[]):
        if not isinstance(c,dict):continue
        detail=c.get('detail')
        if c.get('check')=='python':result['versions']['python']=version(detail)
        if c.get('check')=='candidate_runtime' and isinstance(detail,dict):
            for k in ('torch','torchvision','cuda_runtime'):result['versions'][k]=version(detail.get(k))
        if c.get('check')=='driver_release_gate' and isinstance(detail,dict):
            result['versions']['minimum_candidate_driver']=version(detail.get('minimum_for_this_candidate'))
            result['driver_versions']=[version(x.get('driver_version')) for x in detail.get('devices',[]) if isinstance(x,dict)]
        if c.get('check')=='cuda_bf16_target_kernel' and isinstance(detail,dict):
            for dev in detail.get('devices',[]):
                if not isinstance(dev,dict):continue
                cap=dev.get('capability');cap=cap if isinstance(cap,list) and len(cap)==2 and all(type(n) is int and n>=0 for n in cap) else None
                result['devices'].append({'capability':cap,'total_memory_bytes':number(dev.get('total_memory_bytes')),
                    'bf16_supported':boolean(dev.get('bf16_supported')),'bf16_matmul_ok':boolean(dev.get('bf16_matmul_ok'))})
    return result


def sanitize_smoke(value):
    if not isinstance(value,dict):return {'status':'not_run','target_runtime_accepted':False}
    result={'status':state(value.get('status')),'quality_qualified':False}
    for k in ('actual_model_execution_verified','target_runtime_accepted','model_owners_stopped',
              'coordinator_lease_released','models_parallel','ocr_accuracy_gate','test_dependencies_injected'):
        result[k]=boolean(value.get(k))
    for k in ('call_attempts','neural_generation_calls_attempted','completed_generation_calls',
              'retries','elapsed_seconds'):
        result[k]=number(value.get(k))
    result['implementation_sha256']=fingerprint(value.get('implementation_sha256'))
    return result


def aggregate(summary):
    result=[]
    for arm in sorted({r['arm'] for r in summary['rows']}):
        rows=[r for r in summary['rows'] if r['arm']==arm]
        def total(k):
            vals=[r[k] for r in rows if r.get(k) is not None]
            return sum(vals) if vals else None
        def peaks(key):
            vals=[r[stage][key] for r in rows for stage in ('warmup_resources','document_resources')
                  if isinstance(r.get(stage),dict) and r[stage].get(key) is not None]
            return max(vals) if vals else None
        times=[r['full_row_wall_seconds'] for r in rows if r['full_row_wall_seconds'] is not None]
        item={'arm':arm,'planned':len(rows),'completed':sum(r['technical_execution_passed'] for r in rows),
            'missing':sum(r['status']=='missing' for r in rows),
            'row_wall_median_seconds':statistics.median(times) if times else None,
            'peak_sampled_rss_bytes':peaks('runner_plus_worker_peak_sampled_rss_bytes'),
            'peak_sampled_gpu_bytes':peaks('peak_sampled_attributable_gpu_bytes'),
            'rescue_activations':sum(r['rescue_activated'] is True for r in rows)}
        for k in COUNTS:item[k]=total(k)
        for k in ('correct','correct_validation_pass','wrong','withheld','unknown_reference_fields'):
            vals=[r['score'][k] for r in summary.get('quality_scores',[]) if r['arm']==arm
                  and isinstance(r.get('score'),dict) and r['score'].get(k) is not None]
            item[k]=sum(vals) if vals else None
        scores=[r['score'] for r in summary.get('quality_scores',[]) if r['arm']==arm and isinstance(r.get('score'),dict)]
        has_references=summary.get('references_provided') is True
        item['scored_fields']=sum(s.get('total',0) or 0 for s in scores if s.get('measured') is True) if has_references else None
        planned=summary.get('reference_field_count_per_arm')
        if planned is None and scores:planned=sum((s.get('total',s.get('planned_fields',0)) or 0) for s in scores)
        item['reference_fields_planned']=planned
        if has_references and planned is not None:
            item['unknown_reference_fields']=max(0,planned-item['scored_fields'])
        item['unscored_fields']=max(0,item['requested_fields']-planned) if item.get('requested_fields') is not None and planned is not None else None
        result.append(item)
    return result


def export(run_dir, output):
    root=Path(run_dir).resolve(strict=True);output=Path(output).resolve()
    targets=[output,root/'report.html',root/'metrics.csv',root/'summary-public.json']
    if output.suffix.lower()!='.zip' or any(p.exists() for p in targets):
        raise ValueError('export_outputs_must_be_new_zip_and_sidecars')
    handoff=read_optional(root,'handoff-status.json') or {}
    summary=public_summary(read_optional(root,'comparison/summary.json'))
    packet={'schema_version':'harness-gpu-return/1','created_utc':datetime.now(timezone.utc).isoformat(),
        'privacy_scope':'counts, opaque document IDs, hashes and resource measurements only',
        'contains_documents':False,'contains_ocr_values':False,'contains_prompts_or_logs':False,
        'contains_credentials_or_environment_dump':False,'quality_qualified':False,'gpu_inference_claim':False,
        'handoff':{'status':state(handoff.get('status')),
            'image_id':handoff.get('image_id') if isinstance(handoff.get('image_id'),str)
                and re.fullmatch(r'sha256:[0-9a-f]{64}',handoff['image_id']) else None,
            'stage':handoff.get('stage') if handoff.get('stage') in {'host_preflight','build','lock-init',
                'plan','prepare','verify-models','doctor','smoke','comparison','completed','prepared'} else 'unknown',
            'technical_execution_passed':boolean(handoff.get('technical_execution_passed')) is True,
            'gpu_inference_verified':boolean(handoff.get('gpu_inference_verified')) is True},
        'environment':sanitize_environment(read_optional(root,'doctor.json')),
        'smoke':sanitize_smoke(read_optional(root,'smoke/report.json')),'comparison':summary}
    packet['gpu_inference_claim']=(packet['smoke'].get('target_runtime_accepted') is True
        and packet['smoke'].get('actual_model_execution_verified') is True
        and packet['smoke'].get('test_dependencies_injected') is False
        and packet['handoff']['gpu_inference_verified'] is True)
    rows=aggregate(summary);packet['arm_totals']=rows
    cols=['arm','planned','completed','missing',*COUNTS,'row_wall_median_seconds',
          'peak_sampled_rss_bytes','peak_sampled_gpu_bytes','rescue_activations',
        'reference_fields_planned','scored_fields','unscored_fields','correct','correct_validation_pass',
        'wrong','withheld','unknown_reference_fields']
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=cols);writer.writeheader();writer.writerows(rows)
    csv_bytes=out.getvalue().encode('utf-8-sig')
    esc=lambda v:html.escape(str(v))
    display=lambda v:'—' if v is None else esc(round(v,3) if isinstance(v,float) else v)
    ratio=lambda a,b:display(a)+' / '+display(b)
    gib=lambda v:display(v/2**30 if v is not None else None)
    labels={a:('Qwen 0.8B' if a.startswith('qwen08b') else 'Qwen 4B')+' · DB'+
            (' + YOLO-L' if a.endswith('yolo26l') else ' + YOLO-X' if a.endswith('yolo26x') else '') for a in ARMS}
    headers=['Вариант','Завершено / план','Извлечено полей / запрошено','Формат pass',
             'Точно / оценено','Неверно','Без оценки','Медиана, с','RAM, GiB','VRAM, GiB','Rescue']
    page=['<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Harness OCR — GPU-тест</title><style>body{font:16px/1.5 system-ui;max-width:1280px;margin:32px auto;padding:20px}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #bbb;padding:8px}th{background:#edf2f7}.scroll{overflow:auto}.note{background:#fff1cc;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>Harness OCR: GPU-тест</h1><p>Статус запуска: '+esc(packet['handoff']['status'])+
        '; этап: '+esc(packet['handoff']['stage'])+'.</p><p><a href="metrics.csv">Все показатели CSV</a> · <a href="summary-public.json">JSON</a></p>',
        '<p class="note">Успешное исполнение, охват полей и точность — разные показатели. Review является допустимым результатом. '
        'Этот пакет не квалифицирует продукт и не содержит сканов, распознанных значений или сырых журналов.</p>',
        '<p>'+('Есть отдельные эталоны: ниже дословные оценки только размеченных полей.' if summary.get('correctness_measured')
        else 'Эталоны не предоставлены или оценка ещё не выполнена. Точность распознавания не измерена.')+'</p>',
        '<div class="scroll"><table><tr>'+''.join('<th>'+esc(c)+'</th>' for c in headers)+'</tr>']
    for row in rows:
        unknown=(row['unscored_fields'] or 0)+(row['unknown_reference_fields'] or 0) if summary.get('references_provided') else None
        cells=[esc(labels[row['arm']]),ratio(row['completed'],row['planned']),
            ratio(row['observed_fields'],row['requested_fields']),display(row['validation_pass_fields']),
            ratio(row['correct'],row['scored_fields']),display(row['wrong']),display(unknown),
            display(row['row_wall_median_seconds']),gib(row['peak_sampled_rss_bytes']),
            gib(row['peak_sampled_gpu_bytes']),display(row['rescue_activations'])]
        page.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    page+=['</table></div><p>Время включает холодный старт отдельного воркера; медианы учитывают частичные попытки. '
        'RSS — sampled сумма собственного дерева процессов, общие страницы могут считаться дважды. '
        'VRAM — sampled NVML bytes только сопоставленных PID; это отдельная величина от PyTorch allocated/reserved. '
        'Тире означает отсутствие измерения.</p><details><summary>Проверки окружения и модели</summary><pre>'+esc(json.dumps(
            {'environment':packet['environment'],'smoke':packet['smoke']},ensure_ascii=False,indent=2))+'</pre></details></html>']
    html_bytes=''.join(page).encode('utf-8')
    json_bytes=(json.dumps(packet,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    files={'report.html':html_bytes,'metrics.csv':csv_bytes,'summary-public.json':json_bytes}
    manifest={'schema_version':'harness-gpu-return-manifest/1',
        'files':{name:hashlib.sha256(data).hexdigest() for name,data in files.items()},
        'contains_raw_artifacts':False}
    files['manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').encode()
    output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in files.items():archive.writestr(name,data)
    for name,data in files.items():
        if name!='manifest.json':
            with (root/name).open('xb') as stream:stream.write(data)
    return {'packet':str(output),'files':len(files),'quality_qualified':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    print(json.dumps(export(args.run_dir,args.output),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
