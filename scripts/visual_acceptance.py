"""Run sequential, synthetic VLM functional checks on the target machine.

CUDA/BF16 and both pinned Qwen models are the defaults. --device cpu explicitly
uses float32. Only constructed images enter this diagnostic. It is not an
independent OCR quality study, risk calibration, or an acceptance policy.
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
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
QWENS = ('qwen-0.8b', 'qwen-4b')
LITERALS = {'qwen-0.8b': ('123456', '654321'), 'qwen-4b': ('654321', '123456')}


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _flush(path, report, started, stage=None):
    if stage is not None:report['stage'] = stage
    report['updated_utc'] = _utc()
    report['elapsed_seconds'] = time.perf_counter() - started
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary, path)


def _require(condition, reason):
    if not condition:raise ValueError(reason)


def _validate_generation(result, *, name, settings, spec, provenance):
    """Inspect actual image/generation proof before comparing any proposal."""
    from harness_ocr.core.common import sha256
    from harness_ocr.inference.constraints import build_response_schema, ENGINE
    from harness_ocr.inference.vision import VERSION, PREPROCESS, blind_prompt, compare_visual_capture
    from harness_ocr.util import digest
    _require(isinstance(result, dict), 'visual_backend_result_not_object')
    for key, expected in {'completed':True,'stop_reason':'eos','fresh_generation_state':True,
        'modality':'vision','purpose':'blind_optical_capture','adapter_version':VERSION,
        'model_id':name,'device':settings.device,'dtype':settings.dtype,
        'processor_class':'Qwen3VLProcessor','json_constraint_applied':True,
        'proposed_value_supplied':False,'ground_truth_supplied':False,
        'automatic_acceptance':False,'statistical_independence_claimed':False}.items():
        value = result.get(key)
        _require(value == expected and (not isinstance(expected,bool) or value is expected),
                 'visual_generation_contract_mismatch:' + key)
    _require(result.get('effective_parameter_dtypes') == ['torch.' + settings.dtype],
             'visual_effective_parameter_dtype_mismatch')
    _require(result.get('model_bundle_sha256') == digest(spec), 'visual_pinned_model_sha_mismatch')
    expected_prompt = blind_prompt([p['region_id'] for p in provenance])
    _require(result.get('blind_prompt') == expected_prompt, 'visual_blind_prompt_mismatch')
    _require(result.get('blind_prompt_sha256') == digest(expected_prompt), 'visual_blind_prompt_sha_mismatch')
    _require(result.get('response_schema_sha256') == digest(build_response_schema(expected_prompt,'visual_read')),
             'visual_response_schema_sha_mismatch')
    _require(result.get('constraint_engine') == ENGINE, 'visual_constraint_engine_mismatch')
    _require(result.get('provenance') == provenance, 'visual_provenance_mismatch')
    for key in ('input_ids_sha256','pixel_values_sha256'):sha256(result.get(key),key)
    for key, limit in (('input_tokens',settings.max_input_tokens),('generated_tokens',settings.max_new_tokens)):
        value=result.get(key)
        _require(type(value) is int and 1 <= value <= limit, 'visual_token_budget_mismatch:' + key)
    eos=result.get('effective_eos_token_ids')
    _require(isinstance(eos,list) and eos and all(type(v) is int and v>=0 for v in eos), 'visual_missing_effective_eos')
    grid=result.get('image_grid_thw');shape=result.get('pixel_values_shape')
    _require(isinstance(grid,list) and len(grid)==1 and isinstance(grid[0],list) and len(grid[0])==3
        and all(type(v) is int and v>0 for v in grid[0]) and grid[0][0]==1,
        'visual_invalid_actual_image_grid')
    _require(grid[0][1]*grid[0][2]*16*16<=PREPROCESS['processor_max_pixels'], 'visual_processor_pixel_budget_mismatch')
    _require(isinstance(shape,list) and len(shape)==2 and all(type(v) is int and v>0 for v in shape)
        and shape[0]==math.prod(grid[0]), 'visual_pixel_tensor_shape_mismatch')
    captures=result.get('captures')
    _require(isinstance(captures,list) and len(captures)==1, 'visual_expected_one_optical_capture')
    capture=captures[0]
    _require(capture.get('provenance')==provenance[0] and capture.get('model_id')==name
        and capture.get('model_bundle_sha256')==digest(spec), 'visual_capture_source_or_model_mismatch')
    for key, expected in {'completed':True,'modality':'vision','purpose':'blind_optical_capture',
        'role_verified':False,'automatic_acceptance':False,'statistical_independence_claimed':False}.items():
        _require(capture.get(key) == expected and (not isinstance(expected,bool) or capture.get(key) is expected),
                 'visual_capture_contract_mismatch:' + key)
    proof=capture.get('generation_proof',{})
    _require(all(proof.get(key)==result.get(key) for key in ('input_ids_sha256','pixel_values_sha256',
        'image_grid_thw','response_schema_sha256','effective_eos_token_ids','fresh_generation_state','completed','stop_reason')),
        'visual_capture_generation_proof_mismatch')
    # This checks the sealed capture hash without interpreting a candidate.
    compare_visual_capture(capture,'diagnostic-seal-check')
    return capture


def _fixture(output, name, settings):
    from PIL import Image,ImageDraw,ImageFont
    from harness_ocr.ingest import decode
    from harness_ocr.inference.vision import make_visual_region,prepare_visual_regions
    from harness_ocr.util import file_sha
    target=output/name;target.mkdir()
    font_path=next((p for p in (Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        Path('C:/Windows/Fonts/arial.ttf')) if p.is_file()),None)
    font=ImageFont.truetype(str(font_path),54) if font_path else ImageFont.load_default(size=54)
    literal,wrong=LITERALS[name]
    image=Image.new('RGB',(420,120),'white')
    ImageDraw.Draw(image).text((35,25),literal,font=font,fill='black')
    source=target/'constructed-source.png';image.save(source)
    page=decode(source,target/'pages',settings)['pages'][0]
    descriptors=[make_visual_region(source,page)]
    _,provenance=prepare_visual_regions(descriptors)
    return descriptors,provenance,literal,wrong,{'kind':'constructed_numeric_image',
        'source_sha256':file_sha(source),'font_sha256':file_sha(font_path) if font_path else None,
        'font_implementation':'truetype' if font_path else 'Pillow_default'}


def run_visual_acceptance(output, models, *, qwen=QWENS, device='cuda', seconds=600, threads=4,
                          _owner_factory=None, _verify_models=None):
    """Create a report and execute one real supervised generation per model.

    Private injection hooks are used by tests only; injected execution can never
    set ``target_runtime_accepted`` or ``actual_model_execution_verified``.
    """
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=3600:
        raise ValueError('seconds_must_be_finite_positive_at_most_3600')
    if device not in {'cuda','cpu'}:raise ValueError('unsupported_device')
    if type(threads) is not int or not 1<=threads<=16:raise ValueError('invalid_thread_budget')
    if not isinstance(qwen,(tuple,list)) or not qwen or any(n not in QWENS for n in qwen) or len(set(qwen))!=len(qwen):
        raise ValueError('unique_allowlisted_models_required')
    # Preserve immutable earlier runs, including failures and partial reports.
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    path=output/'report.json';started=time.perf_counter();deadline=started+seconds;owner=None
    injected=_owner_factory is not None or _verify_models is not None
    report={'schema_version':'harness-visual-functional-acceptance/1','status':'running','stage':'initialized',
        'created_utc':_utc(),'device':device,'dtype':'bfloat16' if device=='cuda' else 'float32',
        'models_requested':list(qwen),'planned_arms':len(qwen),'arms':[],
        'total_budget_seconds':seconds,'cpu_threads':threads,'models_parallel':False,
        'neural_generation_calls':0,'test_dependencies_injected':injected,
        'actual_model_execution_verified':False,'target_runtime_accepted':False,
        'quality_qualified':False,'independent_quality_study':False,
        'automatic_acceptance_authority':False,'statistical_independence_claimed':False,
        'input_scope':'only locally constructed images; no user documents or external dataset',
        'interpretation':'Synthetic functional self-test. It does not measure independent OCR quality, '
                         'qualify production risk, or establish independence of Qwen text and vision errors.',
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _flush(path,report,started)
    current=None;owner_closed=False
    try:
        from harness_ocr.inference.supervisor import Supervisor
        from harness_ocr.inference.vision import compare_visual_capture
        from harness_ocr.models import verify_models,model_lock
        from harness_ocr.settings import Settings
        from harness_ocr.util import file_sha,digest
        def remaining():
            left=deadline-time.perf_counter()
            if left<=0:raise TimeoutError('visual_diagnostic_total_deadline')
            return left
        settings=Settings(output/'state',Path(models).resolve(),device=device,dtype=report['dtype'],
            max_new_tokens=192,max_input_tokens=2048,call_timeout_seconds=seconds,
            document_timeout_seconds=seconds,cpu_threads=threads)
        settings.validate();report['lease_namespace']=str(settings.model_lock_dir)
        _flush(path,report,started,'verifying_all_requested_model_files')
        verifier=_verify_models or verify_models
        verification=verifier(settings.models_dir,list(qwen));report['model_file_verification']=verification
        _require(verification.get('ok') is True,'requested_pinned_model_files_not_verified')
        remaining()
        lock=model_lock()['models'];artifacts={}
        for name in qwen:
            files=[]
            for row in lock[name]['files']:
                if row['file'].endswith(('.json','.jinja','.txt')) and 'safetensors' not in row['file']:
                    actual=file_sha(settings.models_dir/name/row['file'])
                    _require(actual==row['sha256'],'pinned_processor_file_sha_mismatch:'+name+'/'+row['file'])
                    files.append({'file':row['file'],'expected_sha256':row['sha256'],'actual_sha256':actual})
            artifacts[name]={'model_id':name,'repo':lock[name]['repo'],'revision':lock[name]['revision'],
                'pinned_model_sha256':digest(lock[name]),'processor_and_tokenizer_artifacts':files}
        report['pinned_artifacts']=artifacts
        _flush(path,report,started,'creating_single_model_owner')
        remaining();owner=(_owner_factory or Supervisor)(settings)
        # One owner is retained while load() evicts the previous Qwen. No model
        # call is dispatched asynchronously, and comparisons never call models.
        for name in qwen:
            current={'model_id':name,'status':'running','stage':'constructing_fixture','generation_calls':0}
            report['arms'].append(current);_flush(path,report,started,'constructing_fixture:'+name)
            remaining()
            descriptors,provenance,literal,wrong,fixture=_fixture(output,name,settings)
            current.update(fixture=fixture,source_provenance=provenance)
            args={'name':name,'regions':descriptors,'max_new_tokens':192}
            serialized=json.dumps(args,ensure_ascii=False)
            _require(literal not in serialized and wrong not in serialized,'proposal_leaked_into_model_arguments')
            current['public_call_arguments_sha256']=digest(args)
            current['generation_calls']=1;report['neural_generation_calls']+=1
            current['stage']='actual_blind_visual_generation';_flush(path,report,started,'actual_blind_visual_generation:'+name)
            result=owner.call('visual_read',args,timeout=remaining())
            current['generation']=result
            _flush(path,report,started,'validating_actual_visual_generation:'+name)
            capture=_validate_generation(result,name=name,settings=settings,spec=lock[name],provenance=provenance)
            good=compare_visual_capture(capture,literal);bad=compare_visual_capture(capture,wrong)
            current.update(correct_proposal_comparison=good,wrong_proposal_comparison=bad,
                same_capture_for_both_proposals=good['capture_id']==bad['capture_id'],
                second_generation_for_wrong_proposal=False)
            _require(good['verdict']=='correct' and bad['verdict']=='incorrect','constructed_literal_selftest_failed')
            _require(good['capture_id']==bad['capture_id'],'comparisons_did_not_share_capture')
            current.update(status='pass',stage='complete')
            _flush(path,report,started,'arm_complete:'+name)
            remaining()
        report['status']='pass';report['stage']='complete'
        report['actual_model_execution_verified']=not injected and isinstance(owner,Supervisor)
    except BaseException as exc:
        report['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'fail'
        report['failed_stage']=report['stage'];report['stage']='stopped'
        report['error']={'type':type(exc).__name__,'message':str(exc)[:1000]}
        if current is not None and current['status']=='running':current.update(status='fail',stage='stopped')
    finally:
        if owner is not None:
            try:owner.close();owner_closed=True
            except BaseException as exc:
                report['status']='fail';report['shutdown_error']={'type':type(exc).__name__,'message':str(exc)[:1000]}
        else:owner_closed=True
        report['model_owner_stopped']=owner_closed
        report['target_runtime_accepted']=report['status']=='pass' and report['actual_model_execution_verified'] and owner_closed
        report['default_both_models_accepted']=report['target_runtime_accepted'] and set(qwen)==set(QWENS)
        _flush(path,report,started)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New report directory, for example /reports/visual-target-001')
    parser.add_argument('--models',default=os.getenv('HARNESS_MODELS','./models'),help='Pinned local model bundle directory')
    parser.add_argument('--qwen',nargs='+',choices=QWENS,default=list(QWENS))
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--seconds',type=float,default=600,help='Total finite study budget, at most 3600 seconds')
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args(argv)
    report=run_visual_acceptance(args.output,args.models,qwen=args.qwen,device=args.device,
        seconds=args.seconds,threads=args.threads)
    print(json.dumps({'status':report['status'],'target_runtime_accepted':report['target_runtime_accepted'],
        'default_both_models_accepted':report['default_both_models_accepted'],'quality_qualified':False,
        'report':str(Path(args.output).resolve()/'report.json')},ensure_ascii=False),flush=True)
    return 0 if report['status']=='pass' and report['target_runtime_accepted'] else 1


if __name__=='__main__':
    if os.name=='posix':
        def interrupted(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
    raise SystemExit(main())
