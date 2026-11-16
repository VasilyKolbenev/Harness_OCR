"""Bounded real CUDA component smoke; constructed pixels, no OCR quality gate.

One Supervisor at a time, a fresh owner per selected Qwen, no retries. Model
files must already be pinned/prepared. This is distinct from doctor (no model
generation) and an independently labelled document-quality experiment.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import stat
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
QWENS=('qwen-0.8b','qwen-4b')
LOCK_FILES=('cpu-admission.lock','cpu-compute.lock','cuda-admission.lock','cuda-compute.lock','gpu-delivery.lock')


def _require(value, reason):
    if not value:raise ValueError(reason)


def _flush(path, report, started, stage=None):
    if stage is not None:report['stage']=stage
    report['updated_utc']=datetime.now(timezone.utc).isoformat()
    report['elapsed_seconds']=time.perf_counter()-started
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _generation(result, name, *, images=False, max_input_tokens=4096, max_new_tokens=256):
    from harness_ocr.core.common import sha256
    _require(isinstance(result,dict),'generation_result_object')
    for key, expected in {'model_id':name,'device':'cuda','dtype':'bfloat16','completed':True,
                          'stop_reason':'eos','fresh_generation_state':True}.items():
        _require(result.get(key)==expected and (type(expected) is not bool or result.get(key) is expected),
                 'generation_contract:'+key)
    for key,maximum in (('input_tokens',max_input_tokens),('generated_tokens',max_new_tokens)):
        _require(type(result.get(key)) is int and 0<result[key]<=maximum,'generation_token_count:'+key)
    sha256(result.get('input_ids_sha256'))
    devices=result.get('input_tensor_devices');dtypes=result.get('input_tensor_dtypes')
    _require(isinstance(devices,dict) and devices and 'input_ids' in devices
             and isinstance(dtypes,dict) and set(devices)==set(dtypes),'actual_input_metadata_missing')
    cuda=lambda v:isinstance(v,str) and re.fullmatch(r'cuda(?::[0-9]+)?',v) is not None
    _require(all(cuda(v) for v in devices.values()),'actual_input_not_cuda')
    _require(all(v in {'torch.bfloat16','torch.float32','torch.int64','torch.int32','torch.bool','torch.uint8'}
                 for v in dtypes.values()),'unexpected_input_dtype')
    params=result.get('effective_parameter_devices');precision=result.get('effective_parameter_dtypes')
    _require(isinstance(params,list) and params and all(cuda(v) for v in params),'actual_parameters_not_cuda')
    _require(isinstance(precision,list) and 'torch.bfloat16' in precision
             and set(precision)<={'torch.bfloat16','torch.float32'},'actual_parameters_not_supported_bf16')
    if images:
        _require('pixel_values' in devices,'actual_image_tensor_missing')
    eos=result.get('effective_eos_token_ids')
    _require(isinstance(eos,list) and eos and all(type(v) is int and v>=0 for v in eos),'effective_eos_missing')
    _require(result.get('json_constraint_applied') is (name!='xiaomi'),'closed_generation_constraint_missing')


def _probe(result, qwen, *, warmup=True):
    _require(result.get('device')=='cuda' and isinstance(result.get('gpu'),str) and result['gpu'],
             'probe_not_actual_cuda')
    capability=result.get('capability')
    _require(isinstance(capability,list) and len(capability)==2
             and all(type(v) is int and v>=0 for v in capability) and capability[0]>0,'probe_capability_missing')
    if warmup:
        _require({'xiaomi',qwen,'detector','cyrillic'}<=set(result.get('loaded_models',[])),
                 'probe_requested_models_not_loaded')


def _detector(result, variant, page, lock):
    _require(type(result.get('completed')) is bool,'detector_completion_metadata_missing')
    diag=result.get('diagnostics',{});runtime=diag.get('runtime',{})
    _require(diag.get('raster_sha256')==page['raster_sha256'],'detector_raster_binding_mismatch')
    _require(runtime.get('text_detector')==variant and runtime.get('recognizer_device')=='cpu',
             'detector_or_ctc_runtime_mismatch')
    _require(type(diag.get('lines_processed')) is int and diag['lines_processed']>=0
             and isinstance(diag.get('ctc'),list),'detector_coverage_metadata_missing')
    for ctc in diag['ctc']:
        shape=ctc.get('actual_output_shape')
        _require(isinstance(shape,list) and len(shape)==3
                 and all(type(n) is int and n>0 for n in shape),'actual_ctc_tensor_shape_missing')
    if variant!='db':
        det=diag.get('detector',{});spec=lock[variant+'-text']
        _require(det.get('device')=='cuda' and det.get('cuda_synchronized') is True,
                 'yolo_actual_cuda_not_verified')
        _require(det.get('checkpoint_sha256')==spec['files'][0]['sha256']
                 and det.get('coordinates_refer_to')=='actual_page_raster_pixels','yolo_checkpoint_or_geometry_mismatch')
    return {'completed':result['completed'],'lines_processed':diag['lines_processed'],
            'ctc_forward_records':len(diag['ctc']),'ctc_path_exercised':bool(diag['ctc']),
            'observations':len(result.get('observations',[])),
            'quality_or_coverage_incomplete':not result['completed'] or diag['lines_processed']==0,
            'quality_gate_applied':False}


def _visual(result, name, descriptors, lock, settings):
    from harness_ocr.core.common import sha256
    from harness_ocr.inference.vision import prepare_visual_regions,blind_prompt,parse_visual_output,compare_visual_capture,VERSION
    from harness_ocr.inference.constraints import build_response_schema,ENGINE
    from harness_ocr.util import digest
    _generation(result,name,images=True,max_input_tokens=settings.max_input_tokens,max_new_tokens=settings.max_new_tokens)
    sha256(result.get('pixel_values_sha256'))
    grid=result.get('image_grid_thw');shape=result.get('pixel_values_shape')
    _require(isinstance(grid,list) and len(grid)==len(descriptors)
             and all(isinstance(g,list) and len(g)==3 and g[0]==1
                     and all(type(n) is int and n>0 for n in g) for g in grid),'visual_image_grid_missing')
    _require(isinstance(shape,list) and len(shape)==2 and all(type(n) is int and n>0 for n in shape)
             and shape[0]==sum(math.prod(g) for g in grid),'visual_pixel_tensor_shape_mismatch')
    _,provenance=prepare_visual_regions(descriptors)
    prompt=blind_prompt([p['region_id'] for p in provenance])
    for key,expected in {'purpose':'blind_optical_capture','modality':'vision','adapter_version':VERSION,
                         'processor_class':'Qwen3VLProcessor','proposed_value_supplied':False,
                         'ground_truth_supplied':False,'automatic_acceptance':False,
                         'statistical_independence_claimed':False,'provenance':provenance,
                         'blind_prompt':prompt,'blind_prompt_sha256':digest(prompt),
                         'model_bundle_sha256':digest(lock[name]),'constraint_engine':ENGINE,
                         'response_schema_sha256':digest(build_response_schema(prompt,'visual_read'))}.items():
        _require(result.get(key)==expected and (type(expected) is not bool or result.get(key) is expected),
                 'visual_contract:'+key)
    captures=parse_visual_output(result,provenance)
    _require(result.get('captures')==captures,'visual_sealed_capture_or_provenance_mismatch')
    for capture in captures:
        compare_visual_capture(capture,'seal-validation-only')  # No accuracy gate.


def _fixture(output, settings):
    from PIL import Image,ImageDraw,ImageFont
    from harness_ocr.ingest import decode
    from harness_ocr.inference.vision import make_visual_region
    from harness_ocr.util import file_sha
    target=output/'constructed';target.mkdir()
    font_path=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font=ImageFont.truetype(str(font_path),38) if font_path.is_file() else ImageFont.load_default(size=38)
    image=Image.new('RGB',(640,170),'white');draw=ImageDraw.Draw(image)
    draw.text((20,15),'SYNTHETIC COMPONENT SMOKE',font=ImageFont.load_default(size=20),fill='black')
    draw.text((20,75),'Номер паспорта: 123456',font=font,fill='black')
    source=target/'constructed-source.png';image.save(source)
    decoded=decode(source,target/'pages',settings);page=decoded['pages'][0]
    return source,page,[make_visual_region(source,page)],{'kind':'constructed_pixels_not_quality_corpus',
        'source_sha256':file_sha(source),'font_sha256':file_sha(font_path) if font_path.is_file() else None}


def run_smoke(models, output, lock_dir, *, qwen=QWENS, include_yolo_x=False, total_seconds=600,
              _owner_factory=None, _verify_models=None, _sampler_factory=None, _cancel=None):
    _require(type(total_seconds) in (int,float) and math.isfinite(total_seconds) and 0<total_seconds<=3600,
             'finite_total_seconds_1_to_3600_required')
    _require(isinstance(qwen,(list,tuple)) and qwen and len(set(qwen))==len(qwen)
             and all(n in QWENS for n in qwen),'unique_allowlisted_qwen_required')
    _require(type(include_yolo_x) is bool,'include_yolo_x_boolean_required')
    lock_dir=Path(lock_dir);_require(lock_dir.is_absolute(),'absolute_shared_lock_directory_required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    path=output/'report.json';started=time.perf_counter();deadline=started+total_seconds
    injected=any(x is not None for x in (_owner_factory,_verify_models,_sampler_factory,_cancel))
    report={'schema_version':'harness-gpu-functional-smoke/1','status':'running','stage':'preflight',
        'created_utc':datetime.now(timezone.utc).isoformat(),'device':'cuda','dtype':'bfloat16',
        'qwen_requested':list(qwen),'include_yolo_x':include_yolo_x,'total_seconds':total_seconds,
        'models_parallel':False,'retries':0,'test_dependencies_injected':injected,
        'actual_model_execution_verified':False,'target_runtime_accepted':False,'quality_qualified':False,
        'ocr_accuracy_gate':False,'independent_quality_study':False,'automatic_acceptance_authority':False,
        'lease_namespace':str(lock_dir.resolve()),'arms':[],'call_attempts':0,
        'neural_generation_calls_attempted':0,'completed_generation_calls':0,
        'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _flush(path,report,started);owner=None;current=None;all_closed=True;coordinator=None
    try:
        from harness_ocr.inference.supervisor import Supervisor,ModelCanceled
        from harness_ocr.inference.locks import FileLease
        from harness_ocr.settings import Settings
        from harness_ocr.models import verify_models,model_lock
        from harness_ocr.util import digest,file_sha
        from harness_ocr.grounding import build_candidates,qwen_prompt,resolve_response
        from harness_ocr.profiles import ProfileRegistry
        from harness_ocr.inference.constraints import build_response_schema,_load,_valid,ENGINE
        from experiments.yolo26.metrics import ResourceSampler
        if not injected:_require(platform.system()=='Linux','linux_target_required')
        for name in LOCK_FILES:
            meta=(lock_dir/name).lstat()
            _require(stat.S_ISREG(meta.st_mode) and not stat.S_ISLNK(meta.st_mode),'shared_lease_file_not_regular')
        coordinator=FileLease(lock_dir/'gpu-delivery.lock').acquire()
        report['coordinator_lease_acquired']=True
        def remaining():
            if _cancel and _cancel():raise ModelCanceled('operator_cancel')
            left=deadline-time.perf_counter()
            if left<=0:raise TimeoutError('smoke_global_deadline')
            return left
        remaining()
        names=['xiaomi','detector','cyrillic',*qwen,'yolo26l-text']+(['yolo26x-text'] if include_yolo_x else [])
        verification=(_verify_models or verify_models)(Path(models).resolve(),names)
        report['model_file_verification']=verification
        _require(verification.get('ok') is True,'requested_model_files_not_verified')
        lock=model_lock()['models'];report['model_bundle_sha256']={n:digest(lock[n]) for n in names}
        base=Settings.from_env()
        settings=replace(base,data_dir=output/'state',models_dir=Path(models).resolve(),device='cuda',dtype='bfloat16',
            runtime_lock_dir=lock_dir.resolve(),reading_policy='adaptive',text_detector='db',rescue_text_detector='yolo26l',
            max_detector_rescues=1,visual_verifier=True,
            max_new_tokens=min(base.max_new_tokens,256),max_input_tokens=min(base.max_input_tokens,4096),
            call_timeout_seconds=total_seconds,document_timeout_seconds=total_seconds)
        settings.validate();remaining()
        report['runtime_configuration']={'attention_backend':settings.attention_backend,
            'max_input_tokens':settings.max_input_tokens,'max_new_tokens':settings.max_new_tokens,
            'reading_policy':settings.reading_policy,'visual_verifier':settings.visual_verifier,
            'rescue_text_detector':settings.rescue_text_detector}
        source,page,descriptors,fixture=_fixture(output,settings);report['fixture']=fixture
        contract=ProfileRegistry().compile('D01',subset=['passport.number'])
        configurations=[(name,'yolo26l',index==0) for index,name in enumerate(qwen)]
        if include_yolo_x:configurations.append((qwen[0],'yolo26x',True))
        for index,(name,detector,do_rescue) in enumerate(configurations):
            only_detector=include_yolo_x and index==len(configurations)-1
            current={'qwen':None if only_detector else name,'detector':detector,'status':'running','steps':[],
                     'model_owner_stopped':False}
            report['arms'].append(current);_flush(path,report,started,'create_owner:'+str(index))
            arm_dir=output/('arm-'+str(index));arm_dir.mkdir()
            chosen=replace(settings,data_dir=arm_dir/'state',qwen=name,rescue_text_detector=detector)
            remaining();owner=(_owner_factory or Supervisor)(chosen)
            monitor=None;body_error=None;cleanup_error=None
            try:
                monitor=(_sampler_factory or ResourceSampler)(owner,'cuda')
                with monitor:
                    def call(op,args):
                        left=remaining();step={'op':op,'status':'running','timeout_seconds':left,'arguments_sha256':digest(args)}
                        current['steps'].append(step);report['call_attempts']+=1
                        if op in ('ocr','select','visual_read'):report['neural_generation_calls_attempted']+=1
                        _flush(path,report,started,op+':'+str(index))
                        result=owner.call(op,args,timeout=left,cancel=_cancel)
                        result_path=arm_dir/(str(len(current['steps']))+'-'+op+'.json')
                        with result_path.open('x',encoding='utf-8') as stream:
                            json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
                            stream.flush();os.fsync(stream.fileno())
                        step.update(result_path=str(result_path),result_sha256=file_sha(result_path))
                        return result,step
                    probe,step=call('probe',{'warmup':not only_detector});_probe(probe,name,warmup=not only_detector)
                    step['status']='pass';current['probe']=probe
                    if only_detector:
                        result,step=call('cheap_rescue',{'page':page,'source_sha256':fixture['source_sha256'],
                            'output_dir':str(arm_dir/'rescue')})
                        step['detector_coverage']=_detector(result,detector,page,lock);step['status']='pass'
                    else:
                        cheap,step=call('cheap',{'page':page,'source_sha256':fixture['source_sha256'],
                            'output_dir':str(arm_dir/'db')})
                        step['detector_coverage']=_detector(cheap,'db',page,lock);step['status']='pass'
                        if do_rescue:
                            result,step=call('cheap_rescue',{'page':page,'source_sha256':fixture['source_sha256'],
                                'output_dir':str(arm_dir/'rescue')})
                            step['detector_coverage']=_detector(result,detector,page,lock);step['status']='pass'
                        ocr,step=call('ocr',{'path':page['path'],'expected_image_sha256':page['raster_sha256']})
                        _generation(ocr,'xiaomi',images=True,max_input_tokens=chosen.max_input_tokens,
                                    max_new_tokens=chosen.max_new_tokens)
                        _require(ocr.get('input_image_sha256')==page['raster_sha256']
                                 and ocr.get('input_image_expected_sha_verified') is True,'ocr_source_bytes_not_verified')
                        step['status']='pass';report['completed_generation_calls']+=1
                        obs={'id':'xiaomi_observed','text':ocr['text'],'source_sha256':fixture['source_sha256'],
                            'evidence_kind':'image_region','completed':True,'page':1,'page_id':page['page_id'],
                            'source_size':page['source_size'],'bbox':[0,0,*page['source_size']]}
                        observations=[*cheap['observations'],obs]
                        candidates=build_candidates(observations,contract)
                        prompt=qwen_prompt(candidates,cheap['regions'],contract,observations=observations,
                            remaining_budget={'reread_regions':0,'reread_rounds':0,'max_new_tokens':chosen.max_new_tokens})
                        selected,step=call('select',{'name':name,'prompt':prompt})
                        _generation(selected,name,max_input_tokens=chosen.max_input_tokens,max_new_tokens=chosen.max_new_tokens)
                        schema=build_response_schema(prompt,'selector')
                        _require(_valid(_load(selected['text']),schema),'selector_response_not_closed')
                        _require(selected.get('response_schema_sha256')==digest(schema)
                                 and selected.get('constraint_engine')==ENGINE,'selector_constraint_proof_mismatch')
                        current['selector_resolution']=resolve_response(selected['text'],candidates,cheap['regions'],contract)
                        step['status']='pass';report['completed_generation_calls']+=1
                        visual,step=call('visual_read',{'name':name,'regions':descriptors})
                        _visual(visual,name,descriptors,lock,chosen)
                        step['status']='pass';report['completed_generation_calls']+=1
                    remaining();current['status']='pass'
            except BaseException as exc:
                body_error=exc
                raise
            finally:
                if monitor is not None:
                    try:current['resources']=monitor.result()
                    except BaseException as exc:
                        cleanup_error=exc;current['resource_error_type']=type(exc).__name__
                try:owner.close();current['model_owner_stopped']=True
                except BaseException as exc:
                    all_closed=False;cleanup_error=exc;current['shutdown_error_type']=type(exc).__name__
                finally:owner=None
                _flush(path,report,started,'owner_closed:'+str(index) if body_error is None else None)
                if body_error is None and cleanup_error is not None:raise cleanup_error
        report['status']='pass';report['stage']='complete'
        report['actual_model_execution_verified']=not injected
    except BaseException as exc:
        from harness_ocr.inference.supervisor import ModelCanceled,ModelDeadline
        report['failed_stage']=report['stage']
        report['status']=('interrupted' if isinstance(exc,(KeyboardInterrupt,ModelCanceled)) else
                          'budget_exhausted' if isinstance(exc,(TimeoutError,ModelDeadline)) else 'fail')
        report['stage']='stopped';report['error']={'type':type(exc).__name__,'message':str(exc)[:600]}
        if current:
            current['status']='fail'
            if current['steps'] and current['steps'][-1]['status']=='running':
                current['steps'][-1].update(status='fail',error_type=type(exc).__name__)
    finally:
        if owner is not None:
            try:
                owner.close()
                if current:current['model_owner_stopped']=True
            except BaseException:all_closed=False;report['status']='fail';report['shutdown_error']=True
        if coordinator is not None:coordinator.close()
        report['coordinator_lease_released']=coordinator is not None
        report['model_owners_stopped']=all_closed
        report['target_runtime_accepted']=report['status']=='pass' and report['actual_model_execution_verified'] and all_closed
        _flush(path,report,started)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',required=True)
    parser.add_argument('--output',required=True,help='New directory; previous attempts are immutable')
    parser.add_argument('--lock-dir',required=True)
    parser.add_argument('--qwen',choices=['both',*QWENS],default='both')
    parser.add_argument('--include-yolo-x',action='store_true')
    parser.add_argument('--total-seconds',type=float,default=600)
    args=parser.parse_args(argv)
    result=run_smoke(args.models,args.output,args.lock_dir,
        qwen=QWENS if args.qwen=='both' else [args.qwen],include_yolo_x=args.include_yolo_x,total_seconds=args.total_seconds)
    print(json.dumps({'schema_version':result['schema_version'],'status':result['status'],
        'target_runtime_accepted':result['target_runtime_accepted'],'quality_qualified':False,
        'report':str(Path(args.output).resolve()/'report.json')}))
    if result['status'] in {'interrupted','budget_exhausted'}:return 2
    return 0 if result['target_runtime_accepted'] else 1


if __name__=='__main__':
    if os.name=='posix':
        def interrupted(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,interrupted)
    raise SystemExit(main())
