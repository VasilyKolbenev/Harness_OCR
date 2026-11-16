"""Fresh-owner adaptive GPU matrix. No HTTP, shared optics, retries or auto-winner.

Default is metadata-only preflight, which accepts missing model weights and
never imports Torch/CUDA. Use --execute only on the prepared target GPU.
Detailed protocol/state/results/logs are PRIVATE. summary.json is a local input
to the allowlist exporter. Only RETURN_TO_TEAM.zip is the public return packet.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
existing=sys.modules.get('harness_ocr')
if existing and not Path(existing.__file__).resolve().is_relative_to(ROOT):
    raise RuntimeError('harness_source_import_mismatch')
sys.path[:0]=[str(ROOT/'src'),str(ROOT)]

from .plan import (COMMON,BudgetExpired,arms,check_deadline,code_pins,digest,exact_score,
    file_sha,load_documents,load_gold_after_inference,pin_models,read,require_budget,
    required_models,schedule,settings_dict,timestamp,utc,validate_gold_preflight,validate_runtime_pins,versions,write)
from .watchdog import VerifiedTree,wait_child

CLEANUP_SECONDS=15
FINAL_HASH_RESERVE_SECONDS=30
SPEC_KEYS={'row_id','row_dir','document','settings','code_pins','runner_sha256',
           'compute_deadline_utc','worker_token_sha256'}


class FailStopOwner:
    """One Supervisor, with no lazy worker restart after the first failed call."""
    def __init__(self,supervisor):
        self.supervisor=supervisor;self.failed=False;self.error_type=None
        self.attempts={};self.refused_after_failure=0
    @property
    def process(self):return self.supervisor.process
    def call(self,operation,args=None,**kwargs):
        if self.failed:
            self.refused_after_failure+=1
            raise RuntimeError('experiment_backend_failed_no_retry')
        self.attempts[operation]=self.attempts.get(operation,0)+1
        try:return self.supervisor.call(operation,args,**kwargs)
        except BaseException as error:
            self.failed=True;self.error_type=type(error).__name__;raise
    def close(self):self.supervisor.close()


def probe_leases(lock_dir):
    from harness_ocr.inference.locks import FileLease
    proofs=[];leases=[]
    try:
        for name in ('cuda-admission.lock','cuda-compute.lock'):
            leases.append(FileLease(Path(lock_dir)/name).acquire())
            proofs.append({'lease':name,'free':True})
        return proofs
    finally:
        for lease in reversed(leases):lease.close()


def model_dependencies():
    from harness_ocr.engine import Engine
    from harness_ocr.inference.supervisor import Supervisor
    from harness_ocr.settings import Settings
    from experiments.yolo26.metrics import ResourceSampler
    return Engine,Supervisor,Settings,ResourceSampler


def perform_row(spec,dependencies=None):
    """Single fresh row; inference data has a closed gold-free whitelist."""
    if set(spec)!=SPEC_KEYS:raise ValueError('worker_spec_not_closed_inference_whitelist')
    document=spec['document']
    if set(document)!={'id','path','source_sha256','profile_id','profile_version','subset','subset_contract_id'}:
        raise ValueError('worker_document_not_closed_inference_whitelist')
    Engine,Supervisor,Settings,Sampler=dependencies or model_dependencies()
    row=Path(spec['row_dir']);deadline=timestamp(spec['compute_deadline_utc'])
    cancel=lambda:utc()>=deadline or (row/'parent-stop.json').exists()
    values=dict(spec['settings'])
    values['data_dir']=row/'state'
    for key in ('data_dir','models_dir','runtime_lock_dir'):values[key]=Path(values[key])
    settings=Settings(**values);settings.validate()
    measurement={'schema_version':'harness-gpu-delivery-row/1','status':'starting','stage':'warmup',
        'started_utc':utc().isoformat(),'worker_pid':os.getpid(),'retry_count':0,
        'source_sha256':document['source_sha256'],'settings_sha256':digest(spec['settings'])}
    owner=None;backend=None;sampler=None
    try:
        if cancel():raise BudgetExpired('global_budget_before_owner')
        owner=Supervisor(settings);backend=FailStopOwner(owner)
        with Sampler(backend,'cuda') as sampler:
            probe=backend.call('probe',{'warmup':True},cancel=cancel,
                timeout=min(settings.call_timeout_seconds,max(.001,(deadline-utc()).total_seconds())))
        write(row/'warmup.json',{'probe':probe,'resources':sampler.result()})
        expected={'xiaomi',settings.qwen,'detector','cyrillic'}
        if probe.get('device')!='cuda' or not expected<=set(probe.get('loaded_models',[])):
            raise RuntimeError('real_cuda_warmup_not_confirmed')
        if cancel():raise BudgetExpired('global_budget_after_warmup')
        if code_pins(deadline)!=spec['code_pins']:raise ValueError('code_changed_during_warmup')
        if file_sha(document['path'],deadline)!=document['source_sha256']:
            raise ValueError('source_changed_before_document')
        measurement['stage']='document'
        engine=Engine(settings,backend)
        with Sampler(backend,'cuda') as sampler:
            result=engine.run('gpu_'+spec['row_id'],document['path'],document['source_sha256'],
                document['profile_id'],subset=document['subset'],qwen=settings.qwen,mode='hybrid',
                allow_reread=True,cancel=cancel,emit_operational_events=False)
        optics=result.pop('optics',None)
        if optics is not None:write(row/'optics.json',optics)
        write(row/'result.json',result)
        if (result.get('source_sha256')!=document['source_sha256']
            or result.get('profile_id')!=document['profile_id']
            or result.get('profile_version')!=document['profile_version']
            or set(result.get('fields',{}))!=set(document['subset'])):
            raise ValueError('result_document_profile_subset_binding_mismatch')
        measurement.update(status='completed' if result.get('completed') is True
            and result.get('terminal_status')=='completed' else 'incomplete',
            document_resources=sampler.result(),terminal_status=result.get('terminal_status'),
            administrative_censor=bool(cancel() and result.get('terminal_status')=='canceled'))
    except BaseException as error:
        measurement.update(status='administratively_censored' if cancel() else 'failed',
            administrative_censor=cancel(),error_type=type(error).__name__)
        # Protected diagnostics; never copied to public summary/default bundle.
        write(row/'private-error.json',{'type':type(error).__name__,'message':str(error)[:1000]})
        if sampler is not None:
            try:measurement['last_stage_resources']=sampler.result()
            except BaseException as sampler_error:measurement['resource_error_type']=type(sampler_error).__name__
    finally:
        if backend is not None:
            measurement['backend_failed_no_retry']=backend.failed
            measurement['backend_error_type']=backend.error_type
            measurement['backend_operation_attempts']=backend.attempts
            measurement['backend_requests_refused_after_failure']=backend.refused_after_failure
        if owner is not None:
            try:owner.close();measurement['supervisor_closed']=True
            except BaseException as error:
                measurement.update(status='failed',close_error_type=type(error).__name__)
        measurement['finished_utc']=utc().isoformat()
        write(row/'measurement.json',measurement)
    return 0 if measurement.get('supervisor_closed') else 1


def row_worker(path):
    spec=read(path);token=os.environ.get('HARNESS_GPU_DELIVERY_WORKER_TOKEN','')
    if len(token)!=64 or hashlib.sha256(token.encode()).hexdigest()!=spec.get('worker_token_sha256'):
        raise RuntimeError('worker_requires_live_parent_authorization')
    deadline=timestamp(spec['compute_deadline_utc'])
    if file_sha(__file__,deadline)!=spec['runner_sha256'] or code_pins(deadline)!=spec['code_pins']:
        raise ValueError('worker_source_pins_changed')
    return perform_row(spec)


def build_plan(args,deadline=None):
    check_deadline(deadline);manifest_sha,documents=load_documents(args.manifest);check_deadline(deadline)
    selected=arms(args.qwen,args.include_yolo_x)
    common={**COMMON,**{key:getattr(args,key) for key in COMMON if hasattr(args,key)}}
    arm_settings={a['id']:settings_dict(common,a,args.models,args.lock_dir,'<per-row-fresh-state>') for a in selected}
    gold_metadata=validate_gold_preflight(args.gold,documents,deadline) if args.gold else None
    return {'schema_version':'harness-gpu-delivery-plan/1','created_utc':utc().isoformat(),
        'manifest_path':str(args.manifest.resolve()),'manifest_sha256':manifest_sha,'documents':documents,
        'arms':selected,'arm_settings':arm_settings,'common_caps':common,'planned_rows':schedule(documents,selected,manifest_sha),
        'required_model_keys':required_models(selected),'model_verification_performed':False,
        'models_dir':str(args.models.resolve()),'lock_dir':str(args.lock_dir.resolve()),
        'versions':versions(),'shared_final_optics_or_claims':False,'retries':0,
        'selection_rule':'fixed document-major rotating arms; no result-based order or automatic winner',
        'group_unit':'input_document; exact duplicate source hashes do not establish independence',
        'quality_qualified':False,'resources_qualified':False,'preflight_neural_calls':0,
        'gold_metadata':gold_metadata,'correctness_unmeasured':True}


def state_pins(row,deadline=None):
    return {p.relative_to(row).as_posix():file_sha(p,deadline)
            for p in sorted((row/'state').rglob('*')) if p.is_file()}


def safe_row(planned,row_dir,measurement):
    """No original IDs, source filenames, values, prompts, exception text or logs."""
    value={key:planned[key] for key in ('row_id','public_document_id','arm')}
    value.update(status=measurement.get('status','missing'),
        administrative_censor=bool(measurement.get('administrative_censor')),
        technical_execution_passed=measurement.get('status')=='completed'
            and measurement.get('supervisor_closed') is True
            and (measurement.get('watchdog') or {}).get('exit_code')==0
            and not (measurement.get('watchdog') or {}).get('inspection_errors')
            and not (measurement.get('watchdog') or {}).get('forced_cleanup'),
        full_row_wall_seconds=measurement.get('row_process_wall_seconds'),
        preflight_wall_seconds=measurement.get('preflight_seconds'),
        document_resources=measurement.get('document_resources'),
        last_stage_resources=measurement.get('last_stage_resources'),
        backend_failed_no_retry=bool(measurement.get('backend_failed_no_retry')))
    attempted=measurement.get('backend_operation_attempts',{})
    value['backend_action_attempts']={op:attempted.get(op,0)
        for op in ('probe','ingest','cheap','cheap_rescue','ocr','select','visual_read')}
    value['backend_requests_refused_after_failure']=measurement.get('backend_requests_refused_after_failure',0)
    value['rescue_attempted']=attempted.get('cheap_rescue',0)>0
    if (row_dir/'warmup.json').is_file():
        warmup=read(row_dir/'warmup.json');probe=warmup.get('probe',{})
        value.update(warmup_resources=warmup.get('resources'),
            gpu={key:probe.get(key) for key in ('device','gpu','capability','torch')})
    if (row_dir/'result.json').is_file():
        result=read(row_dir/'result.json');fields=list(result.get('fields',{}).values())
        calls=result.get('calls',[]);ops={op:sum(c.get('op')==op for c in calls)
            for op in ('ingest','cheap','cheap_rescue','ocr','select','visual_read')}
        peak=[c.get('peak_cuda_allocated_bytes') for c in calls if isinstance(c.get('peak_cuda_allocated_bytes'),int)]
        reserved=[c.get('peak_cuda_reserved_bytes') for c in calls if isinstance(c.get('peak_cuda_reserved_bytes'),int)]
        decision=result.get('decision')
        value.update(completed=result.get('completed') is True,
            decision=decision if decision in {'review','accept_for_autofill','reject_input'} else 'unknown',
            requested_fields=len(fields),observed_fields=sum(f.get('extraction_status')=='observed' for f in fields),
            validation_pass_fields=sum(f.get('validation_status')=='pass' for f in fields),
            review_fields=sum(f.get('decision')=='review' for f in fields),
            quality_review_fields=sum(bool(f.get('quality_review_required')) for f in fields),
            all_required_fields_validated=result.get('all_required_fields_validated'),
            action_counts=ops,rescue_activated=ops['cheap_rescue']>0,
            action_counts_scope='successful calls recorded by Engine; attempted calls include failures separately',
            native_ingestion_present=bool(result.get('native_ingestion')),
            model_generation_calls=sum(ops[k] for k in ('ocr','select','visual_read')),
            zero_generation_document=not any(ops[k] for k in ('ocr','select','visual_read')),
            max_call_cuda_allocated_bytes=max(peak) if peak else None,
            max_call_cuda_reserved_bytes=max(reserved) if reserved else None)
    return value


def make_summary(protocol,output,measurements,status,integrity,elapsed):
    indexed={row['row_id']:row for row in measurements}
    public=[safe_row(p,output/'rows'/p['row_id'],indexed.get(p['row_id'],{})) for p in protocol['planned_rows']]
    for row,planned in zip(public,protocol['planned_rows']):
        document=next(d for d in protocol['documents'] if d['id']==planned['document_id'])
        row.setdefault('requested_fields',len(document['subset']))
        if row['status']=='missing' and status in {'budget_exhausted','stopped'}:
            row.update(administrative_censor=True,missing_reason='not_started_administrative_budget_or_stop')
    if len(indexed)!=len(measurements):raise ValueError('duplicate_attempt_in_measurements')
    all_complete=bool(public) and all(r['technical_execution_passed'] for r in public)
    sampled=[r.get(name) for r in public for name in ('warmup_resources','document_resources')]
    gpu_measured=all(isinstance(stage,dict) and stage.get('gpu_samples',0)>0
                     and stage.get('peak_sampled_attributable_gpu_bytes') is not None for stage in sampled)
    return {'schema_version':'harness-gpu-delivery-summary/1','status':status,
        'protocol_sha256':protocol['protocol_sha256'],'source_code_sha256':digest(protocol['code_pins']),
        'model_registry_subset_sha256':protocol['model_registry_subset_sha256'],
        'model_content_pin_sha256':digest({name:pin['sha256'] for name,pin in protocol['model_file_pins'].items()}),
        'manifest_sha256':protocol['manifest_sha256'],
        'arms':protocol['arms'],'common_caps':protocol['common_caps'],'planned_rows':len(public),
        'recorded_rows':len(measurements),'missing_rows':sum(r['status']=='missing' for r in public),
        'input_documents':len(protocol['documents']),
        'exact_duplicate_source_groups':len({d['source_sha256'] for d in protocol['documents']}),
        'technical_execution_passed':status=='completed' and all_complete and integrity is True,
        'integrity_verification_passed':integrity,'quality_qualified':False,'resources_qualified':False,
        'resource_measurement_complete':bool(sampled) and gpu_measured,
        'new_worker_per_row':True,'model_owner_parallelism':1,'shared_final_optics_or_claims':False,
        'automatic_retries':0,'elapsed_wall_seconds_including_preflight_and_close':elapsed,
        'rows':public,'privacy_scope':'opaque IDs, counts, hashes and resources only; protected raw artifacts excluded',
        'measurement_scope':'Cold warmup, document execution and full fresh row are separate. NVML is sampled '
            'matching-process GPU memory; PyTorch call peaks are different quantities. Null means unmeasured.',
        'selection_decision':'No automatic winner or production qualification; review is independent of technical execution.'}


def score_after_inference(protocol,output,deadline):
    if not protocol.get('gold_metadata'):return {'measured':False,'reason':'no_separate_gold_supplied'}
    check_deadline(deadline)
    metadata=protocol['gold_metadata']
    gold=load_gold_after_inference(metadata['path'],metadata['sha256'],protocol['documents'],deadline)
    rows=[]
    for planned in protocol['planned_rows']:
        check_deadline(deadline)
        reference=gold.get(planned['document_id'])
        if reference is None:continue
        path=output/'rows'/planned['row_id']/'result.json'
        result=read(path) if path.is_file() else None
        document=next(d for d in protocol['documents'] if d['id']==planned['document_id'])
        binding_valid=bool(result and result.get('source_sha256')==document['source_sha256']
            and result.get('profile_id')==document['profile_id']
            and result.get('profile_version')==document['profile_version']
            and set(result.get('fields',{}))==set(document['subset']))
        score=exact_score(result,reference) if binding_valid or result is None else {
            'measured':False,'invalid_result_binding':True,'planned_fields':len(reference)}
        rows.append({'row_id':planned['row_id'],'public_document_id':planned['public_document_id'],
            'arm':planned['arm'],'document_completed':bool(result and result.get('completed')),'score':score})
    value={'schema_version':'harness-gpu-delivery-quality/1',
        'measured':any(row['score'].get('measured') for row in rows),
        'post_inference_only':True,'exact_raw_no_normalization':True,'quality_qualified':False,
        'gold_sha256':metadata['sha256'],'rows':rows,
        'planned_reference_cells':sum(len(gold.get(p['document_id'],{})) for p in protocol['planned_rows']),
        'measured_reference_cells':sum(row['score'].get('total',0) for row in rows if row['score'].get('measured')),
        'missing_result_reference_cells':sum(row['score'].get('planned_fields',0)
            for row in rows if row['score'].get('missing_result')),
        'missing_result_is_not_model_wrong_or_withheld':True}
    write(output/'quality.json',value)
    return value


def stop_running(process,tree,stop_path,hard_end):
    if process is None:return []
    if process.poll() is None:
        write(stop_path,{'reason':'coordinator_stopping_no_retry'})
        try:process.wait(timeout=min(5,max(.01,(hard_end-utc()).total_seconds())))
        except subprocess.TimeoutExpired:pass
    if tree is not None and tree.alive():
        left=max(0,(hard_end-utc()).total_seconds())
        grace=min(3,max(0,left-1))
        return tree.stop(wait_seconds=grace,kill_wait_seconds=min(1,max(0,left-grace)))
    return []


def execute(args):
    require_budget(args.total_seconds)
    started=utc();clock=time.monotonic();hard_end=started+timedelta(seconds=args.total_seconds)
    if args.absolute_end_utc:hard_end=min(hard_end,timestamp(args.absolute_end_utc))
    output=args.output.resolve()
    if output.exists():raise ValueError('fresh_output_required_no_resume_or_retry')
    output.mkdir(parents=True)
    write(output/'status.json',{'schema_version':'harness-gpu-delivery-status/1','status':'preflight',
        'started_utc':started.isoformat(),'hard_deadline_utc':hard_end.isoformat(),'automatic_restart':False})
    protocol=plan=source_pins=None;measurements=[];status='failed';integrity=False;failure=None
    process=tree=active=None;batch=None;final_leases=[];quality={'measured':False}
    compute_end=hard_end-timedelta(seconds=CLEANUP_SECONDS+FINAL_HASH_RESERVE_SECONDS)
    kill_start=hard_end-timedelta(seconds=5)
    try:
        check_deadline(compute_end)
        source_pins=code_pins(compute_end)
        plan=build_plan(args,compute_end)
        if code_pins(compute_end)!=source_pins:raise ValueError('source_changed_during_manifest_preflight')
        from harness_ocr.inference.locks import FileLease
        batch=FileLease(args.lock_dir/'gpu-delivery.lock').acquire()
        probe_leases(args.lock_dir)
        pins,registry_sha=pin_models(args.models,plan['required_model_keys'],compute_end)
        if code_pins(compute_end)!=source_pins:raise ValueError('source_changed_during_model_verification')
        protocol={**plan,'schema_version':'harness-gpu-delivery-protocol/1',
            'created_utc':started.isoformat(),'code_pins':source_pins,'model_file_pins':pins,
            'model_registry_subset_sha256':registry_sha,'model_verification_performed':True,
            'total_seconds':args.total_seconds,'hard_deadline_utc':hard_end.isoformat(),
            'compute_deadline_utc':compute_end.isoformat(),'watchdog_kill_start_utc':kill_start.isoformat(),
            'cleanup_reserve_seconds':CLEANUP_SECONDS,'final_hash_reserve_seconds':FINAL_HASH_RESERVE_SECONDS,
            'automatic_restart':False}
        protocol['protocol_sha256']=digest(protocol)
        write(output/'protocol.json',protocol)
        for planned in protocol['planned_rows']:
            check_deadline(compute_end);preflight=time.monotonic()
            validate_runtime_pins(protocol,compute_end);probe_leases(args.lock_dir)
            row=output/'rows'/planned['row_id'];row.mkdir(parents=True)
            arm=next(a for a in protocol['arms'] if a['id']==planned['arm'])
            document=next(d for d in protocol['documents'] if d['id']==planned['document_id'])
            token=secrets.token_hex(32)
            spec={'row_id':planned['row_id'],'row_dir':str(row),'document':document,
                'settings':protocol['arm_settings'][arm['id']],'code_pins':protocol['code_pins'],
                'runner_sha256':file_sha(__file__,compute_end),'compute_deadline_utc':compute_end.isoformat(),
                'worker_token_sha256':hashlib.sha256(token.encode()).hexdigest()}
            write(row/'inference-spec.json',spec)
            active={**planned,'preflight_seconds':time.monotonic()-preflight}
            write(output/'status.json',{'schema_version':'harness-gpu-delivery-status/1','status':'running',
                'current_row_id':planned['row_id'],'recorded_rows':len(measurements),
                'planned_rows':len(protocol['planned_rows']),'hard_deadline_utc':hard_end.isoformat()})
            env=dict(os.environ);env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                HF_HUB_DISABLE_TELEMETRY='1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',
                YOLO_CONFIG_DIR=str(row/'runtime/yolo'),MPLCONFIGDIR=str(row/'runtime/matplotlib'),
                HARNESS_GPU_DELIVERY_WORKER_TOKEN=token)
            row_clock=time.monotonic()
            with (row/'stdout.log').open('wb') as stdout,(row/'stderr.log').open('wb') as stderr:
                process=subprocess.Popen([sys.executable,'-m','experiments.gpu_delivery.run',
                    '--_row-worker',str(row/'inference-spec.json')],cwd=ROOT,env=env,
                    stdout=stdout,stderr=stderr,start_new_session=os.name=='posix')
                tree=VerifiedTree(process.pid,new_session=os.name=='posix')
                watchdog=wait_child(process,tree,row/'parent-stop.json',compute_end,kill_start)
            measurement=read(row/'measurement.json') if (row/'measurement.json').is_file() else {
                'status':'administratively_censored' if watchdog['administrative_stop_requested'] else 'failed',
                'administrative_censor':watchdog['administrative_stop_requested'],'error_type':'MissingWorkerMeasurement'}
            measurement.update(planned,watchdog=watchdog,preflight_seconds=active['preflight_seconds'],
                row_process_wall_seconds=time.monotonic()-row_clock)
            measurement['state_file_sha256']=state_pins(row,hard_end)
            measurement['result_sha256']=file_sha(row/'result.json',hard_end) if (row/'result.json').is_file() else None
            write(row/'coordinator-measurement.json',measurement);measurements.append(measurement)
            write(output/'measurements.json',measurements)
            if tree.alive():
                write(row/'unexpected-owner-cleanup.json',tree.stop());raise RuntimeError('owned_descendants_left_alive')
            probe_leases(args.lock_dir)
            if watchdog['inspection_errors'] or measurement.get('close_error_type'):
                raise RuntimeError('owner_or_watchdog_integrity_failure')
            if watchdog['forced_cleanup']:
                if not watchdog['administrative_stop_requested']:raise RuntimeError('unexpected_forced_cleanup')
                raise BudgetExpired('global_model_deadline')
            if measurement.get('stage')=='warmup' and measurement.get('status')=='failed':
                raise RuntimeError('warmup_failed_no_retry')
            process=tree=active=None
        status='completed'
    except BudgetExpired:
        status='budget_exhausted';failure='global_administrative_budget'
    except KeyboardInterrupt:
        status='stopped';failure='operator_interrupted_no_retry'
    except BaseException as error:
        status='failed';failure=type(error).__name__
        write(output/'private-coordinator-error.json',{'type':type(error).__name__,'message':str(error)[:1000]})
    finally:
        try:
            if process is not None:
                stop_path=output/'rows'/active['row_id']/'parent-stop.json' if active else output/'parent-stop.json'
                actions=stop_running(process,tree,stop_path,hard_end)
                if actions:write(output/'forced-cleanup.json',actions)
                if tree is not None and tree.alive():raise RuntimeError('owned_process_exit_not_proven')
            if active and not any(r['row_id']==active['row_id'] for r in measurements):
                row=output/'rows'/active['row_id']
                measurement=read(row/'measurement.json') if (row/'measurement.json').is_file() else {'status':'failed'}
                measurement.update(active,administrative_censor=status in {'budget_exhausted','stopped'},
                    coordinator_interruption=failure)
                measurements.append(measurement);write(output/'measurements.json',measurements)
            final_leases=probe_leases(args.lock_dir)
            if protocol is not None:
                validate_runtime_pins(protocol,hard_end,full_models=True);integrity=True
                check_deadline(hard_end)
                quality=score_after_inference(protocol,output,hard_end)
        except BudgetExpired:
            integrity=False
            if status=='completed':status='budget_exhausted'
            failure=failure or 'final_integrity_or_scoring_budget_exhausted'
        except BaseException as error:
            integrity=False;status='failed';failure=failure or type(error).__name__
        finally:
            if batch is not None:
                try:batch.close()
                except BaseException as error:
                    integrity=False;status='failed';failure=failure or type(error).__name__
        elapsed=time.monotonic()-clock
        if protocol is not None:
            summary=make_summary(protocol,output,measurements,status,integrity,elapsed)
            if protocol.get('gold_metadata'):
                # Counts remain available even if the final scoring budget ran
                # out. They are not observations of correct/incorrect answers.
                quality={**quality,'references_provided':True,
                    'gold_sha256':protocol['gold_metadata']['sha256'],
                    'reference_field_count_per_arm':protocol['gold_metadata']['reference_field_count']}
            summary['quality']=quality
            summary['correctness_unmeasured']=not quality.get('measured',False)
            write(output/'summary.json',summary)
        else:
            missing=[{key:row[key] for key in ('row_id','public_document_id','arm')}
                     for row in plan['planned_rows']] if plan else []
            for row in missing:row.update(status='missing',technical_execution_passed=False,
                administrative_censor=status in {'budget_exhausted','stopped'},
                missing_reason='not_started_preflight_failed_or_budget')
            summary={'schema_version':'harness-gpu-delivery-summary/1','status':status,
                'technical_execution_passed':False,'quality_qualified':False,'failure_type':failure,
                'correctness_unmeasured':True,
                'source_code_sha256':digest(source_pins) if source_pins is not None else None,
                'model_verification_performed':False,'arms':plan['arms'] if plan else [],
                'common_caps':plan['common_caps'] if plan else COMMON,
                'planned_rows':len(missing),'recorded_rows':0,'missing_rows':len(missing),'rows':missing,
                'elapsed_wall_seconds_including_preflight_and_close':elapsed}
            write(output/'summary.json',summary)
        write(output/'status.json',{'schema_version':'harness-gpu-delivery-status/1','status':status,
            'finished_utc':utc().isoformat(),'hard_deadline_utc':hard_end.isoformat(),
            'recorded_rows':len(measurements),'planned_rows':summary['planned_rows'],
            'failure_type':failure,'integrity_verification_passed':integrity,
            'final_lease_proofs':final_leases,'automatic_restart':False})
    return summary


def parser():
    value=argparse.ArgumentParser(description=__doc__)
    value.add_argument('--manifest',type=Path)
    value.add_argument('--output',type=Path)
    value.add_argument('--models',type=Path,default=Path(os.getenv('HARNESS_MODELS','models')))
    value.add_argument('--lock-dir',type=Path,default=Path(os.getenv('HARNESS_LOCK_DIR','var/locks')))
    value.add_argument('--qwen',choices=('both','qwen-0.8b','qwen-4b'),default='both')
    value.add_argument('--include-yolo-x','--include-x',action='store_true')
    value.add_argument('--total-seconds',type=float,default=10800)
    value.add_argument('--absolute-end-utc')
    value.add_argument('--gold',type=Path,help='Separate string-only gold: parent schema check before NN; scoring after owners exit')
    value.add_argument('--execute',action='store_true')
    value.add_argument('--dtype',choices=('bfloat16','float32'),default='bfloat16')
    value.add_argument('--attention-backend',choices=('sdpa','eager'),default='sdpa')
    for name in ('cpu_threads','max_pages','max_page_pixels','max_input_tokens','max_new_tokens',
                 'max_rereads','max_detector_rescues','max_whole_page_reads','max_visual_regions'):
        value.add_argument('--'+name.replace('_','-'),type=int,default=COMMON[name])
    for name in ('call_timeout_seconds','document_timeout_seconds'):
        value.add_argument('--'+name.replace('_','-'),type=float,default=COMMON[name])
    value.add_argument('--_row-worker',type=Path,help=argparse.SUPPRESS)
    return value


def main():
    arguments=parser();args=arguments.parse_args()
    if args._row_worker:return row_worker(args._row_worker)
    if args.manifest is None or args.output is None:arguments.error('--manifest and --output are required')
    require_budget(args.total_seconds)
    if args.absolute_end_utc:timestamp(args.absolute_end_utc)
    if args.execute:
        summary=execute(args)
        print(json.dumps({'status':summary['status'],'technical_execution_passed':summary['technical_execution_passed'],
            'planned_rows':summary['planned_rows'],'recorded_rows':summary['recorded_rows'],
            'quality_qualified':False},ensure_ascii=False))
        return 0 if summary['technical_execution_passed'] else (2 if summary['status'] in {'budget_exhausted','stopped'} else 1)
    plan=build_plan(args)
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'plan.json',plan)
    print(json.dumps({'status':'planned','execution_started':False,'preflight_neural_calls':0,
        'documents':len(plan['documents']),'arms':len(plan['arms']),'planned_rows':len(plan['planned_rows']),
        'required_model_keys':plan['required_model_keys'],'model_verification_performed':False},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
