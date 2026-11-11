from __future__ import annotations
from dataclasses import replace
import threading
import os
from .artifacts import ArtifactStore
from .jobs import JobStore,TERMINAL
from .engine import Engine
from .inference.supervisor import Supervisor
from .models import verify_models,model_lock
from .util import write_json,utc
from .core import StateError
from .profiles import project_contract_result

class Service:
    def __init__(self,settings,backend=None,fixture=False):
        settings.validate();self.settings=settings
        if fixture and not settings.allow_fixture:raise ValueError('fixture_mode_must_be_explicit')
        self.fixture=fixture;self.backend=backend or Supervisor(settings)
        self.artifacts=ArtifactStore(settings.data_dir/'artifacts',settings.max_upload_bytes)
        self.jobs=JobStore(settings.data_dir/'jobs.sqlite');self.engine=Engine(settings,self.backend)
        self.stop=threading.Event();self.wake=threading.Event();self.thread=None;self.owner=None
        self._started=False;self._fatal=None
        self._verification_lock=threading.Lock();self._verified={}
        self._cancel_lock=threading.Lock();self._active_cancel={}
        self.readiness={'ready':False,'fixture':fixture,'reasons':['startup_pending']}
    def readiness_status(self):
        value=dict(self.readiness)
        reasons=list(value.get('reasons',[]))
        if self.stop.is_set():reasons.append('service_stopped')
        if self._fatal:reasons.append('scheduler_failed')
        if self._started and value.get('ready') and (not self.thread or not self.thread.is_alive()):
            reasons.append('scheduler_not_running')
        if reasons:value.update(ready=False,reasons=sorted(set(reasons)))
        return value
    def liveness_status(self):
        live=not self._fatal and not (self._started and self.readiness.get('ready')
                                    and (not self.thread or not self.thread.is_alive()))
        return {'live':live,'scheduler_failed':bool(self._fatal)}
    def _fingerprint(self,names):
        lock=model_lock()['models'];result=[]
        for name in names:
            if name not in lock:raise ValueError('unknown_model_bundle')
            for item in lock[name]['files']:
                p=self.settings.models_dir/name/item['file']
                try:
                    s=p.stat();result.append((str(p),s.st_size,s.st_mtime_ns,s.st_ino))
                except OSError:result.append((str(p),None))
        return tuple(result)
    def _ensure_models(self,qwen):
        if qwen not in {'qwen-0.8b','qwen-4b'}:raise ValueError('unknown_qwen')
        if self.fixture:return {'ok':True,'fixture':True}
        names=['detector','cyrillic','xiaomi',qwen]
        with self._verification_lock:
            fingerprint=self._fingerprint(names)
            cached=self._verified.get(qwen)
            if cached and cached['fingerprint']==fingerprint:return cached['verification']
            verification=verify_models(self.settings.models_dir,names)
            if verification['ok']:
                self._verified[qwen]={'fingerprint':fingerprint,'verification':verification}
            else:self._verified.pop(qwen,None)
            return verification
    def start(self):
        if self._started or self.owner is not None:raise RuntimeError('service_already_started')
        if self.stop.is_set():raise RuntimeError('stopped_service_requires_new_instance')
        # One scheduler owns a data directory. Separate HTTP worker processes are forbidden.
        self.owner=(self.settings.data_dir/'scheduler.lock').open('a+b')
        try:
            if os.name=='nt':
                import msvcrt
                self.owner.seek(0);self.owner.write(b'0');self.owner.flush();self.owner.seek(0)
                msvcrt.locking(self.owner.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.owner.close();self.owner=None;raise RuntimeError('scheduler_already_owned')
        try:self.jobs.recover()
        except BaseException:
            self.owner.close();self.owner=None
            self.readiness={'ready':False,'fixture':self.fixture,'reasons':['startup_recovery_failed']}
            raise
        try:
            if self.fixture:self.readiness={'ready':True,'fixture':True,'quality_qualified':False}
            else:
                verification=self._ensure_models(self.settings.qwen)
                if not verification['ok']:self.readiness={'ready':False,'reasons':verification['errors']}
                else:
                    probe=self.backend.call('probe',{'warmup':True})
                    self.readiness={'ready':True,'fixture':False,'quality_qualified':False,
                                    'verification':verification,'runtime':probe}
        except Exception as e:self.readiness={'ready':False,'reasons':[type(e).__name__+':'+str(e)[:200]]}
        # Failed SHA/CUDA verification must hold persisted queued work too.
        # It must never be a mere HTTP readiness warning while the queue runs.
        if self.readiness.get('ready'):
            try:
                self.thread=threading.Thread(target=self._loop,daemon=True,name='harness-single-scheduler');self.thread.start()
            except BaseException:
                self.thread=None
                self.backend.close();self.owner.close();self.owner=None
                self.readiness={'ready':False,'fixture':self.fixture,'reasons':['scheduler_start_failed']}
                raise
        self._started=True
    def close(self):
        self.stop.set();self.wake.set()
        if self.thread:self.thread.join(timeout=self.settings.call_timeout_seconds+5)
        self.backend.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                # Keep ownership until process exit; a second scheduler must not
                # start while this thread could still publish to the database.
                raise RuntimeError('scheduler_shutdown_deadline_exceeded')
        if self.owner:self.owner.close();self.owner=None
    def submit(self,request,idempotency_key=None):
        if not self.readiness_status()['ready']:raise RuntimeError('engine_not_ready')
        self.artifacts.get(request['artifact_id'])
        self.engine.compile_contract(request['profile_id'],subset=request.get('subset'))
        qwen=request.get('qwen') or self.settings.qwen
        verification=self._ensure_models(qwen)
        if not verification['ok']:raise ValueError('requested_model_not_prepared')
        job=self.jobs.create(request,idempotency_key);self.wake.set();return job
    def cancel(self,jid):
        value=self.jobs.cancel(jid)
        # The watchdog callback reads memory only; a contended SQLite read must
        # never delay a hard model deadline or cancellation by busy_timeout.
        with self._cancel_lock:
            event=self._active_cancel.get(jid)
            if event and value['cancel_requested']:event.set()
        self.wake.set();return value
    def _loop(self):
        try:
            while not self.stop.is_set() and self.readiness.get('ready'):
                pending=self.jobs.peek()
                if pending is None:self.wake.wait(.25);self.wake.clear();continue
                verification=self._ensure_models(pending['request'].get('qwen') or self.settings.qwen)
                if not verification['ok']:
                    self.readiness={'ready':False,'fixture':self.fixture,
                        'reasons':['queued_model_verification_failed'],
                        'verification_errors':verification.get('errors',[])}
                    return
                job=self.jobs.next(expected_id=pending['id'])
                if job is None:continue
                event=threading.Event()
                with self._cancel_lock:self._active_cancel[job['id']]=event
                try:
                    if self.jobs.get(job['id'])['cancel_requested']:event.set()
                    path,artifact=self.artifacts.get(job['request']['artifact_id']);r=job['request']
                    result=self.engine.run(job['id'],path,artifact['source_sha256'],r['profile_id'],
                        subset=r.get('subset'),qwen=r.get('qwen'),mode=r.get('mode','hybrid'),
                        emit_operational_events=False,
                        cancel=lambda:self.stop.is_set() or event.is_set())
                except Exception as e:
                    result={'document_id':job['id'],'terminal_status':'failed','completed':False,
                            'decision':'review','fields':{},'errors':[type(e).__name__+':'+str(e)[:200]]}
                finally:
                    with self._cancel_lock:self._active_cancel.pop(job['id'],None)
                self.jobs.finish(job['id'],result)
        except BaseException as e:
            self._fatal={'stage':'scheduler','error_type':type(e).__name__,'utc':utc()}
            self.readiness={'ready':False,'fixture':self.fixture,'reasons':['scheduler_failed']}
            try:write_json(self.settings.data_dir/'service_failure.json',self._fatal)
            except OSError:pass
    def result(self,jid):
        job=self.jobs.get(jid);result=job['result']
        recover_partial=not result and job['status'] in {'interrupted','canceled','failed'}
        if not result and not recover_partial:return None
        try:snapshot=self.engine.documents.get(jid)
        except StateError as e:
            if str(e)!='document_or_revision_not_found':raise
            if not result:return None
            request=job['request'];contract=self.engine.compile_contract(request['profile_id'],subset=request.get('subset'))
            result={**result,'contract_result':project_contract_result(contract,result)}
            return {k:v for k,v in result.items() if k!='optics'}
        # Re-evaluate the trusted server gateway even when the document revision
        # is unchanged: policy revocation and sticky quarantine are live state.
        # Runtime strategy limitations belong to this trial and cannot disappear
        # merely because a human correction increments the document revision.
        sticky=[reason for reason in (result or {}).get('reason_codes',[])
                if isinstance(reason,str) and reason.startswith('unsupported_strategy_')]
        if (result or {}).get('unresolved_rereads'):
            sticky.append('unresolved_reread_requests')
        r=job['request'];contract=self.engine.compile_contract(r['profile_id'],subset=r.get('subset'))
        original_binding=snapshot['metadata']['binding']
        current_binding=self.engine.binding(contract,r.get('qwen') or self.settings.qwen,
            mode=r.get('mode','hybrid'),allow_reread=True)
        binding_changed=current_binding!=original_binding
        if binding_changed:sticky.append('runtime_binding_changed_requires_reprocess')
        # Keep the original model/source provenance; do not relabel old optical
        # evidence as if it had been generated by the current runtime. Passing
        # this barrier into project also prevents an accepted audit/drift event.
        result={**(result or {}),**self.engine.project(snapshot,contract,original_binding,
            terminal=job['status'],complete=False if recover_partial else result.get('completed',False),
            review_reasons=sticky)}
        result['current_projection_binding']=current_binding
        result['runtime_binding_changed']=binding_changed
        if binding_changed:
            result['decision']='review';result['needs_review']=True
            result['reason_codes']=sorted(set(result.get('reason_codes',[])+['runtime_binding_changed_requires_reprocess']))
            for field in result.get('fields',{}).values():field['decision']='review'
        if recover_partial:
            result['reason_codes']=sorted(set(result.get('reason_codes',[])+[job['status'],'durable_partial_result_recovered']))
            result['recovered_partial_result']=True
        result['contract_result']=project_contract_result(contract,result)
        return {k:v for k,v in result.items() if k!='optics'}
    def review(self,jid,correction):
        job=self.jobs.get(jid)
        if job['status'] not in TERMINAL:raise ValueError('review_requires_terminal_job')
        r=job['request'];contract=self.engine.compile_contract(r['profile_id'],subset=r.get('subset'))
        if correction['field_key'] not in contract['field_definitions']:raise ValueError('unknown_review_field')
        self.engine.documents.human_correct(jid,correction['field_key'],correction['raw_value'],
             correction['evidence_ids'],actor='authenticated_reviewer',
             expected_revision=correction['expected_revision'],authorized=True,
             party_role=contract['field_definitions'][correction['field_key']]['role'])
        return self.result(jid)
    def record_audit_outcome(self,task_id,outcome,*,authorized=False):
        """Reviewer-only bridge: source membership before the atomic safety action."""
        if authorized is not True:raise PermissionError('audit_reviewer_authorization_required')
        selections=[event['payload'] for event in self.engine.audit_log.events('audit_selection')
                    if event['payload'].get('task_id')==task_id]
        if not selections:raise KeyError('audit_task_not_found')
        if len(selections)!=1 or selections[0].get('selected') is not True:
            raise ValueError('audit_task_not_selected')
        selection=selections[0]
        snapshot=self.engine.documents.get(selection['document_id'])
        evidence_ids=outcome.get('evidence_ids')
        if (not isinstance(evidence_ids,list) or not evidence_ids
                or any(not isinstance(ref,str) for ref in evidence_ids)
                or len(evidence_ids)!=len(set(evidence_ids))):
            raise ValueError('audit_requires_unique_document_evidence')
        for ref in evidence_ids:
            evidence=snapshot['observations'].get(ref)
            if not evidence or evidence.get('active') is not True or evidence.get('completed') is not True:
                raise ValueError('audit_evidence_missing_inactive_or_incomplete')
        correction=outcome.get('correction_revision')
        if correction is not None:
            if type(correction) is not int or correction<selection['revision']:
                raise ValueError('audit_correction_revision_invalid')
            revisions=self.engine.documents.revisions(selection['document_id'])
            event=next((row['event'] for row in revisions if row['revision']==correction),None)
            if not event or event.get('type')!='human_correction':
                raise ValueError('audit_correction_revision_not_authorized_human_event')
        return self.engine.audit.record_outcome(task_id,critical_error=outcome.get('critical_error'),
            evidence_ids=evidence_ids,correction_revision=correction,
            actor_id='authenticated_reviewer',actor_scopes=['audit:write'])
