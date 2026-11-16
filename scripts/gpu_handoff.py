"""One-command Linux/Docker GPU handoff; no HTTP service is required.

Only the explicit preparation stage has network access. User inputs remain
read-only, reports are host binds, and model operations share the service's
device-lock volume. Existing outputs are never resumed or overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / 'compose.gpu-handoff.yaml'
SCHEMA = 'harness-own-documents/1'
ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z')
VOLUME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
PROJECT = re.compile(r'[a-z0-9][a-z0-9_-]{0,127}\Z')
DOCUMENT_EXTENSIONS = {'.png','.jpg','.jpeg','.pdf','.tif','.tiff','.bmp','.webp','.xml'}
MAX_FILE = 30 * 1024 * 1024


class HandoffError(ValueError):
    """Stable operator-facing reason, never arbitrary subprocess output."""


def require(condition, reason):
    if not condition:
        raise HandoffError(reason)


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def safe_json(path, limit=4*1024*1024):
    require(path.is_file() and path.stat().st_size <= limit, 'expected_report_missing_or_oversize')
    try:
        return json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, UnicodeError):
        raise HandoffError('invalid_report_json') from None


def documents_from_args(args):
    """Host preflight is strict and gold-free; contracts are compiled in image."""
    if args.manifest:
        require(not args.profile and not args.subset, 'profile_subset_belong_in_manifest')
        manifest = args.manifest.resolve()
        value = safe_json(manifest, 1024*1024)
        require(isinstance(value,dict) and set(value)=={'schema_version','documents'}
                and value['schema_version']==SCHEMA, 'manifest_schema')
        base = manifest.parent
        docs = value['documents']
    else:
        require(bool(args.profile), 'input_directory_requires_explicit_profile')
        base = args.input.resolve()
        require(base.is_dir(), 'input_directory_missing')
        # An explicitly supplied reference file is not an input document.
        gold_source=args.gold.resolve() if args.gold else None
        files = sorted((p for p in base.iterdir() if p.resolve()!=gold_source),key=lambda p:p.name)
        require(all(p.is_file() and not p.is_symlink() and p.suffix.lower() in DOCUMENT_EXTENSIONS
                    for p in files), 'input_must_be_flat_documents_or_use_manifest')
        docs = [{'id':f'doc-{i+1:04d}', 'path':p.name, 'profile_id':args.profile,
                 **({'subset':args.subset} if args.subset else {})} for i,p in enumerate(files)]
    require(isinstance(docs,list) and 1 <= len(docs) <= 100, 'document_count')
    checked, seen = [], set()
    for doc in docs:
        require(isinstance(doc,dict) and {'id','path','profile_id'} <= set(doc)
                and set(doc) <= {'id','path','profile_id','subset'}, 'manifest_document_shape')
        ident, name, profile = doc['id'], doc['path'], doc['profile_id']
        require(isinstance(ident,str) and ID.fullmatch(ident) and ident.lower() not in seen,
                'manifest_document_id')
        seen.add(ident.lower())
        require(isinstance(name,str) and 1 <= len(name) <= 1024
                and not Path(name).is_absolute() and '\\' not in name, 'relative_source_path_required')
        source = (base/name).resolve()
        require(source.is_relative_to(base) and source.is_file(), 'source_outside_manifest_or_missing')
        require(0 < source.stat().st_size <= MAX_FILE, 'source_size')
        require(isinstance(profile,str) and 1 <= len(profile) <= 128, 'profile_required')
        subset = doc.get('subset')
        require(subset is None or isinstance(subset,list) and 1 <= len(subset) <= 100
                and all(isinstance(s,str) and s for s in subset)
                and len(set(subset)) == len(subset), 'subset_shape')
        with source.open('rb') as stream:
            raw=stream.read(MAX_FILE+1)
        require(0 < len(raw) <= MAX_FILE, 'source_size')
        checked.append({'id':ident,'source':source,'profile_id':profile,'subset':subset,
                        'source_sha256':hashlib.sha256(raw).hexdigest()})
    return checked


def gold_from_args(path, checked):
    """References remain coordinator-only; values are never returned or logged."""
    if path is None:
        return None
    source=path.resolve()
    require(source.is_file() and source.stat().st_size<=10*1024*1024,'gold_size_or_path')
    with source.open('rb') as stream:
        raw_bytes=stream.read(10*1024*1024+1)
    require(len(raw_bytes)<=10*1024*1024,'gold_size_or_path')
    try:
        value=json.loads(raw_bytes.decode('utf-8-sig'))
    except (UnicodeError,ValueError):
        raise HandoffError('gold_json') from None
    require(isinstance(value,dict) and set(value)=={'schema_version','documents'}
            and value['schema_version']=='harness-gpu-delivery-gold/1'
            and isinstance(value['documents'],dict) and bool(value['documents']), 'gold_schema')
    allowed={d['id']:d for d in checked}
    require(set(value['documents']) <= set(allowed), 'gold_unknown_document')
    count=0
    for ident,fields in value['documents'].items():
        require(isinstance(fields,dict) and bool(fields)
                and all(isinstance(key,str) and 1 <= len(key) <= 128 for key in fields), 'gold_field_shape')
        subset=allowed[ident]['subset']
        require(subset is None or set(fields)<=set(subset), 'gold_field_outside_subset')
        require(all(isinstance(raw,str) and 0 < len(raw) <= 8192 for raw in fields.values()),
                'gold_requires_nonempty_exact_raw_strings_no_typed_values')
        count+=len(fields)
    return {'source':source,'sha256':hashlib.sha256(raw_bytes).hexdigest(),
            'documents':len(value['documents']),'fields':count}


def prepare_references(gold, state):
    if gold is None:
        return None
    target=state/'references'
    target.mkdir(mode=0o700)
    (target/'gold.json').touch(mode=0o600)
    mounts=[{'type':'bind','source':compose_literal(target),'target':'/references',
             'read_only':True,'bind':{'create_host_path':False}},
            {'type':'bind','source':compose_literal(gold['source']),'target':'/references/gold.json',
             'read_only':True,'bind':{'create_host_path':False}}]
    overlay=state/'references.compose.json'
    write_json(overlay,{'services':{'handoff':{'volumes':mounts},'measured':{'volumes':mounts}}})
    return overlay


def check_gold(gold):
    if gold is None:
        return
    try:
        with gold['source'].open('rb') as stream:
            raw=stream.read(10*1024*1024+1)
    except OSError:
        raise HandoffError('gold_changed_after_preflight') from None
    require(len(raw)<=10*1024*1024 and hashlib.sha256(raw).hexdigest()==gold['sha256'],
            'gold_changed_after_preflight')


def compose_literal(text):
    # Compose interpolates strings in JSON too. A filename is never an env var.
    return str(text).replace('$','$$')


def prepare_inputs(checked, target):
    target.mkdir(mode=0o700)
    (target/'documents').mkdir(mode=0o700)
    documents, mounts = [], []
    for i,doc in enumerate(checked):
        suffix = doc['source'].suffix.lower() or '.bin'
        alias = f'documents/{i:04d}{suffix}'
        # Existing mount points permit nested file binds under a read-only root.
        (target/alias).touch(mode=0o600)
        documents.append({'id':doc['id'],'path':alias,'profile_id':doc['profile_id'],
                          **({'subset':doc['subset']} if doc['subset'] is not None else {})})
        mounts.append({'type':'bind','source':compose_literal(doc['source']),
                       'target':'/input/'+alias,'read_only':True,
                       'bind':{'create_host_path':False}})
    write_json(target/'own-documents.json', {'schema_version':SCHEMA,'documents':documents})
    return mounts


def check_sources(checked):
    for doc in checked:
        try:
            with doc['source'].open('rb') as stream:
                raw=stream.read(MAX_FILE+1)
        except OSError:
            raise HandoffError('source_changed_after_preflight') from None
        require(0 < len(raw) <= MAX_FILE and hashlib.sha256(raw).hexdigest()==doc['source_sha256'],
                'source_changed_after_preflight')


class Docker:
    def __init__(self, command, environment, report_dir, compose_files, project):
        self.command=list(command)
        self.environment=dict(environment)
        self.report_dir=report_dir
        self.compose_files=compose_files
        self.project=project
        self.current_container=None
        self.current_service=None
        self.last_cleanup=None

    def raw(self, arguments, *, timeout=30):
        try:
            result=subprocess.run(self.command+arguments, cwd=ROOT, env=self.environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                errors='replace', timeout=timeout, check=False)
        except (OSError,subprocess.TimeoutExpired):
            raise HandoffError('docker_unavailable_or_timed_out') from None
        require(result.returncode==0, 'docker_preflight_command_failed')
        return result.stdout

    def compose(self, extra=()):
        out=self.command+['compose']
        for file in [*self.compose_files,*extra]:
            out += ['--file',str(file)]
        return out+['--project-name',self.project]

    def inspect_owned(self, identity, expected_name):
        """No raw inspect payload leaves this function, especially Config.Env."""
        try:
            found=subprocess.run(self.command+['inspect','--type','container',identity],
                env=self.environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                text=True,encoding='utf-8',errors='replace',timeout=10,check=False)
            if found.returncode:
                # --rm removes a stopped container. A successful empty exact
                # inventory confirms absence; an unreachable daemon does not.
                selector=('id='+identity if re.fullmatch('[0-9a-f]{64}',identity)
                          else 'name=^/'+re.escape(expected_name)+'$')
                inventory=subprocess.run(self.command+['ps','-a','--filter',selector,'--format','{{.ID}}'],
                    env=self.environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                    text=True,encoding='utf-8',errors='replace',timeout=10,check=False)
                return {'state':'absent'} if inventory.returncode==0 and not inventory.stdout.strip() else {'state':'unavailable'}
            value=json.loads(found.stdout)
            if not isinstance(value,list) or len(value)!=1 or not isinstance(value[0],dict):return {'state':'unavailable'}
            item=value[0];container_id=item.get('Id');config=item.get('Config') or {}
            labels=config.get('Labels') if isinstance(config,dict) else None
            running=(item.get('State') or {}).get('Running')
            if (not isinstance(container_id,str) or not re.fullmatch('[0-9a-f]{64}',container_id)
                or item.get('Name')!='/'+expected_name or not isinstance(labels,dict)
                or labels.get('com.docker.compose.project')!=self.project
                or (self.current_service and labels.get('com.docker.compose.service')!=self.current_service)
                or (re.fullmatch('[0-9a-f]{64}',identity) and identity!=container_id)):
                return {'state':'ownership_mismatch'}
            if type(running) is not bool:return {'state':'unavailable'}
            return {'state':'running' if running else 'not_running','id':container_id}
        except (OSError,subprocess.TimeoutExpired,ValueError,AttributeError):
            return {'state':'unavailable'}

    def stop_owned_container(self):
        if not self.current_container:
            return
        name=self.current_container
        self.last_cleanup={'attempted':True,'generated_container_name':name,'expected_project':self.project,
            'validated_container_id':None,'ownership_verified':False,'stopped_by_docker':False,
            'kill_attempted':False,'confirmed_not_running':False,'final_state':'unresolved',
            'scope':'exact_owned_container_only'}
        initial=self.inspect_owned(name,name)
        if initial['state']=='absent':
            self.last_cleanup.update(confirmed_not_running=True,final_state='absent');return
        if initial['state'] not in {'running','not_running'}:
            self.last_cleanup['final_state']=initial['state'];return
        identity=initial['id']
        self.last_cleanup.update(validated_container_id=identity,ownership_verified=True)
        if initial['state']=='not_running':
            self.last_cleanup.update(confirmed_not_running=True,final_state='not_running');return
        try:
            stopped=subprocess.run(self.command+['stop','--time','30',identity],env=self.environment,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=45,check=False)
            self.last_cleanup['stopped_by_docker']=stopped.returncode==0
        except (OSError,subprocess.TimeoutExpired):pass
        after=self.inspect_owned(identity,name)
        if after['state'] in {'absent','not_running'}:
            self.last_cleanup.update(confirmed_not_running=True,final_state=after['state']);return
        if after['state']!='running':
            self.last_cleanup['final_state']=after['state'];return
        self.last_cleanup['kill_attempted']=True
        try:
            killed=subprocess.run(self.command+['kill',identity],env=self.environment,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15,check=False)
            self.last_cleanup['killed_by_docker']=killed.returncode==0
        except (OSError,subprocess.TimeoutExpired):self.last_cleanup['killed_by_docker']=False
        final=self.inspect_owned(identity,name)
        self.last_cleanup.update(confirmed_not_running=final['state'] in {'absent','not_running'},
                                 final_state=final['state'])

    def stage(self, name, command, *, timeout, service='handoff', build=False,compose_extra=()):
        arguments=self.compose(compose_extra)
        if build:
            arguments += ['build','--pull','handoff']
        else:
            self.current_container=f'{self.project}-{name}'
            self.current_service=service
            arguments += ['run','--rm','--no-deps','--pull','never',
                          '--name',self.current_container,service]+command
        log_path=self.report_dir/'logs'/(name+'.log')
        print('Harness OCR: '+name, flush=True)
        try:
            with log_path.open('x', encoding='utf-8') as log:
                child=subprocess.run(arguments, cwd=ROOT, env=self.environment,
                    stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            # A failed Docker client may have left its container alive. Never
            # retry the model stage; verify and close only our generated owner.
            if not build and child.returncode:
                self.stop_owned_container()
            return child.returncode
        except (OSError,subprocess.TimeoutExpired,KeyboardInterrupt):
            self.stop_owned_container()
            raise
        finally:
            if not self.last_cleanup or self.last_cleanup.get('confirmed_not_running'):
                self.current_container=None;self.current_service=None


def gpu_identifier(value):
    """Only canonical host indices or full physical GPU UUIDs prove disjointness.

    Index-vs-UUID, partial UUIDs, MIG IDs and missing/all selections can alias
    the same physical GPU and therefore remain conservative unknowns.
    """
    if isinstance(value,str) and re.fullmatch(r'0|[1-9][0-9]*',value):
        return 'index',int(value)
    if isinstance(value,str) and re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}',value):
        return 'uuid',value.lower()
    return None


def provably_disjoint_gpu(container, selected):
    chosen=gpu_identifier(selected)
    host=container.get('HostConfig')
    if chosen is None or not isinstance(host,dict) or host.get('Privileged') or host.get('Devices'):
        return False
    requests=host.get('DeviceRequests')
    if not isinstance(requests,list) or not requests:return False
    identifiers=[]
    for request in requests:
        if not isinstance(request,dict) or request.get('Driver')!='nvidia':return False
        values=request.get('DeviceIDs')
        if (request.get('Count',0) not in (0,None) or not isinstance(values,list) or not values):return False
        for value in values:
            known=gpu_identifier(value)
            if known is None or known[0]!=chosen[0] or known==chosen:return False
            identifiers.append(known)
    return bool(identifiers)


def check_old_service(docker, service_project, lock_volume, gpu_id='0'):
    ids=docker.raw(['ps','--filter','label=com.docker.compose.service=ocr',
                    '--format','{{.ID}}']).split()
    require(all(re.fullmatch(r'[0-9a-fA-F]{6,64}',i) for i in ids), 'invalid_container_inventory')
    if not ids:
        return
    try:
        containers=json.loads(docker.raw(['inspect',*ids]))
    except ValueError:
        raise HandoffError('invalid_container_inventory') from None
    require(isinstance(containers,list) and len(containers)==len(ids)
            and all(isinstance(c,dict) for c in containers),
            'invalid_container_inventory')
    for index,container in enumerate(containers):
        config=container.get('Config') or {}
        require(isinstance(config,dict),'invalid_container_inventory')
        labels=config.get('Labels') or {}
        require(isinstance(labels,dict)
                and labels.get('com.docker.compose.service')=='ocr','invalid_container_inventory')
        if provably_disjoint_gpu(container,gpu_id):continue
        # Never print/store the inspect payload: Config.Env may contain tokens.
        environment=config.get('Env',[])
        lock_env=[e.split('=',1)[1].rstrip('/') or '/' for e in environment
                  if isinstance(e,str) and e.startswith('HARNESS_LOCK_DIR=')] if isinstance(environment,list) else []
        mounts=container.get('Mounts',[])
        if not isinstance(mounts,list):mounts=[]
        shared=len(lock_env)==1 and any(isinstance(m,dict) and m.get('Type')=='volume'
            and m.get('Name')==lock_volume and isinstance(m.get('Destination'),str)
            and (m['Destination'].rstrip('/') or '/')==lock_env[0] for m in mounts)
        identity=container.get('Id')
        identity=identity if isinstance(identity,str) and re.fullmatch(r'[0-9a-fA-F]{6,64}',identity) \
            and any(identity.startswith(known) for known in ids) else ids[index]
        project=labels.get('com.docker.compose.project')
        command='run docker stop '+identity
        if isinstance(project,str) and PROJECT.fullmatch(project):
            command+=' or docker compose --project-name '+project+' stop ocr'
        reason='running_harness_service' if shared else 'old_harness_lock_namespace'
        raise HandoffError(reason+': '+command+', then rerun with a NEW output directory')


def parser():
    p=argparse.ArgumentParser(description=__doc__,epilog=(
        'With --input, doc-0001, doc-0002... follow filename order (case-sensitive), excluding --gold. '
        'The mapping is private-input-map.json and changes if the input set changes. '
        'Prefer an explicit --manifest with stable IDs when supplying --gold.'))
    source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input',type=Path,help='Flat document directory; requires --profile')
    source.add_argument('--manifest',type=Path,help='Gold-free harness-own-documents/1 manifest')
    p.add_argument('--profile',help='Explicit profile for every --input document; no type guessing')
    p.add_argument('--subset',nargs='+',help='Explicit fields for --input; manifest owns its subsets')
    p.add_argument('--gold',type=Path,help='Separate scalar exact references, coordinator-only; never inline in manifest')
    p.add_argument('--output',type=Path,required=True,help='NEW host report directory')
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--qwen',choices=['both','qwen-0.8b','qwen-4b'],default='both')
    p.add_argument('--include-yolo-x',action='store_true')
    p.add_argument('--gpu-id',default=os.getenv('HARNESS_GPU_ID','0'))
    p.add_argument('--lock-volume',default=os.getenv('HARNESS_LOCK_VOLUME','harness-ocr-device-locks'))
    p.add_argument('--service-project',default=os.getenv('COMPOSE_PROJECT_NAME','harness-ocr'))
    p.add_argument('--total-seconds',type=float,default=10800,help='Comparison budget; build/download/smoke separate')
    p.add_argument('--smoke-seconds',type=float,default=600)
    p.add_argument('--download-seconds',type=float,default=7200)
    p.add_argument('--build-seconds',type=float,default=3600)
    p.add_argument('--skip-download',action='store_true',help='Verify an already prepared pinned cache offline')
    p.add_argument('--prepare-only',action='store_true',help='Plan and prepare pinned models; no model calls')
    return p


def run_handoff(args, *, docker_command=None, identity=None, state_dir=None, export_command=None):
    """Injected executable prefixes are for host contract tests, never NN pass claims."""
    output=args.output.resolve()
    require(not output.exists(), 'output_must_be_new_no_inference_resume')
    if args.input:
        require(not output.is_relative_to(args.input.resolve()), 'output_must_not_be_inside_input')
    output.mkdir(parents=True,mode=0o700)
    (output/'logs').mkdir(mode=0o700)
    injected=any(value is not None for value in (docker_command,identity,export_command))
    report={'schema_version':'harness-gpu-handoff/1','created_utc':utc(),'status':'running',
        'stage':'host_preflight','technical_execution_passed':False,'quality_qualified':False,
        'gpu_inference_verified':False,'documents':0,'qwen':args.qwen,
        'include_yolo_x':args.include_yolo_x,'models_are_parallel':False,'http_service_used':False,
        'inputs_copied':False,'stages':[],'reports_contain_credentials':False,
        'test_dependencies_injected':injected,'orchestration_contract_passed':False,
        'model_inference_performed':False,'model_inference_attempted':False,
        'measurement_pid_namespace':'host_only_for_temporary_smoke_and_comparison',
        'resource_sampling_scope':'owned_process_tree_only_no_arbitrary_host_process_export'}
    report_path=output/'handoff-status.json'
    docker=None
    def flush():
        report['updated_utc']=utc()
        write_json(report_path,report)
    def stage(name,command,timeout,**options):
        report['stage']=name
        item={'name':name,'status':'running','log':'logs/'+name+'.log'}
        report['stages'].append(item)
        flush()
        started=time.perf_counter()
        code=docker.stage(name,command,timeout=timeout,**options)
        item.update(status='pass' if code==0 else 'fail',returncode=code,
                    seconds=time.perf_counter()-started)
        flush()
        require(code==0, 'stage_failed:'+name)
    qwens=['qwen-0.8b','qwen-4b'] if args.qwen=='both' else [args.qwen]
    names=['xiaomi','detector','cyrillic',*qwens,'yolo26l-text']
    if args.include_yolo_x:names.append('yolo26x-text')
    try:
        flush()
        checked=documents_from_args(args)
        gold=gold_from_args(args.gold,checked)
        report['documents']=len(checked)
        report['inputs']=[{'id':d['id'],'source_sha256':d['source_sha256'],
                           'profile_id':d['profile_id']} for d in checked]
        report.update(gold_provided=gold is not None,reference_values_sent_to_workers=False)
        if gold:
            report.update(gold_sha256=gold['sha256'],gold_reference_documents=gold['documents'],
                          gold_reference_fields=gold['fields'])
        input_base=args.manifest.resolve().parent if args.manifest else args.input.resolve()
        write_json(output/'private-input-map.json',{'schema_version':'harness-private-input-map/1',
            'auto_ids':args.manifest is None,'order':'case_sensitive_filename_sort' if args.input else 'explicit_manifest',
            'documents':[{'id':d['id'],'relative_source_path':d['source'].relative_to(input_base).as_posix(),
                           'profile_id':d['profile_id'],'subset':d['subset'],
                           'source_sha256':d['source_sha256']} for d in checked]})
        require(isinstance(args.lock_volume,str) and VOLUME.fullmatch(args.lock_volume)
                and isinstance(args.service_project,str) and VOLUME.fullmatch(args.service_project),
                'invalid_lock_volume_or_service_project')
        require(isinstance(args.gpu_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,128}',args.gpu_id),
                'invalid_gpu_id')
        for value in (args.total_seconds,args.smoke_seconds,args.download_seconds,args.build_seconds):
            require(type(value) in (int,float) and math.isfinite(value) and 0 < value <= 86400,
                    'invalid_time_budget')
        if identity is None:
            require(platform.system()=='Linux' and hasattr(os,'getuid'), 'linux_host_required')
            identity=(os.getuid(),os.getgid())
        require(len(identity)==2 and all(type(i) is int and i>=0 for i in identity),
                'invalid_host_identity')
        report.update(container_uid=identity[0],container_gid=identity[1])
        if docker_command is None:
            executable=shutil.which('docker')
            require(bool(executable), 'docker_required')
            docker_command=(executable,)
        args.models=args.models.resolve()
        args.models.mkdir(parents=True,exist_ok=True)
        run_id=datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:10]
        state_base=state_dir or Path(os.getenv('XDG_STATE_HOME',str(Path.home()/'.local/state')))/'harness-ocr/handoffs'
        state=Path(state_base).resolve()/run_id
        state.mkdir(parents=True,mode=0o700)
        mounts=prepare_inputs(checked,state/'input')
        runtime_compose=state/'inputs.compose.json'
        write_json(runtime_compose, {'services':{'handoff':{'volumes':mounts},
                                               'measured':{'volumes':mounts}}})
        references_compose=prepare_references(gold,state)
        references_extra=[references_compose] if references_compose else []
        report['input_manifest_sha256']=hashlib.sha256((state/'input'/'own-documents.json').read_bytes()).hexdigest()
        env=os.environ.copy()
        # Offline delivery neither needs nor inherits service credentials.
        env.pop('HARNESS_API_TOKEN',None)
        env.pop('HARNESS_REVIEW_TOKEN',None)
        env.update(HARNESS_RUN_UID=str(identity[0]),HARNESS_RUN_GID=str(identity[1]),
            HARNESS_MODELS_HOST=str(args.models),HARNESS_REPORTS_HOST=str(output),
            HARNESS_INPUT_HOST=str(state/'input'),HARNESS_GPU_ID=args.gpu_id,
            HARNESS_LOCK_VOLUME=args.lock_volume)
        docker=Docker(docker_command,env,output,[COMPOSE,runtime_compose],'harness-gpu-'+run_id)
        docker.raw(['version','--format','{{.Server.Version}}'])
        docker.raw(['compose','version','--short'])
        check_old_service(docker,args.service_project,args.lock_volume,args.gpu_id)
        stage('build',[],args.build_seconds,build=True)
        built_tag=docker.environment.get('HARNESS_HANDOFF_IMAGE','harness-ocr:gpu-rc6-yolo')
        image_id=docker.raw(['image','inspect',built_tag,'--format','{{.Id}}']).strip()
        require(re.fullmatch(r'sha256:[0-9a-f]{64}',image_id), 'built_image_id_missing')
        # Later tags may move; all one-off stages use this exact built image.
        docker.environment['HARNESS_HANDOFF_IMAGE']=image_id
        report['image_id']=image_id
        flush()
        docker.raw(['volume','create',args.lock_volume])
        stage('lock-init',[],60,service='lock-init')
        common=['python','-m','experiments.gpu_delivery.run','--manifest','/input/own-documents.json',
                '--models','/models','--lock-dir','/locks','--qwen',args.qwen,
                '--total-seconds',str(args.total_seconds)]
        if args.include_yolo_x:common.append('--include-yolo-x')
        if gold:common+=['--gold','/references/gold.json']
        check_gold(gold)
        stage('plan',common+['--output','/reports/plan'],120,compose_extra=references_extra)
        if not args.skip_download:
            stage('prepare',['python','-m','harness_ocr.cli','prepare-models','--models','/models',
                             '--download','--only',*names],args.download_seconds,service='prepare')
        stage('verify-models',['python','-m','harness_ocr.cli','verify-models','--models','/models',
                               '--only',*names],1200)
        check_sources(checked)
        check_gold(gold)
        if args.prepare_only:
            report.update(status='prepared',stage='prepared',model_inference_performed=False)
            flush()
            return report
        stage('doctor',['python','/app/scripts/doctor.py','--offline','--models','/models',
                        '--output','/reports/doctor.json',*[part for name in names for part in ('--model',name)]],300)
        doctor=safe_json(output/'doctor.json')
        require(isinstance(doctor,dict) and doctor.get('schema_version')=='harness-doctor/1'
                and doctor.get('status')=='pass'
                and doctor.get('model_generation_performed') is False, 'doctor_did_not_pass')
        smoke=['python','/app/scripts/gpu_smoke.py','--models','/models','--lock-dir','/locks',
               '--output','/reports/smoke','--qwen',args.qwen,'--total-seconds',str(args.smoke_seconds)]
        if args.include_yolo_x:smoke.append('--include-yolo-x')
        report['model_inference_attempted']=not injected
        stage('smoke',smoke,args.smoke_seconds+120,service='measured')
        smoke_report=safe_json(output/'smoke'/'report.json')
        require(isinstance(smoke_report,dict)
                and smoke_report.get('schema_version')=='harness-gpu-functional-smoke/1'
                and smoke_report.get('status')=='pass'
                and smoke_report.get('quality_qualified') is False
                and smoke_report.get('actual_model_execution_verified') is True
                and smoke_report.get('target_runtime_accepted') is True
                and smoke_report.get('model_owners_stopped') is True
                and smoke_report.get('test_dependencies_injected') is False, 'gpu_smoke_did_not_pass')
        report['gpu_inference_verified']=not injected
        check_sources(checked)
        check_gold(gold)
        stage('comparison',common+['--output','/reports/comparison','--execute'],args.total_seconds+120,
              service='measured',compose_extra=references_extra)
        summary=safe_json(output/'comparison'/'summary.json')
        require(isinstance(summary,dict) and summary.get('schema_version')=='harness-gpu-delivery-summary/1'
                and summary.get('technical_execution_passed') is True
                and summary.get('quality_qualified') is False, 'comparison_not_technically_complete')
        report.update(status='completed',stage='completed',technical_execution_passed=not injected,
                      orchestration_contract_passed=True,
                      model_inference_performed=not injected)
        flush()
        return report
    except (Exception,KeyboardInterrupt) as exc:
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                      reason=str(exc) if isinstance(exc,HandoffError) else type(exc).__name__,
                      retry_performed=False)
        if docker and docker.last_cleanup:
            report['cleanup']=docker.last_cleanup
        flush()
        raise
    finally:
        # Export uses host stdlib even when Docker/build/preflight failed.
        command=list(export_command or (sys.executable,str(ROOT/'scripts'/'export_gpu_delivery.py')))
        try:
            exported=subprocess.run(command+['--run-dir',str(output),'--output',str(output/'RETURN_TO_TEAM.zip')],
                stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,check=False)
            report['export_status']='pass' if exported.returncode==0 else 'failed'
        except (OSError,subprocess.TimeoutExpired):
            report['export_status']='failed'
        flush()
        if report['export_status']!='pass' and sys.exc_info()[0] is None:
            raise HandoffError('public_report_export_failed')


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        result=run_handoff(args)
    except KeyboardInterrupt:
        print('Interrupted; owned run stopped, partial reports kept. Use a NEW output directory.',file=sys.stderr)
        return 130
    except Exception as exc:
        reason=str(exc) if isinstance(exc,HandoffError) else type(exc).__name__
        print('Harness OCR handoff failed: '+reason,file=sys.stderr)
        return 1
    print('Harness OCR: '+result['status']+'; reports: '+str(args.output.resolve()))
    print('Send RETURN_TO_TEAM.zip. Keep private inputs, rows, traces and logs local.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
