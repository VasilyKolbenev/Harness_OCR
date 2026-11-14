from __future__ import annotations
from pathlib import Path
import time
import json
import os
import platform
from importlib import metadata
from dataclasses import asdict
from . import __version__
from .core import DocumentStore
from .core.trace import TraceRecorder
from .profiles import ProfileRegistry,project_contract_result,assess_completion
from .control.audit import AuditController,AuditLog,AuditSampler,DriftMonitor
from .control.strategies import StrategyStore
from .control.risk import RiskPolicyStore
from .control.runtime import BindingPolicyGateway,unsupported_strategy_actions
from .grounding import build_candidates,qwen_prompt,resolve_response,validate_literal,CANDIDATE_GENERATOR_VERSION as CANDIDATE_VERSION
from .util import digest,utc,file_sha
from .inference.supervisor import ModelCanceled,ModelDeadline
from .typed_fields import value_identity

class Engine:
    def __init__(self,settings,backend):
        settings.validate();self.settings=settings;self.backend=backend
        self.root=settings.data_dir;self.root.mkdir(parents=True,exist_ok=True)
        self.documents=DocumentStore(self.root/'documents.sqlite')
        self.trace=TraceRecorder(self.root/'trace.sqlite');self.profiles=ProfileRegistry()
        self.control_db=self.root/'control.sqlite'
        self.strategies=StrategyStore(self.control_db);self.policies=RiskPolicyStore(self.control_db)
        self.gateway=BindingPolicyGateway.from_file(self.root/'runtime_policy.json',self.policies)
        self.audit_log=AuditLog(self.control_db)
        self.audit=AuditController(AuditSampler(.05,self.audit_log.server_sampling_seed('harness-audit-v1')),self.audit_log)
        package_root=Path(__file__).parent
        self.code_sha256=digest({p.relative_to(package_root).as_posix():file_sha(p)
            for p in sorted(package_root.rglob('*.py'))})
        self.dependency_versions={}
        for name in ('torch','torchvision','transformers','tokenizers','onnxruntime','numpy','Pillow','pypdfium2','defusedxml','lxml','lm-format-enforcer','ultralytics','ultralytics-opencv-headless'):
            try:self.dependency_versions[name]=metadata.version(name)
            except metadata.PackageNotFoundError:self.dependency_versions[name]=None
    def active_cards(self,contract):
        return self.strategies.retrieve(contract['profile_id'],contract['profile_version'],
            field_keys=contract['requested_fields'],actor_scopes={'strategy:read'})
    def compile_contract(self,profile_id,*,subset=None):
        from .profiles.applicability import TrustedApplicabilityRegistry
        return TrustedApplicabilityRegistry.from_file(self.root/'applicability.json').compile(
            self.profiles,profile_id,subset=subset,input_scope_version=self.settings.input_scope_version)
    def binding(self,contract,qwen,mode='hybrid',allow_reread=True,strategy_cards=None):
        from .models import model_lock
        if mode not in {'hybrid','rules'} or type(allow_reread) is not bool:raise ValueError('invalid_bound_runtime_mode')
        cards=self.active_cards(contract) if strategy_cards is None else strategy_cards
        limits={name:getattr(self.settings,name) for name in ('max_upload_bytes','max_pages','max_page_pixels',
            'max_input_tokens','max_new_tokens','call_timeout_seconds','document_timeout_seconds','max_rereads')}
        text_detection={'selector':self.settings.text_detector,'max_lines':200,
            'recognizer':'cyrillic-ctc','recognizer_device':'cpu','word_geometry_is_not_field_role':True}
        if self.settings.text_detector=='db':
            from .inference.db_geometry import DetectorConfig
            text_detection.update(device='cpu',config=asdict(DetectorConfig()))
        else:
            from .inference.yolo_detector import YoloDetectorConfig
            text_detection.update(device=self.settings.device,config=YoloDetectorConfig().binding(),
                model_key=self.settings.text_detector+'-text',
                grouping='deterministic_word_to_line_preserve_source_geometry')
        runtime={'package_version':__version__,'package_code_sha256':self.code_sha256,
            'backend_class':type(self.backend).__module__+'.'+type(self.backend).__qualname__,
            'fixture_permitted':self.settings.allow_fixture,
            'python':platform.python_version(),'machine':platform.machine(),'dependencies':self.dependency_versions,
            'limits':limits,'mode':mode,'allow_reread':allow_reread,'max_reread_rounds':2,
            'text_detection':text_detection,
            'visual_verifier':{'enabled':self.settings.visual_verifier,'max_regions':self.settings.max_visual_regions,
                'protocol':'blind-visual-capture-v1','comparison':'complete_original_observation_literal',
                'max_pixels':589824,'max_side':1536,'processor_min_pixels':1024,
                'processor_alignment':32,'processor_may_resize_to_minimum':True},
            'sampler':{'do_sample':False,'use_cache':True,'thinking':False,'logits_to_keep':1,
                       'output_format':'server-generated-json-schema','constraint_engine':'lm-format-enforcer'},
            'attention':self.settings.attention_backend,'cpu_threads':{'ctc_intraop':self.settings.cpu_threads,'ctc_interop':1,
                'hf_intraop':self.settings.cpu_threads,'hf_interop':1,
                'environment':{k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}},
            'image_preprocess':{'max_pixels':589824,'max_side':1536,'upscale':False},
            'ocr_prompt':'Extract the text in the image.','candidate_generator':CANDIDATE_VERSION,
            'strategy_bundle_sha256':digest(cards)}
        return {'profile_id':contract['profile_id'],'profile_version':contract['profile_version'],
          'subset_contract_id':contract['subset_contract_id'],
          'model_bundle_sha256':digest({'lock':model_lock(),'qwen':qwen,'dtype':self.settings.dtype,
             'device':self.settings.device,'pipeline':'harness-v1','contract':contract['config_sha256'],
             'runtime':runtime}),
          'policy_version':self.settings.policy_version,'input_scope_version':self.settings.input_scope_version}
    def run(self,document_id,path,source_sha,profile_id,*,subset=None,qwen=None,mode='hybrid',
            cancel=None,shared_optics=None,allow_reread=True,emit_operational_events=True):
        if mode not in {'hybrid','rules'}:raise ValueError('unknown_mode')
        qwen=qwen or self.settings.qwen
        if qwen not in {'qwen-0.8b','qwen-4b'}:raise ValueError('unknown_qwen')
        contract=self.compile_contract(profile_id,subset=subset)
        cards=self.active_cards(contract)
        binding=self.binding(contract,qwen,mode,allow_reread,strategy_cards=cards)
        start=time.perf_counter();stage='created';errors=[];calls=[]
        snapshot=self.documents.create_document(document_id,source_sha,contract['profile_id'],
             metadata={'binding':binding,'subset':contract['requested_fields'],'source_path':str(Path(path).resolve())})
        observations=[];regions=[];optics_completed=True;selector_complete=True;optics=None
        strategy_reasons=[];strategy_info={'active_cards':[],'unsupported_actions':[]}
        unresolved_rereads=[]
        visual_checks=[];visual_regions={};visual_regions_by_id={};visual_attempted=False;native_ingestion=None
        def record(name,inputs,outputs,kind='pure',status='completed'):
            self.trace.record(name,inputs,outputs,{'pipeline_version':'1','contract':contract['config_sha256']},
               document_id=document_id,stage_kind=kind,deterministic=kind=='pure',status=status)
        def checkpoint():
            if cancel and cancel():raise ModelCanceled('canceled')
            if time.perf_counter()-start>=self.settings.document_timeout_seconds:raise ModelDeadline('document_deadline')
        def call(op,args):
            checkpoint();left=self.settings.document_timeout_seconds-(time.perf_counter()-start)
            kind='external' if op=='ingest' else 'model'
            wall_start=time.perf_counter()
            try:out=self.backend.call(op,args,cancel=cancel,timeout=min(left,self.settings.call_timeout_seconds))
            except Exception as e:
                record(op,args,{'error':type(e).__name__+':'+str(e)[:250]},kind,'failed');raise
            calls.append({'op':op,'completed':out.get('completed',True),
                          'seconds':out.get('seconds',out.get('diagnostics',{}).get('seconds')),
                          'wall_seconds':time.perf_counter()-wall_start,
                          'timing_scope':'wall includes process/load/preprocess; HF seconds is generate only',
                          'generated_tokens':out.get('generated_tokens'),'peak_cuda_allocated_bytes':out.get('peak_cuda_allocated_bytes'),
                          'peak_cuda_reserved_bytes':out.get('peak_cuda_reserved_bytes'),
                          'cuda_peak_scope':out.get('cuda_peak_scope')})
            record(op,args,out,kind,'completed' if out.get('completed',True) else 'incomplete');return out
        terminal='completed'
        try:
            checkpoint();stage='optics'
            if shared_optics is not None:
                if shared_optics['source_sha256']!=source_sha:raise ValueError('replay_source_mismatch')
                observations=json.loads(json.dumps(shared_optics['observations']));regions=shared_optics['regions']
                optics_completed=shared_optics['completed'];record('frozen_optics',{'source_sha256':source_sha},shared_optics)
            else:
                ingestion=call('ingest',{'path':str(path),'output':str(self.root/'rasters'/document_id),
                    'profile_id':contract['profile_id']})
                if ingestion['source_sha256']!=source_sha:raise ValueError('input_source_sha_mismatch')
                if ingestion.get('native_validation') is not None:
                    from .native.mchd import verify_native_result
                    if contract['profile_id']!='power_of_attorney_evidence_v1' or not verify_native_result(ingestion,Path(path)):
                        raise ValueError('native_adapter_source_replay_failed')
                if ingestion.get('native_validation') is not None:
                    native_ingestion={k:v for k,v in ingestion.items() if k not in {'observations','field_candidates','native_claims','pages'}}
                    strategy_reasons.extend('native_'+r for r in ingestion.get('review_reasons',[]))
                    strategy_reasons.append('native_signature_revocation_authority_not_verified')
                record('ingest',{'source_sha256':source_sha},ingestion,'external')
                observations.extend(ingestion['observations'])
                if observations:
                    snapshot=self.documents.revise(document_id,observations=observations,
                        native_result=ingestion if ingestion.get('native_validation') else None,
                        expected_revision=snapshot['revision'],reason='native_observations')
                if ingestion.get('native_claims'):
                    if not ingestion.get('native_validation'):raise ValueError('unvalidated_native_claims')
                    native=[c for c in ingestion['native_claims'] if c['field_key'] in contract['requested_fields']]
                    if native:snapshot=self.documents.revise(document_id,claims=native,expected_revision=snapshot['revision'],reason='replayed_native_claims')
                if ingestion.get('native_bridge')=='generic_xml_unqualified':
                    # A generic XML parser is not a versioned service adapter.
                    # Its text must not inherit a raster profile's calibration.
                    strategy_reasons.append('unsupported_native_adapter:generic_xml')
                    optics_completed=False
                    raise ValueError('native_adapter_not_registered_for_profile')
                for page in ingestion['pages']:
                    checkpoint()
                    cheap=call('cheap',{'page':page,'source_sha256':source_sha,
                                  'output_dir':str(self.root/'regions'/document_id/page['page_id'])})
                    observations.extend(cheap['observations']);regions.extend(cheap['regions'])
                    if self.settings.visual_verifier:
                        from .inference.vision import make_visual_region
                        for region in cheap['regions']:
                            # A detected but unreadable region still has real
                            # geometry. Register it before any successful text
                            # observation so a later reread can be verified.
                            descriptor=make_visual_region(path,page,region,source_sha256=source_sha)
                            visual_regions_by_id[region['id']]=descriptor
                            for obs in cheap['observations']:
                                if obs.get('page_id')==region['page_id'] and obs.get('quad')==region.get('quad'):
                                    visual_regions[obs['id']]=descriptor
                    snapshot=self.documents.revise(document_id,observations=cheap['observations'],expected_revision=snapshot['revision'],reason='ctc_observations')
                    optics_completed=optics_completed and cheap.get('completed',True)
                    read=call('ocr',{'path':page['path']})
                    observations.append({'id':page['page_id']+'_xiaomi','source_sha256':source_sha,
                        'text':read['text'],'evidence_kind':'image_region','page':page['page'],'page_id':page['page_id'],
                        'source_size':page['source_size'],'bbox':[0,0,*page['source_size']],
                        'completed':read['completed'],'capture_id':document_id+'_xiaomi_'+page['page_id'],
                        'model_id':'xiaomi','raster_transform_proof':page['transform']})
                    if self.settings.visual_verifier:
                        # Page-wide Xiaomi claims have page-wide evidence. They
                        # must be compared against the full page, never a guessed
                        # character crop or a substring of a line reading.
                        visual_regions[observations[-1]['id']]=make_visual_region(path,page,source_sha256=source_sha)
                    optics_completed=optics_completed and read['completed']
                    snapshot=self.documents.revise(document_id,observations=[observations[-1]],expected_revision=snapshot['revision'],reason='xiaomi_observation')
            if shared_optics is not None:
                visual_regions=shared_optics.get('visual_regions',{})
                visual_regions_by_id=shared_optics.get('visual_regions_by_id',
                    {d['id']:d for d in visual_regions.values()})
            optics={'source_sha256':source_sha,'observations':observations,'regions':regions,
                'visual_regions':visual_regions,'visual_regions_by_id':visual_regions_by_id,'completed':optics_completed}
            if shared_optics is not None:
                snapshot=self.documents.revise(document_id,observations=observations,expected_revision=snapshot['revision'],reason='frozen_optical_observations')
            record('strategy_retrieval',{'profile_id':contract['profile_id']},cards,'external')
            supported={'reread','require_role_label','review_on_conflict','preserve_source_punctuation'}
            unsupported=unsupported_strategy_actions(cards,supported)
            strategy_info={'active_cards':[{'card_id':c['card_id'],'version':c['version'],'card_sha256':c['card_sha256']} for c in cards],
                'unsupported_actions':unsupported,'implemented_actions':sorted(supported)}
            strategy_reasons.extend('unsupported_strategy_action:'+a for a in unsupported)
            # Strategy memory can reduce work, never increase hard limits or fill values.
            reread_limit=self.settings.max_rereads if allow_reread else 0
            max_rounds=2
            for card in cards:
                for directive in card.get('directives',[]):
                    if directive['action']=='reread':
                        reread_limit=min(reread_limit,directive['parameters']['max_regions'])
                        max_rounds=min(max_rounds,directive['parameters']['max_rounds'])
                        if directive['parameters']['scale_policy']!='native':
                            strategy_reasons.append('unsupported_strategy_parameter:reread:scale_policy')
            errors.extend(strategy_reasons)
            used=set();round_no=0
            while True:
                checkpoint();stage='candidates';candidates=build_candidates(observations,contract)
                record('candidates',{'observations':observations,'contract':contract},candidates)
                if native_ingestion is not None:break
                # Conflicting anchored observations must reach F01 even if the
                # selector chooses only its preferred reading. Selection is not
                # authority to erase an independently observed contradiction.
                conflicts=self._candidate_conflict_claims(candidates,snapshot)
                if conflicts:
                    snapshot=self.documents.revise(document_id,claims=conflicts,
                        expected_revision=snapshot['revision'],reason='conflicting_literal_observations')
                if mode=='rules':
                    proposals=[]
                    for field in contract['requested_fields']:
                        group=[c for c in candidates.values() if c['field_key']==field and c['role_verified']]
                        values={digest(value_identity(c['raw_value'])) for c in group}
                        if len(values)==1:
                            c=group[0];proposals.append({'field_key':field,'candidate_ids':[c['id']],'party_role':c['party_role']})
                    response=json.dumps({'proposals':proposals,'reread_requests':[]})
                else:
                    stage='selector';out=call('select',{'name':qwen,'prompt':qwen_prompt(candidates,regions,contract,
                        observations=observations,strategies=cards,remaining_budget={
                            'reread_regions':max(0,reread_limit-len(used)),
                            'reread_rounds':max(0,max_rounds-round_no),
                            'max_new_tokens':self.settings.max_new_tokens})})
                    selector_complete=out['completed']
                    if not selector_complete:errors.append('selector_token_limit');break
                    response=out['text']
                resolved=resolve_response(response,candidates,regions,contract)
                record('resolve',{'response':response,'candidates':candidates,'regions':regions,'contract':contract},resolved)
                new=[c for c in resolved['claims'] if c['id'] not in snapshot['claims']]
                if new:snapshot=self.documents.revise(document_id,claims=new,expected_revision=snapshot['revision'],reason='source_grounded_proposals')
                for rejected in resolved['rejected']:errors.append(rejected['reason']+':'+rejected['field_key'])
                pending=[]
                for request in resolved['reread_requests']:
                    pending.extend(x for x in request['region_ids'] if x not in used and x not in pending)
                if resolved['reread_requests'] and (not pending or len(used)>=reread_limit or round_no>=max_rounds):
                    reason=('reread_disabled' if not allow_reread else
                            'reread_no_new_region' if not pending else 'reread_budget_exhausted')
                    unresolved_rereads=[{**r,'unresolved_reason':reason} for r in resolved['reread_requests']]
                    strategy_reasons.append('unresolved_reread_requests')
                    break
                if not pending:break
                by_id={r['id']:r for r in regions};add=[]
                for rid in pending[:reread_limit-len(used)]:
                    region=by_id[rid];used.add(rid);stage='reread'
                    read=call('ocr',{'path':region['path']})
                    add.append({'id':rid+'_reread','source_sha256':source_sha,'text':read['text'],
                         'evidence_kind':'image_region','page':region['page'],'page_id':region['page_id'],
                         'source_size':region['source_size'],'quad':region['quad'],
                         'completed':read['completed'],'capture_id':document_id+'_reread_'+rid,
                         'model_id':'xiaomi','raster_transform_proof':region.get('transform',{})})
                    optics_completed=optics_completed and read['completed']
                snapshot=self.documents.revise(document_id,observations=add,expected_revision=snapshot['revision'],reason='bounded_reread')
                for obs in add:
                    rid=obs['id'].removesuffix('_reread')
                    match=visual_regions_by_id.get(rid)
                    if match:visual_regions[obs['id']]=match
                observations.extend(add);round_no+=1
            if self.settings.visual_verifier and not visual_attempted:
                visual_attempted=True;stage='visual_verifier'
                requests=[];approved_regions={}
                for field in contract['requested_fields']:
                    current=snapshot['fields'].get(field,{})
                    if current.get('raw_value') is None:continue
                    claims=[snapshot['claims'][cid] for cid in current.get('claim_ids',[])
                        if snapshot['claims'][cid]['status']=='supported']
                    # XML has structural proofs, no invented image. Vision is only raster.
                    if claims and all(snapshot['observations'][oid]['evidence_kind']=='structured_xml'
                        for claim in claims for oid in claim['evidence_ids']):continue
                    ids=sorted({oid for claim in claims for oid in claim['evidence_ids']})
                    for oid in ids:
                        if oid not in visual_regions:
                            strategy_reasons.append('visual_verification_unavailable_or_budget:'+field);continue
                        region_key=digest(visual_regions[oid])
                        if region_key not in approved_regions:
                            if len(approved_regions)>=self.settings.max_visual_regions:
                                strategy_reasons.append('visual_verification_unavailable_or_budget:'+field);continue
                            approved_regions[region_key]=visual_regions[oid]
                        requests.append((field,oid,[c['id'] for c in claims if oid in c['evidence_ids']],region_key))
                captures={}
                for field,oid,claim_ids,region_key in requests:
                    if region_key not in captures:
                        out=call('visual_read',{'name':qwen,'regions':[approved_regions[region_key]]})
                        if (out.get('completed') is not True or out.get('stop_reason')!='eos'
                            or out.get('modality')!='vision' or out.get('purpose')!='blind_optical_capture'
                            or not isinstance(out.get('captures'),list) or len(out['captures'])!=1
                            or out['captures'][0].get('provenance',{}).get('region_id')!=visual_regions[oid]['id']):
                            raise ValueError('visual_generation_incomplete_or_provenance_invalid')
                        captures[region_key]=out['captures'][0]
                        snapshot=self.documents.revise(document_id,observations=[captures[region_key]],
                            expected_revision=snapshot['revision'],reason='blind_visual_capture')
                        observations.append(captures[region_key])
                    check={'id':'visual_check_'+digest([field,oid,captures[region_key]['id'],claim_ids])[:24],
                        'field_key':field,'claim_ids':claim_ids,'capture_id':captures[region_key]['id'],'expected_observation_id':oid}
                    snapshot=self.documents.revise(document_id,visual_checks=[check],
                        expected_revision=snapshot['revision'],reason='blind_visual_literal_comparison')
                    visual_checks.append(snapshot['visual_checks'][check['id']])
        except ModelCanceled as e:terminal='canceled';errors.append(str(e))
        except ModelDeadline as e:terminal='budget_exhausted';errors.append(str(e))
        except Exception as e:terminal='failed';errors.append(type(e).__name__+':'+str(e)[:250])
        if optics is not None:
            optics['observations']=observations
            optics['completed']=optics_completed
        snapshot=self.documents.get(document_id)
        result=self.project(snapshot,contract,binding,terminal=terminal,
              complete=terminal=='completed' and optics_completed and selector_complete,
              review_reasons=strategy_reasons,emit_operational_events=emit_operational_events)
        result.update(errors=errors,stopped_at=stage,calls=calls,seconds=time.perf_counter()-start,
                      mode=mode,qwen=qwen,finished_utc=utc(),optics=optics,strategy_runtime=strategy_info,
                      unresolved_rereads=unresolved_rereads)
        result['native_ingestion']=native_ingestion
        result['visual_verification']={'enabled':self.settings.visual_verifier,'checks':visual_checks,
            'max_regions':self.settings.max_visual_regions,'automatic_acceptance_authority':False}
        record('result',{'revision':snapshot['revision']}, {k:v for k,v in result.items() if k!='optics'})
        return result
    @staticmethod
    def _candidate_conflict_claims(candidates,snapshot):
        groups={}
        for candidate in candidates.values():
            if candidate.get('role_verified'):
                groups.setdefault(candidate['field_key'],[]).append(candidate)
        claims=[]
        for field,items in groups.items():
            from .machine_ids import derive_machine_context
            contexts=[derive_machine_context(c['raw_value'],list(snapshot['observations'].values()))
                for c in items if isinstance(c['raw_value'],dict)
                and c['raw_value'].get('contract_id')=='machine-identifiers-v1']
            if (len({digest(value_identity(c['raw_value'])) for c in items})<2
                and len({digest(c) for c in contexts if c is not None})<2):continue
            for c in items:
                ident='observed_conflict_'+digest([field,c['id']])[:24]
                if ident in snapshot['claims']:continue
                claims.append({'id':ident,'field_key':field,'raw_value':c['raw_value'],
                    'evidence_ids':c['evidence_ids'],'spans':c['spans'],
                    'role_evidence_ids':c['role_evidence_ids'],'role_spans':c['role_spans'],
                    'party_role':c['party_role'],'origin':'rule','depends_on':[]})
        return claims
    def project(self,snapshot,contract,binding,*,terminal='completed',complete=True,review_reasons=(),
                emit_operational_events=True):
        native=snapshot.get('native_ingestion')
        if native:
            review_reasons=[*review_reasons,'native_signature_revocation_authority_not_verified',
                *['native_'+r for r in native.get('review_reasons',[])]]
        fields={}
        for key,spec in contract['field_definitions'].items():
            current=snapshot['fields'].get(key,{});raw=current.get('raw_value')
            observed=raw is not None and not current.get('needs_review')
            inactive=key in contract.get('not_applicable_fields',[])
            value={**current,'value':raw,'raw_value':raw,'critical':spec['critical'],
                'party_role':current.get('party_role'),
                'evidence_ids':current.get('evidence_ids',[]),
                'extraction_status':'not_applicable' if inactive else ('observed' if observed else ('ambiguous' if current else 'not_observed'))}
            value.update(self._validate_supports(snapshot,current,raw,spec) if observed else {'canonical_value':None,
                   'transform_ids':[],'validation_status':'not_checked','validation_reasons':['no_unambiguous_literal']})
            fields[key]=value
            if self.settings.visual_verifier and observed:
                live_claims=[snapshot['claims'][cid] for cid in current.get('claim_ids',[])
                    if snapshot['claims'][cid]['status'] in {'supported','human_supported'}]
                raster_ids={oid for claim in live_claims for oid in claim['evidence_ids']
                    if snapshot['observations'][oid]['evidence_kind']=='image_region'}
                verified_ids={check['expected_observation_id'] for check in snapshot.get('visual_checks',{}).values()
                    if check['field_key']==key and check['verdict']=='correct'
                    and any(cid in current.get('claim_ids',[]) for cid in check['claim_ids'])
                    and all(snapshot['observations'].get(oid,{}).get('active') is True
                        and snapshot['observations'][oid].get('completed') is True
                        for oid in (check['capture_id'],check['expected_observation_id']))}
                if raster_ids:value['visual_verification_complete']=raster_ids<=verified_ids
                if raster_ids and not value['visual_verification_complete']:
                    review_reasons=[*review_reasons,'visual_verification_incomplete:'+key]
            if inactive and raw is not None:
                review_reasons=[*review_reasons,'not_applicable_field_observed:'+key]
        result={'document_id':snapshot['document_id'],'source_sha256':snapshot['source_sha256'],
          **binding,'revision':snapshot['revision'],'fields':fields,'observations':snapshot['observations'],
          'conflicts':snapshot['conflicts'],'required_fields':contract['required_fields'],
          'visual_checks':snapshot.get('visual_checks',{}),
          'native_ingestion':native,
          'triggered_conditional_fields':[k for k,v in contract['conditionals'].items() if v['applies'] is True],
          'unresolved_conditions':contract['unknown_conditionals'],
          'conditional_applicability':{k:v['applies'] for k,v in contract['conditionals'].items()},
          'terminal_status':terminal,'completed':complete,'profile_qualified':False,
          'risk_score':None,'score_contract_id':'unqualified-v1','service_approval':False}
        result['completion_report']=assess_completion(contract,fields)
        required=[*contract['required_fields'],*result['triggered_conditional_fields']]
        result['all_required_fields_validated']=bool(required) and not contract['unknown_conditionals'] and all(
            fields[k]['extraction_status']=='observed' and fields[k]['validation_status']=='pass'
            and not fields[k].get('needs_review') for k in required)
        # Qualification/configuration and ranking features come from the trusted
        # server gateway; callers cannot enable acceptance with flags or scores.
        result.update(self.gateway.apply(result,binding))
        if any(f.get('origin')=='human' for f in fields.values()):
            # Machine-qualified risk is not qualification of an operator's new
            # literal. Preserve the correction; the separate human acceptance
            # contract is not silently inherited from the machine policy.
            review_reasons=[*review_reasons,'human_correction_requires_separate_acceptance']
            result['operator_corrected']=True
        if review_reasons:
            result['decision']='review'
            result['reason_codes']=sorted(set(result['reason_codes']+list(review_reasons)))
        if emit_operational_events and result.get('profile_qualified') and result.get('policy_id'):
            stored=self.policies.get(result['policy_id'])['policy']
            baseline=stored['qualification']['document_coverage']
            monitor=DriftMonitor(self.audit_log,binding,baseline_acceptance_rate=baseline)
            drift=monitor.observe(snapshot['document_id'],binding,accepted=result['decision']=='accept_for_autofill',
                event_id='engine_drift_'+digest([snapshot['document_id'],snapshot['revision'],binding,result['decision'],monitor.monitor_sha256]))
            result['drift']=drift
            if drift['quarantined']:
                result['decision']='review'
                result['reason_codes']=sorted(set(result['reason_codes']+['binding_quarantined']))
        for field in fields.values():
            field['decision']='accept_for_autofill' if (result['decision']=='accept_for_autofill'
                and field['extraction_status']=='observed' and field['validation_status']=='pass'
                and not field.get('needs_review')) else 'review'
        result['needs_review']=result['decision']=='review'
        result['audit']=self.audit.plan(snapshot['document_id'],snapshot['revision'],binding,
                     group_id=snapshot['source_sha256'],accepted=result['decision']=='accept_for_autofill',
                     acceptance_context=digest({'policy_id':result.get('policy_id'),'decision':result['decision']})) if emit_operational_events else {
                         'deferred_until_publication':True,'selected':False}
        result['contract_result']=project_contract_result(contract,result)
        return result

    @staticmethod
    def _validate_supports(snapshot,current,raw,spec):
        observations=list(snapshot['observations'].values())
        result=validate_literal(raw,spec,observations=observations)
        if not isinstance(raw,dict):return result
        # Equal literal values can have different manufacturer/format context.
        # Validate every live provenance claim; selecting the favorable reading
        # must not erase an unsupported or contradictory context on another one.
        seen={digest(raw)};checks=list(result.get('validation_checks',[]))
        for cid in current.get('claim_ids',[]):
            claim=snapshot['claims'].get(cid,{})
            if claim.get('status') not in {'supported','human_supported'}:continue
            signature=digest(claim.get('raw_value'))
            if signature in seen:continue
            seen.add(signature)
            checked=validate_literal(claim['raw_value'],spec,observations=observations)
            if checked['validation_status']=='pass':continue
            if result['validation_status']!='fail':result['validation_status']=checked['validation_status']
            result['validation_reasons']=sorted(set(result['validation_reasons']+
                ['supporting_typed_claim_not_validated:'+cid]+checked['validation_reasons']))
            checks.append({'check_id':'typed_support:'+cid,'status':checked['validation_status'],
                'scope':'all_live_typed_claims','reason_codes':checked['validation_reasons'],
                'source_ids':sorted({s for c in checked.get('validation_checks',[]) for s in c.get('source_ids',[])}),
                'evidence_ids':claim['evidence_ids']})
        if len(checks)>256:
            if result['validation_status']=='pass':result['validation_status']='not_checked'
            result['validation_reasons']=sorted(set(result['validation_reasons']+['validation_check_budget_exhausted']))
            checks=[{'check_id':'typed_support_budget','status':'not_checked','scope':'bounded_validation_check_projection',
                'reason_codes':['validation_check_budget_exhausted'],'source_ids':[]}]
        result['validation_checks']=checks
        return result
