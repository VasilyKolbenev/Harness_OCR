"""Paired runner: freeze sources/protocol before compute; gold only in scorer."""
from pathlib import Path
from dataclasses import replace
import json
from .engine import Engine
from .inference.supervisor import Supervisor
from .models import verify_models,model_lock
from .util import digest,file_sha,write_json,utc

def score(result,gold):
    rows=[]
    for key,expected in gold.items():
        f=result['fields'].get(key,{});raw=f.get('raw_value')
        observed=f.get('extraction_status')=='observed' and raw is not None
        rows.append({'field_key':key,'correct_literal':observed and raw==expected,
          'wrong_literal':observed and raw!=expected,'withheld':not observed,
          'critical':f.get('critical',True),'validation_pass':f.get('validation_status')=='pass'})
    return {'fields':rows,'correct':sum(r['correct_literal'] for r in rows),
      'wrong':sum(r['wrong_literal'] for r in rows),'withheld':sum(r['withheld'] for r in rows),
      'total':len(rows),'all_fields_correct':bool(rows) and all(r['correct_literal'] for r in rows),
      'any_wrong_critical':any(r['wrong_literal'] and r['critical'] for r in rows)}

def run_benchmark(manifest_path,output,settings,*,include_controller=True,backend=None):
    manifest_path=Path(manifest_path).resolve()
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if manifest.get('schema_version')!='harness-bench/1' or manifest.get('split') not in {'dev','blind'}:
        raise ValueError('invalid_benchmark_manifest')
    documents=manifest.get('documents',[])
    if not documents or len({d['id'] for d in documents})!=len(documents):raise ValueError('empty_or_duplicate_documents')
    output=Path(output)
    if output.exists():raise ValueError('benchmark_output_must_be_new')
    sources=[]
    for d in documents:
        if not isinstance(d.get('gold'),dict) or not d['gold'] or any(not isinstance(v,str) for v in d['gold'].values()):
            raise ValueError('gold_must_be_nonempty_exact_raw_strings')
        p=(manifest_path.parent/d['path']).resolve()
        sources.append({'id':d['id'],'original_id':d['original_id'],'path':str(p),'sha256':file_sha(p)})
    verification=verify_models(settings.models_dir,['detector','cyrillic','xiaomi','qwen-0.8b','qwen-4b']) if backend is None else {'ok':True,'fixture':True}
    if not verification['ok']:raise ValueError('benchmark_models_not_verified:'+str(verification['errors']))
    output.mkdir(parents=True);arms=['A0','A1','A3']+(['A4'] if include_controller else [])
    protocol={'schema_version':'harness-bench-protocol/1','created_utc':utc(),
      'manifest_sha256':file_sha(manifest_path),'sources':sources,'split':manifest['split'],
      'planned_arms':arms,'arms':{'A0':'DB/CTC + Xiaomi + deterministic selector',
       'A1':'same frozen optics + Qwen0.8B text selector','A3':'same frozen optics + Qwen4B text selector',
       'A4':'same initial optics + Qwen4B + bounded Xiaomi ROI rereads'},
      'models':model_lock(),'device':settings.device,'dtype':settings.dtype,
      'max_input_tokens':settings.max_input_tokens,'max_new_tokens':settings.max_new_tokens,
      'max_rereads':settings.max_rereads,'document_timeout_seconds':settings.document_timeout_seconds,
      'metric':'exact raw literal; no normalization or reference correction',
      'group_unit':'original_id','quality_qualified':False,'fixture':backend is not None}
    protocol['protocol_sha256']=digest(protocol);write_json(output/'protocol.json',protocol)
    settings=replace(settings,data_dir=output/'state');owner=backend or Supervisor(settings)
    engine=Engine(settings,owner);rows=[]
    try:
        for index,(doc,source) in enumerate(zip(documents,sources)):
            frozen=None;frozen_hash=None
            for arm in arms:
                qwen='qwen-0.8b' if arm in {'A0','A1'} else 'qwen-4b'
                jid='b_'+digest([protocol['protocol_sha256'],index,arm])[:30]
                result=engine.run(jid,source['path'],source['sha256'],doc['profile_id'],subset=doc.get('subset'),
                    qwen=qwen,mode='rules' if arm=='A0' else 'hybrid',shared_optics=frozen,allow_reread=arm=='A4',
                    emit_operational_events=False)
                if arm=='A0':
                    frozen=result.pop('optics',None)
                    if frozen is None:
                        write_json(output/f'failed-{index}-{arm}.json',result)
                        raise RuntimeError('initial_optics_failed_preserved_partial_results')
                    frozen_hash=digest(frozen);write_json(output/f'optics-{index}.json',frozen)
                else:result.pop('optics',None)
                row={'id':doc['id'],'original_id':doc['original_id'],'arm':arm,
                   'initial_optics_sha256':frozen_hash,'result':result,'score':score(result,doc['gold'])}
                rows.append(row);write_json(output/f'result-{index}-{arm}.json',row)
    finally:owner.close()
    metrics={}
    for arm in arms:
        subset=[r for r in rows if r['arm']==arm];groups={}
        for row in subset:groups.setdefault(row['original_id'],[]).append(row)
        metrics[arm]={'documents':len(subset),'original_groups':len(groups),
          'fields':sum(r['score']['total'] for r in subset),
          **{k:sum(r['score'][k] for r in subset) for k in ['correct','wrong','withheld','all_fields_correct']},
          'macro_original_field_accuracy':sum(sum(r['score']['correct']/max(1,r['score']['total']) for r in g)/len(g) for g in groups.values())/max(1,len(groups)),
          'documents_with_wrong_critical':sum(r['score']['any_wrong_critical'] for r in subset),
          'all_accept_for_autofill':sum(r['result']['decision']=='accept_for_autofill' for r in subset),
          'seconds_recorded':sum(r['result']['seconds'] for r in subset),
          'latency_note':'A0 includes shared optical cost; A1/A3 time is incremental selector cost',
          'max_peak_cuda_allocated_bytes':max([c.get('peak_cuda_allocated_bytes') or 0 for r in subset for c in r['result']['calls']] or [0])}
    matched_optics_proofs=[]
    for index,(doc,source) in enumerate(zip(documents,sources)):
        paired=[r for r in rows if r['id']==doc['id']]
        hashes={r['arm']:r['initial_optics_sha256'] for r in paired}
        saved=json.loads((output/f'optics-{index}.json').read_text(encoding='utf-8'))
        saved_sha=digest(saved)
        matched_optics_proofs.append({'document_id':doc['id'],'original_id':doc['original_id'],
          'source_sha256':source['sha256'],'initial_optics_sha256':hashes,
          'saved_optics_sha256':saved_sha,
          'match':set(hashes)==set(arms) and set(hashes.values())=={saved_sha}})
    summary={'schema_version':'harness-bench-summary/1','protocol':protocol,'arms':arms,
      'paired_originals':len({d['original_id'] for d in documents}),'metrics':metrics,
      'completed_rows':len(rows),'planned_rows':len(documents)*len(arms),
      'all_generations_completed':all(r['result']['completed'] for r in rows),
      'matched_optics_proofs':matched_optics_proofs,
      'resource_measurement_scope':{
        'peak_cuda_allocated':'Maximum recorded PyTorch allocated bytes per model call; not total GPU process memory or reserved bytes',
        'peak_cuda_reserved_bytes':None,'host_rss_bytes':None,
        'latency':'A0 includes initial optical pipeline; A1/A3 are incremental selectors, A4 incremental selector/rereads. Do not compare these columns as standalone end-to-end latencies.',
        'capacity_qualified':False},
      'quality_qualified':False,'selection_decision':'No automatic winner; independent quality/resource gates required'}
    write_json(output/'summary.json',summary);return summary
