"""Metadata-only planning, pinned bytes, and post-inference exact scoring."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
QWENS=('qwen-0.8b','qwen-4b')
COMMON={'device':'cuda','dtype':'bfloat16','attention_backend':'sdpa','cpu_threads':4,
    'max_pages':10,'max_page_pixels':12_000_000,'max_upload_bytes':30*1024*1024,
    'max_input_tokens':32768,'max_new_tokens':1024,'call_timeout_seconds':300,
    'document_timeout_seconds':900,'max_rereads':2,'max_detector_rescues':1,
    'max_whole_page_reads':1,'visual_verifier':True,'max_visual_regions':2,
    'reading_policy':'adaptive','text_detector':'db','allow_fixture':False}


class BudgetExpired(TimeoutError):pass


def utc():return datetime.now(timezone.utc)
def timestamp(value):
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds()!=0:raise ValueError('UTC_deadline_required')
    return parsed.astimezone(timezone.utc)
def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
def check_deadline(deadline):
    if deadline is not None and utc()>=deadline:raise BudgetExpired('global_budget_exhausted')
def file_sha(path,deadline=None):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            check_deadline(deadline);block=stream.read(4*1024*1024)
            if not block:break
            value.update(block)
    return value.hexdigest()
def _unique(pairs):
    value={}
    for key,item in pairs:
        if key in value:raise ValueError('duplicate_json_key')
        value[key]=item
    return value
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_unique)
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def require_budget(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 1<=value<=86400:
        raise ValueError('total_seconds_must_be_1_to_86400')


def load_documents(path):
    """Reuse the existing strict own-documents validator, including native files.

    Inline gold/unknown keys are rejected by that schema before any process or
    model verification. Its HTTP functions are imported, never executed here.
    """
    # Reject duplicate JSON keys before delegating all existing path/profile
    # rules. The reusable parser's public accepted schema remains unchanged.
    path=Path(path)
    if not path.is_file() or path.stat().st_size>1024*1024:raise ValueError('manifest_size_or_path')
    read(path)
    scripts=ROOT/'scripts';sys.path.insert(0,str(scripts))
    for name in ('acceptance','doctor'):
        existing=sys.modules.get(name)
        if existing and Path(existing.__file__).resolve()!=scripts/(name+'.py'):
            raise RuntimeError('metadata_parser_import_source_mismatch')
    spec=importlib.util.spec_from_file_location('_harness_gpu_own_documents',scripts/'run_documents.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw,checked=module.load_manifest(path)
    documents=[]
    for doc in checked:
        contract=doc['contract']
        documents.append({'id':doc['id'],'path':str(doc['path'].resolve()),
            'source_sha256':doc['source_sha256'],'profile_id':contract['profile_id'],
            'profile_version':contract['profile_version'],'subset':contract['requested_fields'],
            'subset_contract_id':contract['subset_contract_id']})
    return hashlib.sha256(raw).hexdigest(),documents


def arms(qwen='both',include_x=False):
    if qwen not in ('both',*QWENS) or type(include_x) is not bool:raise ValueError('invalid_arm_selection')
    selected=QWENS if qwen=='both' else (qwen,)
    result=[]
    for model in selected:
        for rescue in ('none','yolo26l')+(('yolo26x',) if include_x else ()):
            label=model.replace('.','').replace('-','')+'_'+('db' if rescue=='none' else 'db_'+rescue)
            result.append({'id':label,'qwen':model,'rescue_text_detector':rescue})
    return result


def schedule(documents,selected,manifest_sha):
    result=[]
    for index,doc in enumerate(documents):
        order=selected[index%len(selected):]+selected[:index%len(selected)]
        public='doc_'+digest(['public',manifest_sha,doc['id'],doc['source_sha256']])[:20]
        for arm in order:
            result.append({'row_id':digest([manifest_sha,doc['id'],arm['id']])[:32],
                'document_id':doc['id'],'public_document_id':public,'arm':arm['id']})
    if len({r['row_id'] for r in result})!=len(result):raise ValueError('duplicate_planned_pair')
    return result


def required_models(selected):
    names={'detector','cyrillic','xiaomi'}
    for arm in selected:
        names.add(arm['qwen'])
        if arm['rescue_text_detector']!='none':names.add(arm['rescue_text_detector']+'-text')
    return sorted(names)


def settings_dict(common,arm,models,locks,data_dir):
    from harness_ocr.settings import Settings
    settings=Settings(**{**common,'qwen':arm['qwen'],'rescue_text_detector':arm['rescue_text_detector'],
        'models_dir':Path(models).resolve(),'runtime_lock_dir':Path(locks).resolve(),'data_dir':Path(data_dir)})
    settings.validate()
    return {k:str(v) if isinstance(v,Path) else v for k,v in asdict(settings).items()}


def code_pins(deadline=None):
    paths=[p for p in (ROOT/'src').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc']
    paths += [p for p in (ROOT/'experiments/gpu_delivery').glob('*.py')]
    paths += [ROOT/'experiments/yolo26/metrics.py',ROOT/'scripts/run_documents.py',
              ROOT/'scripts/acceptance.py',ROOT/'scripts/doctor.py',ROOT/'pyproject.toml']
    return {p.relative_to(ROOT).as_posix():file_sha(p,deadline) for p in sorted(set(paths))}


def stat_identity(path):
    stat=Path(path).stat()
    return {'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'inode':stat.st_ino}


def pin_models(models,names,deadline,registry=None):
    if registry is None:
        from harness_ocr.models import model_lock
        registry=model_lock()
    root=Path(models).resolve();pins={}
    for name in names:
        for item in registry['models'][name]['files']:
            path=(root/name/item['file']).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise ValueError('required_model_file_missing')
            actual=file_sha(path,deadline)
            if actual!=item['sha256']:raise ValueError('required_model_file_sha_mismatch')
            pins[path.relative_to(root).as_posix()]={'sha256':actual,'stat':stat_identity(path)}
    return pins,digest({name:registry['models'][name] for name in names})


def validate_runtime_pins(protocol,deadline,full_models=False):
    if code_pins(deadline)!=protocol['code_pins']:raise ValueError('source_changed_during_experiment')
    if versions()!=protocol['versions']:raise ValueError('dependency_versions_changed_during_experiment')
    if file_sha(protocol['manifest_path'],deadline)!=protocol['manifest_sha256']:
        raise ValueError('manifest_changed_during_experiment')
    for doc in protocol['documents']:
        if file_sha(doc['path'],deadline)!=doc['source_sha256']:raise ValueError('document_changed_during_experiment')
    for relative,pin in protocol['model_file_pins'].items():
        path=Path(protocol['models_dir'])/relative
        if stat_identity(path)!=pin['stat']:raise ValueError('model_stat_changed_during_experiment')
        if full_models and file_sha(path,deadline)!=pin['sha256']:raise ValueError('model_sha_changed_during_experiment')


def versions():
    result={'python':sys.version}
    for name in ('harness-ocr','torch','transformers','tokenizers','onnxruntime','numpy','Pillow',
                 'lm-format-enforcer','psutil','nvidia-ml-py','ultralytics-opencv-headless'):
        try:result[name]=metadata.version(name)
        except metadata.PackageNotFoundError:result[name]=None
    return result


def load_exact_gold(path,expected_sha,documents,deadline=None):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>10*1024*1024:raise ValueError('separate_gold_size_or_path')
    if file_sha(path,deadline)!=expected_sha:raise ValueError('gold_file_changed_after_predeclaration')
    value=read(path)
    if (not isinstance(value,dict) or set(value)!={'schema_version','documents'}
        or value['schema_version']!='harness-gpu-delivery-gold/1' or not isinstance(value['documents'],dict)
        or not value['documents']):
        raise ValueError('invalid_separate_gold_schema')
    allowed={d['id']:set(d['subset']) for d in documents}
    from harness_ocr.profiles import ProfileRegistry
    from harness_ocr.typed_fields import STRUCTURED_TYPES
    registry=ProfileRegistry()
    definitions={d['id']:registry.compile(d['profile_id'],subset=d['subset'])['field_definitions']
                 for d in documents}
    if not set(value['documents'])<=set(allowed):raise ValueError('gold_unknown_document')
    for identity,fields in value['documents'].items():
        if not isinstance(fields,dict) or not fields or not set(fields)<=allowed[identity]:
            raise ValueError('gold_field_outside_subset')
        if any(not isinstance(raw,str) or not 0<len(raw)<=8192 for raw in fields.values()):
            raise ValueError('v1_gold_requires_nonempty_exact_raw_strings_no_typed_semantic_shapes')
        if any(definitions[identity][key]['value_type'] in STRUCTURED_TYPES for key in fields):
            raise ValueError('v1_gold_cannot_score_structured_fields_use_separate_typed_semantic_protocol')
    return value['documents']


def validate_gold_preflight(path,documents,deadline=None):
    path=Path(path)
    if not path.is_file() or path.stat().st_size>10*1024*1024:raise ValueError('separate_gold_size_or_path')
    content_sha=file_sha(path,deadline)
    references=load_exact_gold(path,content_sha,documents,deadline)
    check_deadline(deadline)
    return {'path':str(path.resolve()),'sha256':content_sha,'schema_version':'harness-gpu-delivery-gold/1',
        'schema_validated_parent_only_before_inference':True,'reference_values_sent_to_worker':False,
        'reference_documents':len(references),'reference_field_count':sum(len(v) for v in references.values()),
        'reference_type':'nonempty_exact_raw_string_only; typed-semantic references unsupported'}


def load_gold_after_inference(path,expected_sha,documents,deadline=None):
    return load_exact_gold(path,expected_sha,documents,deadline)


def exact_score(result,gold):
    if result is None:return {'measured':False,'missing_result':True,'planned_fields':len(gold)}
    fields=[]
    for key,expected in gold.items():
        field=result.get('fields',{}).get(key,{})
        observed=field.get('extraction_status')=='observed' and field.get('raw_value') is not None
        same=observed and json_exact(field['raw_value'],expected)
        fields.append({'field_key':key,'correct':bool(same),'wrong':bool(observed and not same),
            'withheld':not observed,'validation_pass':field.get('validation_status')=='pass'})
    return {'measured':True,'fields':fields,'total':len(fields),
        **{key:sum(bool(f[key]) for f in fields) for key in ('correct','wrong','withheld','validation_pass')}}


def json_exact(actual,expected):
    """Type-sensitive literal JSON equality, never identity normalization."""
    if type(actual) is not type(expected):return False
    if isinstance(actual,dict):
        return set(actual)==set(expected) and all(json_exact(actual[k],expected[k]) for k in actual)
    if isinstance(actual,list):
        return len(actual)==len(expected) and all(json_exact(a,b) for a,b in zip(actual,expected))
    return actual==expected
