"""Narrow manufacturer PIN/serial checks, never a universal machine-ID codec.

Only a closed, single source line with one machine, an explicit manufacturer,
model and PIN is registered. The original labelled serial may additionally be
compared to the manufacturer's documented PIN suffix; no value is filled.
Unknown families, codes and manual ranges abstain. No VIN checksum is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

VERSION = 'manufacturer-machine-id-v1'
REGISTRY_PATH = Path(__file__).with_name('data') / 'machine_id_sources.json'
REGISTRY_SHA256 = 'd9854c0f805bba3af9c5f7b837c06cfebb1de4fd431c304bf0ff3d452f5105a8'
FAMILY_ID = 'deere-331g-333g-omt362754x19-d4'
_SOURCE_IDS = ['deere-omt362754x19-d4', 'aem-wmc-listing-instructions',
               'aem-wmc-public-record-262aa2ea']
_VERSIONS = {'pin': 'pin-explicit-literal-v1', 'factory_number': 'factory-explicit-literal-v1'}
_ROLE = re.compile(r'(?<!\w)(?:машина|машины|техника|техники)(?!\w)', re.I)
_KIND_LABELS = {'pin': {'pin'}, 'factory_number': {'заводской номер'}}
_CELL_LABELS = {'изготовитель':'manufacturer', 'производитель':'manufacturer',
                'модель':'model', 'pin':'pin', 'заводской номер':'factory_number'}


def _registry():
    data = REGISTRY_PATH.read_bytes()
    if len(data)>20000 or hashlib.sha256(data).hexdigest()!=REGISTRY_SHA256:
        raise ValueError('machine_source_registry_integrity')
    registry=json.loads(data)
    if registry.get('schema_version')!='harness-machine-id-sources/1' or registry['family']['id']!=FAMILY_ID:
        raise ValueError('machine_source_registry_version')
    return registry


def _span(span, by_id, *, bounds=None):
    if not isinstance(span,dict) or set(span)!={'observation_id','start','end'}:
        raise ValueError('machine_invalid_span')
    oid=span['observation_id'];start,end=span['start'],span['end']
    if not isinstance(oid,str) or oid not in by_id or type(start) is not int or type(end) is not int:
        raise ValueError('machine_span_source_missing')
    text=by_id[oid]['text']
    if not 0<=start<end<=len(text):raise ValueError('machine_span_bounds')
    if bounds is not None and not bounds[0]<=start<end<=bounds[1]:raise ValueError('machine_cross_row_span')
    return text[start:end]


def _line(item, by_id):
    assoc=item.get('association')
    if not isinstance(assoc,dict) or set(assoc)!={'kind','observation_id','line_start','line_end'}:
        raise ValueError('machine_association_not_registered')
    if assoc['kind']!='same_line_explicit_labels':raise ValueError('machine_association_not_registered')
    oid=assoc['observation_id'];a,b=assoc['line_start'],assoc['line_end']
    if not isinstance(oid,str) or oid not in by_id or type(a) is not int or type(b) is not int:
        raise ValueError('machine_context_source_missing')
    text=by_id[oid]['text']
    if not 0<=a<b<=len(text) or b-a>1024:raise ValueError('machine_context_bounds')
    if a and text[a-1] not in '\r\n':raise ValueError('machine_context_not_whole_line')
    if b<len(text) and text[b] not in '\r\n':raise ValueError('machine_context_not_whole_line')
    line=text[a:b]
    if '\r' in line or '\n' in line:raise ValueError('machine_context_cross_row')
    if len(list(_ROLE.finditer(line)))!=1:raise ValueError('machine_role_not_unique')
    for key in ('spans','kind_spans','role_spans','version_spans'):
        spans=item.get(key)
        if not isinstance(spans,list) or len(spans)!=1:raise ValueError('machine_requires_single_literal_spans')
        if spans[0].get('observation_id')!=oid:raise ValueError('machine_cross_source_spans')
        literal=_span(spans[0],by_id,bounds=(a,b))
        if key=='spans' and literal!=item['raw_value']:raise ValueError('machine_raw_not_literal')
        if key in {'kind_spans','version_spans'} and literal.casefold() not in _KIND_LABELS[item['kind']]:
            raise ValueError('machine_kind_or_version_label_mismatch')
        if key=='role_spans' and _ROLE.fullmatch(literal) is None:raise ValueError('machine_role_span_mismatch')
    return oid,a,b,line


def _cells(line, offset):
    cells={};cursor=0;role_cells=0
    for part in line.split(';'):
        left=len(part)-len(part.lstrip());right=len(part.rstrip());text=part[left:right]
        if _ROLE.fullmatch(text):
            role_cells+=1;cursor+=len(part)+1;continue
        match=re.fullmatch(r'([^:;|]+):\s*(\S(?:.*\S)?)',text)
        if not match:raise ValueError('machine_context_layout_not_registered')
        kind=_CELL_LABELS.get(match[1].strip().casefold())
        if kind is None or kind in cells:raise ValueError('machine_context_unknown_or_duplicate_cell')
        start=cursor+left+match.start(2);end=cursor+left+match.end(2)
        cells[kind]={'literal':match[2],'span':(offset+start,offset+end)}
        cursor+=len(part)+1
    if role_cells!=1 or not {'manufacturer','model','pin'}<=set(cells):
        raise ValueError('machine_explicit_context_required')
    return cells


def derive_machine_context(raw, observations):
    """Derive explicit source context for conflict identity, not qualification.

    Unknown manufacturers and models are retained literally. Missing context is
    None, never a contradiction or a guessed family. Incoming normalized
    observations may lack the store-owned ``active`` flag; an explicit false
    flag and incomplete evidence are ineligible even for this derivation.
    The validating API below still requires stored ``active is True`` evidence.
    """
    try:
        if (not isinstance(raw,dict) or raw.get('schema_version')!='typed-fields-v1'
            or raw.get('contract_id')!='machine-identifiers-v1' or raw.get('contract_version')!='1.0'
            or raw.get('value_type')!='typed_identifier' or not isinstance(raw.get('items'),list)
            or not 1<=len(raw['items'])<=2):return None
        items=raw['items']
        if any(not isinstance(i,dict) or not isinstance(i.get('kind'),str)
            or i['kind'] not in _VERSIONS or i.get('party_role')!='machine'
            or i.get('version')!=_VERSIONS[i['kind']] for i in items):return None
        if sum(i['kind']=='pin' for i in items)!=1 or sum(i['kind']=='factory_number' for i in items)>1:return None
        observations=list(observations)
        if any(not isinstance(o,dict) for o in observations):return None
        ids=[o.get('id') for o in observations]
        if any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids):return None
        by_id={o['id']:o for o in observations if o.get('active') is not False and o.get('completed') is True
            and isinstance(o.get('text'),str) and isinstance(o.get('source_sha256'),str)
            and re.fullmatch('[0-9a-f]{64}',o['source_sha256'])
            and o.get('evidence_kind') in {'image_region','native_pdf_text'}
            and (o['evidence_kind']!='native_pdf_text' or o.get('text_layer_checks') is True)}
        lines=[_line(i,by_id) for i in items]
        if len({line[:3] for line in lines})!=1:return None
        _,a,_,line=lines[0];cells=_cells(line,a)
        for item in items:
            value=cells[item['kind']];span=item['spans'][0]
            if item['raw_value']!=value['literal'] or (span['start'],span['end'])!=value['span']:return None
        if ('factory_number' in cells)!=any(i['kind']=='factory_number' for i in items):return None
        return {'manufacturer':cells['manufacturer']['literal'].casefold(),'model':cells['model']['literal']}
    except (TypeError,KeyError,ValueError,AttributeError):return None


def validate_machine_identifiers(raw, observations):
    """Return scoped evidence checks without changing raw or claiming issue.

    This helper complements the outer typed-value reconstruction. It performs
    its own literal, role, source-line and context binding checks as well. A
    passing result is the registered layout subset only, not authenticity.
    """
    checks=[];evidence=[]
    def check(name,status,reason=None,scope='registered_manufacturer_layout'):
        checks.append({'check_id':'machine.'+FAMILY_ID+'.'+name,'status':status,'scope':scope,
            'reason_codes':[] if reason is None else [reason],
            'source_ids':list(_SOURCE_IDS),'evidence_ids':list(evidence)})
    def finish(status,reason=None):
        if reason:check('result',status,reason)
        reasons=sorted({r for c in checks for r in c['reason_codes'] if c['status'] in {'fail','not_checked'}})
        return {'validation_status':status,'validation_reasons':reasons,'checks':checks}
    try:registry=_registry()
    except (OSError,ValueError,KeyError,TypeError):return finish('not_checked','machine_source_registry_not_qualified')
    check('source_registry','pass',scope='pinned_primary_source_metadata')
    if (not isinstance(raw,dict) or raw.get('schema_version')!='typed-fields-v1'
        or raw.get('contract_id')!='machine-identifiers-v1' or raw.get('contract_version')!='1.0'
        or raw.get('value_type')!='typed_identifier' or not isinstance(raw.get('items'),list) or not raw['items']):
        return finish('not_checked','machine_typed_contract_not_registered')
    items=raw['items']
    if any(not isinstance(i,dict) or not isinstance(i.get('kind'),str) or i['kind'] not in _VERSIONS for i in items):
        return finish('not_checked','machine_identifier_family_not_registered')
    if len(items)>2 or sum(i['kind']=='pin' for i in items)!=1 or sum(i['kind']=='factory_number' for i in items)>1:
        return finish('not_checked','machine_requires_one_independently_observed_pin')
    if any(i.get('party_role')!='machine' for i in items):return finish('fail','machine_item_role_mismatch')
    if any(i.get('version')!=_VERSIONS[i['kind']] for i in items):return finish('not_checked','machine_document_version_unresolved')
    try:
        observations=list(observations)
        if any(not isinstance(o,dict) for o in observations):raise ValueError('machine_invalid_observations')
        ids=[o.get('id') for o in observations]
        if any(not isinstance(i,str) for i in ids) or len(set(ids))!=len(ids):raise ValueError('machine_duplicate_observation_ids')
        by_id={o['id']:o for o in observations if o.get('active') is True and o.get('completed') is True
            and isinstance(o.get('text'),str) and isinstance(o.get('source_sha256'),str)
            and re.fullmatch('[0-9a-f]{64}',o['source_sha256'])
            and o.get('evidence_kind') in {'image_region','native_pdf_text'}
            and (o['evidence_kind']!='native_pdf_text' or o.get('text_layer_checks') is True)}
        lines=[_line(i,by_id) for i in items]
        if len({line[:3] for line in lines})!=1:raise ValueError('machine_identifiers_not_same_source_line')
        oid,a,b,line=lines[0];evidence.append(oid);cells=_cells(line,a)
        for item in items:
            value=cells[item['kind']]
            span=item['spans'][0]
            if item['raw_value']!=value['literal'] or (span['start'],span['end'])!=value['span']:
                raise ValueError('machine_value_not_complete_labelled_cell')
        if ('factory_number' in cells)!=any(i['kind']=='factory_number' for i in items):
            raise ValueError('machine_source_identifier_omitted')
    except (TypeError,KeyError,ValueError,AttributeError) as exc:
        # Unsupported context cannot lend a manufacturer role or guessed value.
        return finish('not_checked',str(exc) if isinstance(exc,ValueError) else 'machine_invalid_source_binding')
    check('bounded_context','pass',scope='one_complete_labelled_source_line')
    family=registry['family']
    if cells['manufacturer']['literal'].casefold() not in {x.casefold() for x in family['manufacturer_aliases']}:
        return finish('not_checked','machine_manufacturer_not_registered')
    model=cells['model']['literal']
    if model not in family['models']:return finish('not_checked','machine_model_not_registered')
    pin=cells['pin']['literal']
    if re.fullmatch('[A-Z0-9]{17}',pin) is None:return finish('fail','machine_pin_literal_shape')
    check('pin_shape','pass',scope='exact_17_ascii_alphanumeric_literal')
    if pin[:3] not in family['registered_wmc_subset']:return finish('not_checked','machine_wmc_not_in_registered_family_subset')
    check('wmc','pass',scope='pinned_wmc_manufacturer_subset_consistency')
    if pin[3:7]!=model:return finish('fail','machine_pin_model_context_disagreement')
    check('model','pass',scope='independently_labelled_model_matches_pin_positions_4_7')
    if pin[7] not in family['option_codes']:return finish('not_checked','machine_option_code_not_registered')
    check('option','pass',scope='published_option_code_subset')
    if re.fullmatch('[A-Z]',pin[8]) is None:return finish('fail','machine_check_letter_not_alpha')
    check('check_letter','pass',scope='assigned_alpha_only_no_checksum_and_no_authenticity')
    if pin[9] not in family['known_year_codes']:return finish('not_checked','machine_year_code_not_registered')
    check('year_code','pass',scope='published_year_code_subset_2013_2017')
    if pin[10]!=family['registered_emission_code']:
        return finish('not_checked','machine_emission_outside_registered_manual_cover')
    serial=pin[11:]
    if re.fullmatch('[0-9]{6}',serial) is None:return finish('fail','machine_manual_numeric_serial_shape')
    if int(serial)<family['minimum_serial_for_registered_manual']:
        return finish('not_checked','machine_pin_before_registered_manual_range')
    check('manual_range','pass',scope='literal_manual_cover_suffix_F300253_onward_subset')
    if 'factory_number' in cells:
        if cells['factory_number']['literal']!=serial:return finish('fail','machine_pin_factory_number_disagreement')
        check('factory_serial_relation','pass',scope='two_independent_labelled_literals_equal_pin_positions_12_17')
    else:check('factory_serial_relation','not_applicable',scope='no_factory_literal_supplied_or_inferred')
    return finish('pass')
