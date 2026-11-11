"""Versioned, lossless typed values with independently checked literal evidence.

These contracts describe supported *observations*, not document authenticity.
Unknown document editions and unlabelled numbers are deliberately not promoted
to a known kind. Every item retains its own role, label, version and value spans.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
import unicodedata

from .util import digest

VERSION = 'typed-fields-v1'
CONTRACT_VERSION = '1.0'
STRUCTURED_TYPES = {'typed_identifier', 'typed_bank_account_list', 'record_list', 'versioned_policy_identifier'}
CONTRACTS = {
    'party-identifiers-v1': {'type': 'typed_identifier', 'roles': {'principal', 'representative'},
                             'kinds': {'inn', 'ogrn', 'ogrnip', 'rf_passport','snils','kpp'}},
    'machine-identifiers-v1': {'type': 'typed_identifier', 'roles': {'machine'},
                               'kinds': {'vin', 'pin', 'factory_number'}},
    'payee-accounts-v1': {'type': 'typed_bank_account_list', 'roles': {'payee'},
                         'kinds': {'settlement', 'treasury', 'eks', 'correspondent'}},
    'ownership-table-v1': {'type': 'record_list', 'roles': {'ownership_history'}, 'kinds': {'ownership_row'}},
    'oms-identifiers-v1': {'type': 'versioned_policy_identifier', 'roles': {'insured'}, 'kinds': {'enp'}},
}
ROLE_LABELS = {
    'principal': ['доверитель', 'доверителя'], 'representative': ['представитель', 'представителя'],
    'machine': ['машина', 'машины', 'техника', 'техники'],
    'payee': ['получатель', 'получателя'], 'insured': ['застрахованное лицо', 'застрахованного лица'],
    'ownership_history': ['сведения о собственниках', 'сведения о владельцах'],
}
# A version is selected by an observed discriminator, never by a value's length.
# "literal" contracts intentionally promise only literal syntax for variable
# manufacturer identifiers, without inventing a universal PIN/factory format.
RULES = {
    'inn': {'labels': ['ИНН'], 'version': 'inn-label-v1', 'versions': ['ИНН'], 'pattern': r'[0-9]{10}|[0-9]{12}'},
    'ogrn': {'labels': ['ОГРН'], 'version': 'ogrn-label-v1', 'versions': ['ОГРН'], 'pattern': r'[0-9]{13}'},
    'ogrnip': {'labels': ['ОГРНИП'], 'version': 'ogrnip-label-v1', 'versions': ['ОГРНИП'], 'pattern': r'[0-9]{15}'},
    'snils': {'labels':['СНИЛС'],'version':'snils-label-v1','versions':['СНИЛС'],'pattern':r'(?:[0-9]{11}|[0-9]{3}-[0-9]{3}-[0-9]{3} [0-9]{2})'},
    'kpp': {'labels':['КПП'],'version':'kpp-label-v1','versions':['КПП'],'pattern':r'[0-9]{9}'},
    'rf_passport': {'labels': ['паспорт РФ', 'паспорт гражданина Российской Федерации'],
                    'version': 'rf-passport-series-number-v1',
                    'versions': ['паспорт РФ', 'паспорт гражданина Российской Федерации'],
                    'pattern': r'(?:[0-9]{4} [0-9]{6}|[0-9]{2} [0-9]{2} [0-9]{6})'},
    'vin': {'labels': ['VIN'], 'version': 'vin-explicit-syntax-v1', 'versions': ['VIN'],
            'pattern': r'[A-HJ-NPR-Z0-9]{17}'},
    'pin': {'labels': ['PIN'], 'version': 'pin-explicit-literal-v1', 'versions': ['PIN'], 'pattern': r'[A-Za-z0-9][A-Za-z0-9./_-]{0,127}'},
    'factory_number': {'labels': ['заводской номер машины', 'заводской номер'],
                       'version': 'factory-explicit-literal-v1', 'versions': ['заводской номер машины', 'заводской номер'],
                       'pattern': r'[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9./_-]{0,127}'},
    'settlement': {'labels': ['расчётный счёт', 'расчетный счет', 'р/с'], 'version': 'bank-account-cbr-515',
                   'versions': ['расчётный счёт', 'расчетный счет', 'р/с'], 'pattern': r'[0-9]{20}'},
    'correspondent': {'labels': ['корреспондентский счёт', 'корреспондентский счет', 'к/с'],
                      'version': 'bank-account-cbr-515', 'versions': ['корреспондентский счёт', 'корреспондентский счет', 'к/с'], 'pattern': r'[0-9]{20}'},
    'treasury': {'labels': ['казначейский счёт', 'казначейский счет'], 'version': 'treasury-eks-2021',
                 'versions': ['казначейский счёт', 'казначейский счет'], 'pattern': r'[0-9]{20}'},
    'eks': {'labels': ['банковский счёт ЕКС', 'банковский счет ЕКС', 'единый казначейский счет', 'единый казначейский счёт'],
            'version': 'treasury-eks-2021', 'versions': ['единый казначейский счет', 'единый казначейский счёт', 'ЕКС'], 'pattern': r'[0-9]{20}'},
    'enp': {'labels': ['единый номер полиса', 'ЕНП'], 'version': 'oms-enp-unified-v1',
            'versions': ['единый номер полиса', 'ЕНП'], 'pattern': r'[0-9]{16}'},
    'ownership_row': {'labels': ['сведения о собственниках', 'сведения о владельцах'],
                      'version': 'ownership-explicit-table-v1', 'versions': ['сведения о собственниках', 'сведения о владельцах']},
}
TABLE_COLUMNS = {'owner': ['собственник', 'владелец', 'наименование собственника'],
                 'date': ['дата регистрации', 'дата приобретения', 'дата записи'],
                 'document': ['документ о праве собственности', 'документ-основание', 'основание']}

# Account kind alone does not identify where it is held. The location must be
# explicit in the account's own label; never pick the formula that happens to pass.
BANK_LOCATION_LABELS={}
for _kind,_base in {'settlement':['расчетный счет','расчётный счёт'],
                   'correspondent':['корреспондентский счет','корреспондентский счёт']}.items():
    for _label in _base:
        for _suffix,_location in {'в кредитной организации':'credit_institution',
                'в Банке России':'bank_of_russia_unit','в подразделении Банка России':'bank_of_russia_unit',
                'в РКЦ':'bank_of_russia_unit'}.items():
            _qualified=_label+' '+_suffix
            BANK_LOCATION_LABELS[_qualified.casefold()]=_location
            RULES[_kind]['labels'].append(_qualified)


def _bank_row_closed(line):
    """Opening context is a closed labelled record, never arbitrary prose."""
    labels={label.casefold() for kind in ('settlement','correspondent','treasury','eks')
            for label in RULES[kind]['labels']}|{'бик'}
    roles={role.casefold() for role in ROLE_LABELS['payee']}
    for cell in re.split('[;|]',line):
        cell=cell.strip()
        if cell.casefold() in roles:continue
        for role in sorted(roles,key=len,reverse=True):
            if cell.casefold().startswith(role+' '):cell=cell[len(role):].lstrip();break
        match=re.fullmatch(r'(.+?)\s*[:№]\s*(\S(?:.*\S)?)',cell)
        if not match or match[1].strip().casefold() not in labels|roles:return False
    return True


def _unresolved_edition_spans(text, kind):
    # Explicit contradictory/unknown edition markers are evidence of an
    # unresolved version. The server contract version is not a document date.
    patterns=[r'(?<!\w)(?:версия|редакция)\s*(?:формата\s*)?[:№]?\s*[0-9][\w.-]*']
    if kind=='enp':patterns += [r'полис[^\n;|]{0,30}старого образца',r'временное свидетельство']
    return [(m.start(),m.end()) for pattern in patterns for m in re.finditer(pattern,text,re.I)]


def contract_for(spec):
    key = spec.get('typed_contract_id')
    if not isinstance(key,str):return None
    contract = CONTRACTS.get(key)
    if (contract and contract['type'] == spec.get('value_type') and spec.get('role') in contract['roles']
            and spec.get('typed_contract_version',CONTRACT_VERSION)==CONTRACT_VERSION):
        return contract
    return None


def _hits(text, labels):
    found = []
    for label in labels:
        for match in re.finditer(r'(?<!\w)' + re.escape(label) + r'(?!\w)', text, re.I):
            found.append((match.start(), match.end()))
    kept = []
    for a, b in sorted(set(found), key=lambda x: (-(x[1]-x[0]), x[0])):
        if not any(a < d and c < b for c, d in kept): kept.append((a, b))
    return sorted(kept)


def _span(obs, a, b):
    return {'observation_id': obs['id'], 'start': a, 'end': b}


def _item(obs, kind, raw, a, b, label, role, versions, association, party):
    item = {'kind': kind, 'version': RULES[kind]['version'], 'raw_value': raw, 'party_role': party,
            'spans': [_span(obs, a, b)], 'kind_spans': label, 'role_spans': role,
            'version_spans': versions, 'association': association}
    item['item_id'] = 'ti_' + digest(item)[:24]
    return item


def _all_spans(item):
    spans = [s for key in ('spans', 'kind_spans', 'role_spans', 'version_spans') for s in item.get(key, [])]
    for cell in item.get('cells', []):
        spans.extend(cell.get('spans', [])); spans.extend(cell.get('kind_spans', []))
    if item.get('bik'):
        spans.extend(item['bik'].get('spans', [])); spans.extend(item['bik'].get('kind_spans', []))
    if item.get('account_location'):spans.extend(item['account_location'].get('spans',[]))
    return spans


def all_evidence_ids(raw):
    return sorted({s['observation_id'] for item in raw['items'] for s in _all_spans(item)})


def value_identity(raw):
    """Semantic identity without provenance; ordered duplicate rows survive."""
    if not isinstance(raw,dict) or not _structure(raw):return raw
    return {key:deepcopy(raw[key]) for key in ('schema_version','contract_id','contract_version','value_type')} | {
        'items':[{key:deepcopy(item[key]) for key in ('kind','version','party_role','raw_value')} |
                 ({'cells':[{'column':c['column'],'raw_value':c['raw_value']} for c in item['cells']]} if 'cells' in item else {}) |
                 ({'bik':item['bik']['raw_value']} if 'bik' in item else {}) |
                 ({'account_location':item['account_location']['kind']} if 'account_location' in item else {}) for item in raw['items']]}


def _wrap(items, spec):
    return {'schema_version': VERSION, 'contract_id': spec['typed_contract_id'], 'contract_version': CONTRACT_VERSION,
            'value_type': spec['value_type'], 'items': items}


def _parts(text):
    offset = 0
    for line_index, part in enumerate(text.splitlines(keepends=True)):
        line = part.rstrip('\r\n')
        if line.strip(): yield line_index, offset, line
        offset += len(part)


def _observed_items(obs, spec, contract):
    """Only explicit labels in the same bounded text line lend a role.

    Document-version discriminators may occur elsewhere in this observation,
    but exactly one supported version must be observable. Values never borrow
    role, BIK, or number cells from another row.
    """
    items = []
    for _, offset, line in _parts(obs['text']):
        role = _hits(line, ROLE_LABELS[spec['role']])
        other = [h for party, aliases in ROLE_LABELS.items() if party != spec['role'] for h in _hits(line, aliases)]
        if len(role) != 1 or other: continue
        if spec['role']=='machine' and re.search(r'(?<!\w)(?:двигател\w*|шасси|кузов\w*|агрегат\w*)(?!\w)',line,re.I):continue
        all_labels=[(a,b,k) for k in contract['kinds']-{'ownership_row'} for a,b in _hits(line,RULES[k]['labels'])]
        for kind in sorted(contract['kinds'] - {'ownership_row'}):
            rule = RULES[kind]; labels = _hits(line, rule['labels'])
            labels=[(a,b) for a,b in labels if not any(c<=a and b<=d and d-c>b-a for c,d,k in all_labels if k!=kind)]
            if len(labels) != 1: continue
            a, b = labels[0]
            # Require a printed separator and one entire bounded cell. Splitting
            # by semicolon/pipe preserves neighbouring labelled cells.
            tail = re.match(r'\s*(?::|№)\s*', line[b:])
            if not tail and role[0][0]>=b:
                arole,brole=role[0]
                if not line[b:arole].strip():
                    after=re.match(r'\s*(?::|№)\s*',line[brole:])
                    if after:tail=re.match(r'.{'+str(brole-b+after.end())+r'}',line[b:])
            if not tail: continue
            start = b + tail.end(); end = len(line)
            boundary = re.search(r'[;|]', line[start:])
            if boundary: end = start + boundary.start()
            while end > start and line[end-1].isspace(): end -= 1
            value = line[start:end]
            if not value or len(value)>128: continue
            # The label may be embedded in the role phrase, e.g. "ИНН
            # доверителя: ..."; the reverse form has role before kind.
            # Explicit role words between kind label and separator are allowed.
            versions = _hits(obs['text'], rule['versions'])
            if not versions: continue
            assoc = {'kind': 'same_line_explicit_labels', 'observation_id': obs['id'],
                     'line_start': offset, 'line_end': offset+len(line)}
            item = _item(obs, kind, value, offset+start, offset+end,
                         [_span(obs, offset+a, offset+b)], [_span(obs, offset+x, offset+y) for x,y in role],
                         [_span(obs, x,y) for x,y in versions], assoc, spec['role'])
            unresolved=_unresolved_edition_spans(obs['text'],kind)
            if unresolved:
                item['version']='unresolved'
                item['version_spans']=[_span(obs,x,y) for x,y in unresolved]
            if kind in {'settlement', 'correspondent'}:
                location=BANK_LOCATION_LABELS.get(line[a:b].casefold()) if _bank_row_closed(line) else None
                if location:item['account_location']={'kind':location,'spans':[_span(obs,offset+a,offset+b)]}
                hits = list(re.finditer(r'(?<!\w)БИК\s*:\s*([^;|]*)', line, re.I))
                if len(hits)==1:
                    hit=hits[0];cell=hit.group(1);literal=cell.strip()
                    if re.fullmatch('[0-9]{9}',literal):
                        left=hit.start(1)+len(cell)-len(cell.lstrip())
                        item['bik']={'raw_value':literal, 'spans':[_span(obs,offset+left,offset+left+len(literal))],
                              'kind_spans':[_span(obs,offset+hit.start(),offset+hit.start()+3)]}
            items.append(item)
    return items


def _table_items(obs, spec):
    nodes=list(_parts(obs['text'])); items=[]
    for pos, (_, offset, line) in enumerate(nodes):
        roles=_hits(line, ROLE_LABELS['ownership_history'])
        if len(roles)!=1 or pos+1>=len(nodes): continue
        _, head_offset, head=nodes[pos+1]
        if '|' not in head: continue
        parts=[]; cursor=0
        for cell in head.split('|'):
            a=cursor+len(cell)-len(cell.lstrip()); b=cursor+len(cell.rstrip()); cursor+=len(cell)+1
            if a==b: break
            names=[name for name, labels in TABLE_COLUMNS.items() if head[a:b].casefold() in {x.casefold() for x in labels}]
            if len(names)!=1: break
            parts.append((names[0],_span(obs,head_offset+a,head_offset+b)))
        else:
            if not {'owner','date'} <= {x[0] for x in parts} or len({x[0] for x in parts})!=len(parts): continue
            role_spans=[_span(obs,offset+a,offset+b) for a,b in roles]
            table=[]
            for row_index, (line_index,row_offset,row) in enumerate(nodes[pos+2:]):
                # A blank line or changed header ends the declared table.
                previous=nodes[pos+1+row_index]
                if line_index!=previous[0]+1: break
                if '|' not in row:
                    if ':' not in row:table=[]
                    break
                raw_cells=row.split('|')
                if len(raw_cells)!=len(parts) or any(not x.strip() for x in raw_cells):table=[];break
                cells=[]; cursor=0
                for (name,header), cell in zip(parts,raw_cells):
                    a=cursor+len(cell)-len(cell.lstrip()); b=cursor+len(cell.rstrip());cursor+=len(cell)+1
                    cells.append({'column':name,'raw_value':row[a:b], 'spans':[_span(obs,row_offset+a,row_offset+b)],'kind_spans':[header]})
                assoc={'kind':'explicit_table_row','observation_id':obs['id'], 'line_start':row_offset,
                       'line_end':row_offset+len(row),'row_index':row_index,'header_spans':[p[1] for p in parts]}
                item=_item(obs,'ownership_row',row,row_offset,row_offset+len(row),role_spans,role_spans,role_spans,assoc,spec['role'])
                unresolved=_unresolved_edition_spans(obs['text'],'ownership_row')
                if unresolved:
                    item['version']='unresolved';item['version_spans']=[_span(obs,a,b) for a,b in unresolved]
                item['cells']=cells;table.append(item)
            items.extend(table)
    return items


def build_typed_candidates(observations, contract, max_candidates=700):
    candidates={}
    for field,spec in contract['field_definitions'].items():
        definition=contract_for(spec)
        if definition is None: continue
        for obs in observations:
            if obs.get('completed') is not True or obs.get('active') is False:continue
            items=_table_items(obs,spec) if spec['value_type']=='record_list' else _observed_items(obs,spec,definition)
            if not items: continue
            raw=_wrap(items,spec)
            if not typed_supported(raw,[obs]): continue
            cid='c_'+digest([field,raw])[:24]
            role_spans=[s for item in items for s in item['role_spans']]
            candidates[cid]={'id':cid,'field_key':field,'raw_value':raw,'evidence_ids':all_evidence_ids(raw),
                'spans':[s for item in items for s in item['spans']], 'party_role':spec['role'],
                'role_evidence_ids':sorted({s['observation_id'] for s in role_spans}), 'role_spans':role_spans,
                'role_verified':True,'candidate_generator':VERSION,'association_proof':{'kind':'typed_per_item_evidence','contract_id':spec['typed_contract_id']},
                'requires_layout_qualification':True}
            if len(candidates)>max_candidates:raise ValueError('candidate_limit_no_silent_truncation')
    return candidates


def _literal(raw):
    return isinstance(raw,str) and 0<len(raw)<=8192 and bool(raw.strip()) and not any(unicodedata.category(c) in {'Cc','Cf'} and c not in '\t' for c in raw)


def _span_shape(span):
    return (isinstance(span,dict) and set(span)=={'observation_id','start','end'} and isinstance(span['observation_id'],str)
            and type(span['start']) is int and type(span['end']) is int and 0<=span['start']<span['end'])


def _structure(raw):
    if not isinstance(raw,dict) or set(raw)!={'schema_version','contract_id','contract_version','value_type','items'}:return False
    if not all(isinstance(raw[key],str) for key in ('schema_version','contract_id','contract_version','value_type')):return False
    contract=CONTRACTS.get(raw['contract_id'])
    if (raw['schema_version']!=VERSION or raw['contract_version']!=CONTRACT_VERSION or not contract
            or raw['value_type']!=contract['type'] or not isinstance(raw['items'],list) or not 1<=len(raw['items'])<=128):return False
    seen=set()
    for item in raw['items']:
        required={'item_id','kind','version','raw_value','party_role','spans','kind_spans','role_spans','version_spans','association'}
        if not isinstance(item,dict) or not required<=set(item) or set(item)-required-{'cells','bik','account_location'}:return False
        if not isinstance(item['item_id'],str) or item['item_id'] in seen: return False
        seen.add(item['item_id'])
        if not all(isinstance(item[key],str) for key in ('kind','version','party_role')):return False
        if item['kind'] not in contract['kinds'] or item['party_role'] not in contract['roles'] or not _literal(item['raw_value']):return False
        if item['version'] not in {RULES[item['kind']]['version'],'unresolved'}:return False
        for key in ('spans','kind_spans','role_spans','version_spans'):
            if not isinstance(item[key],list) or not item[key] or len(item[key])>64 or not all(_span_shape(s) for s in item[key]):return False
        assoc=item['association']
        if not isinstance(assoc,dict):return False
        if assoc.get('kind')=='validated_native_xml':
            proof=assoc.get('native_validation')
            keys={'adapter_id','schema_version','schema_sha256','mapping_id','source_sha256','xpath','attribute','party_id'}
            if (set(assoc)!={'kind','native_validation'} or not isinstance(proof,dict) or set(proof)!=keys
                    or not all(isinstance(v,str) and v for v in proof.values()) or item['kind'] not in CONTRACTS['party-identifiers-v1']['kinds']
                    or raw['contract_id']!='party-identifiers-v1' or 'cells' in item or 'bik' in item or 'account_location' in item):return False
            continue
        if not isinstance(assoc.get('observation_id'),str):return False
        if type(assoc.get('line_start')) is not int or type(assoc.get('line_end')) is not int or not 0<=assoc['line_start']<assoc['line_end']:return False
        if item['kind']=='ownership_row':
            if set(assoc)!={'kind','observation_id','line_start','line_end','row_index','header_spans'} or assoc['kind']!='explicit_table_row' or type(assoc['row_index']) is not int or assoc['row_index']<0:return False
            if not isinstance(assoc['header_spans'],list) or not all(_span_shape(s) for s in assoc['header_spans']):return False
            cells=item.get('cells')
            if not isinstance(cells,list) or not 2<=len(cells)<=3:return False
            columns=[]
            for cell in cells:
                if not isinstance(cell,dict) or set(cell)!={'column','raw_value','spans','kind_spans'} or not isinstance(cell['column'],str) or cell['column'] not in TABLE_COLUMNS or not _literal(cell['raw_value']):return False
                if any(not isinstance(cell.get(key),list) or len(cell[key])!=1 or not all(_span_shape(s) for s in cell[key]) for key in ('spans','kind_spans')):return False
                columns.append(cell['column'])
            if len(set(columns))!=len(columns) or not {'owner','date'}<=set(columns):return False
        elif set(assoc)!={'kind','observation_id','line_start','line_end'} or assoc['kind']!='same_line_explicit_labels' or 'cells' in item:return False
        if 'bik' in item:
            bik=item['bik']
            if item['kind'] not in {'settlement','correspondent'} or not isinstance(bik,dict) or set(bik)!={'raw_value','spans','kind_spans'}:return False
            if not isinstance(bik['raw_value'],str) or not re.fullmatch(r'[0-9]{9}',bik['raw_value']):return False
            if any(not isinstance(bik[key],list) or len(bik[key])!=1 or not _span_shape(bik[key][0]) for key in ('spans','kind_spans')):return False
        if 'account_location' in item:
            location=item['account_location']
            if (item['kind'] not in {'settlement','correspondent'} or not isinstance(location,dict)
                or set(location)!={'kind','spans'} or not isinstance(location.get('kind'),str)
                or location['kind'] not in {'credit_institution','bank_of_russia_unit'}
                or not isinstance(location['spans'],list) or len(location['spans'])!=1
                or not all(_span_shape(s) for s in location['spans'])):return False
    return True


def typed_supported(raw, observations):
    """Reconstruct all items from active sources; reject fabricated associations.

    Re-extraction also checks that a source-backed table remains complete and
    ordered, and that repeated legitimate rows are preserved rather than merged.
    """
    if not _structure(raw):return False
    try:
        by_id={o['id']:o for o in observations if isinstance(o,dict) and isinstance(o.get('id'),str) and isinstance(o.get('text'),str)
               and o.get('completed') is True and o.get('active') is not False}
    except TypeError:return False
    contract=CONTRACTS[raw['contract_id']]
    role=raw['items'][0]['party_role']
    spec={'typed_contract_id':raw['contract_id'],'value_type':raw['value_type'],'role':role}
    expected=[]; ids=[]
    native=[]
    for item in raw['items']:
        if item['party_role']!=role:return False
        for span in _all_spans(item):
            obs=by_id.get(span['observation_id'])
            if not obs or span['end']>len(obs['text']):return False
        if item['association']['kind']=='validated_native_xml':
            native.append(item);continue
        oid=item['association']['observation_id']
        if oid not in by_id:return False
        if oid not in ids:ids.append(oid)
    for oid in ids:
        obs=by_id[oid]
        expected.extend(_table_items(obs,spec) if raw['value_type']=='record_list' else _observed_items(obs,spec,contract))
    if native:
        if ids:return False
        bindings={(i['association']['native_validation']['adapter_id'],i['association']['native_validation']['schema_version'],
                   i['association']['native_validation']['schema_sha256'],i['association']['native_validation']['source_sha256'],
                   i['association']['native_validation']['party_id']) for i in native}
        if len(bindings)!=1:return False
        if len({digest(i['spans']) for i in native})!=len(native):return False
        try:
            from .native.mchd import verify_typed_xml_value
            return verify_typed_xml_value(raw,list(by_id.values()))
        except (ImportError,ValueError,KeyError,TypeError):return False
    return raw['items']==expected


def validate_typed(raw, spec, *, observations=None):
    canonical=deepcopy(raw)
    result={'canonical_value':canonical,'transform_ids':[], 'validation_status':'not_checked','validation_reasons':[],
            'validation_scope':'typed_literal_evidence_and_format_only','validator_id':VERSION+':'+spec['value_type'],
            'validation_checks':[]}
    def finish(status,reason=None):
        result['validation_status']=status;result['validation_reasons']=[] if reason is None else [reason];return result
    if not isinstance(raw,dict):return finish('not_checked','typed_or_versioned_contract_required')
    if not _structure(raw):return finish('not_checked','unsupported_or_invalid_typed_contract')
    contract=contract_for(spec)
    if contract is None or raw['contract_id']!=spec['typed_contract_id'] or raw['value_type']!=spec['value_type'] or any(i['party_role']!=spec['role'] for i in raw['items']):return finish('fail','typed_field_contract_or_role_mismatch')
    if observations is None:return finish('not_checked','typed_evidence_not_revalidated')
    try:observations=list(observations)
    except TypeError:return finish('fail','invalid_typed_evidence_collection')
    if not typed_supported(raw,observations):return finish('fail','typed_literal_label_role_version_or_layout_not_supported')
    review=[]
    if raw['value_type'] in {'typed_identifier','versioned_policy_identifier'}:
        for kind in {item['kind'] for item in raw['items']}:
            if len({item['raw_value'] for item in raw['items'] if item['kind']==kind})>1:
                review.append('multiple_identifier_values_for_one_kind')
    if raw['value_type']=='typed_bank_account_list':
        kinds={item['kind'] for item in raw['items']}
        if kinds & {'treasury','eks'}:
            if not {'treasury','eks'}<=kinds:review.append('treasury_subtype_requires_treasury_and_eks_accounts')
            if kinds & {'settlement','correspondent'}:review.append('mixed_bank_account_system_requires_review')
        elif 'settlement' not in kinds:review.append('bank_subtype_requires_settlement_account')
        # A visible recipient account cell with no registered kind may be a
        # required account. It cannot disappear while another labelled cell
        # makes the list look complete.
        represented={(i['association'].get('observation_id'),i['association'].get('line_start')) for i in raw['items']}
        for obs in observations:
            if obs.get('completed') is not True or obs.get('active') is False:continue
            for _,offset,line in _parts(obs['text']):
                if (_hits(line,ROLE_LABELS['payee']) and re.search(r'(?<!\w)сч[её]т(?:а|ов)?\s*[:№]',line,re.I)
                        and (obs['id'],offset) not in represented):review.append('unresolved_payee_account_label')
    for item in raw['items']:
        kind=item['kind'];value=item['raw_value']
        if item['version']=='unresolved':review.append('document_edition_unresolved');continue
        if kind=='ownership_row':
            for cell in item['cells']:
                if cell['column']=='date':
                    try:
                        if not re.fullmatch(r'[0-9]{2}\.[0-9]{2}\.[0-9]{4}',cell['raw_value']):raise ValueError()
                        datetime.strptime(cell['raw_value'],'%d.%m.%Y')
                    except ValueError:return finish('fail','ownership_date_format_or_calendar')
            continue
        if kind not in {'settlement','correspondent'} and not re.fullmatch(RULES[kind]['pattern'],value):return finish('fail',kind+'_literal_format')
        if kind in {'settlement','correspondent','treasury','eks'} and not any(c!='0' for c in value):return finish('fail','bank_account_degenerate')
        if kind in {'inn','snils','kpp'}:
            from .validators import validate
            literal_result=validate(value,{'value_type':kind})
            if literal_result['validation_status']=='fail':return finish('fail',kind+'_checksum_or_format')
            if literal_result['validation_status']=='not_checked':review.extend(literal_result['validation_reasons'])
        if kind in {'ogrn','ogrnip'}:
            modulus=11 if kind=='ogrn' else 13
            if not any(c!='0' for c in value) or int(value[:-1])%modulus%10!=int(value[-1]):return finish('fail',kind+'_checksum')
        if kind=='enp':
            # Territorial state OMS fund: ISO/HL7 Mod10 over the 15-digit
            # payload, odd positions counted from the right are doubled.
            odd=value[:-1][::-1][::2][::-1];even=value[:-1][::-1][1::2]
            total=sum(int(c) for c in even+str(int(odd)*2))
            if not any(c!='0' for c in value) or (10-total%10)%10!=int(value[-1]):return finish('fail','enp_mod10_checksum')
        if kind in {'settlement','correspondent'}:
            bik=item.get('bik')
            if not bik:review.append('bank_bik_not_observed');continue
            from .bank_keys import validate_bank_key
            checked=validate_bank_key(value,bik['raw_value'],location=item.get('account_location',{}).get('kind'))
            for check in checked.get('checks',[]):
                result['validation_checks'].append({**check,'check_id':check['check_id']+':'+item['item_id'],
                    'evidence_ids':sorted({s['observation_id'] for s in _all_spans(item)})})
            if checked['validation_status']=='fail':return finish('fail',';'.join(checked['validation_reasons']))
            if checked['validation_status']!='pass':review.extend(checked['validation_reasons'])
    if any(i['kind'] in {'pin','factory_number'} for i in raw['items']):
        from .machine_ids import validate_machine_identifiers
        checked=validate_machine_identifiers(raw,observations)
        result['validation_checks'].extend(checked.get('checks',[]))
        if checked['validation_status']=='fail':return finish('fail',';'.join(checked['validation_reasons']))
        if checked['validation_status']!='pass':review.extend(checked['validation_reasons'])
    if review:return finish('not_checked',';'.join(sorted(set(review))))
    return finish('pass')


def typed_value_schema(spec):
    """Public raw/canonical schema for real nested data (never JSON text)."""
    span={'type':'object','additionalProperties':False,'required':['observation_id','start','end'],
          'properties':{'observation_id':{'type':'string'},'start':{'type':'integer','minimum':0},'end':{'type':'integer','minimum':1}}}
    spans={'type':'array','minItems':1,'maxItems':64,'items':span}
    kinds=sorted(CONTRACTS[spec['typed_contract_id']]['kinds'])
    assoc_props={'kind':{'const':'same_line_explicit_labels'},'observation_id':{'type':'string'},
                 'line_start':{'type':'integer','minimum':0},'line_end':{'type':'integer','minimum':1}}
    association={'type':'object','additionalProperties':False,'required':list(assoc_props),'properties':assoc_props}
    props={'item_id':{'type':'string'},'kind':{'enum':kinds},'version':{'enum':sorted({RULES[k]['version'] for k in kinds}|{'unresolved'})},
           'raw_value':{'type':'string','minLength':1,'maxLength':8192},'party_role':{'const':spec['role']},
           **{key:deepcopy(spans) for key in ('spans','kind_spans','role_spans','version_spans')},
           'association':association}
    if spec['value_type']=='record_list':
        association['properties']['kind']={'const':'explicit_table_row'}
        association['properties'].update(row_index={'type':'integer','minimum':0},header_spans=spans)
        association['required']+=['row_index','header_spans']
        props['cells']={'type':'array','minItems':2,'maxItems':3,'items':{'type':'object','additionalProperties':False,
            'required':['column','raw_value','spans','kind_spans'],'properties':{'column':{'enum':sorted(TABLE_COLUMNS)},'raw_value':{'type':'string'},'spans':spans,'kind_spans':spans}}}
    if spec['value_type']=='typed_bank_account_list':
        props['bik']={'type':'object','additionalProperties':False,'required':['raw_value','spans','kind_spans'],
                      'properties':{'raw_value':{'type':'string','pattern':'^[0-9]{9}$'},'spans':spans,'kind_spans':spans}}
        props['account_location']={'type':'object','additionalProperties':False,'required':['kind','spans'],
            'properties':{'kind':{'enum':['credit_institution','bank_of_russia_unit']},'spans':spans}}
    if spec['typed_contract_id']=='party-identifiers-v1':
        native_keys=['adapter_id','schema_version','schema_sha256','mapping_id','source_sha256','xpath','attribute','party_id']
        native={'type':'object','additionalProperties':False,'required':['kind','native_validation'],
                'properties':{'kind':{'const':'validated_native_xml'},'native_validation':{'type':'object','additionalProperties':False,
                    'required':native_keys,'properties':{key:{'type':'string','minLength':1} for key in native_keys}}}}
        props['association']={'oneOf':[association,native]}
    return {'type':'object','additionalProperties':False,'required':['schema_version','contract_id','contract_version','value_type','items'],
            'properties':{'schema_version':{'const':VERSION},'contract_id':{'const':spec['typed_contract_id']},'contract_version':{'const':CONTRACT_VERSION},
                          'value_type':{'const':spec['value_type']},'items':{'type':'array','minItems':1,'maxItems':128,'items':{'type':'object','additionalProperties':False,
                                'required':[key for key in props if key not in {'bik','account_location'}],'properties':props,
                                'allOf':[{'if':{'properties':{'kind':{'const':kind}}},'then':{'properties':{
                                    'version':{'enum':[RULES[kind]['version'],'unresolved']},
                                    **({'bik':False,'account_location':False} if spec['value_type']=='typed_bank_account_list'
                                       and kind not in {'settlement','correspondent'} else {})}}} for kind in kinds]}}}}
