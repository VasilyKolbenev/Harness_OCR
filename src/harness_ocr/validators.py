"""Versioned checks of observed literals; no digit repair or authenticity claims.

Structured/version-dependent values deliberately need an explicit typed contract.
Sources for arithmetic: CBR method_0921.pdf (SNILS), Roskazna Formats interaction
section 5.1.4 (UIN). See docs/validator-sources.json for exact references.
"""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal,InvalidOperation
import re
import unicodedata

VERSION='literal-validators-v2'
TEXT_TYPES={'multispan_text','party','party_or_payment_reference','term_literal','address','text','string','organization_name'}
STRUCTURED_TYPES={'typed_identifier','typed_bank_account_list','record_list','versioned_policy_identifier'}
PATTERNS={
 'date':r'(?<![0-9])[0-9]{2}\.[0-9]{2}\.[0-9]{4}(?![0-9])',
 'date_or_literal':r'(?<![0-9])[0-9]{2}\.[0-9]{2}\.[0-9]{4}(?![0-9])',
 'subdivision_code':r'(?<![0-9])[0-9]{3}-[0-9]{3}(?![0-9])',
 'inn':r'(?<![0-9])(?:[0-9]{12}|[0-9]{10})(?![0-9])',
 'snils':r'(?<![0-9])(?:[0-9]{3}-[0-9]{3}-[0-9]{3} [0-9]{2}|[0-9]{11})(?![0-9])',
 'money_decimal':r'(?<![0-9.,])(?:[0-9]{1,3}(?:[ \u00a0][0-9]{3})+|[0-9]+)[.,][0-9]{2}(?![0-9])',
 'money':r'(?<![0-9.,])(?:[0-9]{1,3}(?:[ \u00a0][0-9]{3})+|[0-9]+)[.,][0-9]{2}(?![0-9])',
 'amount':r'(?<![0-9.,])(?:[0-9]{1,3}(?:[ \u00a0][0-9]{3})+|[0-9]+)[.,][0-9]{2}(?![0-9])',
 'uin':r'(?<![0-9])(?:[0-9]{25}|[0-9]{20})(?![0-9])',
 'bik':r'(?<![0-9])[0-9]{9}(?![0-9])','kpp':r'(?<![0-9])[0-9]{9}(?![0-9])',
 'kbk':r'(?<![0-9])[0-9]{20}(?![0-9])','oktmo':r'(?<![0-9])(?:[0-9]{11}|[0-9]{8})(?![0-9])',
 'year':r'(?<![0-9])[0-9]{4}(?![0-9])',
 'passport_identifier':r'(?<![0-9])(?:[0-9]{4} [0-9]{6}|[0-9]{2} [0-9]{2} [0-9]{6}|[0-9]{10})(?![0-9])',
}

def pattern_spans(text,kind):
    kind=kind.lower();pattern=PATTERNS.get(kind)
    if kind.startswith('digits_') and kind[7:].isdigit():pattern=r'(?<![0-9])[0-9]{'+kind[7:]+r'}(?![0-9])'
    return [(m.start(),m.end()) for m in re.finditer(pattern,text)] if pattern else []

def validation_contract(spec):
    kind=spec['value_type'].lower()
    if kind in STRUCTURED_TYPES:
        scope='versioned_or_structured_contract_required'
    elif kind in {'inn','snils','uin'}:scope='checksum_and_format'
    elif kind=='date':scope='calendar_and_format'
    else:scope='literal_format_only'
    return {'validator_id':VERSION+':'+kind,'validation_scope':scope,
            'production_qualified':False,'does_not_authenticate_document':True,
            'requires_typed_contract':kind in STRUCTURED_TYPES}

def _clean_text(raw,limit):
    return (isinstance(raw,str) and 0<len(raw)<=limit and bool(raw.strip())
            and not any(unicodedata.category(ch) in {'Cc','Cf'} and ch not in '\n\r\t' for ch in raw))

def validate(raw,spec):
    kind=spec['value_type'].lower();canonical=raw;transforms=[];contract=validation_contract(spec)
    def result(status,reason=None,scope=None):
        return {'canonical_value':canonical,'transform_ids':transforms,
            'validation_status':status,'validation_reasons':[] if reason is None else [reason],
            'validation_scope':scope or contract['validation_scope'],'validator_id':contract['validator_id']}
    if not _clean_text(raw,8192):return result('fail','empty_oversized_or_control_text')
    if kind.startswith('digits_') or kind in {'inn','bik','kpp'}:
        stripped=re.sub('[ \u00a0-]','',raw)
        if stripped!=raw:
            allowed=spec.get('normalization_transform_ids',[])
            if not isinstance(allowed,list) or 'numeric-separators-v1' not in allowed:
                return result('fail','undeclared_numeric_separator_normalization')
            canonical=stripped;transforms=['numeric-separators-v1']
    if kind.startswith('digits_') and kind[7:].isdigit():
        return result('pass' if re.fullmatch(r'[0-9]{'+kind[7:]+r'}',canonical) else 'fail',
             None if re.fullmatch(r'[0-9]{'+kind[7:]+r'}',canonical) else 'ascii_digit_count')
    if kind=='date':
        try:
            if not re.fullmatch(r'[0-9]{2}\.[0-9]{2}\.[0-9]{4}',raw):raise ValueError()
            datetime.strptime(raw,'%d.%m.%Y');return result('pass')
        except ValueError:return result('fail','invalid_calendar_date')
    if kind=='date_or_literal':
        if re.fullmatch(r'[0-9]{2}\.[0-9]{2}\.[0-9]{4}',raw):return validate(raw,{**spec,'value_type':'date'})
        allowed=spec.get('allowed_literals',[])
        return result('pass') if raw in allowed else result('not_checked','unqualified_date_literal_version')
    if kind=='subdivision_code':
        return result('pass') if re.fullmatch(r'[0-9]{3}-[0-9]{3}',raw) else result('fail','subdivision_format')
    if kind=='inn':
        if not re.fullmatch(r'(?:[0-9]{10}|[0-9]{12})',canonical):return result('fail','inn_checksum_or_format')
        d=list(map(int,canonical));ok=False
        if len(d)==10:ok=sum(a*b for a,b in zip(d[:9],[2,4,10,3,5,9,4,6,8]))%11%10==d[9]
        if len(d)==12:ok=(sum(a*b for a,b in zip(d[:10],[7,2,4,10,3,5,9,4,6,8]))%11%10==d[10] and sum(a*b for a,b in zip(d[:11],[3,7,2,4,10,3,5,9,4,6,8]))%11%10==d[11])
        return result('pass') if ok and any(d) else result('fail','inn_checksum_or_format')
    if kind=='snils':
        if not re.fullmatch(r'(?:[0-9]{11}|[0-9]{3}-[0-9]{3}-[0-9]{3} [0-9]{2})',raw):return result('fail','snils_format')
        digits=re.sub('[- ]','',raw)
        if int(digits[:9])<=1001998:return result('not_checked','legacy_snils_checksum_not_applicable_requires_version')
        total=sum(int(d)*w for d,w in zip(digits[:9],range(9,0,-1)))
        expected=total%101
        if expected==100:expected=0
        return result('pass') if expected==int(digits[-2:]) else result('fail','snils_checksum')
    if kind=='uin':
        if not re.fullmatch(r'(?:[0-9]{20}|[0-9]{25})',raw):return result('fail','uin_20_or_25_digits')
        d=list(map(int,raw))
        if not any(d[:-1]):return result('fail','uin_degenerate')
        check=sum(n*((i%10)+1) for i,n in enumerate(d[:-1]))%11
        if check==10:check=sum(n*(((i+2)%10)+1) for i,n in enumerate(d[:-1]))%11
        if check==10:check=0
        return result('pass') if check==d[-1] else result('fail','uin_checksum')
    if kind in {'money_decimal','money','amount'}:
        pattern=r'(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:[ \u00a0][0-9]{3})+)[.,][0-9]{2}'
        if not re.fullmatch(pattern,raw):return result('fail','explicit_decimal_amount_required')
        try:
            if Decimal(raw.replace(' ','').replace('\u00a0','').replace(',','.'))<0:raise InvalidOperation()
        except InvalidOperation:return result('fail','invalid_decimal_amount')
        # Validation may interpret separators, but canonical remains literal.
        return result('pass')
    if kind in {'bik','kpp','kbk','oktmo','year','passport_identifier'}:
        formats={'bik':r'[0-9]{9}','kpp':r'[0-9]{9}','kbk':r'[0-9]{20}',
             'oktmo':r'(?:[0-9]{8}|[0-9]{11})','year':r'[0-9]{4}',
             'passport_identifier':r'(?:[0-9]{4} [0-9]{6}|[0-9]{2} [0-9]{2} [0-9]{6}|[0-9]{10})'}
        ok=bool(re.fullmatch(formats[kind],raw))
        if kind=='year':ok=ok and int(raw)>0
        return result('pass') if ok else result('fail',kind+'_literal_format')
    if kind=='currency_literal':
        allowed=spec.get('allowed_literals',['RUB','RUR','руб.','руб','₽','USD','EUR','CNY','GBP','долл.','евро'])
        return result('pass') if raw in allowed else result('not_checked','currency_literal_not_in_versioned_allowlist')
    if kind=='person_name':
        tokens=raw.split()
        ok=(2<=len(tokens)<=6 and len(raw)<=256 and all(any(ch.isalpha() for ch in t)
             and all(ch.isalpha() or ch in "-'’" for ch in t) for t in tokens))
        return result('pass') if ok else result('fail','person_name_literal_shape')
    if kind=='certificate_series':
        return result('pass') if re.fullmatch(r'[IVXLCDM]+-[А-ЯЁ]{2}',raw) else result('not_checked','certificate_series_version_not_supported')
    if kind=='identifier':
        ok=len(raw)<=128 and any(ch.isalnum() for ch in raw) and all(ch.isalnum() or ch in ' /-._№' for ch in raw)
        return result('pass') if ok else result('fail','identifier_literal_shape')
    if kind=='registration_mark':
        ok=len(raw)<=32 and any(ch.isdigit() and ch.isascii() for ch in raw) and all(ch.isalnum() or ch in ' -' for ch in raw)
        return result('pass') if ok else result('fail','registration_mark_literal_shape')
    if kind in TEXT_TYPES or kind=='status_literal':
        limit=128 if kind=='status_literal' else 4096
        return result('pass') if _clean_text(raw,limit) and any(ch.isalnum() for ch in raw) else result('fail','literal_text_shape')
    if kind in STRUCTURED_TYPES:return result('not_checked','typed_or_versioned_contract_not_implemented')
    return result('not_checked','unknown_value_type')
