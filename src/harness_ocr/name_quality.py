"""Finite name uncertainty signal; never a validator or text repair.

This intentionally detects only an explicit ASCII-Latin/Russian-Cyrillic
subset, not every Unicode confusable or UTS #39 mixed-script identifier.
Other scripts and entirely single-script confusables are outside its scope.
"""
from __future__ import annotations

import hashlib
import json
import string
import unicodedata

VERSION = 'name-script-uncertainty-v1'
TABLE_ID = 'ascii-latin-russian-cyrillic-visible-pairs-v1'
LATIN = frozenset(string.ascii_letters)
CYRILLIC = frozenset('АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
                     'абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
# Membership/diagnostics only. This map is NEVER applied to any value.
PAIRS = (('A','А'),('B','В'),('C','С'),('E','Е'),('H','Н'),('K','К'),
         ('M','М'),('O','О'),('P','Р'),('T','Т'),('X','Х'),('Y','У'),
         ('a','а'),('c','с'),('e','е'),('o','о'),('p','р'),('x','х'),('y','у'))
CONFUSABLE = frozenset(ch for pair in PAIRS for ch in pair)
TABLE_SHA256 = hashlib.sha256(json.dumps(PAIRS,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
MAX_LITERAL = 4096
MAX_WORDS_REPORTED = 32


def _name_scope(spec):
    kind=spec.get('value_type','')
    if kind in {'person_name','organization_name'}:return kind
    # Generic party is not a Russian-person-name contract. This narrower
    # name-leaf rule avoids scanning identities, identifiers and machine models.
    leaf=spec.get('key','').rsplit('.',1)[-1]
    if kind in {'party','multispan_text'} and leaf in {'name','full_name','bank_name'}:
        return 'generic_name_literal'
    return None


def _word_spans(raw):
    """Letter runs with attached combining marks; separators end the run."""
    start=None
    for index,ch in enumerate(raw):
        category=unicodedata.category(ch)
        if category.startswith('L') or (start is not None and category.startswith('M')):
            if start is None:start=index
        elif start is not None:
            yield start,index
            start=None
    if start is not None:yield start,len(raw)


def name_script_uncertainty(raw,spec):
    scope=_name_scope(spec)
    if scope is None or not isinstance(raw,str) or not raw:return None
    if len(raw)>MAX_LITERAL:
        # No scanning, hashing, slicing or script diagnosis beyond the bound.
        # An incomplete check must never look like a successful clean check.
        return {'version':VERSION,'code':'name_quality_check_incomplete_oversize',
            'scope':scope,'check_status':'incomplete','reason':'literal_exceeds_analysis_bound',
            'literal_codepoints':len(raw),'analysis_limit_codepoints':MAX_LITERAL,
            'scanned_codepoints':0,'requires_review':True,'normalization_applied':False,
            'value_authority':False,'does_not_prove_ocr_error':True}
    words=[];truncated=False
    for start,end in _word_spans(raw):
        letters=raw[start:end]
        if (not any(ch in LATIN for ch in letters) or not any(ch in CYRILLIC for ch in letters)
            or not any(ch in CONFUSABLE for ch in letters)):
            continue
        if len(words)==MAX_WORDS_REPORTED:
            truncated=True
            break
        words.append({'start':start,'end':end,
            'latin_offsets':[i for i in range(start,end) if raw[i] in LATIN],
            'cyrillic_offsets':[i for i in range(start,end) if raw[i] in CYRILLIC],
            'confusable_offsets':[i for i in range(start,end) if raw[i] in CONFUSABLE]})
    if not words:return None
    return {'version':VERSION,'code':'mixed_script_confusables_in_name_word',
        'check_status':'completed',
        'scope':scope,'table_id':TABLE_ID,'table_sha256':TABLE_SHA256,
        'unicode_word_category_version':unicodedata.unidata_version,
        'raw_value_sha256':hashlib.sha256(raw.encode('utf-8','surrogatepass')).hexdigest(),
        'hash_encoding':'utf-8-surrogatepass','offset_unit':'unicode_codepoints_zero_based',
        'words':words,'truncated':truncated,'requires_review':True,
        'normalization_applied':False,'value_authority':False,
        'does_not_prove_ocr_error':True}
