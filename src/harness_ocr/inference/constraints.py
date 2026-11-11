"""Fail-closed token-level JSON constraints, built only from server allowlists.

Uses the official LM Format Enforcer core API. The tokenizer preprocessing is
the same encode/decode technique as its Transformers integration, without that
module's version-specific Torch/Transformers imports. Data is cached; parser
states are NEVER shared between generate calls. Upstream's error-to-EOS fallback
and zero-length-array limitation are explicitly guarded below.
"""
from __future__ import annotations
from collections import OrderedDict
from importlib.metadata import version
import json
import re
import threading
from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
from lmformatenforcer.characterlevelparser import CharacterLevelParser
from lmformatenforcer.tokenenforcer import TokenEnforcer,TokenEnforcerTokenizerData
from ..util import digest

VERSION='server-allowlist-json-v1'
ENGINE='lm-format-enforcer/0.11.3'
REASONS=['missing','ambiguous_role','ambiguous_characters','conflict','incomplete','critical_recheck']

class ConstraintError(ValueError):pass

def _load(text):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ConstraintError('duplicate_server_prompt_key')
            out[key]=value
        return out
    return json.loads(text,object_pairs_hook=unique)

def _identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,192}',value):
        raise ConstraintError('unsupported_constraint_identifier')
    return value

def _obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

def _string_enum(values):
    if not values:raise ConstraintError('empty_string_enum')
    return {'type':'string','enum':list(values)}

def build_response_schema(prompt,purpose='selector'):
    if purpose=='verifier':return _obj({'verdict':_string_enum(['correct','incorrect','abstain'])})
    if purpose!='selector':raise ConstraintError('unsupported_constraint_purpose')
    try:material=_load(prompt)
    except (ValueError,TypeError) as exc:raise ConstraintError('selector_requires_server_json_prompt') from exc
    if not isinstance(material,dict) or not {'contract','fields','candidates','regions'}<=set(material):
        raise ConstraintError('selector_prompt_missing_server_allowlists')
    if not isinstance(material['contract'],dict) or not isinstance(material['fields'],list):raise ConstraintError('invalid_server_contract')
    if not 1<=len(material['fields'])<=100 or not isinstance(material['candidates'],list) or len(material['candidates'])>700:
        raise ConstraintError('constraint_allowlist_budget')
    if not isinstance(material['regions'],list) or len(material['regions'])>3000:raise ConstraintError('constraint_region_budget')
    fields={}
    for field in material['fields']:
        if not isinstance(field,dict) or not {'key','role'}<=set(field):raise ConstraintError('invalid_constraint_field')
        key=_identifier(field['key']);role=_identifier(field['role'])
        if key in fields:raise ConstraintError('duplicate_constraint_field')
        fields[key]=role
    choices={key:[] for key in fields};seen=set()
    for candidate in material['candidates']:
        if not isinstance(candidate,dict) or not {'id','field_key','party_role'}<=set(candidate):raise ConstraintError('invalid_constraint_candidate')
        ident=_identifier(candidate['id']);field=_identifier(candidate['field_key'])
        if ident in seen or field not in fields or candidate['party_role']!=fields[field]:raise ConstraintError('candidate_allowlist_scope_mismatch')
        seen.add(ident);choices[field].append(ident)
    regions=[]
    for region in material['regions']:
        if not isinstance(region,dict) or 'id' not in region:raise ConstraintError('invalid_constraint_region')
        ident=_identifier(region['id'])
        if ident in regions:raise ConstraintError('duplicate_constraint_region')
        regions.append(ident)
    proposals=[]
    for field,ids in choices.items():
        if ids:
            proposals.append(_obj({'field_key':_string_enum([field]),
                'candidate_ids':{'type':'array','items':_string_enum(ids),'minItems':1,'maxItems':min(32,len(ids)),'uniqueItems':True},
                'party_role':_string_enum([fields[field]])}))
    # A concrete empty object item is never allowed when maxItems=0. Our strict
    # parser blocks the first item as well as the upstream array comma limit.
    pitem={'oneOf':proposals} if len(proposals)>1 else (proposals[0] if proposals else _obj({}))
    request=_obj({'field_key':_string_enum(list(fields)),
         'region_ids':{'type':'array','items':_string_enum(regions) if regions else {'type':'string'},
                       'minItems':1,'maxItems':min(4,len(regions)),'uniqueItems':True},
         'reason_code':_string_enum(REASONS)})
    return _obj({'proposals':{'type':'array','items':pitem,'minItems':0,'maxItems':len(proposals)},
        'reread_requests':{'type':'array','items':request,'minItems':0,'maxItems':20 if regions else 0}})

def _valid(value,schema):
    """Validate only the closed schema subset emitted above, including uniqueItems."""
    if 'oneOf' in schema:return sum(_valid(value,s) for s in schema['oneOf'])==1
    if schema.get('type')=='object':
        return (isinstance(value,dict) and set(value)==set(schema['required'])
                and all(_valid(value[k],schema['properties'][k]) for k in value))
    if schema.get('type')=='array':
        if not isinstance(value,list) or not schema['minItems']<=len(value)<=schema['maxItems']:return False
        if schema.get('uniqueItems') and len({json.dumps(v,sort_keys=True) for v in value})!=len(value):return False
        return all(_valid(v,schema['items']) for v in value)
    if schema.get('type')=='string':return isinstance(value,str) and ('enum' not in schema or value in schema['enum'])
    return False

class _StrictJsonParser(CharacterLevelParser):
    def __init__(self,schema,inner=None,prefix=''):
        self.schema=schema;self.inner=inner or JsonSchemaParser(schema,config=CharacterLevelParserConfig(
            max_consecutive_whitespaces=4,force_json_field_order=True,max_json_array_length=32))
        self.prefix=prefix
        self.empty_arrays=[name for name,spec in schema.get('properties',{}).items() if spec.get('type')=='array' and spec.get('maxItems')==0]
    def add_character(self,ch):
        if ch not in self.get_allowed_characters():raise ConstraintError('parser_character_outside_closed_schema')
        return _StrictJsonParser(self.schema,self.inner.add_character(ch),self.prefix+ch)
    def get_allowed_characters(self):
        chars=self.inner.get_allowed_characters()
        for key in self.empty_arrays:
            if re.search(re.escape(json.dumps(key))+r'\s*:\s*\[\s*$',self.prefix):
                return ''.join(c for c in chars if c in ' \t\r\n]')
        return chars
    def can_end(self):
        if not self.inner.can_end():return False
        try:return _valid(_load(self.prefix),self.schema)
        except (ValueError,TypeError):return False
    @property
    def config(self):return self.inner.config
    @config.setter
    def config(self,value):
        self.inner.config=CharacterLevelParserConfig(alphabet=value.alphabet,max_consecutive_whitespaces=4,
                         force_json_field_order=True,max_json_array_length=32)

def _tokenizer_data(tokenizer):
    # Mirrors the official integrations/transformers.py preprocessing. Keep this
    # model-independent adapter rather than importing Torch merely to build it.
    size=len(tokenizer);special=set(tokenizer.all_special_ids);eos=tokenizer.eos_token_id
    if isinstance(eos,int):eos_ids=[eos]
    elif isinstance(eos,list) and eos and all(isinstance(t,int) for t in eos):eos_ids=eos
    else:raise ConstraintError('tokenizer_has_no_valid_eos')
    if not all(0<=t<size for t in eos_ids):raise ConstraintError('eos_outside_tokenizer_vocabulary')
    zero=tokenizer.encode('0',add_special_tokens=False)
    if not zero:raise ConstraintError('unsupported_tokenizer_zero_encoding')
    zero=zero[-1];tokens=[]
    for ident in range(size):
        if ident in special:continue
        after=tokenizer.decode([zero,ident])[1:];regular=tokenizer.decode([ident])
        if after:tokens.append((ident,after,len(after)>len(regular)))
    def decoder(ids):return tokenizer.decode(ids).rstrip('\ufffd')
    return TokenEnforcerTokenizerData(tokens,decoder,eos,False,size)

class TokenizerDataCache:
    def __init__(self,max_entries=2):
        if type(max_entries) is not int or not 1<=max_entries<=8:raise ValueError('bounded_tokenizer_cache_required')
        self.max_entries=max_entries;self.entries=OrderedDict();self.lock=threading.Lock();self.builds=0
    def get(self,tokenizer):
        if isinstance(tokenizer,TokenEnforcerTokenizerData):return tokenizer
        key=(id(tokenizer),len(tokenizer),str(tokenizer.eos_token_id),tuple(tokenizer.all_special_ids))
        with self.lock:
            entry=self.entries.get(key)
            if entry is not None and entry[0] is tokenizer:self.entries.move_to_end(key);return entry[1]
            data=_tokenizer_data(tokenizer);self.builds+=1;self.entries[key]=(tokenizer,data)
            while len(self.entries)>self.max_entries:self.entries.popitem(last=False)
            return data

_DEFAULT_CACHE=TokenizerDataCache()

class _PrefixConstraint:
    def __init__(self,data,schema):
        self.data=data;self.schema=schema;self.enforcer=TokenEnforcer(data,_StrictJsonParser(schema));self.initial=None
        self.eos=set(data.eos_token_id if isinstance(data.eos_token_id,list) else [data.eos_token_id])
        self.eos_token_ids=sorted(self.eos)
        self.response_schema_sha256=digest(schema);self.constraint_engine=ENGINE;self.schema_generation_version=VERSION
    def __call__(self,batch_id,sent):
        sequence=sent.tolist() if hasattr(sent,'tolist') else list(sent)
        if not isinstance(sequence,list) or any(type(v) is not int for v in sequence):raise ConstraintError('invalid_generation_token_sequence')
        if self.initial is None:self.initial=tuple(sequence)
        if tuple(sequence[:len(self.initial)])!=self.initial:raise ConstraintError('generation_prompt_changed')
        suffix=sequence[len(self.initial):];parent=tuple(suffix[:-1])
        if suffix:
            if suffix[-1] in self.eos:raise ConstraintError('callback_after_eos_stop_required')
            state=self.enforcer.prefix_states.get(parent)
            if state is None or suffix[-1] not in state.allowed_tokens.allowed_tokens:
                raise ConstraintError('generated_token_outside_allowed_prefix')
        allowed=list(self.enforcer.get_allowed_tokens(suffix).allowed_tokens)
        state=self.enforcer.prefix_states.get(tuple(suffix))
        # Upstream turns unknown exceptions into EOS. Do not let this turn an
        # incomplete/invalid JSON into a seemingly completed generation.
        if (not allowed or state is None or not isinstance(state.parser,_StrictJsonParser)
                or (self.eos.intersection(allowed) and not state.parser.can_end())):
            raise ConstraintError('constraint_parser_failed_closed')
        return allowed

def build_prefix_constraint(tokenizer,prompt,purpose='selector',cache=None):
    if version('lm-format-enforcer')!='0.11.3':raise ConstraintError('unqualified_constraint_engine_version')
    schema=build_response_schema(prompt,purpose)
    if cache is None:cache=_DEFAULT_CACHE
    if isinstance(cache,dict):
        cache=cache.setdefault('harness_tokenizer_data_cache',TokenizerDataCache())
    if not isinstance(cache,TokenizerDataCache):raise ConstraintError('unsupported_constraint_cache')
    return _PrefixConstraint(cache.get(tokenizer),schema)
