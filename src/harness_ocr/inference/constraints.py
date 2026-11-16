"""Fail-closed token-level JSON constraints, built only from server allowlists.

Uses the official LM Format Enforcer core API. The tokenizer preprocessing is
the same encode/decode technique as its Transformers integration, without that
module's version-specific Torch/Transformers imports. Data is cached; parser
states are NEVER shared between generate calls. Upstream's error-to-EOS fallback
and zero-length-array limitation are explicitly guarded below.
"""
from __future__ import annotations
from collections import OrderedDict
from bisect import bisect_left
from importlib.metadata import version
import json
import re
import threading
from lmformatenforcer import JsonSchemaParser,CharacterLevelParserConfig
from lmformatenforcer.characterlevelparser import CharacterLevelParser
from lmformatenforcer.tokenenforcer import TokenEnforcer,TokenEnforcerTokenizerData
from ..util import digest

VERSION='server-allowlist-json-v3-semantic-escapes'
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
    if purpose=='visual_read':
        try:material=_load(prompt)
        except (ValueError,TypeError) as exc:raise ConstraintError('visual_requires_server_json_prompt') from exc
        if not isinstance(material,dict) or set(material)!={'task','region_ids'}:
            raise ConstraintError('visual_prompt_requires_closed_allowlist')
        regions=material['region_ids']
        if (not isinstance(regions,list) or not 1<=len(regions)<=4
            or any(not isinstance(r,str) for r in regions) or len(set(regions))!=len(regions)):
            raise ConstraintError('visual_region_allowlist_budget')
        # The status is a discriminator, not an independent enum. Once the
        # model chooses unreadable, no nonempty literal is a legal prefix.
        # Empty readable/ambiguous captures retain the existing abstention
        # policy in compare_visual_capture; they never corroborate a claim.
        readings={_identifier(rid):{'oneOf':[
            _obj({'status':_string_enum(['unreadable']),'literal':_string_enum([''])}),
            _obj({'status':_string_enum(['readable','ambiguous']),
                  'literal':{'type':'string','maxLength':2048}})]} for rid in regions}
        return _obj({'readings':_obj(readings)})
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
    return _obj({'proposals':{'type':'array','items':pitem,'minItems':0,'maxItems':len(proposals),
                            'x-uniqueBy':'field_key'},
        'reread_requests':{'type':'array','items':request,'minItems':0,'maxItems':20 if regions else 0}})

def _valid(value,schema):
    """Validate the closed server subset, including item and field uniqueness."""
    if 'oneOf' in schema:return sum(_valid(value,s) for s in schema['oneOf'])==1
    if schema.get('type')=='object':
        return (isinstance(value,dict) and set(value)==set(schema['required'])
                and all(_valid(value[k],schema['properties'][k]) for k in value))
    if schema.get('type')=='array':
        if not isinstance(value,list) or not schema['minItems']<=len(value)<=schema['maxItems']:return False
        if schema.get('uniqueItems') and len({json.dumps(v,sort_keys=True) for v in value})!=len(value):return False
        if schema.get('x-uniqueBy'):
            key=schema['x-uniqueBy']
            if any(not isinstance(v,dict) or key not in v for v in value):return False
            if len({json.dumps(v[key],sort_keys=True) for v in value})!=len(value):return False
        return all(_valid(v,schema['items']) for v in value)
    if schema.get('type')=='string':return (isinstance(value,str) and ('enum' not in schema or value in schema['enum'])
            and len(value)<=schema.get('maxLength',len(value)))
    return False


class _EnumChoices:
    """Small indexed enum prefixes; exclusion sets belong to one JSON array."""
    def __init__(self,values):
        self.encoded=tuple(sorted((json.dumps(v,ensure_ascii=False),v) for v in set(values)))
        self.keys=tuple(key for key,_ in self.encoded)
        self.cache=OrderedDict()

    def next_characters(self,prefix,excluded):
        key=(prefix,excluded)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        chars=set()
        for index in range(bisect_left(self.keys,prefix),len(self.keys)):
            encoded,value=self.encoded[index]
            if not encoded.startswith(prefix):break
            if value not in excluded and len(encoded)>len(prefix):chars.add(encoded[len(prefix)])
        result=frozenset(chars);self.cache[key]=result
        # Keep this cache inside one generation. A decorated global method
        # cache would retain old rules/allowlists via its self keys.
        while len(self.cache)>512:self.cache.popitem(last=False)
        return result


class _UniqueRules:
    """Compile the server schema's two uniqueness contracts, never model data."""
    def __init__(self,schema):
        self.by_arrays={};primitive={}
        def visit(spec,name=None):
            if spec.get('type')=='array':
                item=spec['items']
                if spec.get('uniqueItems') and item.get('type')=='string' and 'enum' in item:
                    primitive.setdefault(name,set()).update(item['enum'])
                if spec.get('x-uniqueBy'):
                    key=spec['x-uniqueBy'];values=[]
                    for variant in item.get('oneOf',[item]):
                        values.extend(variant.get('properties',{}).get(key,{}).get('enum',[]))
                    self.by_arrays[name]=(key,_EnumChoices(values))
                visit(item,name)
            for key,value in spec.get('properties',{}).items():visit(value,key)
            for option in spec.get('oneOf',[]):visit(option,name)
        visit(schema)
        self.primitive={name:_EnumChoices(values) for name,values in primitive.items()}


class _SelectorState:
    """Immutable lexical state for selector uniqueness during token traversal.

    The upstream schema parser still owns JSON syntax, field order and enum
    membership. This state only removes already-used enum values, so it cannot
    add a token or turn untrusted document text into a server allowlist. Frames
    and sets are copied on each branch; no parser state crosses generations.
    """
    def __init__(self,rules,frames=(),string=None,is_key=False,escaped=False):
        self.rules=rules;self.frames=frames;self.string=string
        self.is_key=is_key;self.escaped=escaped

    def allowed_string_characters(self):
        if self.string is None or self.is_key or not self.frames:return None
        frame=self.frames[-1]
        if frame['kind']=='array' and frame['name'] in self.rules.primitive:
            return self.rules.primitive[frame['name']].next_characters(self.string,frame['seen'])
        if frame['kind']=='object' and len(self.frames)>1:
            parent=self.frames[-2]
            rule=self.rules.by_arrays.get(parent['name']) if parent['kind']=='array' else None
            if rule and frame['key']==rule[0]:
                return rule[1].next_characters(self.string,parent['seen'])
        return None

    def add(self,ch):
        frames=list(self.frames)
        def finish(value=None,closed=None):
            if not frames:return
            parent=dict(frames[-1])
            if parent['kind']=='object':
                if value is not None:parent['values']=(*parent['values'],(parent['key'],value))
                parent['phase']='comma'
            else:
                if value is not None and parent['name'] in self.rules.primitive:
                    parent['seen']=parent['seen']|{value}
                rule=self.rules.by_arrays.get(parent['name'])
                if closed is not None and rule:
                    observed=dict(closed['values']).get(rule[0])
                    if observed is not None:parent['seen']=parent['seen']|{observed}
            frames[-1]=parent
        if self.string is not None:
            if ch=='"' and not self.escaped:
                value=json.loads(self.string+ch)
                if self.is_key:
                    frame=dict(frames[-1]);frame.update(key=value,phase='colon');frames[-1]=frame
                else:finish(value=value)
                return _SelectorState(self.rules,tuple(frames))
            return _SelectorState(self.rules,tuple(frames),self.string+ch,self.is_key,
                                  not self.escaped if ch=='\\' else False)
        if ch in ' \t\r\n':return self
        if ch=='"':
            is_key=bool(frames and frames[-1]['kind']=='object' and frames[-1]['phase']=='key')
            return _SelectorState(self.rules,tuple(frames),'"',is_key)
        if ch in '{[':
            name=frames[-1]['key'] if frames and frames[-1]['kind']=='object' else None
            frames.append({'kind':'object' if ch=='{' else 'array','name':name,'key':None,
                           'phase':'key' if ch=='{' else 'value','values':(),'seen':frozenset()})
        elif ch in '}]':
            closed=frames.pop();finish(closed=closed if ch=='}' else None)
        elif ch==':':
            frame=dict(frames[-1]);frame['phase']='value';frames[-1]=frame
        elif ch==',':
            frame=dict(frames[-1])
            if frame['kind']=='object':frame.update(phase='key',key=None)
            frames[-1]=frame
        else:raise ConstraintError('unsupported_selector_json_token')
        return _SelectorState(self.rules,tuple(frames))


class _StrictJsonParser(CharacterLevelParser):
    def __init__(self,schema,inner=None,prefix='',semantic=None):
        self.schema=schema;self.inner=inner or JsonSchemaParser(schema,config=CharacterLevelParserConfig(
            max_consecutive_whitespaces=4,force_json_field_order=True,max_json_array_length=32))
        self.prefix=prefix
        self.semantic=semantic or (_SelectorState(_UniqueRules(schema))
            if 'proposals' in schema.get('properties',{}) else None)
        self.empty_arrays=[name for name,spec in schema.get('properties',{}).items() if spec.get('type')=='array' and spec.get('maxItems')==0]
    def add_character(self,ch):
        if ch not in self.get_allowed_characters():raise ConstraintError('parser_character_outside_closed_schema')
        return _StrictJsonParser(self.schema,self.inner.add_character(ch),self.prefix+ch,
                                 self.semantic.add(ch) if self.semantic is not None else None)
    def get_allowed_characters(self):
        chars=self.inner.get_allowed_characters()
        if self.semantic is not None:
            semantic_chars=self.semantic.allowed_string_characters()
            if semantic_chars is not None:chars=''.join(c for c in chars if c in semantic_chars)
        for key in self.empty_arrays:
            if re.search(re.escape(json.dumps(key))+r'\s*:\s*\[\s*$',self.prefix):
                return ''.join(c for c in chars if c in ' \t\r\n]')
        return chars
    def can_end(self):
        if not self.inner.can_end():return False
        try:return _valid(_load(self.prefix),self.schema)
        except (ValueError,TypeError):return False
    def shortcut_key(self):
        # Preserve the library's bounded JSON-string token cache. Hiding this
        # hint forces a full traversal of Qwen's 248k-token trie for every free
        # optical character. This hint only applies *inside* a free string;
        # empty-array guards and our complete-object/EOS validation still run.
        return self.inner.shortcut_key()
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

class _StrictTokenEnforcer(TokenEnforcer):
    def _collect_allowed_tokens(self,parser,tree_node,allowed_tokens,shortcut_key):
        super()._collect_allowed_tokens(parser,tree_node,allowed_tokens,shortcut_key)
        # LMFE's fast free-string cache requires each token's escape sequences
        # to be complete JSON on their own. Qwen has a real token for "\\u",
        # which is a valid prefix, but not a complete escape. Keep the fast
        # cache for ordinary text and additionally traverse only the escape
        # subtree through the same strict parser. No schema/length guard is
        # bypassed, and unreadable's enum string never enters this shortcut.
        if (isinstance(shortcut_key,tuple) and shortcut_key[0]=='json_freetext'
                and '\\' in tree_node.children and '\\' in parser.get_allowed_characters()):
            self._collect_allowed_tokens(parser.add_character('\\'),tree_node.children['\\'],
                                         allowed_tokens,None)

class _PrefixConstraint:
    def __init__(self,data,schema):
        self.data=data;self.schema=schema;self.enforcer=_StrictTokenEnforcer(data,_StrictJsonParser(schema));self.initial=None
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
            if isinstance(state.parser,_StrictJsonParser) and state.parser.can_end():
                raise ConstraintError('token_after_complete_json')
        allowed=list(self.enforcer.get_allowed_tokens(suffix).allowed_tokens)
        state=self.enforcer.prefix_states.get(tuple(suffix))
        # Upstream turns unknown exceptions into EOS. Do not let this turn an
        # incomplete/invalid JSON into a seemingly completed generation.
        if (not allowed or state is None or not isinstance(state.parser,_StrictJsonParser)
                or (self.eos.intersection(allowed) and not state.parser.can_end())):
            raise ConstraintError('constraint_parser_failed_closed')
        # A complete, fully validated response has exactly one continuation:
        # EOS. Do not spend a bounded generation on irrelevant JSON padding.
        if state.parser.can_end():return sorted(self.eos.intersection(allowed))
        return allowed

def build_prefix_constraint(tokenizer,prompt,purpose='selector',cache=None):
    if version('lm-format-enforcer')!='0.11.3':raise ConstraintError('unqualified_constraint_engine_version')
    schema=build_response_schema(prompt,purpose)
    if cache is None:cache=_DEFAULT_CACHE
    if isinstance(cache,dict):
        cache=cache.setdefault('harness_tokenizer_data_cache',TokenizerDataCache())
    if not isinstance(cache,TokenizerDataCache):raise ConstraintError('unsupported_constraint_cache')
    return _PrefixConstraint(cache.get(tokenizer),schema)
