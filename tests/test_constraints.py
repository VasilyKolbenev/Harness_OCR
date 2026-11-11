"""Actual allowed-token tests: schema structure, enums, EOS and parser isolation."""
import json
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from lmformatenforcer.tokenenforcer import TokenEnforcerTokenizerData
from harness_ocr.inference.constraints import (build_prefix_constraint,build_response_schema,
        ConstraintError,TokenizerDataCache)

EOS=128

def data():
    mapping={i:chr(i) for i in range(128)}
    mapping.update({200:'{"proposals":[],"reread_requests":[]}',
                    201:'{"proposals":[{"region_ids":[]}],"reread_requests":[]}'})
    return TokenEnforcerTokenizerData([(i,s,False) for i,s in mapping.items()],
         lambda ids:''.join(mapping.get(i,'') for i in ids),EOS,False,256)

def prompt(*,candidates=True,regions=True,two_fields=False):
    fields=[{'key':'passport.number','role':'document'}]
    choices=[{'id':'c1','field_key':'passport.number','party_role':'document'},
             {'id':'c2','field_key':'passport.number','party_role':'document'}] if candidates else []
    if two_fields:
        fields.append({'key':'payer.inn','role':'payer'})
        if candidates:choices.append({'id':'p1','field_key':'payer.inn','party_role':'payer'})
    return json.dumps({'contract':{'type':'object'},'fields':fields,'candidates':choices,
                       'regions':[{'id':'r1','page':1}] if regions else []})

def accepts(text,*,request=None,purpose='selector'):
    fn=build_prefix_constraint(data(),request or prompt(),purpose=purpose)
    sequence=[999] # prompt tokens are outside the generated JSON and need not decode.
    fn(0,sequence)
    for char in text:
        token=ord(char)
        if token not in fn(0,sequence):return False
        sequence.append(token)
    return EOS in fn(0,sequence)

class GrammarTests(unittest.TestCase):
    def test_valid_choice_and_reread_both_required_arrays(self):
        output={'proposals':[{'field_key':'passport.number','candidate_ids':['c1','c2'],'party_role':'document'}],
             'reread_requests':[{'field_key':'passport.number','region_ids':['r1'],'reason_code':'critical_recheck'}]}
        self.assertTrue(accepts(json.dumps(output,separators=(',',':'))))
        self.assertTrue(accepts('{"proposals":[],"reread_requests":[]}'))
    def test_actual_invalid_model_shape_is_not_a_possible_generation(self):
        wrong={'proposals':[{'field_key':'passport.number','candidate_ids':['c1'],'party_role':'document'},
                           {'region_ids':['r1'],'reason_code':'critical_recheck'}]}
        self.assertFalse(accepts(json.dumps(wrong,separators=(',',':'))))
    def test_no_candidate_and_no_region_arrays_are_forced_empty(self):
        request=prompt(candidates=False,regions=False)
        self.assertTrue(accepts('{"proposals":[],"reread_requests":[]}',request=request))
        self.assertFalse(accepts('{"proposals":[{}],"reread_requests":[]}',request=request))
        self.assertFalse(accepts('{"proposals":[],"reread_requests":[{}]}',request=request))
    def test_cross_field_candidate_ids_and_roles_are_blocked_by_tokens(self):
        request=prompt(two_fields=True)
        for candidate,role in [('p1','document'),('c1','payer'),('made-up','document')]:
            output={'proposals':[{'field_key':'passport.number','candidate_ids':[candidate],'party_role':role}],
                    'reread_requests':[]}
            self.assertFalse(accepts(json.dumps(output,separators=(',',':')),request=request))
    def test_unknown_fields_extra_value_and_missing_key_are_not_allowed(self):
        output={'proposals':[{'field_key':'passport.number','candidate_ids':['c1'],'party_role':'document'}],
                    'reread_requests':[]}
        for mutate in [lambda x:x['proposals'][0].update(field_key='unlisted.field'),
                       lambda x:x['proposals'][0].update(raw_value='123456'),
                       lambda x:x.pop('reread_requests')]:
            value=json.loads(json.dumps(output));mutate(value)
            self.assertFalse(accepts(json.dumps(value,separators=(',',':'))))
    def test_unknown_regions_and_reasons_are_blocked(self):
        for region,reason in [('unlisted','missing'),('r1','do_anything')]:
            self.assertFalse(accepts(json.dumps({'proposals':[],'reread_requests':[
              {'field_key':'passport.number','region_ids':[region],'reason_code':reason}]},separators=(',',':'))))
    def test_verifier_constant_schema_rejects_selector_or_explanations(self):
        self.assertTrue(accepts('{"verdict":"correct"}',purpose='verifier'))
        self.assertTrue(accepts('{"verdict":"abstain"}',purpose='verifier'))
        for text in ['{"verdict":"maybe"}','{"verdict":"correct","reason":"looks good"}',
                     '{"proposals":[],"reread_requests":[]}']:
            self.assertFalse(accepts(text,purpose='verifier'))
    def test_multichar_tokens_are_filtered_not_only_single_character_steps(self):
        fn=build_prefix_constraint(data(),prompt())
        allowed=fn(0,[])
        self.assertIn(200,allowed);self.assertNotIn(201,allowed)
        self.assertIn(EOS,fn(0,[200]))
    def test_eos_not_allowed_before_complete_and_library_fallback_is_rejected(self):
        fn=build_prefix_constraint(data(),prompt());self.assertNotIn(EOS,fn(0,[]))
        with patch.object(fn.enforcer,'get_allowed_tokens',return_value=SimpleNamespace(allowed_tokens=[EOS])):
            with self.assertRaisesRegex(ConstraintError,'failed_closed'):fn(0,[])
    def test_prefix_injection_and_reuse_across_prompts_are_rejected(self):
        fn=build_prefix_constraint(data(),prompt());fn(0,[999])
        with self.assertRaises(ConstraintError):fn(0,[999,ord('x')])
        with self.assertRaises(ConstraintError):fn(0,[998])
    def test_bad_server_prompt_is_never_free_json_fallback(self):
        for request in ['please output JSON','{}','{"contract":{},"fields":[],"candidates":[],"regions":[]}',
                        '{"fields":[],"fields":[]}']:
            with self.assertRaises(ConstraintError):build_prefix_constraint(data(),request)
        with self.assertRaises(ConstraintError):build_prefix_constraint(data(),prompt(),purpose='arbitrary')
    def test_constraint_metadata_is_exposed_for_the_trace(self):
        fn=build_prefix_constraint(data(),prompt())
        self.assertEqual(fn.constraint_engine,'lm-format-enforcer/0.11.3')
        self.assertEqual(len(fn.response_schema_sha256),64)
        self.assertEqual(fn.schema_generation_version,'server-allowlist-json-v1')
        self.assertEqual(fn.eos_token_ids,[EOS])
    def test_callback_after_eos_is_refused_instead_of_parsing_special_token(self):
        fn=build_prefix_constraint(data(),prompt());fn(0,[])
        self.assertIn(EOS,fn(0,[200]))
        with self.assertRaisesRegex(ConstraintError,'after_eos'):fn(0,[200,EOS])
    def test_library_config_assignment_cannot_relax_server_limits(self):
        from lmformatenforcer import CharacterLevelParserConfig
        fn=build_prefix_constraint(data(),prompt())
        parser=fn.enforcer.root_parser
        parser.config=CharacterLevelParserConfig(alphabet='abc{}',max_consecutive_whitespaces=999,
                        force_json_field_order=False,max_json_array_length=999)
        self.assertEqual(parser.config.max_consecutive_whitespaces,4)
        self.assertTrue(parser.config.force_json_field_order)
        self.assertEqual(parser.config.max_json_array_length,32)
        self.assertEqual(parser.config.alphabet,'abc{}')

class TinyTokenizer:
    eos_token_id=EOS;all_special_ids=[EOS]
    def __len__(self):return 129
    def encode(self,text,add_special_tokens=False):return [ord(c) for c in text]
    def decode(self,ids):return ''.join(chr(i) for i in ids if i!=EOS)

class CacheTests(unittest.TestCase):
    def test_tokenizer_preprocessing_cached_but_parser_states_are_fresh(self):
        cache=TokenizerDataCache(max_entries=2);tokenizer=TinyTokenizer()
        one=build_prefix_constraint(tokenizer,prompt(),cache=cache);one(0,[])
        two=build_prefix_constraint(tokenizer,prompt(),cache=cache)
        self.assertEqual(cache.builds,1);self.assertIs(one.data,two.data)
        self.assertIsNot(one.enforcer,two.enforcer);self.assertIsNone(two.initial)
        self.assertEqual(one(0,[]),two(0,[]))
    def test_cache_is_bounded_and_dictionary_form_supported(self):
        cache=TokenizerDataCache(max_entries=2)
        for _ in range(3):build_prefix_constraint(TinyTokenizer(),prompt(),cache=cache)
        self.assertEqual(cache.builds,3);self.assertEqual(len(cache.entries),2)
        mapping={};tokenizer=TinyTokenizer()
        build_prefix_constraint(tokenizer,prompt(),cache=mapping);build_prefix_constraint(tokenizer,prompt(),cache=mapping)
        self.assertEqual(mapping['harness_tokenizer_data_cache'].builds,1)

if __name__=='__main__':unittest.main()
