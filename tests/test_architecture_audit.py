"""Behavioral regressions for the architecture integration audit; no NN calls."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest
from harness_ocr.grounding import build_candidates,qwen_prompt,resolve_response,validate_literal
from harness_ocr.line_graph import lines
from harness_ocr.profiles import ProfileRegistry,ProfileError
from harness_ocr.profiles.applicability import TrustedApplicabilityRegistry
from harness_ocr.core import DocumentStore

SHA='a'*64

def obs(ident,text,box=None,source=SHA,page='p1'):
    value={'id':ident,'text':text,'completed':True,'source_sha256':source,
      'evidence_kind':'image_region','page':1,'page_id':page,'source_size':[1600,1600],
      'capture_id':ident,'model_id':'cyrillic-ctc'}
    if box is not None:value['bbox']=box;value['geometry_granularity']='line'
    else:value['bbox']=[0,0,1600,1600];value['model_id']='xiaomi'
    return value

class SpatialGroundingAuditTests(unittest.TestCase):
    def setUp(self):self.contract=ProfileRegistry().compile('D01',subset=['passport.number'])
    def eligible(self,observations,contract=None):
        return [c for c in build_candidates(observations,contract or self.contract).values() if c['role_verified']]
    def test_label_followed_by_value_on_next_text_line_keeps_literal_offsets(self):
        observation=obs('o1','Номер паспорта\n123456')
        candidates=self.eligible([observation]);self.assertEqual({c['raw_value'] for c in candidates},{'123456'})
        c=candidates[0];self.assertEqual(c['association_proof']['kind'],'consecutive_text_lines')
        self.assertEqual(observation['text'][c['spans'][0]['start']:c['spans'][0]['end']],'123456')
        self.assertTrue(all(n['bbox'] is None for n in lines([observation])))
    def test_blank_line_is_explicitly_bounded_and_intervening_label_blocks(self):
        self.assertEqual({c['raw_value'] for c in self.eligible([obs('o1','Номер паспорта\n\n123456')])},{'123456'})
        self.assertEqual(self.eligible([obs('o1','Номер паспорта\nТелефон: 123456')]),[])
        self.assertEqual(self.eligible([obs('o1','Номер паспорта\nТелефон 123456')]),[])
        self.assertEqual(self.eligible([obs('o1','Номер паспорта\nКомментарий\n123456')]),[])
    def test_unique_below_ctc_line_has_geometric_proof(self):
        values=[obs('label','Номер паспорта',[10,10,250,40]),obs('value','123456',[10,50,140,80])]
        eligible=self.eligible(values);self.assertEqual(len(eligible),1)
        self.assertEqual(eligible[0]['evidence_ids'],['value']);self.assertEqual(eligible[0]['role_evidence_ids'],['label'])
        self.assertEqual(eligible[0]['association_proof']['kind'],'unique_spatial_neighbor')
    def test_two_spatial_cells_abstain_no_distance_or_value_guess(self):
        values=[obs('label','Номер паспорта',[10,10,250,40]),
                obs('v1','123456',[10,50,140,80]),obs('v2','654321',[160,50,280,80])]
        self.assertEqual(self.eligible(values),[])
    def test_cross_page_source_and_far_cells_do_not_borrow_role(self):
        label=obs('label','Номер паспорта',[10,10,250,40])
        for value in [obs('v','123456',[10,50,140,80],page='p2'),
                      obs('v','123456',[10,50,140,80],source='b'*64),
                      obs('v','123456',[10,800,140,830])]:
            with self.subTest(value=value):self.assertEqual(self.eligible([label,value]),[])
    def test_whole_page_box_is_not_advertised_as_line_geometry(self):
        value=obs('x','123456');nodes=lines([value])
        self.assertIsNone(nodes[0]['bbox']);self.assertFalse(nodes[0]['coordinates_exact_for_line'])
        self.assertEqual(nodes[0]['source_region_bbox'],[0,0,1600,1600])
    def test_typed_match_and_full_area_do_not_create_false_conflicts(self):
        contract=ProfileRegistry().compile('D07',subset=['payment.amount_due'])
        label=contract['field_definitions']['payment.amount_due']['label']
        values=self.eligible([obs('o1',label+': 1 000,00 руб.')],contract)
        self.assertEqual({c['raw_value'] for c in values},{'1 000,00'})
    def test_explicit_passport_name_components_join_whitespace_only(self):
        contract=ProfileRegistry().compile('D01',subset=['holder.full_name'])
        value=obs('o1','Фамилия\nИванов\nИмя\nИван\nОтчество\nИванович')
        candidates=self.eligible([value],contract)
        self.assertEqual({c['raw_value'] for c in candidates},{'Иванов Иван Иванович'})
        c=candidates[0];response=json.dumps({'proposals':[{'field_key':'holder.full_name',
             'candidate_ids':[c['id']],'party_role':'holder'}],'reread_requests':[]})
        claim=resolve_response(response,{c['id']:c},[],contract)['claims'][0]
        with tempfile.TemporaryDirectory() as tmp:
            store=DocumentStore(Path(tmp)/'state.sqlite');store.create_document('d1',SHA,contract['profile_id'])
            snapshot=store.revise('d1',observations=[value],claims=[claim])
            self.assertEqual(snapshot['fields']['holder.full_name']['raw_value'],'Иванов Иван Иванович')
    def test_ambiguous_or_unreadable_name_component_is_not_composed(self):
        c=ProfileRegistry().compile('D01',subset=['holder.full_name'])
        for text in ['Фамилия\nИванов\nИмя\nИван\nОтчество',
                     'Фамилия: Иванов\nФамилия: Петров\nИмя: Иван',
                     'Фамилия: Иванов\nИмя: Ив4н']:
            with self.subTest(text=text):self.assertFalse(self.eligible([obs('o1',text)],c))
    def test_qwen_receives_real_graph_strategy_and_budget_without_paths(self):
        observations=[obs('o1','Номер паспорта\n123456')];c=build_candidates(observations,self.contract)
        prompt=json.loads(qwen_prompt(c,[{'id':'r1','page':1,'path':'/private/path'}],self.contract,
            observations=observations,strategies=[{'card_id':'review-v1'}],remaining_budget={'rereads':1}))
        self.assertEqual(len(prompt['source_line_graph']),2)
        self.assertEqual(prompt['approved_strategies'],[{'card_id':'review-v1'}])
        self.assertEqual(prompt['remaining_server_budget'],{'rereads':1})
        self.assertEqual(prompt['profile_instructions'],self.contract['system_prompt'])
        self.assertNotIn('/private/path',json.dumps(prompt))

class LiteralValidatorAuditTests(unittest.TestCase):
    def check(self,text,kind,status,**extra):
        r=validate_literal(text,{'value_type':kind,**extra});self.assertEqual(r['validation_status'],status,r)
        self.assertEqual(r['canonical_value'],text);self.assertFalse(r['transform_ids']);return r
    def test_money_preserves_literal_decimal_separator_no_invented_kopecks(self):
        for text in ['1 000,00','1000.00','0,50']:self.check(text,'money_decimal','pass')
        for text in ['1000','10 00,00','1000,0','1.000,00','1000,00 руб.']:self.check(text,'money_decimal','fail')
    def test_names_keep_yo_case_and_letters_and_reject_missing_or_digits(self):
        self.check('Фёдоров Пётр','person_name','pass')
        self.check('Иванов','person_name','fail');self.check('Иванов Ив4н','person_name','fail')
    def test_snils_checksum_and_legacy_are_distinct(self):
        # Fixed known checksum example, not an OCR corpus/gold reference.
        self.check('112-233-445 95','snils','pass');self.check('11223344594','snils','fail')
        self.check('001-001-998 00','snils','not_checked')
    def test_uin_checksum_fallback_and_no_digit_repair(self):
        for length in [20,25]:
            for prefix in ['1'+'0'*(length-2),'123456789012345678901234'[:length-1]]:
                valid=[]
                for last in '0123456789':
                    value=prefix+last
                    if validate_literal(value,{'value_type':'uin'})['validation_status']=='pass':valid.append(value)
                self.assertEqual(len(valid),1)
                self.check(valid[0],'uin','pass');wrong=valid[0][:-1]+str((int(valid[0][-1])+1)%10)
                self.check(wrong,'uin','fail')
        self.check('0'*20,'uin','fail')
    def test_structured_and_versioned_values_do_not_fake_validation(self):
        for kind in ['typed_identifier','typed_bank_account_list','record_list','versioned_policy_identifier']:
            self.check('1234567890123456',kind,'not_checked')
        self.check('бессрочно','date_or_literal','not_checked')
        self.check('бессрочно','date_or_literal','pass',allowed_literals=['бессрочно'])
    def test_all_declared_types_have_explicit_validation_scope(self):
        registry=ProfileRegistry();unsupported=set()
        for p in registry.list():
            c=registry.compile(p['profile_id'])
            for key,spec in c['field_definitions'].items():
                self.assertIn('validation_scope',c['validator_configuration'][key])
                if c['validator_configuration'][key]['requires_typed_contract']:unsupported.add(spec['value_type'])
        self.assertEqual(unsupported,{'typed_identifier','typed_bank_account_list','record_list','versioned_policy_identifier'})

class ApplicabilityAuditTests(unittest.TestCase):
    def setUp(self):
        self.registry=ProfileRegistry();self.declaration=self.registry.get('D01')
        self.conditional='holder.birth_place'
        self.entry={'profile_id':self.declaration['profile_id'],'profile_version':'1.0.0',
            'input_scope_version':'fixed-service-scope','configuration_version':'1.0.0',
            'condition_context':{self.conditional:False},'basis':'fixed_service_contract','attestation_sha256':'b'*64}
    def trusted(self,entry=None):return TrustedApplicabilityRegistry({'schema_version':'harness-applicability/1','entries':[entry or self.entry]})
    def test_default_and_wrong_scope_keep_unknown_conditions(self):
        for registry,scope in [(TrustedApplicabilityRegistry(),'fixed-service-scope'),(self.trusted(),'other-scope')]:
            c=registry.compile(self.registry,'D01',input_scope_version=scope)
            self.assertIn(self.conditional,c['unknown_conditionals'])
    def test_explicit_server_context_is_bound_and_never_derived_from_missing_ocr(self):
        trusted=self.trusted();c=trusted.compile(self.registry,'D01',input_scope_version='fixed-service-scope')
        self.assertFalse(c['conditionals'][self.conditional]['applies'])
        self.assertIn(self.conditional,c['not_applicable_fields'])
        self.assertTrue(c['applicability_configuration']['matched'])
        changed=copy.deepcopy(self.entry);changed['configuration_version']='2.0.0'
        other=self.trusted(changed).compile(self.registry,'D01',input_scope_version='fixed-service-scope')
        self.assertNotEqual(c['config_sha256'],other['config_sha256'])
    def test_required_field_or_unknown_rule_cannot_be_downgraded(self):
        for context in [{'passport.number':False},{'not_a_field':False}]:
            entry=copy.deepcopy(self.entry);entry['condition_context']=context
            with self.assertRaises(ProfileError):self.trusted(entry).compile(self.registry,'D01',input_scope_version='fixed-service-scope')
        entry=copy.deepcopy(self.entry);entry['condition_context']={self.conditional:'infer_from_ocr'}
        with self.assertRaises(ProfileError):self.trusted(entry)

if __name__=='__main__':unittest.main()
