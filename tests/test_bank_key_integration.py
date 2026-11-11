"""Source-grounded bank context and public result contracts; no NN calls."""
from copy import deepcopy
import unittest

from jsonschema import Draft202012Validator,ValidationError

from harness_ocr.grounding import build_candidates,validate_literal
from harness_ocr.profiles import ProfileRegistry,project_contract_result
from harness_ocr.typed_fields import typed_supported,value_identity,typed_value_schema,RULES


FIELD='payee.accounts'
ACCOUNT='40602810700000000025'
BIK='049805746'


class BankKeyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.contract=ProfileRegistry().compile('D07',subset=[FIELD])
        self.spec=self.contract['field_definitions'][FIELD]
    def extract(self,text):
        obs={'id':'o1','text':text,'completed':True,'active':True,
             'source_sha256':'a'*64,'page':1,'page_id':'p1'}
        candidates=build_candidates([obs],self.contract)
        return obs,candidates
    def one(self,text):
        obs,candidates=self.extract(text)
        self.assertEqual(len(candidates),1,candidates)
        raw=next(iter(candidates.values()))['raw_value']
        return obs,raw,validate_literal(raw,self.spec,observations=[obs])
    @staticmethod
    def line(label='расчетный счет в кредитной организации',account=ACCOUNT,bik=BIK):
        return f'Получатель {label}: {account}; БИК: {bik}'
    def test_explicit_bank_location_is_grounded_in_the_qualified_label(self):
        obs,raw,result=self.one(self.line())
        item=raw['items'][0]
        self.assertEqual(item['account_location']['kind'],'credit_institution')
        span=item['account_location']['spans'][0]
        self.assertEqual(obs['text'][span['start']:span['end']],'расчетный счет в кредитной организации')
        self.assertTrue(typed_supported(raw,[obs]))
        self.assertEqual(result['validation_status'],'pass')
        check=next(c for c in result['validation_checks'] if c['check_id'].startswith('RU20-CBR515-1997-v1:'))
        self.assertEqual(check['status'],'pass')
        self.assertEqual(check['scope'],'arithmetic_consistency_only')
        self.assertEqual(check['evidence_ids'],['o1'])
        self.assertEqual(len(check['source_ids']),3)
    def test_generic_short_labels_do_not_establish_opening_location(self):
        for label in ('расчетный счет','р/с'):
            with self.subTest(label=label):
                _,raw,result=self.one(self.line(label))
                self.assertNotIn('account_location',raw['items'][0])
                self.assertEqual(result['validation_status'],'not_checked')
                self.assertTrue(any('account_opening_location_not_established' in reason for reason in result['validation_reasons']))
    def test_published_rkc_client_location_cannot_use_last_three_bik_digits(self):
        text=self.line('расчетный счет в РКЦ','40102810100000010001','040305000')
        _,raw,result=self.one(text)
        self.assertEqual(raw['items'][0]['account_location']['kind'],'bank_of_russia_unit')
        self.assertEqual(result['validation_status'],'pass')
        _,_,wrong=self.one(self.line('расчетный счет в кредитной организации','40102810100000010001','040305000'))
        self.assertEqual(wrong['validation_status'],'fail')
    def test_location_and_bik_never_borrow_from_another_row(self):
        text=f'Получатель расчетный счет: {ACCOUNT}\nСчет в кредитной организации, БИК: {BIK}'
        _,raw,result=self.one(text)
        self.assertNotIn('account_location',raw['items'][0])
        self.assertNotIn('bik',raw['items'][0])
        self.assertEqual(result['validation_status'],'not_checked')
    def test_bik_must_be_the_whole_bounded_cell(self):
        for suffix in ('X','-1',' или 044525225','/044525225',' 123'):
            with self.subTest(suffix=suffix):
                obs,candidates=self.extract(self.line(bik=BIK+suffix))
                for candidate in candidates.values():
                    result=validate_literal(candidate['raw_value'],self.spec,observations=[obs])
                    self.assertNotEqual(result['validation_status'],'pass')
    def test_multiple_bik_labels_do_not_select_the_one_matching_checksum(self):
        _,raw,result=self.one(self.line()+f'; БИК: 044525225')
        self.assertNotIn('bik',raw['items'][0])
        self.assertNotEqual(result['validation_status'],'pass')
    def test_explicit_negation_or_conflicting_opening_location_is_not_positive_context(self):
        values=[self.line().replace('Получатель расчетный','Получатель НЕ расчетный'),
                self.line()+'; счет открыт в РКЦ']
        for text in values:
            with self.subTest(text=text):
                obs,candidates=self.extract(text)
                for candidate in candidates.values():
                    result=validate_literal(candidate['raw_value'],self.spec,observations=[obs])
                    self.assertNotEqual(result['validation_status'],'pass')
    def test_opening_location_metadata_cannot_be_added_to_a_short_label(self):
        obs,raw,_=self.one(self.line('р/с'))
        forged=deepcopy(raw)
        forged['items'][0]['account_location']={'kind':'credit_institution','spans':deepcopy(forged['items'][0]['kind_spans'])}
        self.assertFalse(typed_supported(forged,[obs]))
        self.assertEqual(validate_literal(forged,self.spec,observations=[obs])['validation_status'],'fail')
    def test_location_kind_and_proof_span_tampering_are_rejected(self):
        obs,raw,_=self.one(self.line())
        for change in ('kind','span'):
            forged=deepcopy(raw)
            if change=='kind':forged['items'][0]['account_location']['kind']='bank_of_russia_unit'
            else:forged['items'][0]['account_location']['spans'][0]['start']+=1
            with self.subTest(change=change):
                self.assertFalse(typed_supported(forged,[obs]))
                self.assertEqual(validate_literal(forged,self.spec,observations=[obs])['validation_status'],'fail')
    def test_location_is_part_of_the_semantic_value_identity(self):
        _,raw,_=self.one(self.line())
        changed=deepcopy(raw);changed['items'][0]['account_location']['kind']='bank_of_russia_unit'
        self.assertNotEqual(value_identity(raw),value_identity(changed))
    def test_public_result_schema_keeps_per_check_scope_and_evidence(self):
        _,raw,checked=self.one(self.line())
        runtime={'needs_review':True,'fields':{FIELD:{'raw_value':raw,**checked,
            'party_role':'payee','extraction_status':'observed','decision':'review','evidence_ids':['o1']}}}
        result=project_contract_result(self.contract,runtime)
        Draft202012Validator(self.contract['json_schema']).validate(result)
        checks=result['fields'][FIELD]['validation_checks']
        self.assertEqual(checks,checked['validation_checks'])
        self.assertEqual(checks[0]['scope'],'arithmetic_consistency_only')
    def test_unrelated_account_kinds_cannot_carry_bank_opening_metadata(self):
        _,raw,_=self.one(self.line())
        forged=deepcopy(raw);item=forged['items'][0]
        item.update(kind='treasury',version=RULES['treasury']['version'])
        self.assertFalse(typed_supported(forged,[]))
        with self.assertRaises(ValidationError):Draft202012Validator(typed_value_schema(self.spec)).validate(forged)


if __name__=='__main__':unittest.main()
