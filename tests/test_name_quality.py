"""Name quality and zero-inference final projection regressions.

The integration tests below use Engine.project on saved-state-shaped data and
a backend that raises on every operation. No OCR or model inference is needed.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness_ocr.engine import Engine
from harness_ocr.name_quality import name_script_uncertainty, VERSION
from harness_ocr.settings import Settings


NAME={'key':'payer.name','value_type':'party'}
MIXED='Лeсникoва Марина Дeнисoвна'


class NameQualityTests(unittest.TestCase):
    def test_observed_case_signals_positions_without_suggesting_a_value(self):
        value=name_script_uncertainty(MIXED,NAME)
        self.assertEqual([i for word in value['words'] for i in word['latin_offsets']],[1,6,18,22])
        self.assertEqual(value['version'],VERSION)
        self.assertTrue(value['requires_review'])
        self.assertFalse(value['value_authority'])
        self.assertFalse(value['normalization_applied'])
        self.assertNotIn('corrected_value',value)
        self.assertNotIn('canonical_value',value)

    def test_separate_words_and_entirely_single_script_names_are_unchanged(self):
        for raw in ('ООО ACME','ООО\u00a0ACME','John Smith','Иванов Иван Иванович',
                    'Jean-Luc Петров','Егор Ёлкин','ACME-ООО'):
            with self.subTest(raw=raw):self.assertIsNone(name_script_uncertainty(raw,NAME))

    def test_signal_is_bidirectional_and_does_not_assert_which_script_is_wrong(self):
        for raw in ('Ивaнов Иван','AСME Holdings'):
            with self.subTest(raw=raw):
                value=name_script_uncertainty(raw,NAME)
                self.assertTrue(value['requires_review'])
                self.assertTrue(value['does_not_prove_ocr_error'])

    def test_legitimate_intra_word_brand_is_review_false_positive_not_repaired(self):
        value=name_script_uncertainty('ОООACME',{'key':'payee.name','value_type':'multispan_text'})
        self.assertIsNotNone(value)
        self.assertEqual(value['scope'],'generic_name_literal')

    def test_only_semantic_name_scope_not_any_field_containing_the_word_name(self):
        for spec in ({'key':'machine.name_make_model','value_type':'multispan_text'},
                     {'key':'principal.identity','value_type':'party'},
                     {'key':'payer.inn','value_type':'inn'},
                     {'key':'passport.number','value_type':'digits_6'},
                     {'key':'text.name','value_type':'text'}):
            with self.subTest(spec=spec):self.assertIsNone(name_script_uncertainty(MIXED,spec))
        for spec in ({'key':'holder.full_name','value_type':'person_name'},
                     {'key':'organization','value_type':'organization_name'},
                     {'key':'payee.bank_name','value_type':'multispan_text'}):
            with self.subTest(spec=spec):self.assertIsNotNone(name_script_uncertainty(MIXED,spec))

    def test_combining_marks_keep_the_run_without_normalizing_codepoints(self):
        raw='И\u0301вaнов Иван'
        value=name_script_uncertainty(raw,NAME)
        self.assertEqual(value['words'][0]['latin_offsets'],[3])
        self.assertEqual(raw,'И\u0301вaнов Иван')
        self.assertIsNone(name_script_uncertainty('И\u0301ванов Иван',NAME))

    def test_bounded_and_fresh_without_global_output_state(self):
        value=name_script_uncertainty(('Ивaнов '*100).strip(),NAME)
        self.assertEqual(len(value['words']),32);self.assertTrue(value['truncated'])
        value['words'].clear()
        self.assertEqual(len(name_script_uncertainty(MIXED,NAME)['words']),2)
        for raw in (None,{},'',123):
            with self.subTest(kind=type(raw).__name__):self.assertIsNone(name_script_uncertainty(raw,NAME))

    def test_oversize_is_explicit_incomplete_review_with_zero_scan_or_hash(self):
        raw='А'*4096+'Лeсникoва'
        with patch('harness_ocr.name_quality._word_spans',side_effect=AssertionError('cannot scan oversize')), \
             patch('harness_ocr.name_quality.hashlib.sha256',side_effect=AssertionError('cannot hash oversize')):
            value=name_script_uncertainty(raw,NAME)
        self.assertEqual(value['code'],'name_quality_check_incomplete_oversize')
        self.assertEqual(value['check_status'],'incomplete');self.assertEqual(value['scanned_codepoints'],0)
        self.assertTrue(value['requires_review']);self.assertFalse(value['value_authority'])
        self.assertNotIn('words',value);self.assertNotIn('table_sha256',value)

    def test_json_unicode_escaping_cannot_change_the_signal_or_literal(self):
        native=json.loads(json.dumps(MIXED,ensure_ascii=False))
        escaped=json.loads(json.dumps(MIXED,ensure_ascii=True))
        self.assertEqual(native,escaped)
        self.assertEqual(name_script_uncertainty(native,NAME),name_script_uncertainty(escaped,NAME))


class NoInference:
    def __init__(self):self.calls=[]
    def call(self,op,*args,**kwargs):
        self.calls.append(op)
        raise AssertionError('post-inference projection attempted a model operation:'+op)


class ProjectionGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name)
        self.backend=NoInference()
        self.engine=Engine(Settings(root/'state',root/'models',device='cpu',dtype='float32',
                                    visual_verifier=False),self.backend)
        # Explicit synthetic SERVER applicability makes payer.name required.
        # Inferring applicability from OCR is deliberately not part of the guard.
        self.contract=self.engine.profiles.compile('D07',subset=['payer.name'],
            condition_context={'payer.name':True})
        # Deliberately permissive fake gateway proves the explicit final guard
        # survives a favorable policy result. No qualified policy is invented.
        self.engine.gateway.apply=lambda result,binding:{'decision':'accept_for_autofill',
            'reason_codes':[],'profile_qualified':False,'all_required_fields_validated':True}

    def snapshot(self,raw=MIXED):
        return {'document_id':'saved-dry-state','source_sha256':'a'*64,'revision':2,
            'fields':{'payer.name':{'raw_value':raw,'status':'supported','needs_review':False,
                'claim_ids':['c1'],'evidence_ids':['o1'],'party_role':'payer','origin':'rule'}},
            'observations':{'o1':{'id':'o1','text':raw,'source_sha256':'a'*64,
                'completed':True,'active':True,'model_id':'cyrillic-ctc','capture_id':'same-crop'}},
            'claims':{'c1':{'id':'c1','field_key':'payer.name','raw_value':raw,'status':'supported',
                           'evidence_ids':['o1'],'origin':'rule'}},'conflicts':[]}

    def test_posthoc_reprojection_changes_review_only_never_reading_or_business_fields(self):
        saved=json.loads(json.dumps(self.snapshot(),ensure_ascii=False));original=copy.deepcopy(saved)
        with patch('harness_ocr.name_quality.name_script_uncertainty',return_value=None):
            before=self.engine.project(saved,self.contract,{},emit_operational_events=False)
        after=self.engine.project(saved,self.contract,{},emit_operational_events=False)
        self.assertEqual(saved,original);self.assertEqual(self.backend.calls,[])
        immutable=('raw_value','value','canonical_value','transform_ids','extraction_status',
                   'validation_status','validation_reasons','validation_scope','validator_id',
                   'party_role','claim_ids','evidence_ids')
        for key in immutable:
            self.assertEqual(before['fields']['payer.name'].get(key),after['fields']['payer.name'].get(key),key)
        for key in ('observations','conflicts','revision','completed','terminal_status'):
            self.assertEqual(before[key],after[key],key)
        for key in ('missing_required','invalid_required','wrong_role_fields',
                    'unknown_conditionals','unexpected_applicable_fields'):
            self.assertEqual(before['completion_report'][key],after['completion_report'][key],key)
        field=after['fields']['payer.name']
        self.assertEqual(field['raw_value'],MIXED);self.assertEqual(field['validation_status'],'pass')
        self.assertEqual(field['extraction_status'],'observed')
        self.assertTrue(field['quality_review_required']);self.assertTrue(field['needs_review'])
        self.assertEqual(field['decision'],'review');self.assertEqual(after['decision'],'review')
        self.assertFalse(after['all_required_fields_validated'])
        self.assertFalse(after['completion_report']['contract_complete'])
        self.assertTrue(after['completion_report']['requires_review'])
        self.assertEqual(after['completion_report']['quality_review_required_fields'],['payer.name'])
        self.assertIn('unicode_name_script_uncertainty:payer.name',after['reason_codes'])
        self.assertEqual(after['contract_result']['fields']['payer.name']['decision'],'review')

    def test_clean_separate_words_have_identical_projection_with_guard_on_or_off(self):
        saved=self.snapshot('ООО ACME')
        with patch('harness_ocr.name_quality.name_script_uncertainty',return_value=None):
            before=self.engine.project(saved,self.contract,{},emit_operational_events=False)
        after=self.engine.project(saved,self.contract,{},emit_operational_events=False)
        self.assertEqual(before,after);self.assertEqual(self.backend.calls,[])

    def test_alternative_favorable_text_cannot_replace_saved_mixed_literal(self):
        saved=self.snapshot();saved['observations']['o2']={**saved['observations']['o1'],
            'id':'o2','text':'Лесникова Марина Денисовна','model_id':'other-reader'}
        original=copy.deepcopy(saved)
        after=self.engine.project(saved,self.contract,{},emit_operational_events=False)
        self.assertEqual(after['fields']['payer.name']['raw_value'],MIXED)
        self.assertEqual(after['observations'],original['observations'])
        self.assertEqual(saved,original);self.assertEqual(self.backend.calls,[])

    def test_required_oversize_with_mixed_word_beyond_bound_cannot_bypass_review(self):
        raw='А'*4096+'Лeсникoва'
        saved=self.snapshot(raw);original=copy.deepcopy(saved)
        with patch('harness_ocr.name_quality.name_script_uncertainty',return_value=None):
            before=self.engine.project(saved,self.contract,{},emit_operational_events=False)
        after=self.engine.project(saved,self.contract,{},emit_operational_events=False)
        field=after['fields']['payer.name']
        self.assertEqual(saved,original);self.assertEqual(self.backend.calls,[])
        self.assertEqual(field['raw_value'],raw)
        for key in ('canonical_value','extraction_status','validation_status','validation_reasons','validator_id'):
            self.assertEqual(before['fields']['payer.name'][key],field[key],key)
        self.assertEqual(field['name_quality_signal']['code'],'name_quality_check_incomplete_oversize')
        self.assertTrue(field['quality_review_required']);self.assertTrue(field['needs_review'])
        self.assertEqual(field['decision'],'review');self.assertEqual(after['decision'],'review')
        self.assertFalse(after['all_required_fields_validated'])
        self.assertFalse(after['completion_report']['contract_complete'])
        self.assertIn('unicode_name_quality_check_incomplete:payer.name',after['reason_codes'])


if __name__=='__main__':unittest.main()
