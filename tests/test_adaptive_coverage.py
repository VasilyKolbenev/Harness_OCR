"""Semantic value spans cannot borrow a valid whole-observation visual check."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from harness_ocr.engine import Engine
from harness_ocr.inference.verification_plan import (
    plan_verification, validated_value_observation_ids, verified_value_observation_ids)
from harness_ocr.settings import Settings
from harness_ocr.util import digest

SHA='a'*64

class NoInferenceBackend:
    def call(self,*args,**kwargs):
        raise AssertionError('Coverage projection must not call a model')

class AdaptiveCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        settings=Settings(self.root/'state',self.root/'models',device='cpu',dtype='float32',
            reading_policy='adaptive',visual_verifier=True)
        self.engine=Engine(settings,NoInferenceBackend())
        self.contract=self.engine.compile_contract('D01',subset=['passport.series'])
        self.binding=self.engine.binding(self.contract,'qwen-0.8b')
        self.engine.documents.create_document('doc',SHA,self.contract['profile_id'])
        self.observation={'id':'original','source_sha256':SHA,'text':'1234',
            'completed':True,'evidence_kind':'image_region','page':1,'page_id':'p1',
            'source_size':[400,240],'bbox':[10,10,100,30],'model_id':'fixture-source'}
        capture={**self.observation,'id':'capture','purpose':'blind_optical_capture',
            'read_status':'readable','model_id':'fixture-vision'}
        capture['capture_sha256']=digest(capture)
        self.engine.documents.revise('doc',observations=[self.observation,capture])

    def add_claim(self,span,*,cid='claim'):
        return self.engine.documents.revise('doc',claims=[{'id':cid,'field_key':'passport.series',
            'raw_value':'1234','evidence_ids':['original'],'spans':[span],
            'party_role':'document','origin':'rule','depends_on':[]}])

    def add_whole_observation_check(self,cid='claim'):
        return self.engine.documents.revise('doc',visual_checks=[{'id':'check_'+cid,
            'field_key':'passport.series','claim_ids':[cid],'capture_id':'capture',
            'expected_observation_id':'original'}])

    def project(self,state):
        return self.engine.project(state,self.contract,self.binding,emit_operational_events=False)

    def test_wrong_localization_cannot_borrow_correct_whole_observation(self):
        # Core supports the raw literal as an actual substring; its declared
        # location is a distinct contract and is deliberately incorrect here.
        self.add_claim({'observation_id':'original','start':0,'end':1})
        state=self.add_whole_observation_check()
        self.assertEqual(state['visual_checks']['check_claim']['verdict'],'correct')
        self.assertEqual(state['claims']['claim']['status'],'supported')
        with self.assertRaises(ValueError):
            validated_value_observation_ids(state['claims']['claim'],state['observations'])
        self.assertFalse(verified_value_observation_ids(state,'passport.series',state['fields']['passport.series']))
        result=self.project(state)
        self.assertEqual(result['fields']['passport.series']['raw_value'],'1234')
        self.assertFalse(result['fields']['passport.series']['visual_verification_complete'])
        self.assertEqual(result['decision'],'review')
        self.assertIn('visual_value_span_proof_missing:passport.series',result['reason_codes'])

    def test_invalid_offsets_unbound_ids_and_nonwhitespace_joins_get_no_credit(self):
        variants=[{'observation_id':'original','start':0,'end':99},
                  {'observation_id':'not-in-evidence','start':0,'end':4},
                  {'observation_id':'original','start':0,'end':4,'separator_after':'X'}]
        for index,span in enumerate(variants):
            with self.subTest(span=span):
                # Use fresh independent state maps to avoid making the test
                # pass merely because a different malformed claim was present.
                state=self.engine.documents.get('doc')
                claim={'id':'adversarial_'+str(index),'field_key':'passport.series','raw_value':'1234',
                    'evidence_ids':['original'],'spans':[span],'party_role':'document','status':'supported'}
                state['claims']={claim['id']:claim}
                state['fields']={'passport.series':{'raw_value':'1234','claim_ids':[claim['id']],
                    'party_role':'document','needs_review':False}}
                with self.assertRaises(ValueError):validated_value_observation_ids(claim,state['observations'])
                self.assertFalse(verified_value_observation_ids(state,'passport.series',state['fields']['passport.series']))
                self.assertFalse(self.project(state)['fields']['passport.series']['visual_verification_complete'])

    def test_a_good_claim_does_not_mask_another_bad_live_span(self):
        self.add_claim({'observation_id':'original','start':0,'end':4},cid='good')
        self.add_whole_observation_check('good')
        state=self.add_claim({'observation_id':'original','start':0,'end':1},cid='bad')
        self.assertEqual(set(state['fields']['passport.series']['claim_ids']),{'good','bad'})
        self.assertFalse(verified_value_observation_ids(state,'passport.series',state['fields']['passport.series']))
        self.assertFalse(self.project(state)['fields']['passport.series']['visual_verification_complete'])

    def test_valid_complete_span_retains_exact_visual_credit(self):
        self.add_claim({'observation_id':'original','start':0,'end':4})
        state=self.add_whole_observation_check()
        self.assertEqual(validated_value_observation_ids(state['claims']['claim'],state['observations']),{'original'})
        self.assertEqual(verified_value_observation_ids(state,'passport.series',state['fields']['passport.series']),{'original'})
        result=self.project(state)
        self.assertTrue(result['fields']['passport.series']['visual_verification_complete'])
        self.assertEqual(result['fields']['passport.series']['validation_status'],'pass')
        self.assertEqual(result['decision'],'review','Optical proof does not grant profile qualification')

if __name__=='__main__':unittest.main()
