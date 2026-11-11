"""All live claims retain their own format context; no model execution."""
from pathlib import Path
import json
import tempfile
import unittest

from harness_ocr.engine import Engine
from harness_ocr.settings import Settings
from harness_ocr.grounding import build_candidates,resolve_response
from harness_ocr.core import EvidenceError
from jsonschema import validate


class IdentifierSupportTests(unittest.TestCase):
    def project(self,manufacturers):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        engine=Engine(Settings(Path(temporary.name),Path(temporary.name)/'models',device='cpu',dtype='float32'),object())
        contract=engine.compile_contract('D05',subset=['machine.main_identifier'])
        binding=engine.binding(contract,'qwen-0.8b')
        state=engine.documents.create_document('same-literal','a'*64,contract['profile_id'],metadata={'binding':binding})
        for n,manufacturer in enumerate(manufacturers):
            obs={'id':'o'+str(n),'text':'Машина; Изготовитель: '+manufacturer+'; Модель: 331G; PIN: 1T0331GJCGF300253; Заводской номер: 300253',
                'source_sha256':'a'*64,'completed':True,'evidence_kind':'image_region',
                'page':1,'page_id':'p1','source_size':[100,100],'bbox':[0,0,100,100]}
            candidates=build_candidates([obs],contract)
            c=next(iter(candidates.values()))
            selected=resolve_response(json.dumps({'proposals':[{'field_key':'machine.main_identifier',
                'candidate_ids':[c['id']],'party_role':'machine'}],'reread_requests':[]}),candidates,[],contract)
            state=engine.documents.revise('same-literal',observations=[obs],claims=selected['claims'])
        self.assertEqual(bool(state['conflicts']),len(set(manufacturers))>1)
        result=engine.project(state,contract,binding,emit_operational_events=False)
        validate(result['contract_result'],contract['json_schema'])
        return result
    def test_same_literal_in_two_qualified_contexts_can_validate(self):
        result=self.project(['John Deere','John Deere'])
        self.assertEqual(result['fields']['machine.main_identifier']['validation_status'],'pass')
    def test_selecting_qualified_context_does_not_hide_an_unknown_manufacturer(self):
        result=self.project(['Unregistered manufacturer','John Deere'])
        field=result['fields']['machine.main_identifier']
        self.assertEqual(field['validation_status'],'not_checked')
        self.assertEqual(result['conflicts'][0]['reason'],'contradictory_format_context')
        self.assertEqual(result['decision'],'review')
    def test_selecting_unsupported_context_cannot_borrow_a_qualified_one(self):
        result=self.project(['John Deere','Unregistered manufacturer'])
        self.assertEqual(result['fields']['machine.main_identifier']['validation_status'],'not_checked')

    def test_multi_candidate_selection_keeps_each_context_in_a_separate_claim(self):
        for manufacturers in (['John Deere','Other maker'],['Other maker','John Deere']):
            with self.subTest(order=manufacturers),tempfile.TemporaryDirectory() as temp:
                engine,contract,binding,observations,candidates=self.inputs(temp,manufacturers)
                chosen=resolve_response(json.dumps({'proposals':[{'field_key':'machine.main_identifier',
                    'candidate_ids':list(candidates),'party_role':'machine'}],'reread_requests':[]}),candidates,[],contract)
                self.assertEqual(len(chosen['claims']),2)
                state=engine.documents.create_document('multi','a'*64,contract['profile_id'])
                state=engine.documents.revise('multi',observations=observations,claims=chosen['claims'])
                result=engine.project(state,contract,binding,emit_operational_events=False)
                self.assertEqual(result['conflicts'][0]['reason'],'contradictory_format_context')
                self.assertEqual(result['fields']['machine.main_identifier']['validation_status'],'not_checked')
    def test_selecting_only_favorable_candidate_cannot_erase_explicit_context_opposition(self):
        with tempfile.TemporaryDirectory() as temp:
            engine,contract,binding,observations,candidates=self.inputs(temp,['Other maker','John Deere'])
            class Favorable:
                def call(self,op,args,**kwargs):
                    if op!='select':raise AssertionError(op)
                    candidate=next(c for c in candidates.values() if c['raw_value']['items'][0]['association']['observation_id']=='o1')
                    return {'completed':True,'text':json.dumps({'proposals':[{'field_key':'machine.main_identifier',
                        'candidate_ids':[candidate['id']],'party_role':'machine'}],'reread_requests':[]})}
            engine.backend=Favorable()
            result=engine.run('selected','no-file-needed','a'*64,'D05',subset=['machine.main_identifier'],
                shared_optics={'source_sha256':'a'*64,'observations':observations,'regions':[],'completed':True})
            self.assertEqual(result['terminal_status'],'completed',result['errors'])
            self.assertEqual(result['conflicts'][0]['reason'],'contradictory_format_context')
            self.assertEqual(result['decision'],'review')
    def test_core_rejects_extra_optical_sources_and_caller_supplied_context(self):
        with tempfile.TemporaryDirectory() as temp:
            engine,contract,binding,observations,candidates=self.inputs(temp,['John Deere','Other maker'])
            first=next(iter(candidates.values()))
            chosen=resolve_response(json.dumps({'proposals':[{'field_key':'machine.main_identifier',
                'candidate_ids':[first['id']],'party_role':'machine'}],'reread_requests':[]}),candidates,[],contract)['claims'][0]
            engine.documents.create_document('direct','a'*64,contract['profile_id'])
            engine.documents.revise('direct',observations=observations)
            bad={**chosen,'evidence_ids':['o0','o1']}
            with self.assertRaises(EvidenceError):engine.documents.add_claim('direct',bad)
            with self.assertRaises(EvidenceError):engine.documents.add_claim('direct',{**chosen,'format_context_identity':'a'*64})

    @staticmethod
    def inputs(temp,manufacturers):
        engine=Engine(Settings(Path(temp),Path(temp)/'models',device='cpu',dtype='float32'),object())
        contract=engine.compile_contract('D05',subset=['machine.main_identifier']);binding=engine.binding(contract,'qwen-0.8b')
        observations=[]
        for n,name in enumerate(manufacturers):
            observations.append({'id':'o'+str(n),'text':'Машина; Изготовитель: '+name+
                '; Модель: 331G; PIN: 1T0331GJCGF300253; Заводской номер: 300253',
                'source_sha256':'a'*64,'completed':True,'evidence_kind':'image_region',
                'page':1,'page_id':'p1','source_size':[100,100],'bbox':[0,0,100,100]})
        return engine,contract,binding,observations,build_candidates(observations,contract)


if __name__=='__main__':unittest.main()
