from pathlib import Path
import tempfile
import json
import unittest
from harness_ocr.engine import Engine
from harness_ocr.settings import Settings
from harness_ocr.control.risk import _hard_guard_reasons
from test_engine import FakeBackend,optics,observation,reply,SHA

class EngineRuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.settings=Settings(self.root/'data',self.root/'models',device='cpu',dtype='float32')
    def run_engine(self,backend=None,**kwargs):
        engine=Engine(self.settings,backend or FakeBackend())
        result=engine.run('d1',self.root/'unused.png',SHA,'D01',subset=['passport.number'],
                          shared_optics=kwargs.pop('shared_optics',optics()),**kwargs)
        return engine,result
    def test_operational_audit_deferred_until_committed_publication(self):
        engine,result=self.run_engine(emit_operational_events=False)
        self.assertEqual(result['terminal_status'],'completed',result['errors'])
        self.assertTrue(result['audit']['deferred_until_publication'])
        self.assertEqual(engine.audit_log.events('audit_selection'),[])
    def test_requested_reread_never_silently_accepted_after_budget(self):
        region={'id':'r1','path':'unused.png','page':1,'page_id':'p1','quad':[[0,0],[100,0],[100,100],[0,100]],
                'source_size':[800,1200]}
        backend=FakeBackend(lambda op,args:{'text':reply(args['prompt'],reread=True),'completed':True} if op=='select' else None)
        engine=Engine(self.settings,backend)
        engine.gateway.apply=lambda result,binding:{'decision':'accept_for_autofill','reason_codes':[],'profile_qualified':False}
        result=engine.run('d1',self.root/'unused',SHA,'D01',subset=['passport.number'],
                          shared_optics=optics(regions=[region]),allow_reread=False)
        self.assertEqual(result['decision'],'review')
        self.assertIn('unresolved_reread_requests',result['reason_codes'])
        self.assertEqual(result['unresolved_rereads'][0]['unresolved_reason'],'reread_disabled')
        self.assertNotIn('ocr',backend.calls)
    def test_contradictory_observation_reaches_core_even_selector_prefers_first(self):
        shared=optics();shared['observations'].append(observation('Номер паспорта: 654321','o2'))
        engine,result=self.run_engine(shared_optics=shared)
        self.assertTrue(result['conflicts'])
        self.assertEqual(result['fields']['passport.number']['extraction_status'],'ambiguous')
        self.assertEqual(result['decision'],'review')
        self.assertGreaterEqual(len(engine.documents.get('d1')['claims']),2)
    def test_missing_claim_role_not_fabricated_from_field_definition(self):
        engine=Engine(self.settings,FakeBackend());contract=engine.compile_contract('D01',subset=['passport.number'])
        state=engine.documents.create_document('d1',SHA,contract['profile_id'])
        state=engine.documents.revise('d1',observations=[observation()],claims=[{'id':'legacy',
            'field_key':'passport.number','raw_value':'123456','evidence_ids':['o1'],'origin':'rule'}])
        out=engine.project(state,contract,engine.binding(contract,'qwen-0.8b'))
        self.assertIsNone(out['fields']['passport.number']['party_role'])
    def test_human_literal_does_not_inherit_machine_qualification(self):
        engine,result=self.run_engine()
        state=engine.documents.human_correct('d1','passport.number','654321',['o1'],actor='operator',
                          expected_revision=result['revision'],authorized=True,party_role='document')
        engine.gateway.apply=lambda result,binding:{'decision':'accept_for_autofill','reason_codes':[],'profile_qualified':False}
        contract=engine.compile_contract('D01',subset=['passport.number'])
        out=engine.project(state,contract,engine.binding(contract,'qwen-0.8b'))
        self.assertEqual(out['fields']['passport.number']['raw_value'],'654321')
        self.assertEqual(out['decision'],'review')
        self.assertIn('human_correction_requires_separate_acceptance',out['reason_codes'])
    def test_false_conditional_cannot_disable_a_static_required_field(self):
        result={'completed':True,'terminal_status':'completed','conflicts':[],
          'required_fields':['x'],'triggered_conditional_fields':[],
          'conditional_applicability':{'x':False,'y':False},'observations':{},
          'fields':{'x':{'value':None,'critical':True},'y':{'value':None,'critical':True}}}
        reasons=_hard_guard_reasons({'hard_contract':{'required_fields':['x'],'conditional_fields':['y']}},result)
        self.assertIn('missing_required_field:x',reasons)
        self.assertNotIn('missing_required_field:y',reasons)
    def test_generic_xml_never_inherits_raster_qualification_or_calls_models(self):
        class XMLBackend:
            def __init__(self):self.calls=[]
            def call(self,op,args,**kwargs):
                self.calls.append(op)
                if op!='ingest':raise AssertionError('unqualified XML must not reach a model')
                return {'source_sha256':SHA,'pages':[],'native_bridge':'generic_xml_unqualified',
                    'observations':[{'id':'xml1','source_sha256':SHA,'text':'Номер паспорта: 123456',
                      'evidence_kind':'structured_xml','xpath':'/x','namespace_map':{},
                      'schema_version':'unqualified-generic-xml-v1','structural_validation_status':'not_checked',
                      'completed':True}]}
        backend=XMLBackend();engine=Engine(self.settings,backend)
        result=engine.run('d1',self.root/'unused.xml',SHA,'D01',subset=['passport.number'])
        self.assertEqual(backend.calls,['ingest'])
        self.assertEqual(result['terminal_status'],'failed')
        self.assertIn('unsupported_native_adapter:generic_xml',result['reason_codes'])
        self.assertIn('xml1',result['observations'])
        self.assertIsNone(result['fields']['passport.number']['raw_value'])

if __name__=='__main__':unittest.main()
