"""Cross-component source contracts; synthetic fixtures, no neural quality claim."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from harness_ocr.engine import Engine
from harness_ocr.settings import Settings
from harness_ocr.ingest import decode
from harness_ocr.util import file_sha,digest
from harness_ocr.inference.vision import make_visual_region,parse_visual_output,prepare_visual_regions
from harness_ocr.benchmark import score
from harness_ocr.typed_fields import value_identity
from harness_ocr.core import EvidenceError
from test_mchd import synthetic_xml


class SourceBackend:
    def __init__(self,settings,*,literal='Номер паспорта: 123456',visual_literal=None):
        self.settings=settings;self.literal=literal;self.visual_literal=visual_literal or literal;self.calls=[]
    def call(self,op,args,**kwargs):
        self.calls.append((op,deepcopy(args)))
        if op=='ingest':return decode(args['path'],args['output'],self.settings,profile_id=args.get('profile_id'))
        if op=='cheap':
            page=args['page'];w,h=page['source_size']
            region={'id':'p1_r1','path':page['path'],'page':1,'page_id':'p1','source_size':[w,h],
                'bbox':[0,0,w,h],'crop_sha256':page['raster_sha256'],'transform':{'version':'axis-aligned-crop-v1'}}
            observation={'id':'p1_ctc1','text':self.literal,'evidence_kind':'image_region',
                'source_sha256':args['source_sha256'],'page':1,'page_id':'p1','source_size':[w,h],
                'bbox':[0,0,w,h],'completed':True,'capture_id':'ctc-fixture','model_id':'fixture'}
            # Engine's production association is exact quad geometry.
            quad=[[0,0],[w,0],[w,h],[0,h]]
            region['quad']=quad;observation['quad']=quad
            # Exercise valid axis-aligned descriptor in a frozen optical replay.
            return {'observations':[observation],'regions':[region],'completed':True}
        if op=='ocr':return {'text':self.literal,'completed':True}
        if op=='select':
            req=json.loads(args['prompt']);choices=[c for c in req['candidates'] if c['role_anchor_observed']]
            proposals=[]
            for field in req['fields']:
                found=next((c for c in choices if c['field_key']==field['key']),None)
                if found:proposals.append({'field_key':found['field_key'],'candidate_ids':[found['id']],'party_role':found['party_role']})
            return {'text':json.dumps({'proposals':proposals,'reread_requests':[]}), 'completed':True}
        if op=='visual_read':
            _,provenance=prepare_visual_regions(args['regions'])
            # Server response fixture: the full literal is never an argument to visual_read.
            out={'text':json.dumps({'readings':{'r'+str(i):{'status':'readable','literal':self.visual_literal}
                for i,p in enumerate(provenance,1)}}),'completed':True,'stop_reason':'eos','model_id':'fixture',
                'model_bundle_sha256':'b'*64,'modality':'vision','purpose':'blind_optical_capture'}
            out['captures']=parse_visual_output(out,provenance)
            return out
        raise AssertionError(op)


class Rc3IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def settings(self,visual=False):
        return Settings(self.root/'state',self.root/'models',device='cpu',dtype='float32',visual_verifier=visual)
    def frozen(self,settings,text):
        from PIL import Image
        path=self.root/'source.png';Image.new('RGB',(320,80),'white').save(path)
        data=decode(path,self.root/'rasters',settings);page=data['pages'][0]
        obs={'id':'o1','text':text,'source_sha256':data['source_sha256'],'evidence_kind':'image_region',
            'page':1,'page_id':'p1','source_size':page['source_size'],'bbox':[0,0,320,80],
            'completed':True,'capture_id':'synthetic','model_id':'fixture'}
        region=make_visual_region(path,page,source_sha256=data['source_sha256'])
        return path,{'source_sha256':data['source_sha256'],'observations':[obs], 'regions':[],
            'visual_regions':{'o1':region},'completed':True}
    def test_native_registered_source_goes_through_engine_without_neural_calls(self):
        settings=self.settings();path=self.root/'mchd.xml';path.write_bytes(synthetic_xml(organization=True))
        backend=SourceBackend(settings);engine=Engine(settings,backend)
        result=engine.run('native-demo',path,file_sha(path),'D04')
        self.assertEqual(result['terminal_status'],'completed',result['errors'])
        self.assertEqual([op for op,_ in backend.calls],['ingest'])
        self.assertIsInstance(result['fields']['principal.identifier']['raw_value'],dict)
        self.assertEqual(result['fields']['principal.identifier']['validation_status'],'pass')
        self.assertEqual(result['fields']['representative.identifier']['validation_status'],'pass')
        self.assertEqual(result['native_ingestion']['structural_validation_status'],'pass')
        self.assertFalse(result['native_ingestion']['signature_verified'])
        self.assertEqual(result['decision'],'review')
        from jsonschema import Draft202012Validator
        Draft202012Validator(engine.compile_contract('D04')['json_schema']).validate(result['contract_result'])
    def test_forged_native_observation_cannot_enter_store_without_original(self):
        settings=self.settings();path=self.root/'mchd.xml';path.write_bytes(synthetic_xml())
        from harness_ocr.native import decode_mchd
        issued=decode_mchd(path);engine=Engine(settings,SourceBackend(settings))
        engine.documents.create_document('unbound',file_sha(path),'D04')
        with self.assertRaises(EvidenceError):engine.documents.revise('unbound',observations=issued['observations'])
    def test_partial_native_recovery_keeps_ordered_parties_restrictions_and_trust_flags(self):
        settings=self.settings();path=self.root/'mchd.xml';path.write_bytes(synthetic_xml(principals=2,machine=True))
        engine=Engine(settings,SourceBackend(settings));result=engine.run('native-many',path,file_sha(path),'D04')
        snapshot=engine.documents.get('native-many')
        recovered=engine.project(snapshot,engine.compile_contract('D04'),snapshot['metadata']['binding'],
            terminal='interrupted',complete=False)
        self.assertEqual(len(recovered['native_ingestion']['native_model']['principals']),2)
        self.assertEqual(recovered['native_ingestion']['native_model']['machine_powers'],
            result['native_ingestion']['native_model']['machine_powers'])
        self.assertFalse(recovered['native_ingestion']['signature_verified'])
        self.assertIsNone(recovered['fields'].get('principal.identity',{}).get('raw_value'))
        self.assertIn('native_signature_revocation_authority_not_verified',recovered['reason_codes'])
    def test_review_cas_rejects_coerced_revision_types(self):
        from harness_ocr.api import Correction
        from pydantic import ValidationError
        data={'field_key':'passport.number','raw_value':'123456','evidence_ids':['o1'],'expected_revision':1}
        self.assertEqual(Correction(**data).expected_revision,1)
        for revision in (True,'1',1.0):
            with self.subTest(revision=revision),self.assertRaises(ValidationError):Correction(**{**data,'expected_revision':revision})
    def test_visual_full_observation_match_persists_fresh_evidence(self):
        settings=self.settings(True);path,frozen=self.frozen(settings,'Номер паспорта: 123456')
        backend=SourceBackend(settings);engine=Engine(settings,backend)
        result=engine.run('visual-match',path,file_sha(path),'D01',subset=['passport.number'],shared_optics=frozen)
        self.assertEqual(result['terminal_status'],'completed',result['errors'])
        self.assertEqual(result['fields']['passport.number']['raw_value'],'123456')
        self.assertTrue(result['fields']['passport.number']['visual_verification_complete'])
        self.assertEqual(len(result['visual_checks']),1)
        check=next(iter(result['visual_checks'].values()))
        self.assertEqual(check['verdict'],'correct')
        self.assertEqual(check['comparison_scope'],'complete_original_observation_literal')
        visual_args=next(args for op,args in backend.calls if op=='visual_read')
        self.assertEqual(set(visual_args),{'name','regions'})
        self.assertNotIn('123456',json.dumps(visual_args))
        self.assertEqual(result['decision'],'review')
    def test_same_exact_roi_for_two_readings_uses_one_capture_and_covers_both(self):
        settings=self.settings(True);path,frozen=self.frozen(settings,'Номер паспорта: 123456')
        second=deepcopy(frozen['observations'][0]);second.update(id='o2',capture_id='other-reader')
        frozen['observations'].append(second);frozen['visual_regions']['o2']=deepcopy(frozen['visual_regions']['o1'])
        class BothReaders(SourceBackend):
            def call(self,op,args,**kwargs):
                if op=='select':
                    self.calls.append((op,deepcopy(args)));req=json.loads(args['prompt'])
                    ids=[c['id'] for c in req['candidates'] if c['role_anchor_observed']]
                    return {'completed':True,'text':json.dumps({'proposals':[{'field_key':'passport.number',
                        'candidate_ids':ids,'party_role':'document'}],'reread_requests':[]})}
                return super().call(op,args,**kwargs)
        backend=BothReaders(settings);engine=Engine(settings,backend)
        result=engine.run('visual-shared-roi',path,file_sha(path),'D01',subset=['passport.number'],shared_optics=frozen)
        self.assertEqual(result['terminal_status'],'completed',result['errors'])
        self.assertTrue(result['fields']['passport.number']['visual_verification_complete'])
        self.assertEqual(len(result['visual_checks']),2)
        self.assertEqual(len({c['capture_id'] for c in result['visual_checks'].values()}),1)
        self.assertEqual(sum(op=='visual_read' for op,_ in backend.calls),1)
    def test_whole_page_observation_gets_a_real_whole_page_visual_descriptor(self):
        from PIL import Image
        settings=self.settings(True);path=self.root/'page-only.png';Image.new('RGB',(320,80),'white').save(path)
        class PageOnly(SourceBackend):
            def call(self,op,args,**kwargs):
                if op=='cheap':
                    self.calls.append((op,deepcopy(args)))
                    return {'observations':[],'regions':[],'completed':True}
                return super().call(op,args,**kwargs)
        backend=PageOnly(settings);engine=Engine(settings,backend)
        result=engine.run('page-only',path,file_sha(path),'D01',subset=['passport.number'])
        self.assertEqual(result['terminal_status'],'completed',result['errors'])
        self.assertTrue(result['fields']['passport.number']['visual_verification_complete'])
        check=next(iter(result['visual_checks'].values()))
        self.assertEqual(check['expected_observation_id'],'p1_xiaomi')
        self.assertEqual(sum(op=='visual_read' for op,_ in backend.calls),1)
    def test_visual_disagreement_blocks_field_and_survives_reprojection(self):
        settings=self.settings(True);path,frozen=self.frozen(settings,'Номер паспорта: 123456')
        backend=SourceBackend(settings,visual_literal='Номер паспорта: 654321');engine=Engine(settings,backend)
        result=engine.run('visual-conflict',path,file_sha(path),'D01',subset=['passport.number'],shared_optics=frozen)
        self.assertEqual(result['terminal_status'],'completed',result['errors'])
        self.assertTrue(result['fields']['passport.number']['needs_review'])
        self.assertEqual(result['fields']['passport.number']['raw_value'],'123456')
        snapshot=engine.documents.get('visual-conflict')
        projected=engine.project(snapshot,engine.compile_contract('D01',subset=['passport.number']),snapshot['metadata']['binding'])
        self.assertEqual(projected['decision'],'review')
        self.assertEqual(projected['conflicts'][0]['reason'],'blind_visual_disagreement_or_abstention')
        self.assertEqual(next(iter(projected['visual_checks'].values()))['verdict'],'incorrect')
    def test_withdrawing_visual_capture_revokes_coverage_without_changing_literal(self):
        settings=self.settings(True);path,frozen=self.frozen(settings,'Номер паспорта: 123456')
        engine=Engine(settings,SourceBackend(settings))
        result=engine.run('visual-revoked',path,file_sha(path),'D01',subset=['passport.number'],shared_optics=frozen)
        self.assertTrue(result['fields']['passport.number']['visual_verification_complete'])
        capture=next(iter(result['visual_checks'].values()))['capture_id']
        snapshot=engine.documents.revise('visual-revoked',invalidate=[capture])
        projected=engine.project(snapshot,engine.compile_contract('D01',subset=['passport.number']),snapshot['metadata']['binding'])
        self.assertEqual(projected['fields']['passport.number']['raw_value'],'123456')
        self.assertFalse(projected['fields']['passport.number']['visual_verification_complete'])
        self.assertIn('visual_verification_incomplete:passport.number',projected['reason_codes'])
    def test_typed_score_compares_literals_and_order_without_provenance(self):
        from harness_ocr.grounding import build_candidates
        settings=self.settings();path,frozen=self.frozen(settings,'Доверитель ИНН: 7707083893')
        engine=Engine(settings,SourceBackend(settings))
        candidate=next(iter(build_candidates(frozen['observations'],engine.compile_contract('D04',subset=['principal.identifier'])).values()))
        raw=candidate['raw_value'];expected=value_identity(raw)
        result={'fields':{'principal.identifier':{'raw_value':raw,'extraction_status':'observed','validation_status':'pass'}}}
        self.assertEqual(score(result,{'principal.identifier':expected})['correct'],1)
        bad=deepcopy(expected);bad['items'][0]['raw_value']='7707083894'
        self.assertEqual(score(result,{'principal.identifier':bad})['wrong'],1)


if __name__=='__main__':unittest.main()
