"""Detector selection and source-evidence integration; no weights or NN calls."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from harness_ocr.cli import _settings
from harness_ocr.engine import Engine
from harness_ocr.ingest import decode
from harness_ocr.inference.cheap import CheapReader
from harness_ocr.inference.db_geometry import rectify_quad
from harness_ocr.inference.vision import parse_visual_output, prepare_visual_regions
from harness_ocr.settings import Settings
from harness_ocr.service import Service
from harness_ocr.util import file_sha


class DetectorSelectionTests(unittest.TestCase):
    def test_default_is_db_and_environment_is_explicitly_validated(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(Settings.from_env().text_detector, 'db')
        for name in ('db', 'yolo26l', 'yolo26x'):
            with patch.dict(os.environ, {'HARNESS_TEXT_DETECTOR':name}):
                settings=Settings.from_env()
                settings.validate()
                self.assertEqual(settings.text_detector,name)
        with patch.dict(os.environ, {'HARNESS_TEXT_DETECTOR':'yolo-auto'}):
            with self.assertRaisesRegex(ValueError,'unknown_text_detector'):
                Settings.from_env().validate()

    def test_cli_override_and_environment_do_not_change_recognition_or_checks(self):
        args=SimpleNamespace(models=Path('models'),device='cpu',max_new_tokens=1024,
            call_timeout=120,document_timeout=600,qwen='qwen-0.8b')
        with patch.dict(os.environ, {'HARNESS_TEXT_DETECTOR':'yolo26l'}):
            inherited=_settings(args,Path('state'))
            overridden=_settings(SimpleNamespace(**vars(args),text_detector='yolo26x'),Path('state'))
        self.assertEqual(inherited.text_detector,'yolo26l')
        self.assertEqual(overridden.text_detector,'yolo26x')
        self.assertEqual(replace(inherited,text_detector='db'),replace(overridden,text_detector='db'))
        self.assertEqual((overridden.max_rereads,overridden.max_visual_regions),(2,2))
        self.assertEqual(overridden.policy_version,'review-default-v1')

    def test_detector_choice_changes_acceptance_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Settings(Path(folder)/'state',Path(folder)/'models',device='cpu',dtype='float32')
            hashes=[]
            for name in ('db','yolo26l','yolo26x'):
                engine=Engine(replace(base,text_detector=name),object())
                contract=engine.compile_contract('D01',subset=['passport.number'])
                bound=engine.binding(contract,base.qwen)
                hashes.append(bound['model_bundle_sha256'])
                self.assertEqual(bound['policy_version'],'review-default-v1')
            self.assertEqual(len(set(hashes)),3)

    def test_service_verifies_selected_detector_and_rejects_its_failed_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            class NoModels:
                def close(self):pass
            for name in ('db','yolo26l','yolo26x'):
                settings=Settings(root/name,root/'models',device='cpu',dtype='float32',text_detector=name)
                service=Service(settings,NoModels())
                expected='detector' if name=='db' else name+'-text'
                try:
                    with patch.object(service,'_fingerprint',return_value=('unprepared',)), \
                         patch('harness_ocr.service.verify_models',return_value={'ok':False,'errors':['missing']}) as verify:
                        result=service._ensure_models('qwen-0.8b')
                    self.assertFalse(result['ok'])
                    self.assertEqual(verify.call_args.args[1],[expected,'cyrillic','xiaomi','qwen-0.8b'])
                    self.assertEqual(service._verified,{})
                finally:service.close()


class _Input:
    name='x'


class _CTC:
    def __init__(self,owner):self.owner=owner
    def get_inputs(self):return [_Input()]
    def run(self,*args):
        self.owner.order.append('ctc')
        return [np.array([[[.05,.9,.05],[.9,.05,.05],[.05,.05,.9]]],np.float32)]


class _Yolo:
    def __init__(self,owner,*,ambiguous=False):self.owner=owner;self.ambiguous=ambiguous
    def detect(self,image):
        self.owner.order.append('detector')
        quad=[[2.,2.],[70.,2.],[70.,22.],[2.,22.]]
        return [{'quad':quad,'word_quads':[quad],'word_ids':['w1'],'detector_score':.8}],{
            'contour_limit_reached':False,'grouping_ambiguous':self.ambiguous}


class _Reader(CheapReader):
    def __init__(self,*args,ambiguous=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.order=[];self.ambiguous=ambiguous
    def _load(self):
        self._detector=_Yolo(self,ambiguous=self.ambiguous)
        self._recognizer=_CTC(self);self.chars=['blank','Я','1']


class ExperimentalReaderTests(unittest.TestCase):
    def test_yolo_only_changes_localization_and_ctc_stays_sequential_source_backed(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'page.png';Image.new('RGB',(80,40),'white').save(source)
            sha=file_sha(source)
            page={'page':1,'page_id':'p1','path':str(source),'source_size':[80,40],
                'raster_sha256':sha,'transform':{'source_sha256':sha,'raster_sha256':sha}}
            for index,name in enumerate(('yolo26l','yolo26x')):
                reader=_Reader(root/'models',text_detector=name)
                with patch('harness_ocr.inference.cheap.detector_preprocess',side_effect=AssertionError('DB path called')):
                    result=reader.read(page,sha,root/f'crops{index}')
                self.assertEqual(reader.order,['detector','ctc'])
                self.assertTrue(result['completed'])
                self.assertEqual(result['observations'][0]['text'],'Я1')
                self.assertEqual(result['observations'][0]['model_id'],'cyrillic-ctc')
                self.assertEqual(result['regions'][0]['text_localization']['word_ids'],['w1'])
                proof=result['observations'][0]['raster_transform_proof']
                self.assertEqual(proof['original_source_sha256'],sha)
                self.assertEqual(proof['crop_sha256'],file_sha(result['regions'][0]['path']))
                self.assertFalse(result['diagnostics']['runtime']['models_parallel'])

    def test_uncertain_word_grouping_cannot_claim_optical_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'page.png';Image.new('RGB',(80,40),'white').save(source)
            sha=file_sha(source)
            page={'page':1,'page_id':'p1','path':str(source),'source_size':[80,40],
                'transform':{'source_sha256':sha}}
            reader=_Reader(root/'models',text_detector='yolo26l',ambiguous=True)
            result=reader.read(page,sha,root/'crops')
            self.assertFalse(result['completed'])
            self.assertTrue(result['diagnostics']['needs_review'])
            self.assertEqual(result['stop_reason'],'detector_grouping_incomplete')
            self.assertEqual(result['observations'][0]['text'],'Я1')


class EmptyCtcRereadBackend:
    literal='Номер паспорта: 123456'
    def __init__(self,settings):self.settings=settings;self.calls=[];self.ocr_calls=0
    def call(self,op,args,**kwargs):
        self.calls.append((op,deepcopy(args)))
        if op=='ingest':return decode(args['path'],args['output'],self.settings)
        if op=='cheap':
            page=args['page'];quad=[[5.,5.],[300.,5.],[300.,55.],[5.,55.]]
            with Image.open(page['path']) as image:crop,proof=rectify_quad(image,quad)
            output=Path(args['output_dir']);output.mkdir(parents=True)
            path=output/'line.png';crop.save(path)
            region={'id':'p1_r1','path':str(path),'page':1,'page_id':'p1','source_size':page['source_size'],
                'quad':quad,'bbox':[5.,5.,300.,55.],'crop_sha256':file_sha(path),'transform':proof}
            return {'observations':[],'regions':[region],'completed':True}
        if op=='ocr':
            self.ocr_calls+=1
            return {'text':'' if self.ocr_calls==1 else self.literal,'completed':True}
        if op=='select':
            request=json.loads(args['prompt']);choices=[c for c in request['candidates'] if c['role_anchor_observed']]
            if not choices:
                answer={'proposals':[],'reread_requests':[{'field_key':'passport.number',
                    'region_ids':['p1_r1'],'reason_code':'missing'}]}
            else:
                candidate=choices[0]
                answer={'proposals':[{'field_key':candidate['field_key'],'candidate_ids':[candidate['id']],
                    'party_role':candidate['party_role']}],'reread_requests':[]}
            return {'text':json.dumps(answer),'completed':True}
        if op=='visual_read':
            _,proofs=prepare_visual_regions(args['regions'])
            out={'text':json.dumps({'readings':{'r1':{'status':'readable','literal':self.literal}}}),
                'completed':True,'stop_reason':'eos','model_id':'fixture','model_bundle_sha256':'b'*64,
                'modality':'vision','purpose':'blind_optical_capture'}
            out['captures']=parse_visual_output(out,proofs)
            return out
        raise AssertionError(op)


class EmptyCtcVisualRegressionTests(unittest.TestCase):
    def test_successful_reread_of_previously_empty_ctc_has_full_visual_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source.png';Image.new('RGB',(320,80),'white').save(source)
            settings=Settings(root/'state',root/'models',device='cpu',dtype='float32',visual_verifier=True)
            backend=EmptyCtcRereadBackend(settings);engine=Engine(settings,backend)
            result=engine.run('empty-ctc',source,file_sha(source),'D01',subset=['passport.number'])
            self.assertEqual(result['terminal_status'],'completed',result['errors'])
            self.assertEqual(result['fields']['passport.number']['raw_value'],'123456')
            self.assertTrue(result['fields']['passport.number']['visual_verification_complete'])
            self.assertEqual(sum(op=='visual_read' for op,_ in backend.calls),1)
            check=next(iter(result['visual_checks'].values()))
            self.assertEqual(check['expected_observation_id'],'p1_r1_reread')
            self.assertEqual(check['verdict'],'correct')
            self.assertIn('p1_r1',result['optics']['visual_regions_by_id'])
            self.assertEqual(result['decision'],'review')


if __name__=='__main__':unittest.main()
