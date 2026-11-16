"""Adaptive/rescue runtime contracts with fake readers, no neural weights."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from harness_ocr.cli import _settings, main
from harness_ocr.inference.cheap import CheapReader
from harness_ocr.inference.supervisor import _worker
from harness_ocr.service import Service
from harness_ocr.settings import Settings
from harness_ocr.util import file_sha


class SettingsTests(unittest.TestCase):
    def base(self):
        return Settings(Path('state'), Path('models'), device='cpu', dtype='float32')

    def test_defaults_leave_optional_yolo_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        settings.validate()
        self.assertEqual((settings.reading_policy, settings.rescue_text_detector), ('adaptive', 'none'))
        self.assertEqual((settings.max_detector_rescues, settings.max_whole_page_reads), (1, 1))
        self.assertFalse(settings.rescue_enabled)

    def test_enum_and_budget_validation_is_strict_including_bool(self):
        bad = [('reading_policy', 'best-effort'), ('reading_policy', True), ('reading_policy', []),
               ('rescue_text_detector', 'db'), ('rescue_text_detector', False),
               ('max_detector_rescues', True), ('max_detector_rescues', -1),
               ('max_detector_rescues', 3), ('max_detector_rescues', 1.0),
               ('max_whole_page_reads', False), ('max_whole_page_reads', -1),
               ('max_whole_page_reads', 11), ('max_whole_page_reads', '1')]
        for name, value in bad:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                replace(self.base(), **{name: value}).validate()
        replace(self.base(), max_detector_rescues=2, max_whole_page_reads=10).validate()
        replace(self.base(), max_detector_rescues=0, max_whole_page_reads=0).validate()
        with self.assertRaisesRegex(ValueError, 'whole_page_read_budget'):
            replace(self.base(), max_pages=2, max_whole_page_reads=3).validate()

    def test_same_enabled_primary_and_rescue_is_rejected(self):
        settings = replace(self.base(), text_detector='yolo26l', rescue_text_detector='yolo26l')
        with self.assertRaisesRegex(ValueError, 'must_differ'):
            settings.validate()
        replace(settings, max_detector_rescues=0).validate()
        replace(settings, reading_policy='legacy').validate()

    def test_environment_has_explicit_zero_and_no_truthy_string_coercion(self):
        with patch.dict(os.environ, {'HARNESS_READING_POLICY': 'adaptive',
            'HARNESS_RESCUE_TEXT_DETECTOR': 'yolo26x', 'HARNESS_MAX_DETECTOR_RESCUES': '2',
            'HARNESS_MAX_WHOLE_PAGE_READS': '0'}, clear=True):
            settings = Settings.from_env()
        settings.validate()
        self.assertTrue(settings.rescue_enabled)
        self.assertEqual((settings.max_detector_rescues, settings.max_whole_page_reads), (2, 0))
        for value in ('true', '', '1.5'):
            with self.subTest(value=value), patch.dict(os.environ, {'HARNESS_MAX_DETECTOR_RESCUES': value}):
                with self.assertRaises(ValueError): Settings.from_env()

    def test_cli_zero_overrides_environment_and_disabled_mode_is_explicit(self):
        args = SimpleNamespace(models=Path('models'), device='cpu', max_new_tokens=512,
            call_timeout=10, document_timeout=60, qwen='qwen-0.8b', reading_policy='legacy',
            rescue_text_detector='none', max_detector_rescues=0, max_whole_page_reads=0)
        with patch.dict(os.environ, {'HARNESS_RESCUE_TEXT_DETECTOR': 'yolo26x',
             'HARNESS_MAX_DETECTOR_RESCUES': '2', 'HARNESS_MAX_WHOLE_PAGE_READS': '3'}):
            settings = _settings(args, Path('state'))
        self.assertEqual((settings.reading_policy, settings.rescue_text_detector), ('legacy', 'none'))
        self.assertEqual((settings.max_detector_rescues, settings.max_whole_page_reads), (0, 0))
        self.assertFalse(settings.rescue_enabled)


class _Connection:
    def __init__(self, requests): self.requests = list(requests) + [None]; self.replies = []; self.closed = False
    def recv(self): return self.requests.pop(0)
    def send(self, value): self.replies.append(value)
    def close(self): self.closed = True


class _FakeReader:
    instances = []
    def __init__(self, models_dir, **kwargs):
        self.models_dir = models_dir; self.kwargs = kwargs; self._detector = None; self.loads = 0; self.calls = []
        self.instances.append(self)
    def load(self):
        if self._detector is None: self.loads += 1; self._detector = object()
        return self
    def read(self, **args):
        self.load(); self.calls.append(args)
        return {'text_detector': self.kwargs['text_detector'], 'namespace': self.kwargs.get('id_namespace')}


class _FakeHF:
    loads = []
    def __init__(self, settings): self.settings = settings
    def load(self, name): self.loads.append(name)


class WorkerDispatchTests(unittest.TestCase):
    def setUp(self): _FakeReader.instances = []; _FakeHF.loads = []
    def settings(self, **changes):
        return replace(Settings(Path('state'), Path('models'), device='cpu', dtype='float32',
            rescue_text_detector='yolo26l'), **changes)
    def run_worker(self, settings, requests, *, available=True):
        connection = _Connection(requests)
        fake_torch = SimpleNamespace(__version__='fake', cuda=SimpleNamespace(is_available=lambda: False))
        with patch('harness_ocr.inference.cheap.CheapReader', _FakeReader), \
             patch('harness_ocr.inference.hf.HFReaders', _FakeHF), \
             patch.dict(sys.modules, {'torch': fake_torch}), \
             patch('importlib.util.find_spec', return_value=object() if available else None), \
             patch('multiprocessing.get_context', side_effect=AssertionError('second owner forbidden')):
            _worker(connection, settings)
        self.assertTrue(connection.closed)
        return connection.replies

    def test_probe_loads_only_primary_and_rescue_reuses_the_same_worker(self):
        args = {'page': {'page_id': 'p1'}, 'source_sha256': 'a'*64, 'output_dir': 'crops'}
        replies = self.run_worker(self.settings(max_detector_rescues=2), [
            {'op': 'probe', 'args': {'warmup': True}}, {'op': 'cheap', 'args': args},
            {'op': 'cheap_rescue', 'args': args}, {'op': 'cheap_rescue', 'args': args},
            {'op': 'probe', 'args': {}}])
        self.assertTrue(all(r['ok'] for r in replies))
        self.assertEqual(replies[0]['value']['loaded_models'], ['xiaomi', 'qwen-0.8b', 'detector', 'cyrillic'])
        self.assertFalse(replies[0]['value']['rescue_initialized'])
        self.assertFalse(replies[0]['value']['rescue_loaded'])
        self.assertTrue(replies[-1]['value']['rescue_loaded'])
        self.assertEqual(_FakeHF.loads, ['xiaomi', 'qwen-0.8b'])
        self.assertEqual(len(_FakeReader.instances), 2)
        primary, rescue = _FakeReader.instances
        self.assertEqual(primary.kwargs['text_detector'], 'db')
        self.assertIsNone(primary.kwargs.get('id_namespace'))
        self.assertEqual(rescue.kwargs['text_detector'], 'yolo26l')
        self.assertEqual(rescue.kwargs['id_namespace'], 'yolo26l')
        self.assertEqual((primary.loads, rescue.loads, len(rescue.calls)), (1, 1, 2))
        self.assertEqual((primary.kwargs['threads'], rescue.kwargs['threads']), (4, 4))

    def test_none_legacy_or_zero_refuses_before_creating_a_reader(self):
        args = {'page': {}, 'source_sha256': 'a'*64, 'output_dir': 'crops'}
        for changes in ({'rescue_text_detector': 'none'}, {'reading_policy': 'legacy'},
                        {'max_detector_rescues': 0}):
            replies = self.run_worker(self.settings(**changes), [{'op': 'cheap_rescue', 'args': args}])
            self.assertFalse(replies[0]['ok'])
            self.assertIn('detector_rescue_not_enabled', replies[0]['error'])
            self.assertEqual(_FakeReader.instances, [])

    def test_request_cannot_override_server_detector_or_namespace(self):
        args = {'page': {}, 'source_sha256': 'a'*64, 'output_dir': 'crops', 'text_detector': 'yolo26x'}
        replies = self.run_worker(self.settings(), [{'op': 'cheap_rescue', 'args': args}])
        self.assertFalse(replies[0]['ok'])
        self.assertIn('invalid_detector_rescue_arguments', replies[0]['error'])
        self.assertEqual(_FakeReader.instances, [])

    def test_missing_optional_runtime_fails_probe_without_loading_weights(self):
        replies = self.run_worker(self.settings(), [{'op': 'probe', 'args': {'warmup': True}}], available=False)
        self.assertFalse(replies[0]['ok'])
        self.assertIn('configured_rescue_runtime_not_installed', replies[0]['error'])
        self.assertEqual(_FakeReader.instances, [])
        self.assertEqual(_FakeHF.loads, [])


class ModelVerificationTests(unittest.TestCase):
    def test_service_cache_separates_rescue_policy_even_with_same_file_fingerprint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            settings = Settings(root/'state', root/'models', device='cpu', dtype='float32',
                                rescue_text_detector='yolo26l')
            service = Service(settings, SimpleNamespace(close=lambda: None))
            try:
                with patch.object(service, '_fingerprint', return_value=('fixed-fixture',)), \
                     patch('harness_ocr.service.verify_models', return_value={'ok': True}) as verify:
                    service._ensure_models('qwen-0.8b'); service._ensure_models('qwen-0.8b')
                    self.assertEqual(verify.call_count, 1)
                    self.assertIn('yolo26l-text', verify.call_args.args[1])
                    service.settings = replace(settings, rescue_text_detector='yolo26x')
                    service._ensure_models('qwen-0.8b')
                    self.assertEqual(verify.call_count, 2)
                    self.assertIn('yolo26x-text', verify.call_args.args[1])
                    service.settings = replace(service.settings, reading_policy='legacy')
                    service._ensure_models('qwen-0.8b')
                    self.assertEqual(verify.call_count, 3)
                    self.assertEqual(verify.call_args.args[1], ['detector', 'cyrillic', 'xiaomi', 'qwen-0.8b'])
            finally: service.close()

    def test_cli_verifies_configured_rescue_before_any_owner_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); source = root/'input.bin'; source.write_bytes(b'fixture')
            args = ['harness-ocr', 'infer', '--input', str(source), '--profile', 'D01',
                '--output', str(root/'result.json'), '--models', str(root/'models'), '--device', 'cpu',
                '--reading-policy', 'adaptive', '--rescue-text-detector', 'yolo26l',
                '--max-detector-rescues', '2', '--max-whole-page-reads', '0']
            with patch.dict(os.environ, {}, clear=True), patch('sys.argv', args), \
                 patch('harness_ocr.models.verify_models', return_value={'ok': False, 'errors': ['missing_rescue']}) as verify, \
                 patch('harness_ocr.inference.supervisor.Supervisor') as supervisor:
                with self.assertRaisesRegex(ValueError, 'missing_rescue'): main()
                self.assertIn('yolo26l-text', verify.call_args.args[1])
                supervisor.assert_not_called()

    def test_cli_settings_reach_fake_owner_without_losing_zero_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'input.bin'; source.write_bytes(b'fixture')
            args=['harness-ocr','infer','--input',str(source),'--profile','D01',
                '--output',str(root/'out.json'),'--models',str(root/'models'),'--device','cpu',
                '--rescue-text-detector','yolo26x','--max-detector-rescues','1','--max-whole-page-reads','0']
            with patch.dict(os.environ, {}, clear=True), patch('sys.argv',args), \
                 patch('harness_ocr.models.verify_models',return_value={'ok':True}), \
                 patch('harness_ocr.inference.supervisor.Supervisor') as supervisor, \
                 patch('harness_ocr.engine.Engine') as engine, redirect_stdout(io.StringIO()):
                engine.return_value.run.return_value={'fields':{},'completed':True}
                main()
            settings=supervisor.call_args.args[0]
            self.assertEqual((settings.reading_policy,settings.rescue_text_detector),('adaptive','yolo26x'))
            self.assertEqual((settings.max_detector_rescues,settings.max_whole_page_reads),(1,0))
            supervisor.return_value.close.assert_called_once()


class NamespaceTests(unittest.TestCase):
    def test_primary_ids_stay_unchanged_secondary_streams_are_distinct_but_correlated(self):
        class Input: name='x'
        class CTC:
            def get_inputs(self):return [Input()]
            def run(self,*args):return [np.array([[[.05,.9,.05],[.9,.05,.05],[.05,.05,.9]]],np.float32)]
        class Detector:
            def get_inputs(self):return [Input()]
            def run(self,*args):return [np.ones((1,1,4,4),np.float32)]
            def detect(self,image):return rows,{'grouping_ambiguous':False}
        quad=[[2.,2.],[70.,2.],[70.,22.],[2.,22.]]; rows=[{'quad':quad}]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'page.png'; Image.new('RGB',(80,40),'white').save(path); sha=file_sha(path)
            page={'page':1,'page_id':'p1','path':str(path),'source_size':[80,40],
                'raster_sha256':sha,'transform':{'source_sha256':sha,'raster_sha256':sha}}
            results=[]
            for detector,namespace in [('db',None),('yolo26l','yolo26l')]:
                reader=CheapReader(root/'models',text_detector=detector,id_namespace=namespace)
                reader._detector=Detector();reader._recognizer=CTC();reader.chars=['blank','Я','1']
                with patch('harness_ocr.inference.cheap.detector_preprocess',return_value=(np.zeros((1,3,32,64)),{})), \
                     patch('harness_ocr.inference.cheap.detector_postprocess',return_value=(rows,{})):
                    results.append(reader.read(page,sha,root/(namespace or 'primary')))
            primary,secondary=results; a,b=primary['observations'][0],secondary['observations'][0]
            self.assertEqual((a['id'],primary['regions'][0]['id']),('p1_ctc1','p1_r1'))
            self.assertEqual((b['id'],secondary['regions'][0]['id']),('p1_yolo26l_ctc1','p1_yolo26l_r1'))
            self.assertEqual(Path(secondary['regions'][0]['path']).name,'p1_yolo26l_line1.png')
            self.assertEqual((a['reading_stream_id'],b['reading_stream_id']),('ctc:p1:db','ctc:p1:yolo26l'))
            self.assertEqual(a['model_id'],b['model_id']);self.assertEqual(b['model_id'],'cyrillic-ctc')
            self.assertEqual(a['capture_id'],b['capture_id'])
            self.assertEqual(a['raster_transform_proof']['crop_sha256'],b['raster_transform_proof']['crop_sha256'])
            self.assertFalse(secondary['diagnostics']['runtime']['models_parallel'])
        for namespace in ('../other','arbitrary',True,[]):
            with self.subTest(namespace=namespace),self.assertRaises(ValueError):
                CheapReader(Path('models'),id_namespace=namespace)


if __name__=='__main__':unittest.main()
