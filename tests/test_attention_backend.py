"""Explicit attention routing, fail closed without a NN or Torch import."""
from dataclasses import replace
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import Mock,patch

from harness_ocr.engine import Engine
from harness_ocr.inference.hf import HFReaders
from harness_ocr.settings import Settings


class AttentionBackendTests(unittest.TestCase):
    def test_default_environment_and_unsupported_values(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(Settings.from_env().attention_backend,'eager')
        for backend in ('eager','sdpa'):
            with patch.dict(os.environ,{'HARNESS_ATTENTION_BACKEND':backend}):
                settings=Settings.from_env();settings.validate()
                self.assertEqual(settings.attention_backend,backend)
        with patch.dict(os.environ,{'HARNESS_ATTENTION_BACKEND':'automatic'}):
            with self.assertRaisesRegex(ValueError,'unsupported_attention_backend'):
                Settings.from_env().validate()

    def test_xiaomi_and_qwen_receive_selected_backend_without_fallback(self):
        transformers=ModuleType('transformers')
        transformers.AutoModelForImageTextToText=Mock()
        transformers.AutoProcessor=Mock();transformers.AutoTokenizer=Mock()
        model=Mock();model.eval.return_value=model;model.to.return_value=model
        transformers.AutoModelForImageTextToText.from_pretrained.return_value=model
        with tempfile.TemporaryDirectory() as folder,patch.dict('sys.modules',{'transformers':transformers}):
            for backend in ('eager','sdpa'):
                reader=HFReaders.__new__(HFReaders)
                reader.settings=Settings(Path(folder)/'state',Path(folder)/'models',
                    device='cpu',dtype='float32',attention_backend=backend)
                reader.torch=SimpleNamespace(float32='explicit-float32')
                reader.models={};reader.processors={};reader.tokenizers={}
                for name in ('xiaomi','qwen-0.8b'):
                    reader.load(name)
                    call=transformers.AutoModelForImageTextToText.from_pretrained.call_args
                    self.assertEqual(call.kwargs['attn_implementation'],backend)
                    self.assertEqual(call.kwargs['torch_dtype'],'explicit-float32')
                    self.assertTrue(call.kwargs['local_files_only'])
                    self.assertFalse(call.kwargs['trust_remote_code'])
            reader.models={}
            transformers.AutoModelForImageTextToText.from_pretrained.side_effect=ValueError('backend unavailable')
            with self.assertRaisesRegex(ValueError,'backend unavailable'):
                reader.load('xiaomi')
            self.assertEqual(reader.models,{})

    def test_backend_change_invalidates_previous_acceptance_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            settings=Settings(Path(folder)/'state',Path(folder)/'models',device='cpu',dtype='float32')
            eager=Engine(settings,object());contract=eager.compile_contract('D01',subset=['passport.number'])
            sdpa=Engine(replace(settings,attention_backend='sdpa'),object())
            a=eager.binding(contract,'qwen-0.8b');b=sdpa.binding(contract,'qwen-0.8b')
            self.assertNotEqual(a['model_bundle_sha256'],b['model_bundle_sha256'])
            self.assertEqual(a['policy_version'],b['policy_version'])


if __name__=='__main__':unittest.main()
