"""CLI handoff preserves server safety settings; no neural calls."""
import contextlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from harness_ocr.cli import _settings, main
from harness_ocr.profiles import ProfileRegistry


class CliHandoffTests(unittest.TestCase):
    def test_direct_inference_honors_visual_and_shared_lock_environment(self):
        args=SimpleNamespace(models=Path('models'),device='cpu',max_new_tokens=512,
            call_timeout=30,document_timeout=100,qwen='qwen-4b')
        with patch.dict(os.environ,{'HARNESS_VISUAL_VERIFIER':'1','HARNESS_CPU_THREADS':'3',
                                   'HARNESS_LOCK_DIR':str(Path('shared-locks').resolve())}):
            settings=_settings(args,Path('state'))
        self.assertTrue(settings.visual_verifier)
        self.assertEqual(settings.cpu_threads,3)
        self.assertEqual(settings.model_lock_dir,Path('shared-locks').resolve())
        self.assertEqual((settings.device,settings.dtype,settings.qwen),('cpu','float32','qwen-4b'))
        self.assertEqual(settings.max_new_tokens,512)
        settings.validate()

    def test_invalid_visual_environment_does_not_silently_disable_verifier(self):
        args=SimpleNamespace(models=Path('models'),device='cuda',max_new_tokens=512,
            call_timeout=30,document_timeout=100)
        with patch.dict(os.environ,{'HARNESS_VISUAL_VERIFIER':'yes'}):
            with self.assertRaisesRegex(ValueError,'invalid_visual_verifier_environment'):
                _settings(args,Path('state'))

    def test_profile_export_contains_the_generated_schema_and_all_fields(self):
        for profile in ProfileRegistry().list():
            with self.subTest(profile=profile['profile_id']):
                output=io.StringIO()
                with patch('sys.argv',['harness-ocr','profiles','--profile',profile['document_id']]),contextlib.redirect_stdout(output):
                    main()
                value=json.loads(output.getvalue())
                expected=ProfileRegistry().compile(profile['profile_id'])
                self.assertEqual(value['json_schema'],expected['json_schema'])
                self.assertEqual(value['field_definitions'],expected['field_definitions'])
                self.assertEqual(value['config_sha256'],expected['config_sha256'])
                self.assertFalse(value['automatic_acceptance_authorized'])


if __name__=='__main__':unittest.main()
