import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from harness_ocr.models import model_download_url, model_lock, prepare_models


class ReleaseSourceTests(unittest.TestCase):
    def test_yolo_release_is_explicit_allowlisted_and_hash_pinned(self):
        for name in ('yolo26l-text', 'yolo26x-text'):
            spec = model_lock()['models'][name]
            item = spec['files'][0]
            self.assertEqual(spec['license'], 'AGPL-3.0')
            self.assertEqual(len(item['sha256']), 64)
            self.assertTrue(model_download_url(spec, item).startswith(
                'https://github.com/ultralytics/assets/releases/download/v8.4.0/'))
            for bad in ('http://github.com/ultralytics/assets/releases/download/v8.4.0/x.pt',
                        'https://example.com/x.pt', item['download_url'] + '?redirect=1'):
                with self.assertRaises(ValueError):
                    model_download_url(spec, {**item, 'download_url': bad})

    def test_bad_download_cannot_replace_pinned_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'yolo26l-text/yolo26l-text.pt'
            target.parent.mkdir()
            target.write_bytes(b'previous invalid local file')
            with patch('harness_ocr.models.urllib.request.urlopen',
                       return_value=io.BytesIO(b'corrupt release download')):
                with self.assertRaisesRegex(ValueError, 'model_checksum_mismatch'):
                    prepare_models(directory, ['yolo26l-text'], download=True)
            self.assertEqual(target.read_bytes(), b'previous invalid local file')

    def test_existing_hf_revision_download_route_unchanged(self):
        spec = model_lock()['models']['detector']
        item = spec['files'][0]
        self.assertEqual(model_download_url(spec, item),
                         f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}/{item["file"]}')
