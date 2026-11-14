"""Offline/runtime-boundary tests with fake outputs, never a neural execution."""
import hashlib
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from harness_ocr.inference.yolo_detector import YoloTextDetector
from harness_ocr.inference.yolo_geometry import YoloDetectorConfig


class Array:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def result(quads=None, *, shape=(100, 200), scores=None, classes=None):
    quads = quads if quads is not None else [[[10., 20.], [40., 20.], [40., 32.], [10., 32.]]]
    return SimpleNamespace(orig_shape=shape, obb=SimpleNamespace(
        xyxyxyxy=Array(quads), conf=Array(scores if scores is not None else [.9] * len(quads)),
        cls=Array(classes if classes is not None else [0.] * len(quads))))


class FakeModel:
    task = 'obb'

    def __init__(self, output=None):
        self.output = output if output is not None else result()
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.output]


class YoloDetectorTests(unittest.TestCase):
    def detector(self, output=None, **kwargs):
        detector = YoloTextDetector('unused', **kwargs)
        detector._model = FakeModel(output)
        detector._checkpoint_sha = 'a' * 64
        detector._runtime_version = 'test-fake'
        return detector

    def test_missing_or_modified_checkpoint_fails_before_optional_import(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'yolo26l-text/yolo26l-text.pt'
            checkpoint.parent.mkdir()
            lock = {'models': {'yolo26l-text': {'files': [
                {'file': checkpoint.name, 'sha256': 'a' * 64}]}}}
            with patch('harness_ocr.models.model_lock', return_value=lock), patch.dict(sys.modules, {'ultralytics': None}):
                for existing in (False, True):
                    if existing:
                        checkpoint.write_bytes(b'not-the-registered-checkpoint')
                    with self.subTest(existing=existing), self.assertRaisesRegex(ValueError, 'artifact_sha_mismatch'):
                        YoloTextDetector(temp).load()

    def test_load_uses_only_absolute_pinned_file_and_disables_automatic_network(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'yolo26l-text/yolo26l-text.pt'
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b'fake local weights')
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            lock = {'models': {'yolo26l-text': {'files': [{'file': checkpoint.name, 'sha256': digest}]}}}
            runtime, utils = ModuleType('ultralytics'), ModuleType('ultralytics.utils')
            torch_utils = ModuleType('ultralytics.utils.torch_utils')
            fake_torch = SimpleNamespace(set_num_threads=lambda n: None, get_num_threads=lambda: 4)
            runtime.__version__, runtime.utils = 'test-fake', utils
            calls = []

            def create(path, **kwargs):
                calls.append((path, kwargs))
                return FakeModel()

            runtime.YOLO = create
            with patch('harness_ocr.models.model_lock', return_value=lock), patch.dict(sys.modules,
                    {'ultralytics': runtime, 'ultralytics.utils': utils,
                     'ultralytics.utils.torch_utils': torch_utils, 'torch': fake_torch}), patch.dict(os.environ,
                    {'YOLO_CONFIG_DIR': str(Path(temp) / 'config')}):
                detector = YoloTextDetector(temp)
                self.assertIs(detector.load(), detector)
                detector.load()
                self.assertEqual(calls, [(str(checkpoint.resolve()), {'task': 'obb'})])
                self.assertEqual(os.environ['YOLO_AUTOINSTALL'], 'false')
                self.assertEqual(os.environ['YOLO_OFFLINE'], 'true')
                self.assertFalse(utils.AUTOINSTALL)
                self.assertFalse(utils.ONLINE)
                self.assertEqual(utils.NUM_THREADS, 4)
                self.assertEqual(torch_utils.NUM_THREADS, 4)

    def test_original_pixel_coordinates_are_not_scaled_twice(self):
        detector = self.detector()
        rows, diagnostics = detector.detect(Image.new('L', (200, 100)))
        self.assertEqual(rows[0]['quad'], [[10., 20.], [40., 20.], [40., 32.], [10., 32.]])
        call = detector._model.calls[0]
        self.assertEqual(call['source'].mode, 'RGB')
        self.assertEqual(call['source'].size, (200, 100))
        self.assertFalse(call['half'])
        self.assertFalse(call['augment'])
        self.assertFalse(call['save'])
        self.assertEqual(call['imgsz'], 1280)
        self.assertEqual(call['max_det'], 1000)
        self.assertEqual(diagnostics['recognizer'], 'none_detector_only')
        self.assertIsNone(diagnostics['cuda_peak_allocated_bytes'])

    def test_detection_saturation_is_explicit_even_when_it_could_be_exact_count(self):
        detector = self.detector(config=YoloDetectorConfig(max_det=1))
        _, diagnostics = detector.detect(Image.new('RGB', (200, 100)))
        self.assertTrue(diagnostics['detection_limit_reached'])
        self.assertTrue(diagnostics['contour_limit_reached'])

    def test_obb_crossing_border_is_clipped_and_requires_review(self):
        detector = self.detector(result([[[-2., 20.], [40., 20.], [40., 32.], [-2., 32.]]]))
        rows, diagnostics = detector.detect(Image.new('RGB', (200, 100)))
        self.assertEqual(min(p[0] for p in rows[0]['quad']), 0)
        self.assertEqual(diagnostics['word_bounds_clipped'], 1)
        self.assertTrue(diagnostics['grouping_ambiguous'])

    def test_nonfinite_raw_points_cannot_be_hidden_by_border_clipping(self):
        bad = result([[[float('nan'), 20.], [40., 20.], [40., 32.], [10., 32.]]])
        with self.assertRaisesRegex(ValueError, 'finite_4x2'):
            self.detector(bad).detect(Image.new('RGB', (200, 100)))

    def test_shape_and_class_contracts_fail_closed(self):
        for output in (result(shape=(99, 200)), result(classes=[1]), result(scores=[])):
            with self.subTest(output=output), self.assertRaises(ValueError):
                self.detector(output).detect(Image.new('RGB', (200, 100)))

    def test_empty_obb_output_is_valid_and_never_creates_text(self):
        rows, diagnostics = self.detector(result([])).detect(Image.new('RGB', (200, 100)))
        self.assertEqual(rows, [])
        self.assertEqual(diagnostics['words_observed'], 0)
        self.assertFalse(diagnostics['detection_limit_reached'])

    def test_unavailable_cuda_does_not_silently_fallback_to_cpu(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': fake_torch}), self.assertRaisesRegex(RuntimeError, 'cuda_unavailable'):
            self.detector(device='cuda').detect(Image.new('RGB', (200, 100)))

    def test_cuda_metrics_are_synchronized_process_peaks_not_detector_only_estimates(self):
        calls = []
        fake_cuda = SimpleNamespace(is_available=lambda: True, synchronize=lambda: calls.append('sync'),
            memory_allocated=lambda: 100, memory_reserved=lambda: 200,
            reset_peak_memory_stats=lambda: calls.append('reset'),
            max_memory_allocated=lambda: 300, max_memory_reserved=lambda: 400)
        with patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=fake_cuda)}):
            _, diagnostics = self.detector(device='cuda').detect(Image.new('RGB', (200, 100)))
        self.assertEqual(calls, ['sync', 'reset', 'sync'])
        self.assertEqual(diagnostics['cuda_peak_allocated_bytes'], 300)
        self.assertEqual(diagnostics['cuda_allocated_before_bytes'], 100)
        self.assertIn('other_resident_models', diagnostics['cuda_peak_scope'])


if __name__ == '__main__':
    unittest.main()
