"""Portable arithmetic/contract checks; never executes a neural model."""
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from harness_ocr.core import DocumentStore
from harness_ocr.inference.cheap import CheapReader
from harness_ocr.inference.db_geometry import (ctc_decode, ctc_preprocess,
    detector_postprocess, map_points, order_quad, rectify_quad)

SHA = 'a' * 64
QUAD = [[0., 0.], [40., 0.], [40., 20.], [0., 20.]]


class ArithmeticTests(unittest.TestCase):
    def test_import_has_no_nn_runtime_side_effect(self):
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
        completed = subprocess.run([sys.executable, '-c',
            'import sys; import harness_ocr.inference.cheap; '
            'assert not any(n in sys.modules for n in ["torch","onnxruntime","openvino"])'],
            env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_ctc_collapse_then_blank_removal_and_top_two(self):
        # Repeated A separated by a blank must survive as AA, adjacent BB not.
        indexes = [1, 1, 0, 1, 2, 2, 0]
        prediction = np.zeros((1, len(indexes), 3), np.float32)
        for step, value in enumerate(indexes):
            prediction[0, step] = [.05, .05, .05]
            prediction[0, step, value] = .9
        result = ctc_decode(prediction, ['blank', 'А', 'Б'])
        self.assertEqual(result['text'], 'ААБ')
        self.assertEqual([c['time_step'] for c in result['characters']], [0, 3, 4])
        self.assertFalse(result['score_calibrated_as_correctness'])
        self.assertTrue(all(c['score'] >= c['second_score'] for c in result['characters']))

    def test_ctc_logits_finite_and_dictionary_fail_closed(self):
        result = ctc_decode(np.array([[[0., 5., 1.], [5., 0., 1.]]]), ['blank', 'Я', ' '])
        self.assertEqual(result['text'], 'Я')
        self.assertTrue(result['softmax_applied'])
        for value in [np.zeros((1, 1, 4)), np.full((1, 1, 3), np.nan), np.zeros((1, 0, 3))]:
            with self.assertRaises(ValueError):
                ctc_decode(value, ['blank', 'Я', ' '])

    def test_ctc_preprocess_bgr_padding_and_width_compression_explicit(self):
        tensor, metadata = ctc_preprocess(Image.new('RGB', (20, 20), (255, 0, 0)))
        self.assertEqual(tensor.shape, (1, 3, 48, 320))
        np.testing.assert_allclose(tensor[0, :, 0, 0], [-1., -1., 1.])
        self.assertTrue(np.all(tensor[0, :, :, 48:] == 0))
        self.assertEqual(metadata['resized_width'], 48)
        _, wide = ctc_preprocess(Image.new('RGB', (4000, 10)))
        self.assertEqual(wide['canvas_width'], 3200)
        self.assertTrue(wide['width_compressed_to_limit'])
        self.assertFalse(wide['input_text_clipped'])

    def test_quad_order_and_non_axis_aligned_homography_round_trip(self):
        np.testing.assert_array_equal(order_quad([QUAD[2], QUAD[0], QUAD[3], QUAD[1]]), QUAD)
        matrix = np.array([[1., .2, 5.], [.1, .8, -3.], [.001, .002, 1.]])
        projected = map_points(QUAD, matrix)
        np.testing.assert_allclose(map_points(projected, np.linalg.inv(matrix)), QUAD, atol=1e-10)
        with self.assertRaises(ValueError):
            map_points(QUAD, np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            order_quad([[0, 1], [1, 2], [2, 3], [np.nan, 0]])

    @unittest.skipUnless(importlib.util.find_spec('cv2') and importlib.util.find_spec('pyclipper'), 'optional OpenCV+pyclipper geometry dependencies unavailable')
    def test_real_geometry_synthetic_map_and_crop_inverse(self):
        heatmap = np.zeros((64, 128), np.float32)
        heatmap[20:35, 20:90] = .95
        rows, diagnostics = detector_postprocess(heatmap, [256, 128])
        self.assertEqual(len(rows), 1)
        self.assertFalse(diagnostics['contour_limit_reached'])
        crop, proof = rectify_quad(Image.new('RGB', (256, 128), 'white'), rows[0]['quad'])
        self.assertGreater(crop.width, crop.height)
        projected = map_points(rows[0]['quad'], proof['transform_source_to_view'])
        recovered = map_points(projected, proof['transform_view_to_source'])
        np.testing.assert_allclose(recovered, rows[0]['quad'], atol=1e-5)


class FakeInput:
    name = 'x'


class FakeSession:
    def __init__(self, owner, kind):
        self.owner, self.kind = owner, kind

    def get_inputs(self):
        return [FakeInput()]

    def run(self, names, values):
        self.owner.calls.append(self.kind)
        if self.kind == 'detector':
            return [np.ones((1, 1, 4, 4), np.float32)]
        return [np.array([[[.05, .9, .05], [.9, .05, .05], [.05, .05, .9]]], np.float32)]


class FakeReader(CheapReader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls, self.loads = [], 0

    def _load(self):
        if self._detector is None:
            self.loads += 1
            self._detector = FakeSession(self, 'detector')
            self._recognizer = FakeSession(self, 'recognizer')
            self.chars = ['blank', 'Я', '1']
            self._artifacts = {'fake': 'test_only'}


class ReaderContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.path = self.folder / 'source.png'
        Image.new('RGB', (80, 40), 'white').save(self.path)
        raster = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.page = {'page': 1, 'page_id': 'p1', 'path': str(self.path), 'source_size': [80, 40],
                     'raster_sha256': raster, 'transform': {'source_sha256': SHA, 'raster_sha256': raster}}

    def tearDown(self):
        self.temp.cleanup()

    def read_fake(self, reader, count=1, folder='crops'):
        rows = [{'quad': QUAD, 'detector_score': .9} for _ in range(count)]
        proof = {'transform_source_to_view': np.eye(3).tolist(), 'transform_view_to_source': np.eye(3).tolist(),
                 'quad_source_xy': QUAD, 'rotation_degrees_counterclockwise': 0}
        with patch('harness_ocr.inference.cheap.detector_preprocess', return_value=(np.zeros((1, 3, 32, 64), np.float32), {})), \
             patch('harness_ocr.inference.cheap.detector_postprocess', return_value=(rows, {'contour_limit_reached': False})), \
             patch('harness_ocr.inference.cheap.rectify_quad', side_effect=lambda image, quad: (Image.new('RGB', (40, 20), 'white'), proof)):
            return reader.read(self.page, SHA, self.folder / folder)

    def test_sequential_sessions_source_proofs_and_core_compatible_observations(self):
        reader = FakeReader(self.folder)
        result = self.read_fake(reader, count=2)
        self.assertEqual(reader.calls, ['detector', 'recognizer', 'recognizer'])
        self.assertEqual(reader.loads, 1)
        self.assertEqual([o['text'] for o in result['observations']], ['Я1', 'Я1'])
        self.assertEqual([o['id'] for o in result['observations']], ['p1_ctc1', 'p1_ctc2'])
        self.assertTrue(result['diagnostics']['completed'])
        store = DocumentStore(self.folder / 'state.sqlite')
        store.create('d1', SHA, 'p1')
        snapshot = store.revise('d1', observations=result['observations'])
        self.assertEqual(len(snapshot['observations']), 2)
        for region in result['regions']:
            self.assertTrue(Path(region['path']).is_file())
            self.assertEqual(region['transform']['raster_sha256'], self.page['raster_sha256'])
        self.assertFalse(result['diagnostics']['scores_are_correctness_probabilities'])

    def test_line_limit_is_explicit_and_not_document_completion(self):
        reader = FakeReader(self.folder, max_lines=1)
        result = self.read_fake(reader, count=2)
        self.assertEqual(len(result['observations']), 1)
        self.assertFalse(result['diagnostics']['completed'])
        self.assertFalse(result['completed'])
        self.assertTrue(result['diagnostics']['needs_review'])
        self.assertTrue(result['diagnostics']['line_limit_reached'])
        self.assertTrue(result['observations'][0]['completed'])

    def test_source_binding_failures_precede_model_load(self):
        for update in [{'source_size': [10, 10]}, {'raster_sha256': 'b' * 64}, {'transform': {}}]:
            reader = FakeReader(self.folder)
            page = {**self.page, **update}
            with self.assertRaises(ValueError):
                reader.read(page, SHA, self.folder / 'invalid')
            self.assertEqual(reader.loads, 0)

    def test_output_immutable_and_parallel_call_guard(self):
        reader = FakeReader(self.folder)
        self.read_fake(reader)
        with self.assertRaises(ValueError):
            reader.read(self.page, SHA, self.folder / 'crops')
        reader._guard.acquire()
        try:
            with self.assertRaises(RuntimeError):
                reader.read(self.page, SHA, self.folder / 'other')
        finally:
            reader._guard.release()

    def test_missing_model_or_modified_artifact_rejected_before_ort_import(self):
        reader = CheapReader(self.folder / 'models')
        with self.assertRaisesRegex(ValueError, 'artifact_sha_mismatch'):
            reader._load()
        self.assertNotIn('onnxruntime', sys.modules)


if __name__ == '__main__':
    unittest.main()
