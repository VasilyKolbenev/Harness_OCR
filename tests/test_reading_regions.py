"""Source pixels and hard geometry bounds for adaptive reading; no models."""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from harness_ocr.inference.region_tasks import (
    context_box, materialize_region, page_from_visual_descriptor, same_region_geometry)
from harness_ocr.inference.vision import make_visual_region, prepare_visual_regions
from harness_ocr.util import file_sha


class ReadingRegionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.raster = self.root/'source.png', self.root/'raster.png'
        image = Image.new('RGB', (400, 240), 'white')
        ImageDraw.Draw(image).text((30, 40), '001234', fill='black')
        image.save(self.source); image.save(self.raster)
        self.sha = file_sha(self.source)
        self.page = {'path': str(self.raster), 'page': 1, 'page_id': 'p1', 'source_size': [400, 240],
            'raster_sha256': file_sha(self.raster), 'transform': {'version': 'raster-v1',
            'orientation': 'EXIF_transposed', 'source_sha256': self.sha, 'raster_sha256': file_sha(self.raster)}}

    def test_exact_context_crop_replays_against_the_original_source(self):
        region = materialize_region(self.page, [20, 30, 120, 70], self.root/'crops', source_sha256=self.sha)
        descriptor = make_visual_region(self.source, self.page, region, source_sha256=self.sha)
        _, proof = prepare_visual_regions([descriptor])
        self.assertEqual(proof[0]['bbox'], [20, 30, 120, 70])
        self.assertEqual(descriptor['crop_kind'], 'axis_aligned')
        self.assertNotIn('quad', descriptor)
        self.assertTrue(proof[0]['raster_transform_proof']['source_to_raster_verification']['exact_raster_pixels_verified'])
        self.assertFalse(proof[0]['raster_transform_proof']['invented_text_or_superresolution'])
        self.assertFalse(region['semantic_field_role_established'])
        recovered = page_from_visual_descriptor(descriptor)
        self.assertEqual(Path(recovered['path']).resolve(), self.raster.resolve())
        self.assertEqual({k:v for k,v in recovered.items() if k != 'path'},
                         {k:v for k,v in self.page.items() if k != 'path'})

    def test_changed_raster_and_foreign_source_cannot_make_a_bound_crop(self):
        bad = deepcopy(self.page); bad['transform']['source_sha256'] = 'f'*64
        with self.assertRaisesRegex(ValueError, 'not_bound'):
            materialize_region(bad, [20, 30, 120, 70], self.root/'bad', source_sha256=self.sha)
        Image.new('RGB', (400, 240), 'black').save(self.raster)
        with self.assertRaisesRegex(ValueError, 'raster_sha'):
            materialize_region(self.page, [20, 30, 120, 70], self.root/'bad', source_sha256=self.sha)

    def test_no_hash_then_second_open_race_can_crop_different_pixels(self):
        real_read = Path.read_bytes
        swap = BytesIO(); Image.new('RGB', (400, 240), 'black').save(swap, format='PNG')
        changed = False
        def checked_then_changed(path):
            nonlocal changed
            data = real_read(path)
            if Path(path).resolve() == self.raster.resolve() and not changed:
                changed = True; self.raster.write_bytes(swap.getvalue())
            return data
        with patch.object(Path, 'read_bytes', checked_then_changed):
            try:
                region = materialize_region(self.page, [200, 100, 300, 150], self.root/'race', source_sha256=self.sha)
            except ValueError:
                return  # Refusing changed source bytes is also a safe result.
        with Image.open(region['path']) as crop:
            self.assertEqual(crop.convert('RGB').getpixel((10, 10)), (255, 255, 255),
                             'A crop may not use black replacement pixels under the old white-raster SHA')

    def test_geometry_has_no_negative_nan_or_silent_outside_source_crop(self):
        for box in [[-1, 0, 80, 20], [0, 0, 401, 20], [20, 20, 20, 30],
                    [0, 0, float('nan'), 20], [False, 0, 80, 20]]:
            with self.subTest(box=box), self.assertRaises(ValueError):
                materialize_region(self.page, box, self.root/'bad', source_sha256=self.sha)

    def test_existing_region_file_is_immutable_and_not_relabelled_with_new_hash(self):
        region = materialize_region(self.page, [20, 30, 120, 70], self.root/'crops', source_sha256=self.sha)
        path = Path(region['path']); Image.new('RGB', (100, 40), 'black').save(path)
        tampered = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'immutable'):
            materialize_region(self.page, [20, 30, 120, 70], self.root/'crops', source_sha256=self.sha)
        self.assertEqual(path.read_bytes(), tampered)

    def test_context_adds_label_room_without_authorizing_a_value_or_role(self):
        region = {'quad': [[40, 40], [100, 40], [100, 60], [40, 60]], 'source_size': [400, 240]}
        before = deepcopy(region); box = context_box(region)
        self.assertLess(box[0], 40); self.assertLess(box[1], 40)
        self.assertGreater(box[2], 100); self.assertGreater(box[3], 60)
        self.assertTrue(0 <= box[0] < box[2] <= 400 and 0 <= box[1] < box[3] <= 240)
        self.assertEqual(region, before)
        self.assertIsNone(context_box({'bbox': [0, 0, 400, 240], 'source_size': [400, 240]}))

    def test_fake_metadata_cannot_turn_an_unrelated_crop_into_the_expected_roi(self):
        region = materialize_region(self.page, [20, 30, 120, 70], self.root/'crops', source_sha256=self.sha)
        descriptor = make_visual_region(self.source, self.page, region, source_sha256=self.sha)
        bad = deepcopy(descriptor); bad['bbox'] = [200, 100, 300, 140]
        with self.assertRaisesRegex(ValueError, 'not_pixels_of_declared_region'):
            prepare_visual_regions([bad])

    def test_tall_multiline_context_preserves_upright_exact_pixels_and_axis_transform(self):
        # A tall box is deliberately above the line-rectifier rotation ratio.
        # Unequal colored corners make a rotation or resampling detectable.
        with Image.open(self.raster) as original:
            image = original.convert('RGB')
        draw = ImageDraw.Draw(image)
        draw.rectangle([60, 10, 69, 19], fill='red')
        draw.rectangle([150, 220, 159, 229], fill='blue')
        image.save(self.source); image.save(self.raster)
        self.sha = file_sha(self.source)
        self.page['raster_sha256'] = file_sha(self.raster)
        self.page['transform'].update(source_sha256=self.sha, raster_sha256=self.page['raster_sha256'])
        box = [60, 10, 160, 230]
        region = materialize_region(self.page, box, self.root/'tall', source_sha256=self.sha)
        with Image.open(region['path']) as crop:
            self.assertEqual(crop.size, (100, 220))
            self.assertEqual(crop.convert('RGB').tobytes(), image.crop(box).tobytes())
            self.assertEqual(crop.getpixel((0, 0)), (255, 0, 0))
            self.assertEqual(crop.getpixel((99, 219)), (0, 0, 255))
        proof = region['transform']
        self.assertEqual(proof['version'], 'axis-aligned-crop-v1')
        self.assertEqual(proof['bbox_source_xy'], box)
        self.assertEqual(proof['view_size'], [100, 220])
        self.assertEqual(proof['operations'], [{'operation':'axis_aligned_crop'}])
        self.assertIn('_task_v2_', region['id'])
        descriptor = make_visual_region(self.source, self.page, region, source_sha256=self.sha)
        prepared, replay = prepare_visual_regions([descriptor])
        self.assertEqual(prepared[0].size, (100, 220))
        self.assertEqual(prepared[0].tobytes(), image.crop(box).tobytes())
        self.assertEqual(replay[0]['raster_transform_proof']['bbox_source_xy'], box)
        self.assertTrue(replay[0]['raster_transform_proof']['source_to_raster_verification']['exact_raster_pixels_verified'])
        forged = deepcopy(descriptor); forged['bbox'] = [61, 10, 161, 230]
        with self.assertRaisesRegex(ValueError, 'not_pixels_of_declared_region'):
            prepare_visual_regions([forged])

    def test_geometry_relatedness_requires_real_exact_source_bound_bbox_or_quad(self):
        region = materialize_region(self.page, [20, 30, 120, 70], self.root/'crops', source_sha256=self.sha)
        observation = {key:deepcopy(region[key]) for key in ('source_sha256','page','page_id','source_size','bbox')}
        self.assertTrue(same_region_geometry(observation, region))
        for changes in ({'bbox':[21, 30, 121, 70]}, {'source_sha256':'f'*64}, {'page_id':'p2'},
                        {'bbox':None}, {'source_size':None}, {'page':True}):
            with self.subTest(changes=changes):
                self.assertFalse(same_region_geometry({**observation, **changes}, region))
        self.assertFalse(same_region_geometry({k:v for k,v in observation.items() if k!='bbox'},
                                             {k:v for k,v in region.items() if k!='bbox'}))
        quad = [[20, 30], [120, 30], [120, 70], [20, 70]]
        quad_region = {**region, 'quad':quad}
        quad_observation = {**observation, 'quad':deepcopy(quad)}; quad_observation.pop('bbox')
        self.assertTrue(same_region_geometry(quad_observation, quad_region))
        self.assertFalse(same_region_geometry(observation, quad_region),
            'An equivalent enclosing bbox cannot substitute for quad geometry')

    def test_ocr_refuses_changed_bytes_and_pixel_overflow_before_loading_weights(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from harness_ocr.inference.hf import HFReaders
        reader = HFReaders.__new__(HFReaders)
        reader.settings = SimpleNamespace(max_upload_bytes=30000000, max_page_pixels=10, max_new_tokens=32)
        reader.load = Mock(side_effect=AssertionError('A source failure must not load a model'))
        with self.assertRaisesRegex(ValueError, 'sha_mismatch'):
            reader.ocr(self.raster, expected_image_sha256='f'*64)
        with self.assertRaisesRegex(ValueError, 'pixel_limit'):
            reader.ocr(self.raster, expected_image_sha256=file_sha(self.raster))
        reader.load.assert_not_called()

    def test_ocr_processor_receives_the_checked_buffer_even_if_path_changes_after_read(self):
        from types import SimpleNamespace
        from harness_ocr.inference.hf import HFReaders
        from harness_ocr.inference import vision
        reader = HFReaders.__new__(HFReaders)
        reader.settings = SimpleNamespace(max_upload_bytes=30000000, max_page_pixels=100000, max_new_tokens=32)
        sampled = []
        class Processor:
            def apply_chat_template(self, *args, **kwargs): return 'fixture-image-prompt'
            def __call__(self, **kwargs):
                sampled.append(kwargs['images'][0].getpixel((200, 150)))
                return {}
        reader.processors = {'xiaomi':Processor()}; reader.load = lambda name: None
        reader.generate = lambda *args, **kwargs: {'text':'fixture', 'completed':True}
        original = vision._bytes; expected = file_sha(self.raster)
        def checked_then_changed(path, sha, **kwargs):
            data = original(path, sha, **kwargs)
            Image.new('RGB', (400, 240), 'black').save(self.raster)
            return data
        with patch.object(vision, '_bytes', checked_then_changed):
            output = reader.ocr(self.raster, expected_image_sha256=expected)
        self.assertEqual(sampled, [(255, 255, 255)])
        self.assertEqual(output['input_image_sha256'], expected)
        self.assertTrue(output['input_image_expected_sha_verified'])


if __name__ == '__main__':
    unittest.main()
