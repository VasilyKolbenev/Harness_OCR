"""Text localisation contracts: no model/weights or expected strings involved."""
import copy
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

from harness_ocr.inference.yolo_geometry import YoloDetectorConfig, group_word_quads


def word(x, y, width=30, height=12, *, ident='a', angle=0, score=.9):
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    quad = [[x + dx * cosine - dy * sine, y + dx * sine + dy * cosine]
            for dx, dy in ((0, 0), (width, 0), (width, height), (0, height))]
    return {'quad': quad, 'detector_score': score, 'word_id': ident}


class YoloGeometryTests(unittest.TestCase):
    def test_import_does_not_load_neural_libraries(self):
        env = {**os.environ, 'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src')}
        run = subprocess.run([sys.executable, '-c',
            'import sys; import harness_ocr.inference.yolo_detector; '
            'assert not any(x in sys.modules for x in ("torch", "ultralytics", "onnxruntime", "cv2"))'],
            env=env, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_multitoken_label_and_value_preserve_word_provenance(self):
        words = [word(10, 20, ident='label1'), word(45, 20, ident='label2'),
                 word(80, 20, width=70, ident='value'), word(10, 55, width=130, ident='seller')]
        rows, info = group_word_quads(words, (600, 400))
        self.assertEqual([r['word_ids'] for r in rows], [['label1', 'label2', 'value'], ['seller']])
        self.assertEqual(len(rows[0]['word_quads']), 3)
        self.assertFalse(info['text_or_dictionary_used_for_grouping'])
        self.assertFalse(info['semantic_field_role_established'])

    def test_columns_and_large_whitespace_are_not_joined(self):
        words = [word(10, 20, ident='left'), word(50, 20, ident='leftvalue'),
                 word(300, 20, ident='right'), word(340, 20, ident='rightvalue')]
        rows, _ = group_word_quads(words, (600, 400))
        self.assertEqual([r['word_ids'] for r in rows], [['left', 'leftvalue'], ['right', 'rightvalue']])

    def test_adjacent_rows_remain_separate(self):
        rows, _ = group_word_quads([word(10, 20, ident='top'), word(50, 20, ident='top2'),
                                   word(10, 34, ident='bottom'), word(50, 34, ident='bottom2')], (600, 400))
        self.assertEqual([r['word_ids'] for r in rows], [['top', 'top2'], ['bottom', 'bottom2']])

    def test_rotated_line_uses_text_axis_not_global_y_sort(self):
        angle = 30
        words = [word(100 + d * math.cos(math.radians(angle)),
                      100 + d * math.sin(math.radians(angle)), ident=str(i), angle=angle)
                 for i, d in enumerate((0, 35, 70))]
        rows, info = group_word_quads(list(reversed(words)), (600, 400))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['word_ids'], ['0', '1', '2'])
        self.assertFalse(info['grouping_ambiguous'])

    def test_vertical_line_can_be_rectified_by_existing_crop_path(self):
        from PIL import Image
        from harness_ocr.inference.db_geometry import map_points, rectify_quad
        words = [word(200, 100, angle=90, ident='a'), word(200, 135, angle=90, ident='b')]
        rows, _ = group_word_quads(words, (600, 400))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['word_ids'], ['a', 'b'])
        for point in rows[0]['quad']:
            self.assertTrue(0 <= point[0] <= 600 and 0 <= point[1] <= 400)
        crop, proof = rectify_quad(Image.new('RGB', (600, 400)), rows[0]['quad'])
        self.assertGreater(crop.width, crop.height)
        pixel = [[194., 120.]]
        view_pixel = map_points(pixel, proof['transform_source_to_view'])
        reconstructed = map_points(view_pixel, proof['transform_view_to_source'])
        self.assertAlmostEqual(reconstructed[0][0], pixel[0][0], places=5)
        self.assertAlmostEqual(reconstructed[0][1], pixel[0][1], places=5)

    def test_rotated_and_horizontal_neighbours_do_not_merge(self):
        rows, _ = group_word_quads([word(100, 100), word(135, 100, angle=25, ident='b')], (600, 400))
        self.assertEqual(len(rows), 2)

    def test_duplicate_detections_keep_one_geometry_and_record_identity(self):
        original = word(10, 20, ident='weak', score=.7)
        duplicate = word(10, 20, ident='strong', score=.95)
        rows, info = group_word_quads([original, duplicate], (600, 400))
        self.assertEqual(rows[0]['word_ids'], ['strong'])
        self.assertEqual(info['duplicate_detections'], [{'word_id': 'weak', 'kept_word_id': 'strong'}])
        self.assertEqual(info['words_observed'], 2)

    def test_distinct_overlapping_words_stay_separate_and_require_review(self):
        rows, info = group_word_quads([word(10, 20, width=50, ident='a'),
                                      word(35, 20, width=50, ident='b')], (600, 400))
        self.assertEqual(len(rows), 2)
        self.assertTrue(info['grouping_ambiguous'])
        self.assertEqual(len(info['overlapping_word_pairs']), 1)

    def test_near_tie_next_words_are_not_arbitrarily_chosen(self):
        # Two equally plausible following rows for one detected word.
        words = [word(10, 50, height=20, ident='a'), word(45, 43, height=20, ident='b'),
                 word(45, 57, height=20, ident='c')]
        rows, info = group_word_quads(words, (600, 400))
        self.assertEqual(len(rows), 3)
        self.assertTrue(info['grouping_ambiguous'])
        self.assertIn('a', info['ambiguous_word_ids'])

    def test_transitive_baseline_drift_cannot_bridge_rows(self):
        rows, info = group_word_quads([word(10, 20, height=20, ident='a'),
                                      word(45, 28, height=20, ident='b'),
                                      word(80, 36, height=20, ident='c')], (600, 400))
        self.assertEqual(len(rows), 3)
        self.assertTrue(info['grouping_ambiguous'])

    def test_box_corner_permutation_does_not_change_crop(self):
        original = word(100, 100, angle=25)
        permuted = copy.deepcopy(original)
        permuted['quad'] = [original['quad'][i] for i in (2, 0, 3, 1)]
        before, _ = group_word_quads([original], (600, 400))
        after, _ = group_word_quads([permuted], (600, 400))
        self.assertEqual(before, after)

    def test_no_detections_is_valid_empty_geometry(self):
        rows, info = group_word_quads([], (600, 400))
        self.assertEqual(rows, [])
        self.assertEqual(info['words_observed'], 0)
        self.assertFalse(info['grouping_ambiguous'])

    def test_invalid_geometry_is_rejected_instead_of_becoming_evidence(self):
        for bad in ([[0, 0]] * 4,
                    [[0, 0], [20, 0], [20, float('nan')], [0, 20]],
                    [[-1, 0], [20, 0], [20, 20], [0, 20]],
                    [[0, 0], [20, 0], [40, 0], [0, 20]]):
            with self.subTest(quad=bad), self.assertRaises(ValueError):
                group_word_quads([{'quad': bad, 'detector_score': .9}], (600, 400))

    def test_detector_scores_cannot_be_nonfinite_or_correctness_labels(self):
        for value in (float('nan'), float('inf'), -1, 1.1, True):
            with self.subTest(score=value), self.assertRaises(ValueError):
                group_word_quads([word(10, 20, score=value)], (600, 400))

    def test_input_limits_do_not_silently_drop_words(self):
        with self.assertRaisesRegex(ValueError, 'word_count'):
            group_word_quads([word(10, 20, ident='a'), word(45, 20, ident='b')],
                             (600, 400), YoloDetectorConfig(max_det=1))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            group_word_quads([], (4000, 4000))

    def test_registered_config_rejects_unbounded_geometry_tuning(self):
        for config in (YoloDetectorConfig(imgsz=123), YoloDetectorConfig(max_det=5000),
                       YoloDetectorConfig(max_gap_height=20), YoloDetectorConfig(confidence=float('nan'))):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()


if __name__ == '__main__':
    unittest.main()
