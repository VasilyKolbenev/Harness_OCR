"""Portable DB+CTC reader. Two lazy CPU sessions, one sequential owner.

Only page raster pixels enter OCR. Neither ground truth, guessed document
fields nor Qwen output can become CTC observations. Scores are diagnostics;
no calibrated document/field acceptance is inferred from them.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import threading
import time

from .db_geometry import (DetectorConfig, ctc_decode, ctc_preprocess,
                          detector_postprocess, detector_preprocess, rectify_quad)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class CheapReader:
    def __init__(self, models_dir, *, max_lines=200, threads=4):
        if type(max_lines) is not int or not 1 <= max_lines <= 200:
            raise ValueError('cheap_reader_max_lines_must_be_1_to_200')
        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError('cheap_reader_invalid_thread_budget')
        self.models_dir = Path(models_dir)
        self.max_lines, self.threads = max_lines, threads
        self.config = DetectorConfig()
        self._detector = self._recognizer = None
        self._guard = threading.Lock()
        self._artifacts = {}
        self.chars = None

    def _load(self):
        if self._detector is not None:
            return
        # Check only files actually consumed here; the bundle preparation tool
        # verifies the entire allowlisted model bundle, including its README.
        from ..models import model_lock
        import yaml
        lock = model_lock()['models']
        artifacts = {}
        for name in ('detector', 'cyrillic'):
            spec = {row['file']: row for row in lock[name]['files']}
            for filename in ('inference.onnx', 'inference.yml'):
                path = self.models_dir / name / filename
                if not path.is_file() or _sha(path) != spec[filename]['sha256']:
                    raise ValueError(f'cheap_reader_artifact_sha_mismatch:{name}/{filename}')
                artifacts[f'{name}/{filename}'] = spec[filename]['sha256']
        detection = yaml.safe_load((self.models_dir / 'detector/inference.yml').read_text(encoding='utf-8'))
        post = detection['PostProcess']
        if (post.get('name'), post.get('thresh'), post.get('box_thresh'), post.get('max_candidates'), post.get('unclip_ratio')) != ('DBPostProcess', .3, .6, 1000, 1.5):
            raise ValueError('unsupported_detector_postprocess_configuration')
        recognition = yaml.safe_load((self.models_dir / 'cyrillic/inference.yml').read_text(encoding='utf-8'))
        chars = recognition['PostProcess']['character_dict']
        if not isinstance(chars, list) or not chars or any(not isinstance(c, str) or not c for c in chars):
            raise ValueError('invalid_cyrillic_character_dictionary')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        detector = ort.InferenceSession(str(self.models_dir / 'detector/inference.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        recognizer = ort.InferenceSession(str(self.models_dir / 'cyrillic/inference.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        if len(detector.get_inputs()) != 1 or len(recognizer.get_inputs()) != 1:
            raise ValueError('unsupported_cheap_reader_input_signature')
        self.chars = ['blank', *chars, ' ']
        self._artifacts = artifacts
        self._detector, self._recognizer = detector, recognizer

    def read(self, page, source_sha256, output_dir):
        if not self._guard.acquire(blocking=False):
            raise RuntimeError('cheap_reader_concurrent_call_forbidden')
        try:
            return self._read(page, source_sha256, output_dir)
        finally:
            self._guard.release()

    def _read(self, page, source_sha256, output_dir):
        import numpy as np
        from PIL import Image
        from ..core.common import identifier, sha256
        sha256(source_sha256)
        number = page.get('page')
        if type(number) is not int or number < 1:
            raise ValueError('invalid_page_number')
        page_id = identifier(page.get('page_id'), 'page_id')
        source = Path(page['path']).resolve(strict=True)
        output = Path(output_dir).resolve()
        # A fresh page-specific folder makes every crop immutable on disk.
        if output.exists() and any(output.iterdir()):
            raise ValueError('cheap_reader_output_must_be_empty')
        output.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with Image.open(source) as opened:
            if int(opened.getexif().get(274, 1)) != 1:
                raise ValueError('cheap_reader_requires_upright_page_raster')
            image = opened.convert('RGB')
        if list(image.size) != page.get('source_size'):
            raise ValueError('page_source_size_does_not_match_actual_raster')
        raster_sha = _sha(source)
        if page.get('raster_sha256') and page['raster_sha256'] != raster_sha:
            raise ValueError('page_raster_sha_mismatch')
        transform = page.get('transform')
        if not isinstance(transform, dict) or transform.get('source_sha256') != source_sha256:
            raise ValueError('page_transform_not_bound_to_original_source')
        if transform.get('raster_sha256') and transform['raster_sha256'] != raster_sha:
            raise ValueError('page_transform_raster_sha_mismatch')
        self._load()
        tensor, preprocessing = detector_preprocess(image, self.config)
        outputs = self._detector.run(None, {self._detector.get_inputs()[0].name: tensor})
        prediction = np.asarray(outputs[0], np.float32)
        if prediction.ndim != 4 or prediction.shape[:2] != (1, 1):
            raise ValueError('expected_db_output_1x1xHxW')
        quads, detection = detector_postprocess(prediction[0, 0], list(image.size), self.config)
        line_limited = len(quads) > self.max_lines
        observations, regions, ctc_diagnostics = [], [], []
        for index, row in enumerate(quads[:self.max_lines], 1):
            crop, proof = rectify_quad(image, row['quad'])
            crop_path = output / f'{page_id}_line{index}.png'
            if crop_path.resolve() == source:
                raise ValueError('crop_would_overwrite_source')
            crop.save(crop_path, format='PNG')
            array, metadata = ctc_preprocess(crop)
            prediction = self._recognizer.run(None, {self._recognizer.get_inputs()[0].name: array})[0]
            decoded = ctc_decode(prediction, self.chars)
            quad = row['quad']
            bbox = [min(p[0] for p in quad), min(p[1] for p in quad), max(p[0] for p in quad), max(p[1] for p in quad)]
            region_id, observation_id = f'{page_id}_r{index}', f'{page_id}_ctc{index}'
            crop_sha = _sha(crop_path)
            geometry = {**proof, 'raster_sha256': raster_sha, 'crop_sha256': crop_sha,
                        'original_source_sha256': source_sha256, 'page_transform': transform,
                        'coordinates_refer_to': 'actual_page_raster_pixels', 'source_size': list(image.size)}
            regions.append({'id': region_id, 'page': number, 'page_id': page_id,
                'path': str(crop_path), 'quad': quad, 'bbox': bbox, 'source_size': list(image.size),
                'source_sha256': source_sha256, 'crop_sha256': crop_sha, 'transform': geometry,
                'role_hint_routing_only': True, 'semantic_field_role_established': False})
            if decoded['text']:
                observations.append({'id': observation_id, 'source_sha256': source_sha256,
                    'text': decoded['text'], 'evidence_kind': 'image_region', 'page': number,
                    'page_id': page_id, 'quad': quad, 'source_size': list(image.size), 'completed': True,
                    'capture_id': f'{page_id}:ctc:{crop_sha[:20]}', 'model_id': 'cyrillic-ctc',
                    'source_transform': transform, 'raster_transform_proof': geometry,
                    'score_calibrated': False})
            ctc_diagnostics.append({'region_id': region_id, 'observation_id': observation_id if decoded['text'] else None,
                **decoded, 'preprocess': metadata, 'actual_output_shape': list(np.asarray(prediction).shape),
                'semantic_field_role_established': False, 'character_time_steps_are_not_pixel_boxes': True})
        incomplete = line_limited or bool(detection.get('contour_limit_reached'))
        return {'observations': observations, 'regions': regions, 'completed': not incomplete,
            'stop_reason': 'line_or_contour_limit' if incomplete else 'db_ctc_finished', 'diagnostics': {
            'completed': not incomplete, 'needs_review': incomplete, 'line_limit_reached': line_limited,
            'lines_detected': len(quads), 'lines_processed': len(regions), 'max_lines': self.max_lines,
            'stop_reason': 'line_or_contour_limit' if incomplete else 'db_ctc_finished',
            'seconds': time.perf_counter() - started, 'runtime': {'backend': 'onnxruntime', 'device': 'CPU',
                'intra_op_threads': self.threads, 'inter_op_threads': 1, 'execution': 'sequential', 'models_parallel': False},
            'artifact_sha256': dict(self._artifacts), 'raster_sha256': raster_sha,
            'preprocess': preprocessing, 'detector': detection, 'ctc': ctc_diagnostics,
            'scores_are_correctness_probabilities': False, 'dictionary_repair_or_ground_truth_used': False}}
