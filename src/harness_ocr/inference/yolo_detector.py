"""Opt-in, offline YOLO26-Text detector; recognition stays in Harness.

Only SHA-pinned local checkpoints can load. Ultralytics is imported lazily and
must be installed by the deployment preparation step, never by inference.
Primary interface reference: https://docs.ultralytics.com/modes/predict/ and
https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.OBB
OBB.xyxyxyxy coordinates already refer to the original input image.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
import os
import tempfile
import time

from .yolo_geometry import YoloDetectorConfig, group_word_quads


def _file_sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


class YoloTextDetector:
    def __init__(self, models_dir, *, variant='yolo26l', device='cpu', config=YoloDetectorConfig(), threads=4):
        if variant not in ('yolo26l', 'yolo26x'):
            raise ValueError('unsupported_yolo_text_variant')
        if device not in ('cpu', 'cuda'):
            raise ValueError('yolo_device_must_be_cpu_or_cuda')
        if type(threads) is not int or not 1 <= threads <= 16:
            raise ValueError('yolo_invalid_thread_budget')
        config.validate()
        self.models_dir, self.variant, self.device = Path(models_dir), variant, device
        self.config, self._model = config, None
        self._checkpoint_sha, self._runtime_version = None, None
        self._config_dir = None
        self.threads, self._torch = threads, None

    def load(self):
        if self._model is not None:
            return self
        from ..models import model_lock
        name = self.variant + '-text'
        spec = model_lock()['models'].get(name)
        if not spec:
            raise ValueError('yolo_checkpoint_not_registered')
        filename = name + '.pt'
        registered = {row['file']: row for row in spec['files']}.get(filename)
        if not registered:
            raise ValueError('yolo_checkpoint_file_not_registered')
        checkpoint = self.models_dir / name / filename
        # Requiring the exact existing file prevents Ultralytics' name-based
        # automatic download fallback. Resolve before importing its runtime.
        if not checkpoint.is_file() or _file_sha(checkpoint) != registered['sha256']:
            raise ValueError(f'yolo_artifact_sha_mismatch:{name}/{filename}')
        checkpoint = checkpoint.resolve(strict=True)
        os.environ['YOLO_AUTOINSTALL'] = 'false'
        os.environ['YOLO_OFFLINE'] = 'true'
        if 'YOLO_CONFIG_DIR' not in os.environ:
            self._config_dir = tempfile.TemporaryDirectory(prefix='harness-yolo-config-')
            os.environ['YOLO_CONFIG_DIR'] = self._config_dir.name
        import torch
        torch.set_num_threads(self.threads)
        import ultralytics
        from ultralytics import YOLO
        # Also disable these switches if another local component imported the
        # optional runtime earlier in this same isolated worker.
        import ultralytics.utils as yolo_utils
        import ultralytics.utils.torch_utils as yolo_torch_utils
        yolo_utils.AUTOINSTALL, yolo_utils.ONLINE = False, False
        # select_device('cpu') otherwise resets Torch to the library default
        # (often eight), silently invalidating a four-thread DB/YOLO comparison.
        yolo_utils.NUM_THREADS = yolo_torch_utils.NUM_THREADS = self.threads
        model = YOLO(str(checkpoint), task='obb')
        if getattr(model, 'task', None) != 'obb':
            raise ValueError('yolo_text_checkpoint_must_be_obb')
        self._model, self._checkpoint_sha = model, registered['sha256']
        self._runtime_version = ultralytics.__version__
        self._torch = torch
        return self

    def detect(self, image):
        size = list(image.size)
        if len(size) != 2 or any(type(v) is not int or v < 16 for v in size) or size[0] * size[1] > self.config.max_source_pixels:
            raise ValueError('yolo_source_dimensions_out_of_bounds')
        self.load()
        resources = {'cuda_peak_allocated_bytes': None, 'cuda_peak_reserved_bytes': None,
                     'cuda_allocated_before_bytes': None, 'cuda_reserved_before_bytes': None,
                     'cuda_synchronized': False,
                     'cuda_peak_scope': 'whole_worker_process_during_detector_call_including_other_resident_models'}
        if self.device == 'cuda':
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError('yolo_requested_cuda_unavailable')
            torch.cuda.synchronize()
            resources['cuda_allocated_before_bytes'] = torch.cuda.memory_allocated()
            resources['cuda_reserved_before_bytes'] = torch.cuda.memory_reserved()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        # PIL input is explicitly RGB. The library maps letterboxed predictions
        # back to this exact original image; applying another inverse would be
        # an erroneous double mapping.
        results = self._model.predict(source=image.convert('RGB'), imgsz=self.config.imgsz,
                                      conf=self.config.confidence, iou=self.config.iou,
                                      max_det=self.config.max_det, device=self.device,
                                      half=False, augment=False, rect=False, batch=1,
                                      stream=False, save=False, save_txt=False,
                                      verbose=False)
        if not isinstance(results, list) or len(results) != 1:
            raise ValueError('yolo_expected_one_page_result')
        result = results[0]
        if tuple(result.orig_shape) != (size[1], size[0]) or result.obb is None:
            raise ValueError('yolo_original_raster_shape_or_obb_mismatch')
        quads = result.obb.xyxyxyxy.detach().cpu().tolist()
        scores = result.obb.conf.detach().cpu().tolist()
        classes = result.obb.cls.detach().cpu().tolist()
        if not (len(quads) == len(scores) == len(classes)) or len(quads) > self.config.max_det:
            raise ValueError('yolo_invalid_obb_output_count')
        if any(value != 0 for value in classes):
            raise ValueError('yolo_text_checkpoint_nontext_class')
        words, clipped_words = [], 0
        for i, (quad, score) in enumerate(zip(quads, scores), 1):
            # A rotated rectangle can straddle the raster border. Explicitly
            # clip (and flag) it before creating evidence crops in our code.
            if len(quad) != 4 or any(len(point) != 2 or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in point) for point in quad):
                raise ValueError('yolo_expected_finite_4x2_quad')
            clipped = [[min(size[0], max(0., x)), min(size[1], max(0., y))] for x, y in quad]
            clipped_words += clipped != quad
            words.append({'quad': clipped, 'detector_score': score, 'word_id': f'w{i:04d}'})
        rows, geometry = group_word_quads(words, size, self.config)
        if self.device == 'cuda':
            torch.cuda.synchronize()
            resources.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                             cuda_synchronized=True)
        saturated = len(quads) >= self.config.max_det
        return rows, {**geometry, **resources, 'backend': 'ultralytics-pytorch', 'model': self.variant + '-text',
                      'checkpoint_sha256': self._checkpoint_sha, 'ultralytics_version': self._runtime_version,
                      'device': self.device, 'precision': 'fp32', 'source_size': size,
                      'requested_cpu_threads': self.threads,
                      'actual_torch_cpu_threads': self._torch.get_num_threads() if self._torch is not None else None,
                      'detection_limit_reached': saturated, 'contour_limit_reached': saturated,
                      'word_bounds_clipped': clipped_words,
                      'grouping_ambiguous': bool(geometry['grouping_ambiguous'] or clipped_words),
                      'source_mapping': 'Ultralytics_OBB_xyxyxyxy_original_image_pixels',
                      'retained_quads': len(rows), 'seconds': time.perf_counter() - started,
                      'recognizer': 'none_detector_only', 'automatic_downloads_permitted': False}

    def close(self):
        self._model = None
