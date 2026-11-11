"""Bounded, model-free DB geometry and Cyrillic CTC arithmetic.

Adapted from this project's detector/CTC research implementation. DB reference:
PaddlePaddle Authors, PaddleX text_detection/processors.py (2024), Apache-2.0:
https://github.com/PaddlePaddle/PaddleX/blob/develop/paddlex/inference/models/text_detection/processors.py
https://github.com/PaddlePaddle/PaddleX/blob/develop/LICENSE

No bitwise equivalence with PaddleX is claimed. The registered detector uses
max-side 1920/nearest32 rather than its YAML's resize_long960; CTC resizing is
Pillow bilinear. Neural runtimes and optional geometry libraries load lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class DetectorConfig:
    max_side: int = 1920
    stride: int = 32
    map_threshold: float = .3
    box_threshold: float = .6
    unclip_ratio: float = 1.5
    max_candidates: int = 1000
    max_source_pixels: int = 12_000_000

    def validate(self):
        if self.max_side != 1920 or self.stride != 32:
            raise ValueError('detector_registered_resize_is_max1920_stride32')
        if not 0 < self.map_threshold < 1 or not 0 < self.box_threshold <= 1:
            raise ValueError('invalid_db_threshold')
        if not 0 < self.unclip_ratio <= 3:
            raise ValueError('invalid_db_unclip')
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 1000:
            raise ValueError('detector_contour_limit')
        if type(self.max_source_pixels) is not int or not 1 <= self.max_source_pixels <= 12_000_000:
            raise ValueError('detector_pixel_limit')


def _libraries():
    import cv2
    import pyclipper
    return cv2, pyclipper


def array_sha(array):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def detector_preprocess(image, config=DetectorConfig()):
    import numpy as np
    cv2, _ = _libraries()
    config.validate()
    rgb = np.asarray(image.convert('RGB'))
    height, width = rgb.shape[:2]
    if min(height, width) < 16 or width * height > config.max_source_pixels:
        raise ValueError('detector_source_dimensions_out_of_bounds')
    ratio = min(1., config.max_side / max(height, width))
    scaled_height, scaled_width = int(height * ratio), int(width * ratio)
    target_height = max(32, int(round(scaled_height / 32) * 32))
    target_width = max(32, int(round(scaled_width / 32) * 32))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if (target_width, target_height) != (width, height):
        bgr = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    values = bgr.astype(np.float32) / np.float32(255.)
    values = (values - np.asarray([.485, .456, .406], np.float32)) / np.asarray([.229, .224, .225], np.float32)
    tensor = np.ascontiguousarray(values.transpose(2, 0, 1)[None])
    return tensor, {'source_size': [width, height], 'resized_size': [target_width, target_height],
        'ratio_width': target_width / width, 'ratio_height': target_height / height,
        'uniform_scale_before_stride': ratio,
        'stride_rounding_delta_pixels': [target_width - scaled_width, target_height - scaled_height],
        'resize_policy': 'max1920/no_deliberate_uniform_upscale/nearest32_min32',
        'model_yaml_default_difference': 'max1920 instead of resize_long960',
        'channel_order': 'BGR', 'normalization': '(pixel/255-ImageNet_mean)/ImageNet_std',
        'mean': [.485, .456, .406], 'std': [.229, .224, .225],
        'resize_interpolation': 'OpenCV INTER_LINEAR', 'input_tensor_sha256': array_sha(tensor)}


def order_quad(points):
    """Return min-area-rectangle corners in TL, TR, BR, BL order."""
    import numpy as np
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError('expected_finite_4x2_quad')
    by_x = values[np.argsort(values[:, 0], kind='stable')]
    left = by_x[:2][np.argsort(by_x[:2, 1], kind='stable')]
    right = by_x[2:][np.argsort(by_x[2:, 1], kind='stable')]
    return np.asarray([left[0], right[0], right[1], left[1]], np.float32)


def map_points(points, matrix):
    """Project actual raster-pixel points; reject singular/infinite mappings."""
    import numpy as np
    points = np.asarray(points, np.float64)
    matrix = np.asarray(matrix, np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError('invalid_homography_shape')
    if not np.isfinite(points).all() or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError('invalid_homography')
    projected = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    if np.any(np.abs(projected[:, 2]) < 1e-12):
        raise ValueError('homography_point_at_infinity')
    return projected[:, :2] / projected[:, 2:]


def _mini_quad(contour):
    import numpy as np
    cv2, _ = _libraries()
    rectangle = cv2.minAreaRect(np.asarray(contour, np.float32).reshape(-1, 1, 2))
    return order_quad(cv2.boxPoints(rectangle)), float(min(rectangle[1]))


def _quad_score(heatmap, points):
    import numpy as np
    cv2, _ = _libraries()
    height, width = heatmap.shape
    points = np.asarray(points, np.float32)
    x0 = max(0, min(math.floor(float(points[:, 0].min())), width - 1))
    x1 = max(0, min(math.ceil(float(points[:, 0].max())), width - 1))
    y0 = max(0, min(math.floor(float(points[:, 1].min())), height - 1))
    y1 = max(0, min(math.ceil(float(points[:, 1].max())), height - 1))
    if x1 < x0 or y1 < y0:
        return 0.
    mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    local = points - np.asarray([x0, y0], np.float32)
    cv2.fillPoly(mask, local.reshape(1, -1, 2).astype(np.int32), 1)
    return float(cv2.mean(heatmap[y0:y1 + 1, x0:x1 + 1], mask)[0])


def _unclip_quad(points, ratio):
    import numpy as np
    cv2, pyclipper = _libraries()
    polygon = np.asarray(points, np.float32)
    area = abs(float(cv2.contourArea(polygon)))
    perimeter = float(cv2.arcLength(polygon, True))
    if area <= 0 or perimeter <= 0:
        return None
    precision = 1024.
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(np.rint(polygon * precision).astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = offset.Execute(area * ratio / perimeter * precision)
    if len(expanded) != 1 or len(expanded[0]) < 4:
        return None
    return np.asarray(expanded[0], np.float32) / precision


def detector_postprocess(heatmap, source_size, config=DetectorConfig()):
    """DB contours are routing geometry, never evidence of a semantic role."""
    import numpy as np
    cv2, _ = _libraries()
    config.validate()
    values = np.asarray(heatmap, np.float32)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1.00001:
        raise ValueError('expected_finite_sigmoid_db_map')
    width, height = source_size
    if any(type(x) is not int or x <= 0 for x in (width, height)):
        raise ValueError('invalid_detector_source_size')
    bitmap = (values > config.map_threshold).astype(np.uint8) * 255
    contours = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    output = []
    for contour in contours[:config.max_candidates]:
        quad, side = _mini_quad(contour)
        if side < 3:
            continue
        score = _quad_score(values, quad)
        if score < config.box_threshold:
            continue
        expanded = _unclip_quad(quad, config.unclip_ratio)
        if expanded is None:
            continue
        quad, side = _mini_quad(expanded)
        if side < 5:
            continue
        mapped = np.empty_like(quad)
        mapped[:, 0] = np.clip(np.rint(quad[:, 0] * width / values.shape[1]), 0, width)
        mapped[:, 1] = np.clip(np.rint(quad[:, 1] * height / values.shape[0]), 0, height)
        mapped = order_quad(mapped)
        if abs(float(cv2.contourArea(mapped))) <= 1:
            continue
        output.append({'quad': mapped.tolist(), 'quad_map_xy': quad.tolist(), 'detector_score': score})
    output.sort(key=lambda row: (sum(p[1] for p in row['quad']) / 4, min(p[0] for p in row['quad'])))
    return output, {'heatmap_shape': list(values.shape), 'contours_observed': len(contours),
        'contours_considered': min(len(contours), config.max_candidates),
        'contour_limit_reached': len(contours) > config.max_candidates,
        'retained_quads': len(output), 'binary_map_sha256': array_sha(bitmap),
        'map_threshold': config.map_threshold, 'box_threshold': config.box_threshold,
        'source_mapping': 'actual_raster_width/map_width_and_actual_raster_height/map_height',
        'unclip_integer_subpixel_precision': 1024, 'dilation': False,
        'reading_order': 'quad_center_y_then_min_x_geometry_only'}


def rectify_quad(image, quad):
    import numpy as np
    from PIL import Image
    cv2, _ = _libraries()
    rgb = np.asarray(image.convert('RGB'))
    points = order_quad(quad)
    width = max(1, int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))))
    height = max(1, int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))))
    destination = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    forward = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(rgb, forward, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotation = 0
    if height / width >= 1.5:
        crop = np.rot90(crop)
        rotation = 90
        forward = np.asarray([[0, 1, 0], [-1, 0, width], [0, 0, 1]], np.float64) @ forward
    inverse = np.linalg.inv(forward)
    actual = Image.fromarray(np.ascontiguousarray(crop), 'RGB')
    return actual, {'quad_source_xy': points.tolist(), 'view_size': list(actual.size),
        'transform_source_to_view': forward.tolist(), 'transform_view_to_source': inverse.tolist(),
        'coordinate_convention': 'actual_page_raster_pixel_edges_to_rectified_canvas',
        'rotation_degrees_counterclockwise': rotation,
        'operations': [{'operation': 'perspective_rectification', 'interpolation': 'INTER_CUBIC', 'border': 'BORDER_REPLICATE'},
            *([{'operation': 'orthogonal_rotation', 'degrees_counterclockwise': 90}] if rotation else [])],
        'invented_text_or_superresolution': False}


def ctc_preprocess(image):
    import numpy as np
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError('invalid_ctc_image_dimensions')
    canvas = min(3200, int(48 * max(320 / 48, width / height)))
    resized = min(canvas, math.ceil(48 * width / height))
    array = np.asarray(image.resize((resized, 48), Image.Resampling.BILINEAR), np.float32)[:, :, ::-1]
    array = (array.transpose(2, 0, 1) / 255. - .5) / .5
    result = np.zeros((1, 3, 48, canvas), np.float32)
    result[0, :, :, :resized] = array
    return result, {'resize_implementation': 'Pillow_bilinear_not_bitwise_cv2_equivalent',
        'channel_order': 'BGR', 'normalization': '(x/255-0.5)/0.5', 'input_size': [width, height],
        'resized_width': resized, 'canvas_width': canvas, 'valid_ratio': min(1., resized / canvas),
        'width_compressed_to_limit': math.ceil(48 * width / height) > 3200,
        'input_text_clipped': False, 'input_tensor_sha256': array_sha(result)}


def ctc_decode(prediction, chars):
    """Greedy CTC: collapse adjacent IDs, then remove blank; no text repair."""
    import numpy as np
    values = np.asarray(prediction)
    if not isinstance(chars, list) or len(chars) < 2 or any(not isinstance(c, str) for c in chars):
        raise ValueError('invalid_ctc_dictionary')
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] == 0 or values.shape[2] != len(chars) or not np.isfinite(values).all():
        raise ValueError('ctc_output_dictionary_mismatch_or_nonfinite')
    probabilities = values[0].astype(np.float64)
    transformed = probabilities.min() < 0 or probabilities.max() > 1.00001 or not np.allclose(probabilities.sum(-1), 1., atol=1e-3)
    if transformed:
        shifted = probabilities - probabilities.max(-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(-1, keepdims=True)
    indexes = probabilities.argmax(-1)
    chosen, previous = [], None
    for step, index in enumerate(indexes):
        index = int(index)
        if index != previous and index != 0:
            order = np.argsort(probabilities[step], kind='stable')[-2:][::-1]
            chosen.append({'character': chars[index], 'index': index, 'time_step': step,
                'score': float(probabilities[step, index]), 'second_character': chars[int(order[1])],
                'second_score': float(probabilities[step, order[1]])})
        previous = index
    return {'text': ''.join(c['character'] for c in chosen), 'characters': chosen,
        'mean_ctc_score': sum(c['score'] for c in chosen) / len(chosen) if chosen else None,
        'score_calibrated_as_correctness': False, 'softmax_applied': bool(transformed),
        'blank_id': 0, 'collapse_order': 'collapse_adjacent_ids_then_remove_blank'}
