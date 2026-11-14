"""Conservative, text-blind word-to-line geometry for the YOLO experiment.

All coordinates are actual page-raster pixel edges. Grouping establishes no
semantic field role. It does not insert text, repair characters, infer labels,
or join across a large whitespace gap. Uncertain joins remain separate and are
reported, rather than choosing the candidate that makes a validator pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


GROUPING_VERSION = 'yolo-word-lines/1'


@dataclass(frozen=True)
class YoloDetectorConfig:
    imgsz: int = 1280
    confidence: float = .25
    iou: float = .7
    max_det: int = 1000
    max_source_pixels: int = 12_000_000
    max_angle_degrees: float = 10.
    max_baseline_offset_height: float = .5
    max_gap_height: float = 1.5
    max_height_ratio: float = 2.
    duplicate_iou: float = .85

    def validate(self):
        if type(self.imgsz) is not int or not 320 <= self.imgsz <= 2560 or self.imgsz % 32:
            raise ValueError('yolo_imgsz_must_be_stride32_320_to_2560')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in
               (self.confidence, self.iou, self.max_angle_degrees,
                self.max_baseline_offset_height, self.max_gap_height,
                self.max_height_ratio, self.duplicate_iou)):
            raise ValueError('yolo_nonfinite_configuration')
        if not 0 < self.confidence < 1 or not 0 < self.iou <= 1:
            raise ValueError('yolo_invalid_threshold')
        if type(self.max_det) is not int or not 1 <= self.max_det <= 2000:
            raise ValueError('yolo_invalid_detection_limit')
        if type(self.max_source_pixels) is not int or not 1 <= self.max_source_pixels <= 12_000_000:
            raise ValueError('yolo_invalid_pixel_limit')
        if not 0 < self.max_angle_degrees <= 15 or not 0 < self.max_baseline_offset_height <= .5:
            raise ValueError('yolo_invalid_line_angle_or_baseline')
        if not 0 < self.max_gap_height <= 1.5 or not 1 <= self.max_height_ratio <= 2:
            raise ValueError('yolo_invalid_line_gap_or_height')
        if not .8 <= self.duplicate_iou < 1:
            raise ValueError('yolo_invalid_duplicate_threshold')

    def binding(self):
        self.validate()
        return {**asdict(self), 'grouping_version': GROUPING_VERSION}


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _area(points):
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(points, points[1:] + points[:1]))) / 2


def _quad(points, source_size):
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError('yolo_expected_4x2_quad')
    output = []
    width, height = source_size
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in point):
            raise ValueError('yolo_expected_finite_4x2_quad')
        x, y = map(float, point)
        if not 0 <= x <= width or not 0 <= y <= height:
            raise ValueError('yolo_quad_outside_actual_raster')
        output.append((x, y))
    center = (sum(p[0] for p in output) / 4, sum(p[1] for p in output) / 4)
    output.sort(key=lambda p: math.atan2(p[1] - center[1], p[0] - center[0]))
    if _area(output) <= 1 or any(_cross(output[i], output[(i + 1) % 4],
                                                   output[(i + 2) % 4]) <= 1e-8 for i in range(4)):
        raise ValueError('yolo_quad_degenerate_or_nonconvex')
    edges = [(math.dist(output[i], output[(i + 1) % 4]), i) for i in range(4)]
    length, i = max(edges)
    a, b = output[i], output[(i + 1) % 4]
    axis = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
    if axis[0] < -1e-8 or (abs(axis[0]) <= 1e-8 and axis[1] < 0):
        axis = (-axis[0], -axis[1])
    perpendicular = (-axis[1], axis[0])
    bounds = _bounds(output, axis, perpendicular)
    return output, center, axis, bounds[3] - bounds[2]


def _bounds(points, axis, perpendicular):
    horizontal = [_dot(p, axis) for p in points]
    vertical = [_dot(p, perpendicular) for p in points]
    return min(horizontal), max(horizontal), min(vertical), max(vertical)


def _intersection(subject, clip):
    result = list(subject)
    for a, b in zip(clip, clip[1:] + clip[:1]):
        incoming, result = result, []
        if not incoming:
            break
        for p, q in zip(incoming, incoming[1:] + incoming[:1]):
            inside_p, inside_q = _cross(a, b, p) >= -1e-8, _cross(a, b, q) >= -1e-8
            if inside_p:
                result.append(p)
            if inside_p != inside_q:
                cp, cq = _cross(a, b, p), _cross(a, b, q)
                fraction = cp / (cp - cq)
                result.append((p[0] + fraction * (q[0] - p[0]),
                               p[1] + fraction * (q[1] - p[1])))
    return result


def _iou(a, b):
    if max(p[0] for p in a) <= min(p[0] for p in b) or max(p[0] for p in b) <= min(p[0] for p in a):
        return 0.
    if max(p[1] for p in a) <= min(p[1] for p in b) or max(p[1] for p in b) <= min(p[1] for p in a):
        return 0.
    polygon = _intersection(a, b)
    intersection = _area(polygon) if len(polygon) >= 3 else 0.
    return intersection / max(1e-12, _area(a) + _area(b) - intersection)


def group_word_quads(words, source_size, config=YoloDetectorConfig()):
    """Return (DB-compatible line rows, diagnostics), retaining word evidence.

    Joins require mutual unique nearest neighbours in the word's long-axis
    direction. Thus a near-tie, overlapping word, large gap, or column boundary
    is not resolved using a recognised string or expected field value.
    """
    config.validate()
    if not isinstance(source_size, (tuple, list)) or len(source_size) != 2 or any(
            type(v) is not int or v < 16 for v in source_size) or math.prod(source_size) > config.max_source_pixels:
        raise ValueError('yolo_source_dimensions_out_of_bounds')
    if not isinstance(words, list) or len(words) > config.max_det:
        raise ValueError('yolo_word_count_exceeds_registered_limit')
    retained, duplicate_ids = [], []
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            raise ValueError('yolo_invalid_word_record')
        score = word.get('detector_score')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('yolo_invalid_detection_score')
        quad, center, axis, height = _quad(word.get('quad'), source_size)
        word_id = word.get('word_id', f'w{index + 1:04d}')
        if not isinstance(word_id, str) or not word_id or len(word_id) > 100:
            raise ValueError('yolo_invalid_word_identifier')
        retained.append({'quad': quad, 'center': center, 'axis': axis, 'height': height,
                         'detector_score': float(score), 'word_id': word_id})
    if len({w['word_id'] for w in retained}) != len(retained):
        raise ValueError('yolo_duplicate_word_identifier')
    # Prefer the highest-confidence *geometry* for duplicate detections. Every
    # discarded identity is recorded; distinct overlapping words are retained.
    retained.sort(key=lambda w: (-w['detector_score'], w['center'][1], w['center'][0], str(w['word_id'])))
    unique = []
    for word in retained:
        duplicate = next((old for old in unique if _iou(word['quad'], old['quad']) >= config.duplicate_iou), None)
        if duplicate is None:
            unique.append(word)
        else:
            duplicate_ids.append({'word_id': word['word_id'], 'kept_word_id': duplicate['word_id']})
    unique.sort(key=lambda w: (w['center'][1], w['center'][0], str(w['word_id'])))
    ambiguous, overlaps = set(), set()
    neighbours = {i: {1: [], -1: []} for i in range(len(unique))}
    cosine = math.cos(math.radians(config.max_angle_degrees))
    for i, word in enumerate(unique):
        axis, height = word['axis'], word['height']
        perpendicular = (-axis[1], axis[0])
        bounds = _bounds(word['quad'], axis, perpendicular)
        for j, other in enumerate(unique):
            if i == j or abs(_dot(axis, other['axis'])) < cosine:
                continue
            short = min(height, other['height'])
            if max(height, other['height']) / max(short, 1e-8) > config.max_height_ratio:
                continue
            delta = (other['center'][0] - word['center'][0], other['center'][1] - word['center'][1])
            if abs(_dot(delta, perpendicular)) > config.max_baseline_offset_height * short:
                continue
            target = _bounds(other['quad'], axis, perpendicular)
            direction = 1 if _dot(delta, axis) > 0 else -1
            gap = target[0] - bounds[1] if direction == 1 else bounds[0] - target[1]
            if gap < -.15 * short:
                overlaps.add(tuple(sorted((i, j))))
                continue
            if gap <= config.max_gap_height * short:
                neighbours[i][direction].append((max(0., gap) / short, j))
    choices = {}
    for i in neighbours:
        for direction, candidates in neighbours[i].items():
            candidates.sort(key=lambda row: (row[0], row[1]))
            if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= .25:
                ambiguous.add(i)
                continue
            if candidates:
                choices[i, direction] = candidates[0][1]
    parent = list(range(len(unique)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (i, direction), j in choices.items():
        if choices.get((j, -direction)) == i and i not in ambiguous and j not in ambiguous:
            parent[find(j)] = find(i)
    groups = {}
    for i in range(len(unique)):
        groups.setdefault(find(i), []).append(unique[i])
    checked_groups = []
    for group in groups.values():
        anchor = max(group, key=lambda w: max(math.dist(w['quad'][i], w['quad'][(i + 1) % 4]) for i in range(4)))
        axis, short = anchor['axis'], min(w['height'] for w in group)
        perpendicular = (-axis[1], axis[0])
        offsets = [_dot(w['center'], perpendicular) for w in group]
        if (max(offsets) - min(offsets) > config.max_baseline_offset_height * short + 1e-6 or
                any(abs(_dot(axis, w['axis'])) < cosine for w in group)):
            # Do not let pairwise-compatible words form a curved chain that
            # crosses rows. Keeping singleton crops is the conservative result.
            ambiguous.update(i for i, word in enumerate(unique) if word in group)
            checked_groups.extend([[word] for word in group])
        else:
            checked_groups.append(group)
    rows, clipped_lines = [], 0
    width, height = source_size
    for group in checked_groups:
        axis = max(group, key=lambda w: max(math.dist(w['quad'][i], w['quad'][(i + 1) % 4]) for i in range(4)))['axis']
        perpendicular = (-axis[1], axis[0])
        points = [p for word in group for p in word['quad']]
        x0, x1, y0, y1 = _bounds(points, axis, perpendicular)
        corners = [(x * axis[0] + y * perpendicular[0], x * axis[1] + y * perpendicular[1])
                   for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        clipped = [[min(width, max(0., x)), min(height, max(0., y))] for x, y in corners]
        clipped_lines += any(abs(a - b) > 1e-5 for old, new in zip(corners, clipped) for a, b in zip(old, new))
        group.sort(key=lambda w: (_dot(w['center'], axis), str(w['word_id'])))
        rows.append({'quad': clipped, 'detector_score': min(w['detector_score'] for w in group),
                     'word_quads': [[list(p) for p in w['quad']] for w in group],
                     'word_ids': [w['word_id'] for w in group],
                     'semantic_field_role_established': False})
    rows.sort(key=lambda row: (sum(p[1] for p in row['quad']) / 4, min(p[0] for p in row['quad'])))
    return rows, {'grouping_version': GROUPING_VERSION, 'grouping_parameters': config.binding(),
                  'words_observed': len(words), 'words_after_duplicate_removal': len(unique),
                  'duplicate_detections': duplicate_ids, 'lines_grouped': len(rows),
                  'ambiguous_word_ids': [unique[i]['word_id'] for i in sorted(ambiguous)],
                  'overlapping_word_pairs': [[unique[i]['word_id'], unique[j]['word_id']] for i, j in sorted(overlaps)],
                  'line_bounds_clipped': clipped_lines,
                  'grouping_ambiguous': bool(ambiguous or overlaps or clipped_lines),
                  'coordinates_refer_to': 'actual_page_raster_pixels',
                  'semantic_field_role_established': False,
                  'scores_are_correctness_probabilities': False,
                  'text_or_dictionary_used_for_grouping': False}
