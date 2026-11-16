"""Server-authored, source-bound reading regions. Geometry routes work, never values."""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

from ..util import digest

VERSION = 'source-bound-field-regions-v2-upright'


def region_box(region):
    quad = region.get('quad')
    value = ([min(p[0] for p in quad), min(p[1] for p in quad),
              max(p[0] for p in quad), max(p[1] for p in quad)] if quad else region.get('bbox'))
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)
            or value[2] <= value[0] or value[3] <= value[1]):
        raise ValueError('reading_region_requires_real_geometry')
    return list(value)


def same_region_geometry(observation, region):
    """Exact source/page/geometry identity; missing geometry never matches.

    A bounding box is not interchangeable with a perspective quad. Detector
    regions may carry an auxiliary enclosing bbox, but their quad remains the
    optical geometry when present. This helper routes work/descriptors only.
    """
    if not isinstance(observation, dict) or not isinstance(region, dict):
        return False
    if (not isinstance(observation.get('source_sha256'), str)
            or not observation['source_sha256']
            or observation.get('source_sha256') != region.get('source_sha256')
            or not isinstance(observation.get('page_id'), str) or not observation['page_id']
            or observation.get('page_id') != region.get('page_id')
            or type(observation.get('page')) is not int or observation['page'] < 1
            or observation.get('page') != region.get('page')
            or observation.get('source_size') != region.get('source_size')):
        return False
    size = observation.get('source_size')
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        return False
    left = 'quad' if observation.get('quad') is not None else 'bbox'
    right = 'quad' if region.get('quad') is not None else 'bbox'
    if left != right or observation.get(left) is None or region.get(right) is None:
        return False
    try:
        for item in (observation, region):
            if left == 'quad' and (not isinstance(item[left], list) or len(item[left]) != 4
                    or any(not isinstance(p, (list, tuple)) or len(p) != 2 for p in item[left])):
                return False
            box = region_box(item)
            if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
                return False
    except (ValueError, TypeError, IndexError, KeyError):
        return False
    return observation[left] == region[right]


def context_box(region, *, max_area_fraction=.30):
    """Add bounded label context around a real region, including vertical text.

    No character positions or field roles are inferred. A crop may recover a
    printed label; until it does, the candidate remains unanchored.
    """
    x0, y0, x1, y1 = region_box(region)
    size = region.get('source_size')
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError('reading_region_requires_source_size')
    width, height = size
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('reading_region_outside_source')
    w, h = x1-x0, y1-y0
    if w >= h:
        dx, dy = min(.12*width, max(2*h, .2*w)), min(.08*height, max(4*h, .02*height))
    else:
        dx, dy = min(.08*width, max(4*w, .02*width)), min(.12*height, max(2*w, .2*h))
    box = [max(0, math.floor(x0-dx)), max(0, math.floor(y0-dy)),
           min(width, math.ceil(x1+dx)), min(height, math.ceil(y1+dy))]
    if (box[2]-box[0])*(box[3]-box[1]) > max_area_fraction*width*height:
        # A broad page reading is a separate, explicit operation, not an ROI.
        return None
    return box


def materialize_region(page, bbox, output_dir, *, source_sha256, purpose='field_context',
                       max_page_pixels=12_000_000):
    """Crop exact ingest pixels and retain a replayable original-page transform."""
    from PIL import Image
    from ..core.common import identifier, sha256

    sha256(source_sha256)
    page_id = identifier(page['page_id'], 'page_id')
    raster_path = Path(page['path']).resolve(strict=True)
    data = raster_path.read_bytes()
    raster_sha = hashlib.sha256(data).hexdigest()
    if raster_sha != page.get('raster_sha256'):
        raise ValueError('reading_page_raster_sha_mismatch')
    transform = page.get('transform', {})
    if (transform.get('source_sha256') != source_sha256
            or transform.get('raster_sha256') != raster_sha):
        raise ValueError('reading_page_not_bound_to_source')
    with Image.open(io.BytesIO(data)) as opened:
        if opened.width*opened.height > max_page_pixels:
            raise ValueError('reading_page_pixel_limit')
        if int(opened.getexif().get(274, 1)) != 1:
            raise ValueError('reading_page_must_be_upright')
        image = opened.convert('RGB')
    if list(image.size) != page.get('source_size'):
        raise ValueError('reading_page_size_mismatch')
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in bbox)):
        raise ValueError('reading_crop_requires_finite_bbox')
    x0, y0, x1, y1 = [math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])]
    if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
        raise ValueError('reading_crop_outside_source')
    box = [x0,y0,x1,y1]
    ident = page_id + '_task_v2_' + digest([VERSION, source_sha256, raster_sha, box, purpose])[:20]
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    target = output / (ident + '.png')
    if target == raster_path:
        raise ValueError('reading_crop_would_overwrite_source')
    # Context may contain multiple upright lines. The detector's line
    # rectifier rotates tall boxes, which is inappropriate for this crop.
    crop = image.crop(tuple(box))
    proof = {'version':'axis-aligned-crop-v1', 'bbox_source_xy':box,
        'view_size':list(crop.size), 'coordinate_convention':'actual_page_raster_pixel_edges',
        'operations':[{'operation':'axis_aligned_crop'}], 'invented_text_or_superresolution':False}
    encoded=io.BytesIO();crop.save(encoded,format='PNG');crop_bytes=encoded.getvalue()
    crop_sha=hashlib.sha256(crop_bytes).hexdigest()
    try:
        with target.open('xb') as stream:
            stream.write(crop_bytes)
    except FileExistsError:
        if hashlib.sha256(target.read_bytes()).hexdigest()!=crop_sha:
            raise ValueError('immutable_reading_crop_changed')
    proof = {**proof, 'raster_sha256':raster_sha, 'crop_sha256':crop_sha,
        'original_source_sha256':source_sha256, 'page_transform':transform,
        'coordinates_refer_to':'actual_page_raster_pixels', 'source_size':list(image.size)}
    return {'id':ident, 'path':str(target), 'page':page['page'], 'page_id':page_id,
        'source_size':list(image.size), 'source_sha256':source_sha256,
        'bbox':box, 'crop_sha256':crop_sha, 'transform':proof,
        'reading_purpose':purpose, 'role_hint_routing_only':True,
        'semantic_field_role_established':False}


def page_from_visual_descriptor(descriptor):
    """Recover only the already source-bound ingest page, without new geometry."""
    return {'page':descriptor['page'], 'page_id':descriptor['page_id'],
        'path':descriptor['raster_path'], 'source_size':descriptor['source_size'],
        'raster_sha256':descriptor['raster_sha256'], 'transform':descriptor['page_transform']}
