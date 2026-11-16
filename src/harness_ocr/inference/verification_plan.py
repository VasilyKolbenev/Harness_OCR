"""Plan bounded blind reads of value-bearing, pixel-localized observations.

An evidence record is not necessarily a verification ROI. In particular a
whole-page OCR reading may corroborate a date whose actual value span belongs
to a CTC crop. This planner preserves that corroboration, but does not send the
whole page to a blind field verifier. Unsupported required spans remain explicit
and cannot be counted as verified. Pixel reconstruction and literal comparison
are still performed by ``vision`` and the evidence core, respectively.

The plan is not a role proof, field-completeness certificate or acceptance rule.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math

from ..core.common import identifier, sha256
from ..core.evidence import _box, _quad
from ..typed_fields import typed_supported, value_identity
from ..util import digest

VERSION = 'value-span-visual-plan-v1'
LIVE = {'supported', 'human_supported'}
SCOPE = 'complete_value_bearing_observation_literals'
UNION_SCOPE = 'source_value_span_union_v1'


def _polygon_key(points):
    return digest(sorted([[0.0 if v == 0 else float(v) for v in point] for point in points]))


def _value_spans(claim):
    raw = claim.get('raw_value')
    if not isinstance(raw, dict):
        return claim.get('spans')
    spans = []
    for item in raw['items']:
        spans.extend(item['spans'])
        for cell in item.get('cells', []):
            spans.extend(cell['spans'])
        for key in ('bik', 'account_location'):
            if item.get(key):
                spans.extend(item[key]['spans'])
    return spans


def value_observation_ids(claim):
    """Actual value sources, never role-only or arbitrary corroboration IDs.

    Missing localization raises instead of yielding an empty set which a caller
    might mistakenly regard as vacuously complete optical verification.
    """
    try:
        spans = _value_spans(claim)
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError('invalid_value_span_proof') from exc
    if not isinstance(spans, list) or not spans:
        raise ValueError('value_span_missing')
    ids = set()
    for span in spans:
        if (not isinstance(span, dict) or not isinstance(span.get('observation_id'), str)
                or type(span.get('start')) is not int or type(span.get('end')) is not int
                or not 0 <= span['start'] < span['end']):
            raise ValueError('invalid_value_span_proof')
        identifier(span['observation_id'], 'observation_id')
        ids.add(span['observation_id'])
    return ids


def _span_problem(span, observations, evidence_ids):
    if (not isinstance(span, dict)
            or not {'observation_id', 'start', 'end'} <= set(span)
            or set(span) - {'observation_id', 'start', 'end', 'separator_after'}):
        return 'invalid_value_span_shape'
    oid = span['observation_id']
    if not isinstance(oid, str) or oid not in evidence_ids or oid not in observations:
        return 'value_span_not_bound_to_claim'
    obs = observations[oid]
    if obs.get('active') is not True or obs.get('completed') is not True:
        return 'value_observation_not_active_complete'
    a, b = span['start'], span['end']
    if (type(a) is not int or type(b) is not int or not isinstance(obs.get('text'), str)
            or not 0 <= a < b <= len(obs['text'])):
        return 'invalid_value_span_offsets'
    separator = span.get('separator_after', '')
    if not isinstance(separator, str) or separator.strip():
        return 'value_span_join_invents_characters'
    return None


def _claim_spans(claim, observations):
    """Every declared fragment must be valid; never select the favorable span."""
    evidence = claim.get('evidence_ids')
    if (not isinstance(evidence, list) or not evidence
            or any(not isinstance(oid, str) for oid in evidence)
            or len(evidence) != len(set(evidence))):
        return [], 'invalid_claim_evidence'
    raw = claim.get('raw_value')
    if isinstance(raw, dict):
        try:
            spans = _value_spans(claim)
            if not typed_supported(raw, [observations[oid] for oid in evidence]):
                return spans, 'typed_value_not_supported'
        except (KeyError, ValueError, TypeError, IndexError):
            return [], 'invalid_typed_value_proof'
    else:
        spans = claim.get('spans')
    if not isinstance(spans, list) or not spans:
        return [], 'value_span_missing'
    for span in spans:
        problem = _span_problem(span, observations, evidence)
        if problem:
            return spans, problem
    if isinstance(raw, str):
        slices = [observations[s['observation_id']]['text'][s['start']:s['end']]
                  for s in spans]
        joined = ''.join(piece + span.get('separator_after', '')
                         for piece, span in zip(slices, spans))
        # Repeated exact value spans can be independent corroborating readings;
        # otherwise the complete, ordered span join must reproduce the literal.
        if joined != raw and not all(piece == raw and not span.get('separator_after')
                                     for piece, span in zip(slices, spans)):
            return spans, 'value_spans_do_not_reproduce_literal'
    elif not isinstance(raw, dict):
        return spans, 'unsupported_claim_value'
    return spans, None


def value_span_union(snapshot, canonical_claim_id):
    """Recompute an exact bounding rectangle of complete source observations.

    Character positions never become pixel positions. A partial text-line span
    therefore cannot make a union ROI; its complete observation must be checked
    by the original whole-observation comparison instead.
    """
    claim = snapshot.get('claims', {}).get(canonical_claim_id, {})
    if claim.get('status') not in LIVE or claim.get('id') != canonical_claim_id:
        raise ValueError('visual_union_claim_not_live')
    field = claim.get('field_key')
    current = snapshot.get('fields', {}).get(field, {})
    raw = claim.get('raw_value')
    if (not isinstance(raw, str) or not raw or len(raw) > 2048
            or current.get('raw_value') != raw or current.get('needs_review')):
        raise ValueError('visual_union_requires_unambiguous_scalar_field')
    for other in snapshot.get('claims', {}).values():
        if other.get('field_key') != field:
            continue
        if other.get('status') == 'conflict' or (other.get('status') in LIVE and (
                other.get('raw_value') != raw or other.get('party_role') != claim.get('party_role')
                or other.get('evidence_conflict'))):
            raise ValueError('visual_union_active_field_conflict')
    observations = snapshot.get('observations', {})
    spans, problem = _claim_spans(claim, observations)
    if problem:
        raise ValueError('visual_union_' + problem)
    ids = [s['observation_id'] for s in spans]
    if len(ids) < 2 or len(set(ids)) != len(ids):
        raise ValueError('visual_union_requires_distinct_source_observations')
    joined = ''.join(observations[s['observation_id']]['text'][s['start']:s['end']]
                     + s.get('separator_after', '') for s in spans)
    if joined != raw:
        # Agreeing duplicate readings are alternatives, not adjacent words of
        # one literal. They need separate original-observation comparisons.
        raise ValueError('visual_union_requires_ordered_literal_join')
    allowed = {snapshot.get('source_sha256'),
               *snapshot.get('metadata', {}).get('allowed_source_sha256', [])}
    identity, raster, boxes = None, None, []
    geometries = set()
    for span in spans:
        obs = observations[span['observation_id']]
        if span['start'] != 0 or span['end'] != len(obs['text']):
            raise ValueError('visual_union_partial_observation_span')
        if (obs.get('evidence_kind') != 'image_region' or obs.get('source_sha256') not in allowed):
            raise ValueError('visual_union_requires_bound_image_observations')
        size = obs.get('source_size')
        if (not isinstance(size, list) or len(size) != 2
                or any(type(v) is not int or v <= 0 for v in size)):
            raise ValueError('visual_union_invalid_source_size')
        key = (obs.get('source_sha256'), obs.get('page'), obs.get('page_id'), tuple(size))
        if identity is None:
            identity = key
        if key != identity:
            raise ValueError('visual_union_cross_source_or_page')
        proof = obs.get('raster_transform_proof', {})
        if not isinstance(proof, dict):
            raise ValueError('visual_union_missing_raster_proof')
        rsha = proof.get('raster_sha256')
        sha256(rsha, 'raster_sha256')
        if raster is None:
            raster = rsha
        if raster != rsha:
            raise ValueError('visual_union_different_source_rasters')
        if obs.get('quad') is not None:
            _quad(obs['quad'], size)
            points = obs['quad']
            box = [min(p[0] for p in points), min(p[1] for p in points),
                   max(p[0] for p in points), max(p[1] for p in points)]
            geometry = _polygon_key(points)
        else:
            box = obs.get('bbox')
            _box(box, size)
            x0, y0, x1, y1 = box
            geometry = _polygon_key([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        if box == [0, 0, *size]:
            raise ValueError('visual_union_whole_page_observation')
        if geometry in geometries:
            raise ValueError('visual_union_duplicate_pixel_region')
        geometries.add(geometry)
        boxes.append(box)
    for previous, current_box in zip(boxes, boxes[1:]):
        overlap = min(previous[3],current_box[3])-max(previous[1],current_box[1])
        height = min(previous[3]-previous[1],current_box[3]-current_box[1])
        if overlap < .6*height or current_box[0] <= previous[0]:
            # Blind transcription preserves printed line breaks. A scalar
            # assembled across vertical lines must use separate source checks;
            # comparing it to a space-joined union would create a false conflict.
            raise ValueError('visual_union_requires_one_horizontal_baseline')
    bbox = [math.floor(min(b[0] for b in boxes)), math.floor(min(b[1] for b in boxes)),
            math.ceil(max(b[2] for b in boxes)), math.ceil(max(b[3] for b in boxes))]
    if bbox == [0, 0, *identity[3]]:
        raise ValueError('visual_union_whole_page_extent')
    union = {'version': UNION_SCOPE, 'canonical_claim_id': canonical_claim_id,
             'field_key': field, 'value_spans': deepcopy(spans), 'observation_ids': ids,
             'raw_value': raw, 'source_sha256': identity[0], 'page': identity[1],
             'page_id': identity[2], 'source_size': list(identity[3]),
             'raster_sha256': raster, 'bbox': bbox,
             'role_verified': False, 'automatic_acceptance': False}
    union['id'] = 'value_union_' + digest(union)[:24]
    return union


def validated_value_observation_ids(claim, observations):
    """Require the declared source spans to reproduce this claim's value."""
    _, problem = _claim_spans(claim, observations)
    if problem:
        raise ValueError(problem)
    return value_observation_ids(claim)


def verified_value_observation_ids(snapshot, field_key, current):
    """Replay checks before counting coverage; stale/forged checks give no credit."""
    from ..core.state import DocumentStore
    ids = current.get('claim_ids', [])
    if not isinstance(ids, list) or any(not isinstance(cid, str) for cid in ids):
        return set()
    refs = set(ids)
    live = {cid for cid in refs if snapshot.get('claims', {}).get(cid, {}).get('status') in LIVE
            and snapshot['claims'][cid].get('field_key') == field_key}
    required = set()
    try:
        for cid in live:
            required.update(validated_value_observation_ids(snapshot['claims'][cid],
                                                          snapshot.get('observations',{})))
    except (ValueError,KeyError,TypeError):
        return set()
    verified = set()
    for check in snapshot.get('visual_checks', {}).values():
        if not isinstance(check, dict):
            continue
        ids = check.get('claim_ids', [])
        if (check.get('field_key') != field_key or check.get('verdict') != 'correct'
                or not isinstance(ids, list) or not ids or any(not isinstance(cid, str) for cid in ids)
                or not set(ids) <= live):
            continue
        if check.get('comparison_scope') == UNION_SCOPE:
            keys = ('id', 'field_key', 'claim_ids', 'capture_id', 'comparison_scope',
                    'canonical_claim_id', 'value_spans')
        elif check.get('comparison_scope') == 'complete_original_observation_literal':
            keys = ('id', 'field_key', 'claim_ids', 'capture_id', 'expected_observation_id')
        else:
            continue
        try:
            replayed = DocumentStore._visual_check({k: check[k] for k in keys}, snapshot)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
        if replayed.get('verdict') != 'correct':
            continue
        expected = (replayed['expected_observation_ids'] if replayed['comparison_scope'] == UNION_SCOPE
                    else [replayed['expected_observation_id']])
        verified.update(set(expected) & required)
    return verified


def _region_problem(obs, region):
    """Cheap binding checks only. vision._prepare still proves actual pixels."""
    if not isinstance(region, dict):
        return 'value_roi_unavailable'
    required = {'id', 'path', 'crop_sha256', 'source_path', 'source_sha256',
                'raster_path', 'raster_sha256', 'page', 'page_id', 'source_size',
                'page_transform', 'crop_kind', 'transform'}
    key = 'quad' if region.get('crop_kind') == 'perspective' else 'bbox'
    if set(region) != required | {key}:
        return 'value_roi_descriptor_not_closed'
    try:
        identifier(region['id'], 'visual_region_id')
        identifier(region['page_id'], 'page_id')
        for name in ('source_sha256', 'raster_sha256', 'crop_sha256'):
            sha256(region[name])
        json.dumps(region, allow_nan=False)
        if any(not isinstance(region[name], str) or not region[name]
               for name in ('path', 'source_path', 'raster_path')):
            return 'value_roi_path_invalid'
        if (not isinstance(region['source_size'], list) or len(region['source_size']) != 2
                or any(type(v) is not int or v <= 0 for v in region['source_size'])):
            return 'value_roi_descriptor_invalid'
        if region.get('crop_kind') not in {'axis_aligned', 'perspective'}:
            return 'value_roi_crop_kind_invalid'
        if not isinstance(region.get('transform'), dict):
            return 'value_roi_transform_invalid'
        if type(region['page']) is not int or region['page'] < 1:
            return 'value_roi_page_invalid'
        if key == 'quad':
            _quad(region[key], region['source_size'])
        else:
            _box(region[key], region['source_size'])
            if any(type(v) is not int for v in region[key]):
                return 'value_roi_axis_coordinates_must_be_pixels'
            if region.get('transform', {}).get('version') != 'axis-aligned-crop-v1':
                return 'value_roi_axis_transform_invalid'
    except (ValueError, TypeError, KeyError):
        return 'value_roi_descriptor_invalid'
    for name in ('source_sha256', 'page', 'page_id', 'source_size'):
        if obs.get(name) != region[name]:
            return 'value_roi_source_geometry_mismatch'
    # A quad and its enclosing bounding box are not interchangeable pixel ROIs.
    if obs.get(key) != region[key] or (key == 'bbox' and obs.get('quad') is not None):
        return 'value_roi_source_geometry_mismatch'
    width, height = region['source_size']
    if key == 'bbox':
        whole = region[key] == [0, 0, width, height]
    else:
        whole = set(map(tuple, region[key])) == {(0, 0), (width, 0),
                                               (width, height), (0, height)}
    if whole:
        return 'whole_page_value_span_requires_localization'
    transform = region['page_transform']
    if (not isinstance(transform, dict) or transform.get('version') != 'raster-v1'
            or transform.get('orientation') != 'EXIF_transposed'
            or transform.get('source_sha256') != region['source_sha256']
            or transform.get('raster_sha256') != region['raster_sha256']):
        return 'value_roi_page_transform_not_bound'
    proof = obs.get('raster_transform_proof')
    if not isinstance(proof, dict) or proof != region['transform']:
        return 'value_roi_observation_transform_not_bound'
    for name, expected in (('original_source_sha256', region['source_sha256']),
                           ('raster_sha256', region['raster_sha256']),
                           ('crop_sha256', region['crop_sha256']),
                           ('source_size', region['source_size'])):
        if name in proof and proof[name] != expected:
            return 'value_roi_observation_transform_not_bound'
    return None


def _region_key(region):
    # Only the opaque alias may differ. Paths, source/pixel digests, geometry and
    # transforms must agree; text equality alone never deduplicates two crops.
    return digest({k: v for k, v in region.items() if k != 'id'})


def plan_verification(snapshot, requested_fields, visual_regions, *, max_regions=2):
    """Return blind-read requests, fair ROI allocation and explicit coverage.

    ``supporting_only_observation_ids`` remain evidence and are never labelled as
    visually verified. ``required_value_observation_ids`` cover all value spans
    of all supported claims, including every typed fragment. An unsupported or
    deferred required span means incomplete coverage, irrespective of any
    favorable capture from a different claim.
    """
    if not isinstance(snapshot, dict) or not isinstance(visual_regions, dict):
        raise ValueError('visual_plan_requires_server_state')
    if (not isinstance(requested_fields, (list, tuple)) or not requested_fields
            or len(requested_fields) > 100
            or any(not isinstance(f, str) for f in requested_fields)
            or len(requested_fields) != len(set(requested_fields))):
        raise ValueError('visual_plan_requires_unique_fields')
    for field in requested_fields:
        identifier(field, 'field_key')
    if type(max_regions) is not int or not 0 <= max_regions <= 4:
        raise ValueError('invalid_visual_plan_region_budget')
    observations, claims, fields = (snapshot.get(k, {}) for k in ('observations', 'claims', 'fields'))
    if not all(isinstance(x, dict) for x in (observations, claims, fields)):
        raise ValueError('visual_plan_state_maps_required')
    allowed = {snapshot.get('source_sha256'),
               *snapshot.get('metadata', {}).get('allowed_source_sha256', [])}
    coverage, units, descriptors, union_regions, queues = {}, {}, {}, {}, {}
    review_reasons = []
    for field in requested_fields:
        current = fields.get(field, {})
        item = {'scope': SCOPE, 'live_claim_ids': [], 'required_value_observation_ids': [],
                'all_raster_evidence_ids': [], 'supporting_only_observation_ids': [],
                'role_only_observation_ids': [], 'unsupported_required': [],
                'deferred_budget': [], 'selected_observation_ids': [],
                'role_verified_by_plan': False, 'field_completeness_proven': False,
                'automatic_acceptance': False}
        coverage[field] = item
        if current.get('raw_value') is None:
            item['not_applicable_reason'] = 'no_unambiguous_observed_value'
            queues[field] = []
            continue
        refs = current.get('claim_ids', [])
        live = [claims[cid] for cid in refs if cid in claims and claims[cid].get('status') in LIVE]
        required_ids, raster_ids, role_ids = set(), set(), set()
        field_keys = []
        if not live:
            item['unsupported_required'].append({'reason': 'supported_value_claim_missing'})
        if any(cid not in claims or claims[cid].get('status') not in LIVE for cid in refs):
            item['unsupported_required'].append({'reason': 'field_claim_reference_not_live'})
        if any(value_identity(c.get('raw_value')) != value_identity(current.get('raw_value')) for c in live):
            item['unsupported_required'].append({'reason': 'field_live_claim_disagreement'})
        for claim in live:
            cid = claim.get('id')
            item['live_claim_ids'].append(cid)
            if claim.get('field_key') != field:
                item['unsupported_required'].append({'claim_id': cid, 'reason': 'claim_field_mismatch'})
                continue
            evidence = claim.get('evidence_ids', [])
            for oid in evidence:
                obs = observations.get(oid, {})
                if obs.get('evidence_kind') == 'image_region':
                    raster_ids.add(oid)
                if (obs.get('active') is not True or obs.get('completed') is not True
                        or obs.get('source_sha256') not in allowed):
                    item['unsupported_required'].append({'claim_id': cid, 'observation_id': oid,
                                                        'reason': 'claim_evidence_not_active_complete_source'})
            role_ids.update(claim.get('role_evidence_ids', []))
            spans, problem = _claim_spans(claim, observations)
            if problem:
                # Missing or corrupt localization cannot turn all its raster
                # evidence into optional corroboration and a false empty pass.
                required_ids.update(oid for oid in evidence
                                    if observations.get(oid, {}).get('evidence_kind') == 'image_region')
                item['unsupported_required'].append({'claim_id': cid, 'reason': problem})
                continue
            span_ids = sorted({s['observation_id'] for s in spans})
            required_ids.update(oid for oid in span_ids
                                if observations[oid].get('evidence_kind') == 'image_region')
            # A whole, lossless multi-observation value gets one real union
            # crop rather than one model slot per word. Partial-line spans
            # deliberately fall back to the original whole-observation check.
            union = None
            if len(span_ids) > 1 and all(not _region_problem(observations[oid], visual_regions.get(oid))
                                         for oid in span_ids):
                try:
                    union = value_span_union(snapshot, cid)
                except ValueError:
                    pass
            if union is not None:
                geometry = {k: deepcopy(union[k]) for k in ('version', 'source_sha256', 'raster_sha256',
                            'page', 'page_id', 'source_size', 'bbox')}
                key = 'union_' + digest(geometry)
                union_regions.setdefault(key, geometry)
                units[(field, 'union:' + cid)] = {'kind': 'value_span_union', 'field_key': field,
                    'canonical_claim_id': cid, 'claim_ids': [cid],
                    'observation_ids': union['observation_ids'], 'value_spans': union['value_spans'],
                    'union': geometry, 'region_key': key}
                if key not in field_keys:
                    field_keys.append(key)
                continue
            for oid in span_ids:
                obs = observations[oid]
                if obs.get('evidence_kind') != 'image_region':
                    continue
                required_ids.add(oid)
                problem = _region_problem(obs, visual_regions.get(oid))
                if problem:
                    item['unsupported_required'].append({'claim_id': cid, 'observation_id': oid,
                                                        'reason': problem})
                    continue
                region = visual_regions[oid]
                key = _region_key(region)
                descriptors.setdefault(key, deepcopy(region))
                unit = units.setdefault((field, oid), {'kind': 'observation', 'field_key': field,
                    'observation_id': oid, 'observation_ids': [oid], 'claim_ids': [], 'region_key': key})
                if cid not in unit['claim_ids']:
                    unit['claim_ids'].append(cid)
                if key not in field_keys:
                    field_keys.append(key)
        item['live_claim_ids'].sort()
        item['required_value_observation_ids'] = sorted(required_ids)
        item['all_raster_evidence_ids'] = sorted(raster_ids)
        item['supporting_only_observation_ids'] = sorted(raster_ids - required_ids)
        item['role_only_observation_ids'] = sorted(role_ids - required_ids)
        queues[field] = field_keys
        if item['unsupported_required']:
            review_reasons.append('visual_verification_localization_required:' + field)

    selected = []
    for index in range(max((len(q) for q in queues.values()), default=0)):
        for field in requested_fields:
            if index >= len(queues[field]):
                continue
            key = queues[field][index]
            if key not in selected and len(selected) < max_regions:
                selected.append(key)
    requests = []
    for key in selected:
        for field in requested_fields:
            for (unit_field, _), unit in sorted(units.items()):
                if unit_field == field and unit['region_key'] == key:
                    unit['claim_ids'].sort()
                    requests.append(unit)
                    coverage[field]['selected_observation_ids'].extend(unit['observation_ids'])
    for field, item in coverage.items():
        item['selected_observation_ids'] = sorted(set(item['selected_observation_ids']))
        item['deferred_budget'] = sorted({oid for (f, _), unit in units.items()
                                         if f == field and unit['region_key'] not in selected
                                         for oid in unit['observation_ids']})
        if item['deferred_budget']:
            review_reasons.append('visual_verification_unavailable_or_budget:' + field)
        item['all_required_regions_scheduled'] = bool(item['required_value_observation_ids']) and not (
            item['unsupported_required'] or item['deferred_budget'])
    plan = {'version': VERSION, 'scope': SCOPE, 'document_id': snapshot.get('document_id'),
            'source_sha256': snapshot.get('source_sha256'), 'planned_revision': snapshot.get('revision'),
            'max_regions': max_regions, 'regions': {k: descriptors[k] for k in selected if k in descriptors},
            'union_regions': {k: union_regions[k] for k in selected if k in union_regions},
            'requests': requests, 'coverage': coverage,
            'review_reasons': sorted(set(review_reasons)), 'automatic_acceptance': False,
            'pixel_verification_required': True}
    plan['id'] = 'visual_plan_' + digest(plan)[:24]
    return plan


plan_visual_verification = plan_verification
