"""Value-span routing and exact pixel-union contracts; no NN or network."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from harness_ocr.inference.verification_plan import (
    UNION_SCOPE, plan_verification, value_observation_ids, value_span_union,
    verified_value_observation_ids)
from harness_ocr.core.state import DocumentStore
from harness_ocr.util import digest, file_sha


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.state = {'document_id': 'doc', 'source_sha256': 'a'*64, 'revision': 1,
                      'metadata': {}, 'observations': {}, 'claims': {}, 'fields': {}}
        self.regions = {}

    def obs(self, oid, text, box=None, *, page=1):
        box = box or [10, 10, 100, 30]
        crop = digest([page, box]); size = [400, 240]
        page_transform = {'version': 'raster-v1', 'orientation': 'EXIF_transposed',
                          'source_sha256': 'a'*64, 'raster_sha256': 'b'*64}
        transform = {'version': 'axis-aligned-crop-v1', 'bbox_source_xy': box,
                     'raster_sha256': 'b'*64, 'crop_sha256': crop,
                     'original_source_sha256': 'a'*64, 'source_size': size}
        self.state['observations'][oid] = {'id': oid, 'text': text, 'active': True,
            'completed': True, 'evidence_kind': 'image_region', 'source_sha256': 'a'*64,
            'page': page, 'page_id': 'p'+str(page), 'source_size': size, 'bbox': box,
            'raster_transform_proof': transform}
        self.regions[oid] = {'id': 'region_'+oid, 'path': '/fixture/'+crop+'.png',
            'crop_sha256': crop, 'source_path': '/fixture/source.png',
            'source_sha256': 'a'*64, 'raster_path': '/fixture/page.png',
            'raster_sha256': 'b'*64, 'page': page, 'page_id': 'p'+str(page),
            'source_size': size, 'page_transform': page_transform,
            'crop_kind': 'axis_aligned', 'transform': transform, 'bbox': box}
        return self.state['observations'][oid]

    def span(self, oid, *, start=0, end=None, separator=None):
        span = {'observation_id': oid, 'start': start,
                'end': len(self.state['observations'][oid]['text']) if end is None else end}
        if separator is not None:
            span['separator_after'] = separator
        return span

    def claim(self, field, raw, spans, *, cid=None, evidence=None):
        cid = cid or 'claim_'+str(len(self.state['claims'])+1)
        claim = {'id': cid, 'status': 'supported', 'field_key': field,
                 'raw_value': raw, 'spans': spans, 'party_role': 'holder',
                 'evidence_ids': evidence or list(dict.fromkeys(s['observation_id'] for s in spans)),
                 'role_evidence_ids': [], 'role_spans': []}
        self.state['claims'][cid] = claim
        current = self.state['fields'].setdefault(field, {'raw_value': raw, 'claim_ids': [], 'needs_review': False})
        current['claim_ids'].append(cid)
        return claim

    def plan(self, *, budget=2):
        return plan_verification(self.state, list(self.state['fields']), self.regions, max_regions=budget)

    def test_corroborating_page_is_retained_but_does_not_consume_a_date_slot(self):
        self.obs('ctc', '10.11.2026')
        self.obs('page', 'Договор от 10.11.2026\nОстальной текст', [0, 0, 400, 240])
        self.claim('contract.date', '10.11.2026', [self.span('ctc')], evidence=['ctc', 'page'])
        before = deepcopy(self.state)
        p = self.plan(); c = p['coverage']['contract.date']
        self.assertEqual([r['observation_id'] for r in p['requests']], ['ctc'])
        self.assertEqual(c['required_value_observation_ids'], ['ctc'])
        self.assertEqual(c['supporting_only_observation_ids'], ['page'])
        self.assertEqual(c['all_raster_evidence_ids'], ['ctc', 'page'])
        self.assertTrue(c['all_required_regions_scheduled'])
        self.assertFalse(c['field_completeness_proven'])
        self.assertFalse(c['role_verified_by_plan'])
        self.assertEqual(self.state, before)
        self.assertFalse(p['automatic_acceptance'])

    def test_a_whole_page_value_span_needs_localization_instead_of_a_blind_read(self):
        self.obs('page', 'Дата: 10.11.2026', [0, 0, 400, 240])
        self.claim('contract.date', '10.11.2026', [self.span('page', start=6)])
        p = self.plan(); c = p['coverage']['contract.date']
        self.assertFalse(p['requests']); self.assertFalse(p['regions'])
        self.assertEqual(c['required_value_observation_ids'], ['page'])
        self.assertIn('whole_page_value_span_requires_localization', str(c['unsupported_required']))
        self.assertFalse(c['all_required_regions_scheduled'])

    def test_missing_localization_cannot_turn_a_value_into_optional_context(self):
        self.obs('o', '1234'); self.claim('passport.series', '1234', [], evidence=['o'])
        p = self.plan(); c = p['coverage']['passport.series']
        self.assertEqual(c['required_value_observation_ids'], ['o'])
        self.assertIn('value_span_missing', str(c['unsupported_required']))
        self.assertFalse(p['requests'])
        with self.assertRaises(ValueError):
            value_observation_ids(self.state['claims']['claim_1'])

    def test_every_span_must_be_bound_even_when_another_contains_the_right_value(self):
        self.obs('o', '1234'); self.obs('bad', '9999', [120, 10, 170, 30])
        c = self.claim('passport.series', '1234', [self.span('o'), self.span('bad')])
        p = self.plan()
        self.assertFalse(p['requests'])
        self.assertIn('do_not_reproduce', str(p['coverage']['passport.series']['unsupported_required']))
        c['spans'] = [self.span('o', end=99)]
        self.assertIn('invalid_value_span_offsets', str(self.plan()['coverage']))
        c['spans'] = [self.span('o', separator='X')]
        self.assertIn('invents_characters', str(self.plan()['coverage']))

    def test_missing_or_forged_roi_binding_is_explicitly_unavailable(self):
        self.obs('o', '1234'); self.claim('passport.series', '1234', [self.span('o')])
        original = deepcopy(self.regions['o'])
        variants = [None, {**original, 'source_sha256': 'd'*64},
                    {**original, 'bbox': [11, 10, 100, 30]},
                    {**original, 'path': None},
                    {**original, 'transform': None},
                    {**original, 'bbox': [10.0, 10.0, 100.0, 30.0]},
                    {**original, 'page_transform': {**original['page_transform'], 'raster_sha256': 'e'*64}},
                    {**original, 'path': original['path'], 'raw_value': '1234'}]
        for bad in variants:
            with self.subTest(bad=bad):
                self.regions['o'] = bad
                p = self.plan(); self.assertFalse(p['requests'])
                self.assertTrue(p['coverage']['passport.series']['unsupported_required'])

    def test_round_robin_preserves_all_claims_without_spending_every_slot_on_one_field(self):
        self.obs('a1', 'Альфа', [10, 10, 70, 30]); self.obs('a2', 'Альфа', [10, 40, 70, 60])
        self.obs('b', '1234', [100, 10, 150, 30])
        self.claim('payee.name', 'Альфа', [self.span('a1')])
        self.claim('payee.name', 'Альфа', [self.span('a2')])
        self.claim('passport.series', '1234', [self.span('b')])
        p = self.plan(); selected = {r['observation_id'] for r in p['requests']}
        self.assertEqual(selected, {'a1', 'b'})
        self.assertEqual(p['coverage']['payee.name']['deferred_budget'], ['a2'])
        self.assertFalse(p['coverage']['payee.name']['all_required_regions_scheduled'])
        self.assertTrue(p['coverage']['passport.series']['all_required_regions_scheduled'])

    def test_identical_pixel_descriptor_aliases_share_one_capture_not_just_equal_text(self):
        self.obs('a', '1234'); self.obs('b', '1234')
        self.claim('passport.series', '1234', [self.span('a')])
        self.claim('other.number', '1234', [self.span('b')])
        p = self.plan(budget=1)
        self.assertEqual(len(p['regions']), 1); self.assertEqual(len(p['requests']), 2)
        self.assertTrue(all(c['all_required_regions_scheduled'] for c in p['coverage'].values()))
        self.regions['b']['bbox'] = [120, 10, 170, 30]
        self.state['observations']['b']['bbox'] = [120, 10, 170, 30]
        p = self.plan(budget=1)
        self.assertEqual(len(p['requests']), 1)
        self.assertFalse(p['coverage']['other.number']['all_required_regions_scheduled'])

    def test_whole_word_union_is_one_slot_and_has_no_candidate_literal_in_crop_metadata(self):
        self.obs('a', 'ООО', [10, 10, 45, 30]); self.obs('b', 'Альфа', [50, 10, 100, 30])
        self.claim('payee.name', 'ООО Альфа', [self.span('a', separator=' '), self.span('b')])
        p = self.plan(budget=1); unit = p['requests'][0]
        self.assertEqual(unit['kind'], 'value_span_union')
        self.assertEqual(unit['observation_ids'], ['a', 'b'])
        self.assertEqual(len(p['union_regions']), 1); self.assertFalse(p['regions'])
        self.assertEqual(unit['union']['bbox'], [10, 10, 100, 30])
        self.assertNotIn('ООО Альфа', json.dumps(p['union_regions'], ensure_ascii=False))
        self.assertTrue(p['coverage']['payee.name']['all_required_regions_scheduled'])

    def test_partial_line_spans_get_whole_observation_checks_not_invented_character_boxes(self):
        self.obs('a', 'Фамилия: Иванов', [10, 10, 180, 30])
        self.obs('b', 'Имя: Иван', [10, 40, 180, 60])
        self.claim('holder.full_name', 'Иванов Иван', [self.span('a', start=9, separator=' '), self.span('b', start=5)])
        p = self.plan(budget=1)
        self.assertFalse(p['union_regions'])
        self.assertEqual(p['requests'][0]['kind'], 'observation')
        self.assertEqual(p['coverage']['holder.full_name']['deferred_budget'], ['b'])
        self.assertFalse(p['coverage']['holder.full_name']['all_required_regions_scheduled'])

    def test_vertical_name_words_keep_separate_rois_instead_of_a_space_joined_union(self):
        self.obs('a', 'Иванов', [10, 10, 110, 30])
        self.obs('b', 'Иван', [10, 60, 90, 80])
        claim = self.claim('holder.full_name', 'Иванов Иван',
                           [self.span('a', separator=' '), self.span('b')])
        before = deepcopy(self.state)
        with self.assertRaisesRegex(ValueError, 'one_horizontal_baseline'):
            value_span_union(self.state, claim['id'])
        limited = self.plan(budget=1)
        self.assertFalse(limited['union_regions'])
        self.assertEqual(limited['requests'][0]['kind'], 'observation')
        self.assertEqual(limited['coverage']['holder.full_name']['deferred_budget'], ['b'])
        self.assertFalse(limited['coverage']['holder.full_name']['all_required_regions_scheduled'])
        sufficient = self.plan(budget=2)
        self.assertFalse(sufficient['union_regions'])
        self.assertEqual({r['observation_id'] for r in sufficient['requests']}, {'a', 'b'})
        self.assertTrue(sufficient['coverage']['holder.full_name']['all_required_regions_scheduled'])
        self.assertFalse(sufficient['coverage']['holder.full_name']['field_completeness_proven'])
        self.assertEqual(self.state, before)

    def test_duplicate_agreeing_readings_are_not_concatenated_into_a_field_union(self):
        self.obs('a', '1234', [10, 10, 70, 30]); self.obs('b', '1234', [10, 60, 70, 80])
        self.claim('passport.series', '1234', [self.span('a'), self.span('b')])
        p = self.plan()
        self.assertFalse(p['union_regions'])
        self.assertEqual(len(p['regions']), 2)
        self.assertEqual({r['observation_id'] for r in p['requests']}, {'a', 'b'})

    def test_typed_compound_keeps_all_value_sources_under_a_small_budget(self):
        from harness_ocr.profiles import ProfileRegistry
        from harness_ocr.typed_fields import build_typed_candidates
        self.obs('a', 'Доверитель ИНН: 7707083893', [10, 10, 230, 30])
        self.obs('b', 'Доверитель ОГРН: 1027700132195', [10, 40, 280, 60])
        contract = ProfileRegistry().compile('D04', subset=['principal.identifier'])
        candidates = build_typed_candidates(list(self.state['observations'].values()), contract, 700)
        pieces = list(candidates.values())
        raw = deepcopy(pieces[0]['raw_value'])
        raw['items'] = [item for candidate in pieces for item in candidate['raw_value']['items']]
        claim = self.claim('principal.identifier', raw,
                           [span for candidate in pieces for span in candidate['spans']])
        self.assertEqual(value_observation_ids(claim), {'a', 'b'})
        p = self.plan(budget=1)
        self.assertFalse(p['union_regions'])
        self.assertEqual(p['coverage']['principal.identifier']['required_value_observation_ids'], ['a', 'b'])
        self.assertFalse(p['coverage']['principal.identifier']['all_required_regions_scheduled'])

    def test_conflicting_live_claim_cannot_be_hidden_by_a_valid_first_claim(self):
        self.obs('a', '1234'); self.obs('b', '9999', [110, 10, 170, 30])
        self.claim('passport.series', '1234', [self.span('a')])
        self.claim('passport.series', '9999', [self.span('b')])
        p = self.plan()
        self.assertIn('field_live_claim_disagreement', str(p['coverage']))
        self.assertFalse(p['coverage']['passport.series']['all_required_regions_scheduled'])
        self.assertEqual({r['observation_id'] for r in p['requests']}, {'a', 'b'})

    def test_zero_budget_and_invalid_limits_cannot_look_like_verified_success(self):
        self.obs('a', '1234'); self.claim('passport.series', '1234', [self.span('a')])
        p = self.plan(budget=0)
        self.assertFalse(p['requests']); self.assertEqual(p['coverage']['passport.series']['deferred_budget'], ['a'])
        for budget in [-1, True, 5]:
            with self.assertRaises(ValueError): self.plan(budget=budget)
        with self.assertRaises(ValueError):
            plan_verification(self.state, ['passport.series', 'passport.series'], self.regions)


class PixelUnionTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image, ImageDraw
        from harness_ocr.inference.vision import make_visual_region
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = Image.new('RGB', (400, 240), 'white')
        draw = ImageDraw.Draw(image); draw.text((10, 10), 'ACME', fill='black'); draw.text((80, 10), 'CITY', fill='black')
        self.source = self.root/'source.png'; self.raster = self.root/'raster.png'
        image.save(self.source); image.save(self.raster)
        self.sha, self.rsha = file_sha(self.source), file_sha(self.raster)
        self.page = {'path': str(self.raster), 'page': 1, 'page_id': 'p1', 'source_size': [400, 240],
            'raster_sha256': self.rsha, 'transform': {'version': 'raster-v1', 'orientation': 'EXIF_transposed',
                'source_sha256': self.sha, 'raster_sha256': self.rsha}}
        self.observations, self.regions = [], {}
        for oid, text, box in [('a', 'ACME', [10, 10, 60, 30]), ('b', 'CITY', [80, 10, 130, 30])]:
            roi = self.line_region(box, oid)
            self.regions[oid] = make_visual_region(self.source, self.page, roi, source_sha256=self.sha)
            self.observations.append({'id': oid, 'text': text, 'source_sha256': self.sha,
                'evidence_kind': 'image_region', 'page': 1, 'page_id': 'p1', 'source_size': [400, 240],
                'quad': roi['quad'], 'raster_transform_proof': roi['transform'], 'completed': True})
        self.claim = {'id': 'name_claim', 'field_key': 'payee.name', 'raw_value': 'ACME CITY',
            'evidence_ids': ['a', 'b'], 'spans': [{'observation_id': 'a', 'start': 0, 'end': 4, 'separator_after': ' '},
            {'observation_id': 'b', 'start': 0, 'end': 4}], 'role_evidence_ids': [], 'role_spans': [], 'party_role': 'payee'}
        self.store = DocumentStore(self.root/'state.sqlite')
        state = self.store.create_document('doc', self.sha, 'D07', {'field_keys': ['payee.name']})
        self.state = self.store.revise('doc', observations=self.observations, claims=[self.claim], expected_revision=state['revision'])

    def line_region(self, box, ident):
        """Actual detector-style quad pixels; context cropping is a separate path."""
        from PIL import Image
        from harness_ocr.inference.db_geometry import rectify_quad
        x0,y0,x1,y1 = box; quad = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
        with Image.open(self.raster) as image:
            crop, proof = rectify_quad(image, quad)
        target = self.root/(ident+'.png'); crop.save(target)
        proof = {**proof, 'raster_sha256':self.rsha, 'crop_sha256':file_sha(target),
            'original_source_sha256':self.sha, 'page_transform':self.page['transform'],
            'source_size':[400, 240], 'coordinates_refer_to':'actual_page_raster_pixels'}
        return {'id':ident, 'path':str(target), 'page':1, 'page_id':'p1', 'source_size':[400, 240],
            'source_sha256':self.sha, 'quad':quad, 'crop_sha256':file_sha(target), 'transform':proof}

    def capture(self, *, literal='ACME CITY', kind='quad'):
        from PIL import Image
        from harness_ocr.inference.region_tasks import materialize_region
        from harness_ocr.inference.vision import make_visual_region, prepare_visual_regions, parse_visual_output
        union = value_span_union(self.state, 'name_claim')
        if kind == 'quad':
            roi = self.line_region(union['bbox'], 'quad_union')
        else:
            roi = materialize_region(self.page, union['bbox'], self.root/'union',
                source_sha256=self.sha, purpose='verification_union')
        descriptor = make_visual_region(self.source, self.page, roi, source_sha256=self.sha)
        _, provenance = prepare_visual_regions([descriptor])
        output = {'text': json.dumps({'readings': {'r1': {'status': 'readable', 'literal': literal}}}),
                  'completed': True, 'stop_reason': 'eos', 'model_id': 'fixture-qwen', 'fresh_generation_state': True,
                  'model_bundle_sha256': 'd'*64}
        return parse_visual_output(output, provenance)[0]

    def check(self, capture):
        state = self.store.revise('doc', observations=[capture], expected_revision=self.state['revision'])
        item = {'id': 'union_check', 'field_key': 'payee.name', 'claim_ids': ['name_claim'],
                'capture_id': capture['id'], 'comparison_scope': UNION_SCOPE,
                'canonical_claim_id': 'name_claim', 'value_spans': deepcopy(self.claim['spans'])}
        return state, item

    def test_real_union_pixels_bind_one_blind_capture_to_every_word(self):
        capture = self.capture(); state, item = self.check(capture)
        state = self.store.revise('doc', visual_checks=[item], expected_revision=state['revision'])
        checked = state['visual_checks']['union_check']
        self.assertEqual(checked['verdict'], 'correct')
        self.assertEqual(checked['expected_observation_ids'], ['a', 'b'])
        self.assertFalse(checked['role_verified']); self.assertFalse(checked['automatic_acceptance'])
        self.assertEqual(verified_value_observation_ids(state, 'payee.name', state['fields']['payee.name']), {'a', 'b'})
        changed = self.store.revise('doc', invalidate=['b'], expected_revision=state['revision'])
        self.assertEqual(verified_value_observation_ids(changed, 'payee.name', changed['fields']['payee.name']), set())

    def test_exact_axis_aligned_union_is_also_valid(self):
        capture = self.capture(kind='bbox'); state, item = self.check(capture)
        self.assertEqual(DocumentStore._visual_check(item, state)['verdict'], 'correct')

    def test_wrong_optical_literal_stays_a_disagreement_and_cannot_credit_coverage(self):
        state, item = self.check(self.capture(literal='ACME C1TY'))
        state = self.store.revise('doc', visual_checks=[item], expected_revision=state['revision'])
        self.assertEqual(state['visual_checks']['union_check']['verdict'], 'incorrect')
        self.assertTrue(state['fields']['payee.name']['needs_review'])
        self.assertEqual(verified_value_observation_ids(state, 'payee.name', state['fields']['payee.name']), set())

    def test_forged_spans_extra_keys_and_geometry_are_rejected(self):
        state, item = self.check(self.capture())
        for mutate in [lambda x: x['value_spans'][0].update(end=3),
                       lambda x: x.update(raw_value='guessed'),
                       lambda x: x.update(claim_ids=['name_claim', 'foreign'])]:
            bad = deepcopy(item); mutate(bad)
            with self.assertRaises(ValueError): DocumentStore._visual_check(bad, state)
        cap = state['observations'][item['capture_id']]
        cap['quad'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'exact_source_extent'): DocumentStore._visual_check(item, state)

    def test_missing_capture_pixel_proof_is_not_an_optical_verification(self):
        state, item = self.check(self.capture())
        cap = state['observations'][item['capture_id']]
        cap['raster_transform_proof']['source_to_raster_verification']['exact_raster_pixels_verified'] = False
        with self.assertRaisesRegex(ValueError, 'pixel_proof'): DocumentStore._visual_check(item, state)

    def test_core_recomputes_horizontal_baseline_before_recording_a_union_disagreement(self):
        state, item = self.check(self.capture())
        # A valid capture is not authority to merge vertically arranged source
        # words. Recompute source layout rather than accepting a stale union.
        state['observations']['b']['quad'] = [[10, 60], [60, 60], [60, 80], [10, 80]]
        with self.assertRaisesRegex(ValueError, 'visual_union_source_proof_invalid') as caught:
            DocumentStore._visual_check(item, state)
        self.assertIn('one_horizontal_baseline', str(caught.exception.__cause__))
        self.assertFalse(state.get('visual_checks'))
        self.assertFalse(state.get('conflicts'))

    def test_geometry_aliases_cannot_duplicate_one_printed_word(self):
        state = deepcopy(self.state)
        state['observations']['b']['quad'] = deepcopy(state['observations']['a']['quad'][1:] + state['observations']['a']['quad'][:1])
        with self.assertRaisesRegex(ValueError, 'duplicate_pixel_region'):
            value_span_union(state, 'name_claim')

    def test_malformed_or_relabelled_check_does_not_credit_value_coverage(self):
        state, item = self.check(self.capture())
        forged = {**item, 'verdict': 'correct', 'claim_ids': [{}]}
        state['visual_checks'] = {'forged': forged}
        self.assertEqual(verified_value_observation_ids(state, 'payee.name', state['fields']['payee.name']), set())

    def test_partial_line_cross_page_and_conflicting_claims_cannot_make_union(self):
        for mutate in [lambda s: s['claims']['name_claim']['spans'][0].update(end=3),
                       lambda s: s['observations']['b'].update(page=2, page_id='p2'),
                       lambda s: s['claims'].update(opposition={**deepcopy(s['claims']['name_claim']),
                           'id': 'opposition', 'raw_value': 'OTHER', 'status': 'conflict'})]:
            state = deepcopy(self.state); mutate(state)
            with self.assertRaises(ValueError): value_span_union(state, 'name_claim')

    def test_old_whole_observation_check_keeps_its_original_closed_contract(self):
        from harness_ocr.inference.vision import prepare_visual_regions, parse_visual_output
        _, prov = prepare_visual_regions([self.regions['a']])
        capture = parse_visual_output({'text': '{"readings":{"r1":{"status":"readable","literal":"ACME"}}}',
            'completed': True, 'stop_reason': 'eos', 'model_id': 'fixture'}, prov)[0]
        state = self.store.revise('doc', observations=[capture], expected_revision=self.state['revision'])
        old = {'id': 'old_check', 'field_key': 'payee.name', 'claim_ids': ['name_claim'],
               'capture_id': capture['id'], 'expected_observation_id': 'a'}
        state = self.store.revise('doc', visual_checks=[old], expected_revision=state['revision'])
        self.assertEqual(state['visual_checks']['old_check']['comparison_scope'], 'complete_original_observation_literal')
        self.assertEqual(verified_value_observation_ids(state, 'payee.name', state['fields']['payee.name']), {'a'})
        self.assertNotEqual(verified_value_observation_ids(state, 'payee.name', state['fields']['payee.name']), {'a', 'b'})


if __name__ == '__main__':
    unittest.main()
