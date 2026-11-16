"""Adaptive engine contracts with real pixels and explicit fake OCR/VLM outputs.

These tests verify scheduling, source proofs and abstention, not recognition
accuracy. No model, weights, network or GPU is loaded.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw

from harness_ocr.engine import Engine
from harness_ocr.ingest import decode
from harness_ocr.inference.region_tasks import materialize_region
from harness_ocr.inference.supervisor import ModelCanceled, ModelDeadline
from harness_ocr.inference.vision import make_visual_region, parse_visual_output, prepare_visual_regions
from harness_ocr.settings import Settings
from harness_ocr.util import file_sha


class PixelFakeBackend:
    def __init__(self, settings, *, primary=(), rescue=(), ocr=(), selector=None,
                 visual=(), fail_first_visual=False):
        self.settings = settings; self.primary = list(primary); self.rescue_rows = list(rescue)
        self.ocr_answers = list(ocr); self.selector = selector; self.visual_answers = list(visual)
        self.fail_first_visual = fail_first_visual
        self.calls = []; self.active_calls = 0; self.max_active_calls = 0
        self.ocr_count = 0; self.visual_count = 0; self.cancel_requested = False
        self.region_text = {}

    def cheap_result(self, args, rows, namespace):
        from harness_ocr.inference.db_geometry import rectify_quad
        page = args['page']; observations = []; regions = []
        for index, (literal, box) in enumerate(rows):
            # Fake detected lines retain the production line rectifier. Only
            # server-created multi-line context uses upright bbox cropping.
            x0,y0,x1,y1 = box; quad = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
            with Image.open(page['path']) as image:
                crop, proof = rectify_quad(image, quad)
            target = Path(args['output_dir'])/(namespace+'_'+str(index)+'.png')
            target.parent.mkdir(parents=True, exist_ok=True); crop.save(target)
            proof = {**proof, 'raster_sha256':page['raster_sha256'], 'crop_sha256':file_sha(target),
                'original_source_sha256':args['source_sha256'], 'page_transform':page['transform'],
                'source_size':page['source_size'], 'coordinates_refer_to':'actual_page_raster_pixels'}
            region = {'id':namespace+'_r'+str(index), 'path':str(target), 'page':page['page'],
                'page_id':page['page_id'], 'source_size':page['source_size'],
                'source_sha256':args['source_sha256'], 'quad':quad, 'bbox':box,
                'crop_sha256':file_sha(target), 'transform':proof}
            regions.append(region); self.region_text[region['id']] = literal
            if literal:
                observations.append({'id':namespace+'_o'+str(index), 'source_sha256':args['source_sha256'],
                    'text':literal, 'evidence_kind':'image_region', 'page':page['page'],
                    'page_id':page['page_id'], 'source_size':page['source_size'], 'quad':region['quad'],
                    'completed':True, 'capture_id':namespace+'_capture_'+str(index), 'model_id':'cyrillic-ctc',
                    'reading_stream_id':'ctc:'+page['page_id']+':'+namespace,
                    'raster_transform_proof':region['transform']})
        return {'observations':observations, 'regions':regions, 'completed':True}

    def call(self, op, args, **kwargs):
        if self.active_calls:
            raise AssertionError('Two inference operations overlapped on one backend')
        self.active_calls += 1; self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.calls.append((op, deepcopy(args), kwargs))
        try:
            if op == 'ingest':
                return decode(args['path'], args['output'], self.settings)
            if op in ('cheap', 'cheap_rescue'):
                return self.cheap_result(args, self.primary if op == 'cheap' else self.rescue_rows,
                                         'db' if op == 'cheap' else 'yolo26l')
            if op == 'ocr':
                self.ocr_count += 1
                if args.get('expected_image_sha256') != file_sha(args['path']):
                    raise AssertionError('OCR crop bytes must be SHA-bound before the call')
                answer = self.ocr_answers.pop(0) if self.ocr_answers else ''
                if callable(answer): answer = answer(self)
                if isinstance(answer, BaseException): raise answer
                return {'text':answer, 'completed':True, 'input_image_sha256':args['expected_image_sha256']}
            if op == 'select':
                if self.selector is None: raise AssertionError('Selector was unnecessary for this fixture')
                return {'text':self.selector(json.loads(args['prompt'])), 'completed':True}
            if op == 'visual_read':
                self.visual_count += 1
                if self.fail_first_visual and self.visual_count == 1:
                    raise ValueError('controlled_visual_field_failure')
                _, provenance = prepare_visual_regions(args['regions'])
                literal = (self.visual_answers.pop(0) if self.visual_answers
                           else self.region_text[args['regions'][0]['id']])
                if isinstance(literal, BaseException): raise literal
                output = {'text':json.dumps({'readings':{'r1':{'status':'readable','literal':literal}}}),
                    'completed':True, 'stop_reason':'eos', 'modality':'vision', 'purpose':'blind_optical_capture',
                    'model_id':'fixture-qwen', 'model_bundle_sha256':'d'*64, 'fresh_generation_state':True}
                output['captures'] = parse_visual_output(output, provenance)
                return output
            raise AssertionError('Unexpected model operation: '+op)
        finally:
            self.active_calls -= 1


class AdaptiveReadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.source = self.root/'source.png'
        image = Image.new('RGB', (400, 240), 'white')
        draw = ImageDraw.Draw(image)
        draw.text((20, 25), '001234', fill='black')
        draw.text((120, 100), 'ACME', fill='black'); draw.text((180, 100), 'CITY', fill='black')
        image.save(self.source); self.sha = file_sha(self.source)

    def settings(self, name='run', **changes):
        return replace(Settings(self.root/name, self.root/'models', device='cpu', dtype='float32',
            reading_policy='adaptive', max_whole_page_reads=0), **changes)

    def run_engine(self, settings=None, *, backend=None, profile='D01', subset=None, **kwargs):
        settings = settings or self.settings()
        backend = backend or PixelFakeBackend(settings)
        engine = Engine(settings, backend)
        result = engine.run('doc', self.source, self.sha, profile,
                            subset=subset or ['passport.number'], **kwargs)
        return engine, backend, result

    def test_unique_anchored_fact_skips_both_selector_and_redundant_page_ocr(self):
        settings = self.settings()
        backend = PixelFakeBackend(settings, primary=[('Номер паспорта: 001234', [10, 20, 220, 50])])
        _, _, result = self.run_engine(settings, backend=backend)
        self.assertEqual(result['terminal_status'], 'completed', result['errors'])
        self.assertTrue(result['completed']); self.assertEqual(result['fields']['passport.number']['raw_value'], '001234')
        self.assertEqual(result['fields']['passport.number']['validation_status'], 'pass')
        self.assertEqual([op for op, _, _ in backend.calls], ['ingest', 'cheap'])
        self.assertEqual(result['decision'], 'review')

    def test_business_invalid_printed_values_are_retained_without_repair_reads(self):
        for index, (profile, field, text, raw) in enumerate([
            ('D01', 'holder.birth_date', 'Дата рождения: 31.02.2016', '31.02.2016'),
            ('D07', 'payer.inn', 'ИНН плательщика: 7707083894', '7707083894')]):
            with self.subTest(field=field):
                settings = self.settings('invalid'+str(index))
                backend = PixelFakeBackend(settings, primary=[(text, [10, 20, 300, 50])])
                _, _, result = self.run_engine(settings, backend=backend, profile=profile, subset=[field])
                self.assertEqual(result['fields'][field]['raw_value'], raw)
                self.assertEqual(result['fields'][field]['validation_status'], 'fail')
                self.assertFalse(any(op in {'ocr', 'select', 'cheap_rescue'} for op, _, _ in backend.calls))

    def test_a_bare_number_gets_role_only_after_real_context_is_read(self):
        settings = self.settings(max_rereads=1)
        rows = [('001234', [20, 20, 100, 45])]
        bare = PixelFakeBackend(settings, primary=rows)
        _, _, first = self.run_engine(settings, backend=bare, allow_reread=False)
        self.assertEqual(first['fields']['passport.number']['extraction_status'], 'not_observed')
        settings = self.settings('context', max_rereads=1)
        backend = PixelFakeBackend(settings, primary=rows, ocr=['Номер паспорта: 001234'])
        _, _, result = self.run_engine(settings, backend=backend)
        self.assertEqual(result['fields']['passport.number']['raw_value'], '001234', result['errors'])
        self.assertEqual(backend.ocr_count, 1)
        self.assertFalse(any(op == 'select' for op, _, _ in backend.calls))
        claims = [c for c in result['observations'].values() if c.get('reading_purpose') == 'field_label_context']
        self.assertEqual(len(claims), 1); self.assertIn('Номер паспорта', claims[0]['text'])
        self.assertTrue(any(o['text'] == '001234' for o in result['observations'].values()))

    def test_structured_identifier_gets_kind_and_party_only_from_the_context_read(self):
        settings = self.settings(max_rereads=1)
        backend = PixelFakeBackend(settings, primary=[('7707083893', [20, 20, 140, 45])],
            ocr=['Доверитель ИНН: 7707083893'])
        _, _, result = self.run_engine(settings, backend=backend, profile='D04', subset=['principal.identifier'])
        self.assertEqual(backend.ocr_count, 1, 'A visible but untyped value needs bounded label context, not an invented identifier kind')
        field = result['fields']['principal.identifier']
        self.assertEqual(field['extraction_status'], 'observed', result['errors'])
        self.assertEqual(field['raw_value']['items'][0]['kind'], 'inn')
        self.assertEqual(field['raw_value']['items'][0]['party_role'], 'principal')
        self.assertEqual(field['validation_status'], 'pass')
        for index, context in enumerate(['Представитель ИНН: 7707083893', '7707083893']):
            with self.subTest(context=context):
                settings = self.settings('wrong-kind-role'+str(index), max_rereads=1)
                backend = PixelFakeBackend(settings, primary=[('7707083893', [20, 20, 140, 45])], ocr=[context])
                _, _, result = self.run_engine(settings, backend=backend, profile='D04', subset=['principal.identifier'])
                self.assertEqual(backend.ocr_count, 1)
                self.assertNotEqual(result['fields']['principal.identifier']['extraction_status'], 'observed')

    def test_model_preference_cannot_erase_a_genuine_source_conflict(self):
        def prefer_first(prompt):
            candidate = next(c for c in prompt['candidates'] if c['text'] == '001234')
            return json.dumps({'proposals':[{'field_key':candidate['field_key'],
                'candidate_ids':[candidate['id']], 'party_role':candidate['party_role']}], 'reread_requests':[]})
        settings = self.settings(max_rereads=1, max_input_tokens=32768)
        backend = PixelFakeBackend(settings, primary=[('Номер паспорта: 001234', [10, 10, 220, 35]),
            ('Номер паспорта: 654321', [10, 60, 220, 85])], selector=prefer_first,
            ocr=['Номер паспорта: 001234'])
        _, _, result = self.run_engine(settings, backend=backend, qwen='qwen-4b')
        self.assertEqual(result['fields']['passport.number']['extraction_status'], 'ambiguous')
        self.assertIsNone(result['fields']['passport.number']['raw_value'])
        self.assertTrue(result['conflicts'])
        for value in ['001234', '654321']:
            self.assertTrue(any(value in o['text'] for o in result['observations'].values()))
        selected = [args for op, args, _ in backend.calls if op == 'select']
        self.assertTrue(selected); self.assertTrue(all(args['name'] == 'qwen-4b' for args in selected))

    def test_previously_empty_ctc_region_can_recover_and_receive_visual_proof(self):
        settings = self.settings(max_rereads=1, visual_verifier=True)
        literal = 'Номер паспорта: 001234'
        backend = PixelFakeBackend(settings, primary=[('', [20, 20, 220, 50])], ocr=[literal], visual=[literal])
        _, _, result = self.run_engine(settings, backend=backend, qwen='qwen-4b')
        self.assertEqual(result['fields']['passport.number']['raw_value'], '001234', result['errors'])
        self.assertTrue(result['fields']['passport.number']['visual_verification_complete'])
        self.assertEqual(backend.ocr_count, 1); self.assertEqual(backend.visual_count, 1)
        self.assertEqual([args['name'] for op, args, _ in backend.calls if op == 'visual_read'], ['qwen-4b'])
        self.assertTrue(result['optics']['visual_regions_by_id'])

    def test_yolo_rescue_is_gap_triggered_sequential_and_bounded(self):
        settings = self.settings(max_rereads=0, rescue_text_detector='yolo26l', max_detector_rescues=1)
        backend = PixelFakeBackend(settings, primary=[('001234', [20, 20, 100, 45])],
            rescue=[('Номер паспорта: 001234', [10, 10, 240, 50])])
        _, _, result = self.run_engine(settings, backend=backend)
        self.assertEqual(result['fields']['passport.number']['raw_value'], '001234', result['errors'])
        self.assertEqual([op for op, _, _ in backend.calls], ['ingest', 'cheap', 'cheap_rescue'])
        self.assertEqual(backend.max_active_calls, 1)
        self.assertEqual(result['reading_runtime']['rescue_pages'], ['p1'])
        good = PixelFakeBackend(self.settings('already', max_rereads=0, rescue_text_detector='yolo26l'),
            primary=[('Номер паспорта: 001234', [10, 10, 240, 50])], rescue=backend.rescue_rows)
        _, _, output = self.run_engine(good.settings, backend=good)
        self.assertTrue(output['completed']); self.assertFalse(any(op == 'cheap_rescue' for op, _, _ in good.calls))

    def test_hard_read_budgets_stop_even_when_every_read_adds_no_text(self):
        settings = self.settings(max_rereads=2, rescue_text_detector='yolo26l',
                                 max_detector_rescues=1, max_whole_page_reads=1)
        backend = PixelFakeBackend(settings, primary=[('', [20, 20, 180, 45]), ('', [20, 130, 180, 155])])
        _, _, result = self.run_engine(settings, backend=backend)
        self.assertLessEqual(backend.ocr_count, 3)
        self.assertEqual(sum(op == 'cheap_rescue' for op, _, _ in backend.calls), 1)
        self.assertLessEqual(len(result['reading_runtime']['used_reread_regions']), 2)
        self.assertEqual(result['reading_runtime']['whole_page_reads'], ['p1'])
        self.assertEqual(result['fields']['passport.number']['extraction_status'], 'not_observed')

    def test_generated_context_is_not_the_next_reread_anchor_for_the_same_value(self):
        settings = self.settings(max_rereads=2)
        backend = PixelFakeBackend(settings,
            primary=[('06.02.1966', [200, 80, 320, 100]), ('200254', [350, 120, 380, 200])],
            ocr=['06.02.1966', '200254'])
        _, _, result = self.run_engine(settings, backend=backend,
            subset=['holder.birth_date', 'passport.number'])
        self.assertTrue(result['completed'], result['errors'])
        self.assertEqual(backend.ocr_count, 2)
        decisions = [d for d in result['reading_runtime']['decisions'] if d['action']=='targeted_xiaomi_read']
        self.assertEqual([d['field_key'] for d in decisions], ['holder.birth_date', 'passport.number'])
        generated = {d['region_id'] for d in decisions}
        self.assertFalse(generated & {d['basis_region_id'] for d in decisions},
            'A generated multi-line context cannot silently become a fresh detector anchor')
        self.assertEqual(set(result['reading_runtime']['generated_context_regions']), generated)
        self.assertEqual(len(result['reading_runtime']['used_reread_regions']), 2)
        self.assertTrue(all(field['extraction_status']=='not_observed' for field in result['fields'].values()))
        self.assertEqual(result['decision'], 'review')

    def test_bare_repeat_from_whole_page_is_not_role_evidence(self):
        settings = self.settings(max_rereads=0, max_whole_page_reads=1)
        backend = PixelFakeBackend(settings, primary=[('001234', [20, 20, 100, 45])], ocr=['001234'])
        _, _, result = self.run_engine(settings, backend=backend)
        self.assertEqual(result['fields']['passport.number']['extraction_status'], 'not_observed')
        self.assertEqual(result['reading_runtime']['whole_page_reads'], ['p1'])

    def test_overbroad_context_cannot_bypass_the_explicit_whole_page_budget(self):
        for page_budget in (0, 1):
            with self.subTest(page_budget=page_budget):
                settings = self.settings('broad'+str(page_budget), max_rereads=1,
                                         max_whole_page_reads=page_budget)
                backend = PixelFakeBackend(settings, primary=[('001234', [0, 0, 400, 240])],
                                           ocr=['Номер паспорта: 001234'])
                _, _, result = self.run_engine(settings, backend=backend)
                self.assertEqual(result['terminal_status'], 'completed', result['errors'])
                self.assertEqual(backend.ocr_count, page_budget,
                    'An overbroad ROI is not a free page read under the local budget')
                runtime = result['reading_runtime']
                self.assertIn('context_crop_too_broad:passport.number', runtime['review_reasons'])
                self.assertFalse(any(d['action'] == 'targeted_xiaomi_read' for d in runtime['decisions']))
                self.assertEqual(runtime['whole_page_reads'], ['p1'] if page_budget else [])
                self.assertTrue(any(d['action'] == 'targeted_read_deferred' for d in runtime['decisions']))
                if page_budget:
                    page = next(args['page'] for op, args, _ in backend.calls if op == 'cheap')
                    ocr = next(args for op, args, _ in backend.calls if op == 'ocr')
                    self.assertEqual(ocr['path'], page['path'])
                    self.assertEqual(ocr['expected_image_sha256'], page['raster_sha256'])
                    self.assertTrue(any(d['action'] == 'whole_page_fallback' for d in runtime['decisions']))
                    self.assertEqual(result['fields']['passport.number']['raw_value'], '001234')
                else:
                    self.assertEqual(result['fields']['passport.number']['extraction_status'], 'not_observed')
                self.assertEqual(result['decision'], 'review')

    def test_secondary_partial_projection_failure_cannot_replace_cancel_or_deadline(self):
        from harness_ocr.reading import ReadingController
        # Exercise only the exception-preservation boundary. No engine/model or
        # fabricated state is needed to distinguish the two exception objects.
        for original in (ModelCanceled('original_cancel'), ModelDeadline('original_deadline')):
            with self.subTest(original=type(original).__name__):
                controller = ReadingController.__new__(ReadingController)
                commits = []
                def interrupted():
                    raise original
                def broken_partial(*, commit=False):
                    commits.append(commit)
                    raise ValueError('secondary_projection_failure')
                controller._run = interrupted
                controller.refresh = broken_partial
                with self.assertRaises(type(original)) as caught:
                    controller.run()
                self.assertIs(caught.exception, original)
                self.assertEqual(commits, [True])
                self.assertIn('partial_projection_failed:ValueError', caught.exception.__notes__)

    def test_cancel_preserves_already_read_source_fact_and_starts_no_more_calls(self):
        def cancel_on_read(backend):
            backend.cancel_requested = True
            return ModelCanceled('controlled_cancel')
        settings = self.settings(max_rereads=1)
        backend = PixelFakeBackend(settings, primary=[('Номер паспорта: 001234', [10, 10, 230, 35]),
            ('12.03.2000', [20, 80, 130, 105])], ocr=[cancel_on_read])
        _, _, result = self.run_engine(settings, backend=backend,
            subset=['passport.number', 'holder.birth_date'], cancel=lambda: backend.cancel_requested)
        self.assertEqual(result['terminal_status'], 'canceled')
        self.assertFalse(result['completed'])
        self.assertEqual(result['fields']['passport.number']['raw_value'], '001234')
        self.assertEqual(backend.ocr_count, 1)
        self.assertEqual(backend.calls[-1][0], 'ocr')

    def test_one_bad_visual_unit_does_not_discard_other_fields(self):
        settings = self.settings(visual_verifier=True, max_visual_regions=2)
        backend = PixelFakeBackend(settings, primary=[('Номер паспорта: 001234', [10, 10, 250, 35]),
            ('Дата рождения: 12.03.2000', [10, 80, 280, 105])], fail_first_visual=True)
        _, _, result = self.run_engine(settings, backend=backend, subset=['passport.number', 'holder.birth_date'])
        self.assertEqual(backend.visual_count, 2)
        self.assertFalse(result['completed'])
        self.assertTrue(any('controlled_visual_field_failure' in error for error in result['errors']))
        self.assertEqual(result['fields']['passport.number']['raw_value'], '001234')
        self.assertEqual(result['fields']['holder.birth_date']['raw_value'], '12.03.2000')
        self.assertEqual(sum(f.get('visual_verification_complete') is True for f in result['fields'].values()), 1)

    def test_assembled_name_gets_one_pixel_union_instead_of_one_visual_slot_per_word(self):
        settings = self.settings(visual_verifier=True, max_visual_regions=1, max_rereads=0)
        backend = PixelFakeBackend(settings, primary=[('Получатель', [10, 100, 100, 125]),
            ('ACME', [120, 100, 170, 125]), ('CITY', [180, 100, 240, 125])], visual=['ACME CITY'])
        _, _, result = self.run_engine(settings, backend=backend, profile='D07', subset=['payee.name'])
        self.assertEqual(result['fields']['payee.name']['raw_value'], 'ACME CITY', result['errors'])
        self.assertTrue(result['fields']['payee.name']['visual_verification_complete'], result['errors'])
        self.assertEqual(backend.visual_count, 1)
        self.assertEqual(len(result['visual_checks']), 1)
        check = next(iter(result['visual_checks'].values()))
        self.assertEqual(check['comparison_scope'], 'source_value_span_union_v1')
        self.assertEqual(set(check['expected_observation_ids']), {'db_o1', 'db_o2'})
        self.assertFalse(check['role_verified']); self.assertFalse(check['automatic_acceptance'])

    def test_page_corroboration_is_preserved_without_a_second_blind_page_read(self):
        settings = self.settings(visual_verifier=True, max_visual_regions=2, max_rereads=0)
        literal = 'Дата рождения: 12.03.2000'
        backend = PixelFakeBackend(settings, primary=[(literal, [10, 10, 280, 35])], visual=[literal])
        ingestion = decode(self.source, self.root/'shared', settings); page = ingestion['pages'][0]
        cheap = backend.cheap_result({'page':page, 'source_sha256':self.sha,
            'output_dir':str(self.root/'shared-crops')}, backend.primary, 'db')
        ctc = cheap['observations'][0]
        whole = {'id':'page_support', 'source_sha256':self.sha, 'text':literal+'\nOther printed context',
            'evidence_kind':'image_region', 'page':1, 'page_id':'p1', 'source_size':page['source_size'],
            'bbox':[0, 0, *page['source_size']], 'completed':True, 'model_id':'xiaomi',
            'raster_transform_proof':page['transform']}
        descriptor = make_visual_region(self.source, page, cheap['regions'][0], source_sha256=self.sha)
        shared = {'source_sha256':self.sha, 'observations':[ctc, whole], 'regions':cheap['regions'],
            'completed':True, 'visual_regions':{ctc['id']:descriptor,
                whole['id']:make_visual_region(self.source, page, source_sha256=self.sha)},
            'visual_regions_by_id':{descriptor['id']:descriptor}}
        engine = Engine(settings, backend); contract = engine.compile_contract('D01', subset=['holder.birth_date'])
        binding = engine.binding(contract, settings.qwen, strategy_cards=engine.active_cards(contract))
        state = engine.documents.create_document('doc', self.sha, contract['profile_id'],
            {'binding':binding, 'subset':contract['requested_fields'], 'source_path':str(self.source.resolve())})
        label_end = len('Дата рождения'); start = len('Дата рождения: ')
        engine.documents.revise('doc', observations=shared['observations'], expected_revision=state['revision'],
            claims=[{'id':'corroborated', 'field_key':'holder.birth_date', 'raw_value':'12.03.2000',
                'evidence_ids':[ctc['id'], whole['id']], 'spans':[{'observation_id':ctc['id'], 'start':start, 'end':len(literal)}],
                'role_evidence_ids':[ctc['id']], 'role_spans':[{'observation_id':ctc['id'], 'start':0, 'end':label_end}],
                'party_role':contract['field_definitions']['holder.birth_date']['role']}])
        result = engine.run('doc', self.source, self.sha, 'D01', subset=['holder.birth_date'], shared_optics=shared)
        self.assertTrue(result['completed'], result['errors'])
        self.assertEqual(backend.visual_count, 1)
        field = result['fields']['holder.birth_date']
        self.assertTrue(field['visual_verification_complete'])
        self.assertEqual(field['visual_required_observation_ids'], [ctc['id']])
        self.assertNotIn(whole['id'], field['visual_verified_observation_ids'])
        self.assertTrue(result['observations'][whole['id']]['active'])
        self.assertTrue(any(whole['id'] in c['evidence_ids'] for c in engine.documents.get('doc')['claims'].values()))

    def test_large_ambiguous_field_is_deferred_without_killing_a_small_proven_field(self):
        settings = self.settings(max_rereads=1, max_input_tokens=2048)
        backend = PixelFakeBackend(settings)
        text = 'A'*12000
        observations = [
            {'id':'small', 'text':'Номер паспорта: 001234', 'bbox':[10, 10, 200, 30]},
            {'id':'long1', 'text':'Кем выдан: '+text, 'bbox':[10, 60, 350, 85]},
            {'id':'long2', 'text':'Кем выдан: '+'B'*12000, 'bbox':[10, 120, 350, 145]}]
        shared = {'source_sha256':self.sha, 'regions':[], 'completed':True,
            'observations':[{**o, 'source_sha256':self.sha, 'evidence_kind':'image_region', 'page':1,
                             'page_id':'p1', 'source_size':[400, 240], 'completed':True} for o in observations]}
        _, _, result = self.run_engine(settings, backend=backend,
            subset=['passport.issuer', 'passport.number'], shared_optics=shared)
        self.assertEqual(result['terminal_status'], 'completed', result['errors'])
        self.assertEqual(result['fields']['passport.number']['raw_value'], '001234')
        self.assertTrue(any('selector_field_exceeds_input_budget:passport.issuer' in r for r in result['reason_codes']))
        self.assertFalse(any(op == 'select' for op, _, _ in backend.calls))

    def test_model_calls_do_not_receive_gold_prior_values_or_candidate_literals(self):
        settings = self.settings(max_rereads=1, visual_verifier=True)
        literal = 'Номер паспорта: 001234'
        backend = PixelFakeBackend(settings, primary=[('', [20, 20, 220, 50])], ocr=[literal], visual=[literal])
        _, _, result = self.run_engine(settings, backend=backend)
        self.assertTrue(result['fields']['passport.number']['visual_verification_complete'])
        for op, args, _ in backend.calls:
            if op == 'ocr':
                self.assertEqual(set(args), {'path', 'expected_image_sha256'})
            elif op == 'visual_read':
                self.assertEqual(set(args), {'name', 'regions'})
            else:
                continue
            encoded = json.dumps(args, ensure_ascii=False)
            self.assertNotIn(literal, encoded)
            for forbidden in ('ground_truth', 'proposed_raw', 'candidate_correct', 'expected_literal'):
                self.assertNotIn(forbidden, encoded)


if __name__ == '__main__':
    unittest.main()
