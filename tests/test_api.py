"""HTTP lifecycle/security integration with an explicitly enabled non-neural fixture."""
import io
import json
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from harness_ocr.api import create_app
from harness_ocr.service import Service
from harness_ocr.settings import Settings
from harness_ocr.inference.supervisor import ModelCanceled

API_TOKEN = 'a' * 32
REVIEW_TOKEN = 'r' * 32
PROFILE = 'rf_passport_main_v1'


class FixtureBackend:
    """Returns known OCR observations; never imports or claims model quality."""
    def __init__(self, block=False):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = block
        self.closed = False

    def call(self, op, args=None, cancel=None, timeout=None):
        self.calls.append(op)
        if op == 'ingest':
            from harness_ocr.ingest import decode
            return decode(args['path'], args['output'], self.settings)
        if op == 'cheap':
            if self.block:
                self.entered.set()
                until = time.monotonic() + 5
                while not self.release.wait(.01):
                    if cancel and cancel():
                        raise ModelCanceled('fixture_canceled')
                    if time.monotonic() >= until:
                        raise RuntimeError('fixture_gate_timeout')
            return {'observations': [], 'regions': [], 'completed': True, 'diagnostics': {'fixture': True}}
        if op == 'ocr':
            return {'text': 'Серия паспорта: 1234\nНомер паспорта: 567890', 'completed': True, 'generated_tokens': 12}
        if op == 'select':
            prompt = json.loads(args['prompt'])
            proposals = []
            for field in prompt['fields']:
                found = next((c for c in prompt['candidates'] if c['field_key'] == field['key'] and c['role_anchor_observed']), None)
                if found:
                    proposals.append({'field_key': field['key'], 'candidate_ids': [found['id']], 'party_role': field['role']})
            return {'text': json.dumps({'proposals': proposals, 'reread_requests': []}), 'completed': True}
        raise ValueError('unexpected_fixture_operation:' + op)

    def close(self):
        self.release.set()
        self.closed = True


class ApiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.backend = FixtureBackend()
        self.settings = Settings(self.root / 'data', self.root / 'models', device='cpu', dtype='float32', allow_fixture=True)
        self.backend.settings = self.settings
        self.service = Service(self.settings, backend=self.backend, fixture=True)
        self.client = None

    def start(self):
        self.client = TestClient(create_app(self.service, API_TOKEN, REVIEW_TOKEN))
        self.client.__enter__()
        return self.client

    def tearDown(self):
        self.backend.release.set()
        if self.client:
            self.client.__exit__(None, None, None)
        else:
            self.service.close()
        self.temp.cleanup()

    @staticmethod
    def headers(review=False, **extra):
        return {'Authorization': 'Bearer ' + (REVIEW_TOKEN if review else API_TOKEN), **extra}

    def upload(self):
        image = io.BytesIO()
        Image.new('RGB', (320, 100), 'white').save(image, format='PNG')
        response = self.client.post('/v1/artifacts', content=image.getvalue(),
            headers=self.headers(**{'Content-Type': 'application/octet-stream'}))
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()['artifact_id']

    def request(self, aid=None, **extra):
        return {'artifact_id': aid or self.upload(), 'profile_id': PROFILE, 'subset': ['passport.number'], **extra}

    def wait_terminal(self, jid):
        until = time.monotonic() + 8
        while time.monotonic() < until:
            response = self.client.get(f'/v1/jobs/{jid}', headers=self.headers())
            value = response.json()
            if value['status'] not in {'queued', 'running'}:
                return value
            time.sleep(.02)
        self.fail('fixture_job_did_not_finish')

    def test_authentication_upload_and_forged_authority_fail_closed(self):
        client = self.start()
        self.assertEqual(client.get('/health/live').status_code, 200)
        self.assertEqual(client.get('/v1/profiles').status_code, 401)
        self.assertEqual(client.get('/health/ready', headers={'Authorization': 'Bearer wrong'}).status_code, 401)
        self.assertEqual(client.post('/v1/artifacts', content=b'abc', headers=self.headers()).status_code, 415)
        request = self.request()
        for forged in [{'path': '/tmp/source.pdf'}, {'auto_accept': True}, {'authorized': True}, {'policy_id': 'bypass'}]:
            response = client.post('/v1/jobs', json={**request, **forged}, headers=self.headers())
            self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(self.backend.calls, [])

    def test_fixture_extraction_review_scope_cas_and_trace(self):
        client = self.start()
        ready = client.get('/health/ready', headers=self.headers()).json()
        self.assertTrue(ready['ready'])
        self.assertTrue(ready['fixture'])
        self.assertFalse(ready['quality_qualified'])
        job = client.post('/v1/jobs', json=self.request(), headers=self.headers()).json()
        state = self.wait_terminal(job['id'])
        self.assertEqual(state['status'], 'completed', state)
        self.assertNotIn('result', state)
        response = client.get(f'/v1/jobs/{job["id"]}/result', headers=self.headers())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['fields']['passport.number']['raw_value'], '567890', result)
        self.assertEqual(result['decision'], 'review')
        self.assertFalse(result['service_approval'])
        self.assertNotIn('optics', result)
        from jsonschema import Draft202012Validator
        contract = self.service.engine.profiles.compile(PROFILE, subset=['passport.number'])
        Draft202012Validator(contract['json_schema']).validate(result['contract_result'])
        correction = {'field_key': 'passport.number', 'raw_value': '567891',
            'evidence_ids': result['fields']['passport.number']['evidence_ids'], 'expected_revision': result['revision']}
        self.assertEqual(client.post(f'/v1/jobs/{job["id"]}/review', json=correction, headers=self.headers()).status_code, 403)
        corrected = client.post(f'/v1/jobs/{job["id"]}/review', json=correction, headers=self.headers(review=True))
        self.assertEqual(corrected.status_code, 200, corrected.text)
        self.assertEqual(corrected.json()['fields']['passport.number']['raw_value'], '567891')
        self.assertEqual(corrected.json()['fields']['passport.number']['origin'], 'human')
        stale = client.post(f'/v1/jobs/{job["id"]}/review', json=correction, headers=self.headers(review=True))
        self.assertEqual(stale.status_code, 409, stale.text)
        forged = client.post(f'/v1/jobs/{job["id"]}/review', json={**correction, 'authorized': True}, headers=self.headers(review=True))
        self.assertEqual(forged.status_code, 422)
        trace = client.get(f'/v1/jobs/{job["id"]}/trace', headers=self.headers())
        self.assertEqual(trace.status_code, 200)
        self.assertTrue(any(row['stage'] == 'select' for row in trace.json()))

    def test_idempotent_submission_and_payload_mismatch(self):
        self.start()
        request = self.request()
        headers = self.headers(**{'Idempotency-Key': 'document-42'})
        first = self.client.post('/v1/jobs', json=request, headers=headers)
        repeated = self.client.post('/v1/jobs', json=request, headers=headers)
        self.assertEqual(first.status_code, 202, first.text)
        self.assertEqual(repeated.json()['id'], first.json()['id'])
        conflict = self.client.post('/v1/jobs', json={**request, 'subset': ['passport.series']}, headers=headers)
        self.assertEqual(conflict.status_code, 422)
        self.assertIn('idempotency_key_payload_mismatch', conflict.text)
        self.wait_terminal(first.json()['id'])

    def test_queued_cancellation_never_enters_backend(self):
        self.backend.block = True
        self.start()
        request = self.request()
        first = self.client.post('/v1/jobs', json=request, headers=self.headers()).json()
        self.assertTrue(self.backend.entered.wait(3))
        second = self.client.post('/v1/jobs', json=request, headers=self.headers()).json()
        canceled = self.client.post(f'/v1/jobs/{second["id"]}/cancel', headers=self.headers())
        self.assertEqual(canceled.status_code, 200)
        self.assertEqual(canceled.json()['status'], 'canceled')
        self.backend.release.set()
        self.wait_terminal(first['id'])
        self.assertEqual(self.backend.calls.count('cheap'), 1)
        unavailable = self.client.get(f'/v1/jobs/{second["id"]}/result', headers=self.headers())
        self.assertEqual(unavailable.status_code, 409)

    def test_running_job_becomes_interrupted_on_restart_without_recompute(self):
        request = {'artifact_id': self.service.artifacts.put(b'fixture')['artifact_id'],
            'profile_id': PROFILE, 'subset': ['passport.number']}
        job = self.service.jobs.create(request)
        self.service.jobs.next()
        self.start()
        response = self.client.get(f'/v1/jobs/{job["id"]}', headers=self.headers())
        self.assertEqual(response.json()['status'], 'interrupted')
        self.assertEqual(self.backend.calls, [])

    def partial_job(self):
        artifact = self.service.artifacts.put(b'fixture-partial-source')
        request = {'artifact_id': artifact['artifact_id'], 'profile_id': PROFILE,
                   'subset': ['passport.number']}
        job = self.service.jobs.create(request)
        contract = self.service.engine.profiles.compile(PROFILE, subset=request['subset'])
        binding = self.service.engine.binding(contract, self.settings.qwen)
        self.service.engine.documents.create(job['id'], artifact['source_sha256'], PROFILE,
            metadata={'binding': binding, 'subset': contract['requested_fields']})
        snapshot = self.service.engine.documents.revise(job['id'], observations=[{
            'id': 'partial_ocr', 'source_sha256': artifact['source_sha256'], 'text': 'Номер паспорта: 567890',
            'evidence_kind': 'image_region', 'page': 1, 'page_id': 'p1', 'bbox': [0, 0, 200, 40],
            'source_size': [320, 100], 'completed': True, 'capture_id': 'fixture-partial', 'model_id': 'fixture'}],
            claims=[{'id': 'partial_claim', 'field_key': 'passport.number', 'raw_value': '567890',
                'evidence_ids': ['partial_ocr'], 'depends_on': [], 'party_role': 'document', 'origin': 'rule'}])
        return job, snapshot

    def test_restart_exposes_durable_partial_evidence_without_completing_or_recomputing(self):
        job, snapshot = self.partial_job()
        self.service.jobs.next()
        self.start()
        self.assertEqual(self.service.jobs.get(job['id'])['status'], 'interrupted')
        response = self.client.get(f'/v1/jobs/{job["id"]}/result', headers=self.headers())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['terminal_status'], 'interrupted')
        self.assertEqual(result['revision'], snapshot['revision'])
        self.assertEqual(result['fields']['passport.number']['raw_value'], '567890')
        self.assertEqual(result['observations']['partial_ocr']['text'], 'Номер паспорта: 567890')
        self.assertFalse(result['completed'])
        self.assertEqual(result['decision'], 'review')
        self.assertTrue(result['needs_review'])
        self.assertFalse(result['service_approval'])
        self.assertTrue(result['recovered_partial_result'])
        self.assertIn('interrupted', result['reason_codes'])
        self.assertEqual(self.backend.calls, [])
        again = self.client.get(f'/v1/jobs/{job["id"]}/result', headers=self.headers()).json()
        self.assertEqual(again['revision'], snapshot['revision'])
        self.assertIsNone(self.service.jobs.get(job['id'])['result'])

    def test_canceled_failed_partial_snapshots_remain_review_and_missing_snapshot_unavailable(self):
        seeded = []
        for terminal in ['canceled', 'failed']:
            job, snapshot = self.partial_job()
            # Simulate a process exiting after the durable terminal status but
            # before it could store the separate result projection.
            with self.service.jobs.connect() as db:
                db.execute('UPDATE jobs SET status=? WHERE id=?', (terminal, job['id']))
            seeded.append((terminal, job, snapshot))
        self.start()
        for terminal, job, snapshot in seeded:
            response = self.client.get(f'/v1/jobs/{job["id"]}/result', headers=self.headers())
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result['terminal_status'], terminal)
            self.assertFalse(result['completed'])
            self.assertEqual(result['decision'], 'review')
            self.assertIn(terminal, result['reason_codes'])
            self.assertEqual(result['revision'], snapshot['revision'])
        self.assertEqual(self.backend.calls, [])

    def finish_fixture_job(self, job, snapshot, **extra):
        self.service.jobs.next()
        self.service.jobs.finish(job['id'], {
            'document_id': job['id'], 'terminal_status': 'completed', 'completed': True,
            'revision': snapshot['revision'], 'decision': 'accept_for_autofill',
            'reason_codes': [], 'seconds': 1.25, 'calls': [{'op': 'fixture_only'}], **extra})

    def test_unchanged_revision_rechecks_policy_and_retains_original_metrics(self):
        job, snapshot = self.partial_job()
        self.finish_fixture_job(job, snapshot)
        self.start()
        gate = {'decision': 'review', 'reason_codes': ['policy_not_active_or_unknown'],
                'profile_qualified': False, 'policy_id': None}
        with patch.object(self.service.engine.gateway, 'apply', return_value=gate) as current_gate:
            response = self.client.get(f'/v1/jobs/{job["id"]}/result', headers=self.headers())
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(current_gate.called)
        self.assertEqual(result['revision'], snapshot['revision'])
        self.assertEqual(result['decision'], 'review')
        self.assertIn('policy_not_active_or_unknown', result['reason_codes'])
        self.assertEqual(result['seconds'], 1.25)
        self.assertEqual(result['calls'], [{'op': 'fixture_only'}])
        self.assertEqual(self.backend.calls, [])

    def test_human_revision_cannot_erase_trial_strategy_limitations(self):
        job, snapshot = self.partial_job()
        reason = 'unsupported_strategy_action:contrast_view'
        self.finish_fixture_job(job, snapshot, decision='review', reason_codes=[reason])
        self.start()
        # Isolate the per-trial strategy barrier from the independently tested
        # policy gateway: even an otherwise accepting gateway must retain it.
        gate = {'decision': 'accept_for_autofill', 'reason_codes': [],
                'profile_qualified': False, 'policy_id': None}
        correction = {'field_key': 'passport.number', 'raw_value': '567891',
            'evidence_ids': ['partial_ocr'], 'expected_revision': snapshot['revision']}
        with patch.object(self.service.engine.gateway, 'apply', return_value=gate):
            response = self.client.post(f'/v1/jobs/{job["id"]}/review', json=correction,
                headers=self.headers(review=True))
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertGreater(result['revision'], snapshot['revision'])
        self.assertEqual(result['fields']['passport.number']['raw_value'], '567891')
        self.assertEqual(result['decision'], 'review')
        self.assertIn(reason, result['reason_codes'])
        self.assertEqual(result['seconds'], 1.25)
        self.assertEqual(self.backend.calls, [])

    def test_runtime_changes_cannot_reuse_old_qualification_or_relabel_provenance(self):
        job, snapshot = self.partial_job()
        original = snapshot['metadata']['binding']
        self.finish_fixture_job(job, snapshot)
        self.start()
        accepting_gate = {'decision': 'accept_for_autofill', 'reason_codes': [],
                          'profile_qualified': False, 'policy_id': None}
        engine = self.service.engine
        changes = [
            ('code_sha256', 'c' * 64),
            ('settings', replace(engine.settings, max_new_tokens=engine.settings.max_new_tokens + 1)),
            ('active_cards', lambda contract: [{'id': 'changed-server-strategy-card'}]),
        ]
        for attribute, changed in changes:
            with self.subTest(runtime_attribute=attribute):
                with patch.object(engine, attribute, changed), patch.object(engine.gateway, 'apply', return_value=accepting_gate):
                    response = self.client.get(f'/v1/jobs/{job["id"]}/result', headers=self.headers())
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertTrue(result['runtime_binding_changed'])
                self.assertNotEqual(result['current_projection_binding'], original)
                self.assertEqual(result['source_sha256'], snapshot['source_sha256'])
                for key, value in original.items():
                    self.assertEqual(result[key], value)
                self.assertEqual(result['decision'], 'review')
                self.assertTrue(result['needs_review'])
                self.assertIn('runtime_binding_changed_requires_reprocess', result['reason_codes'])
                self.assertTrue(all(field['decision'] == 'review' for field in result['fields'].values()))
                self.assertTrue(all(field['decision'] == 'review' for field in result['contract_result']['fields'].values()))
                self.assertEqual(result['fields']['passport.number']['raw_value'], '567890')
                self.assertEqual(result['seconds'], 1.25)
        self.assertEqual(engine.documents.get(job['id'])['metadata']['binding'], original)
        self.assertEqual(self.backend.calls, [])

    def test_missing_models_readiness_503_and_submission_blocked(self):
        self.service = Service(self.settings, backend=self.backend, fixture=False)
        self.start()
        ready = self.client.get('/health/ready', headers=self.headers())
        self.assertEqual(ready.status_code, 503)
        self.assertFalse(ready.json()['ready'])
        rejected = self.client.post('/v1/jobs', json=self.request(), headers=self.headers())
        self.assertEqual(rejected.status_code, 503)
        self.assertEqual(self.backend.calls, [])

    def test_fixture_requires_explicit_server_setting_and_distinct_tokens(self):
        with self.assertRaises(ValueError):
            Service(Settings(self.root / 'other', self.root / 'models', device='cpu', dtype='float32'), backend=self.backend, fixture=True)
        with self.assertRaises(RuntimeError):
            create_app(self.service, API_TOKEN, API_TOKEN)


if __name__ == '__main__':
    unittest.main()
