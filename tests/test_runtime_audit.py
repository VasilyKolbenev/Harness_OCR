"""Independent runtime regression tests; no neural models or network sockets."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from harness_ocr.service import Service
from harness_ocr.jobs import JobStore
from harness_ocr.settings import Settings


class ProbeOnlyBackend:
    def __init__(self):self.calls=[]
    def call(self,op,args=None,**kwargs):
        self.calls.append(op)
        if op=='probe':return {'loaded_models':['xiaomi','qwen-0.8b'],'fixture':True}
        raise AssertionError('No real inference is permitted in this test')
    def close(self):pass


class RuntimeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.settings=Settings(self.root/'data',self.root/'models',device='cpu',dtype='float32',allow_fixture=True)
        self.backend=ProbeOnlyBackend();self.service=Service(self.settings,self.backend,fixture=True)
    def tearDown(self):
        self.service.close();self.temp.cleanup()
    def request(self,qwen='qwen-0.8b'):
        artifact=self.service.artifacts.put(b'explicitly synthetic source')
        return {'artifact_id':artifact['artifact_id'],'profile_id':'D01','subset':['passport.number'],'qwen':qwen}
    def wait_stopped(self):
        until=time.perf_counter()+3
        while self.service.thread and self.service.thread.is_alive() and time.perf_counter()<until:
            time.sleep(.01)
        self.assertFalse(self.service.thread and self.service.thread.is_alive(),'scheduler did not stop')
    @staticmethod
    def verification(ok=True):
        return {'ok':ok,'errors':[] if ok else ['synthetic_sha_mismatch'],'checked_files':1}
    @staticmethod
    def completed(jid):
        return {'document_id':jid,'terminal_status':'completed','completed':True,'decision':'review','fields':{},'reason_codes':[]}

    def test_failed_startup_verification_never_consumes_persisted_queue(self):
        self.service.fixture=False
        job=self.service.jobs.create(self.request())
        with patch('harness_ocr.service.verify_models',return_value=self.verification(False)):
            self.service.start()
        self.assertEqual(self.service.jobs.get(job['id'])['status'],'queued')
        self.assertEqual(self.backend.calls,[])
        self.assertFalse(self.service.readiness_status()['ready'])
        self.assertIsNone(self.service.thread)

    def test_startup_recovery_failure_releases_scheduler_owner(self):
        with patch.object(self.service.jobs,'recover',side_effect=sqlite3.OperationalError('fixture recovery failure')):
            with self.assertRaises(sqlite3.OperationalError):self.service.start()
        self.assertIsNone(self.service.owner)
        replacement=Service(self.settings,ProbeOnlyBackend(),fixture=True)
        try:replacement.start();self.assertTrue(replacement.readiness_status()['ready'])
        finally:replacement.close()

    def test_double_start_cannot_create_second_scheduler_or_close_the_first_owner(self):
        self.service.start();original=self.service.thread
        with self.assertRaisesRegex(RuntimeError,'already_started'):self.service.start()
        self.assertIs(self.service.thread,original);self.assertTrue(original.is_alive())

    def test_scheduler_start_failure_releases_owner_without_joining_unstarted_thread(self):
        with patch('harness_ocr.service.threading.Thread.start',side_effect=RuntimeError('fixture thread start failure')):
            with self.assertRaises(RuntimeError):self.service.start()
        self.assertIsNone(self.service.thread);self.assertIsNone(self.service.owner)
        replacement=Service(self.settings,ProbeOnlyBackend(),fixture=True)
        try:replacement.start();self.assertTrue(replacement.readiness_status()['ready'])
        finally:replacement.close()

    def test_restart_verifies_queued_nondefault_model_before_claiming_it(self):
        self.service.fixture=False
        job=self.service.jobs.create(self.request('qwen-4b'))
        checked=[]
        def verify(root,names):
            checked.append(tuple(names));return self.verification('qwen-4b' not in names)
        with patch('harness_ocr.service.verify_models',side_effect=verify):
            self.service.start();self.wait_stopped()
        self.assertEqual(self.service.jobs.get(job['id'])['status'],'queued')
        self.assertTrue(any('qwen-4b' in names for names in checked))
        self.assertEqual(self.backend.calls,['probe'])
        self.assertIn('queued_model_verification_failed',self.service.readiness_status()['reasons'])

    def test_model_file_changes_invalidate_cached_verification(self):
        self.service.fixture=False
        with patch('harness_ocr.service.verify_models',return_value=self.verification()) as verify:
            self.service.start()
            self.assertEqual(verify.call_count,1)
            config=self.settings.models_dir/'qwen-0.8b'/'config.json'
            config.parent.mkdir(parents=True);config.write_text('{"changed":true}')
            verify.return_value=self.verification(False)
            job=self.service.jobs.create(self.request());self.service.wake.set();self.wait_stopped()
            self.assertEqual(verify.call_count,2)
        self.assertEqual(self.service.jobs.get(job['id'])['status'],'queued')
        self.assertFalse(self.service.readiness_status()['ready'])

    def test_scheduler_queue_failure_is_observable_and_blocks_new_submissions(self):
        request=self.request()
        with patch.object(self.service.jobs,'peek',side_effect=sqlite3.OperationalError('fixture queue failure')):
            self.service.start();self.wait_stopped()
        self.assertFalse(self.service.readiness_status()['ready'])
        self.assertFalse(self.service.liveness_status()['live'])
        with self.assertRaisesRegex(RuntimeError,'not_ready'):self.service.submit(request)
        diagnostic=json.loads((self.settings.data_dir/'service_failure.json').read_text())
        self.assertEqual(diagnostic['error_type'],'OperationalError')
        self.assertNotIn('fixture queue failure',json.dumps(diagnostic))

    def test_scheduler_publication_failure_holds_remaining_work(self):
        first=self.service.jobs.create(self.request());second=self.service.jobs.create(self.request())
        with patch.object(self.service.engine,'run',return_value=self.completed(first['id'])) as run, \
             patch.object(self.service.jobs,'finish',side_effect=sqlite3.OperationalError('fixture publication failure')):
            self.service.start();self.wait_stopped()
        self.assertEqual(run.call_count,1)
        self.assertEqual(self.service.jobs.get(first['id'])['status'],'running')
        self.assertEqual(self.service.jobs.get(second['id'])['status'],'queued')
        self.assertFalse(self.service.readiness_status()['ready'])

    def test_watchdog_cancel_callback_never_queries_sqlite(self):
        job=self.service.jobs.create(self.request());observed=threading.Event()
        def run(*args,**kwargs):
            callback=kwargs['cancel']
            with patch.object(self.service.jobs,'get',side_effect=AssertionError('watchdog touched SQLite')):
                self.assertFalse(callback())
            self.service.cancel(job['id'])
            with patch.object(self.service.jobs,'get',side_effect=AssertionError('watchdog touched SQLite')):
                self.assertTrue(callback())
            self.assertFalse(kwargs['emit_operational_events'])
            observed.set();return self.completed(job['id'])
        with patch.object(self.service.engine,'run',side_effect=run):
            self.service.start();self.assertTrue(observed.wait(3))
            until=time.perf_counter()+3
            while self.service.jobs.get(job['id'])['status']=='running' and time.perf_counter()<until:time.sleep(.01)
        self.assertEqual(self.service.jobs.get(job['id'])['status'],'canceled')
        self.assertEqual(self.backend.calls,[])

    def test_cancel_between_queue_claim_and_callback_registration_is_retained(self):
        job=self.service.jobs.create(self.request());seen=threading.Event()
        original=self.service.jobs.next
        def claim(*args,**kwargs):
            value=original(*args,**kwargs)
            if value:self.service.cancel(value['id'])
            return value
        def run(*args,**kwargs):
            self.assertTrue(kwargs['cancel']());seen.set();return self.completed(job['id'])
        with patch.object(self.service.jobs,'next',side_effect=claim),patch.object(self.service.engine,'run',side_effect=run):
            self.service.start();self.assertTrue(seen.wait(3))

    def test_verified_peek_cannot_claim_another_model_when_head_is_canceled(self):
        store=JobStore(self.root/'standalone.sqlite')
        first=store.create({'qwen':'qwen-0.8b'});second=store.create({'qwen':'qwen-4b'})
        pending=store.peek();self.assertEqual(pending['id'],first['id'])
        store.cancel(first['id'])
        self.assertIsNone(store.next(expected_id=pending['id']))
        self.assertEqual(store.get(second['id'])['status'],'queued')

    def test_late_cancel_retracts_nested_projection_without_mutating_input(self):
        store=JobStore(self.root/'late-cancel.sqlite');job=store.create({'profile':'synthetic'})
        store.next();store.cancel(job['id'])
        result={'terminal_status':'completed','completed':True,'decision':'accept_for_autofill',
                'needs_review':False,'reason_codes':['preserve_me'],
                'fields':{'x':{'decision':'accept_for_autofill'}},
                'contract_result':{'needs_review':False,'fields':{'x':{'decision':'accept_for_autofill'}}},
                'audit':{'selected':True},'drift':{'accepted':True}}
        original=deepcopy(result);saved=store.finish(job['id'],result)['result']
        self.assertEqual(result,original)
        self.assertEqual(saved['terminal_status'],'canceled');self.assertFalse(saved['completed'])
        self.assertTrue(saved['contract_result']['needs_review'])
        self.assertEqual(saved['contract_result']['fields']['x']['decision'],'review')
        self.assertIn('preserve_me',saved['reason_codes']);self.assertIn('canceled',saved['reason_codes'])
        self.assertNotIn('audit',saved);self.assertNotIn('drift',saved)

    def test_result_reprojection_cannot_erase_unperformed_reread_requirement(self):
        request=self.request();job=self.service.jobs.create(request)
        _,artifact=self.service.artifacts.get(request['artifact_id'])
        engine=self.service.engine
        contract=engine.compile_contract('D01',subset=request['subset'])
        binding=engine.binding(contract,self.settings.qwen)
        engine.documents.create_document(job['id'],artifact['source_sha256'],contract['profile_id'],
             metadata={'binding':binding,'subset':contract['requested_fields']})
        snapshot=engine.documents.revise(job['id'],observations=[{
            'id':'source1','source_sha256':artifact['source_sha256'],'text':'Номер паспорта: 123456',
            'evidence_kind':'image_region','page':1,'page_id':'p1','source_size':[500,100],
            'bbox':[0,0,500,100],'completed':True}],claims=[{
            'id':'claim1','field_key':'passport.number','raw_value':'123456','party_role':'document',
            'evidence_ids':['source1'],'origin':'rule','depends_on':[]}])
        self.service.jobs.next()
        stored={**self.completed(job['id']),'revision':snapshot['revision'],
            'unresolved_rereads':[{'field_key':'passport.number','region_ids':['r1'],
                                   'reason_code':'critical_recheck','unresolved_reason':'budget_exhausted'}]}
        self.service.jobs.finish(job['id'],stored)
        accepting={'decision':'accept_for_autofill','reason_codes':[],'profile_qualified':False,'policy_id':None}
        with patch.object(engine.gateway,'apply',return_value=accepting):
            result=self.service.result(job['id'])
        self.assertEqual(result['decision'],'review')
        self.assertIn('unresolved_reread_requests',result['reason_codes'])
        self.assertEqual(result['contract_result']['fields']['passport.number']['decision'],'review')
        self.assertFalse(result['audit']['eligible_accepted_result'])


if __name__=='__main__':unittest.main()
