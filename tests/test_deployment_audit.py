"""Deployment regressions; these never load or download neural models."""
from dataclasses import replace
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from harness_ocr.settings import Settings
from harness_ocr.inference.supervisor import Supervisor,ModelCallError
from harness_ocr.inference.locks import FileLease,LeaseBusy
from test_supervisor import fake_worker

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('deployment_healthcheck',ROOT/'scripts'/'healthcheck.py')
healthcheck=importlib.util.module_from_spec(spec);spec.loader.exec_module(healthcheck)

def orphan_worker(conn,settings):
    request=conn.recv()
    ready=Path(request['args']['ready']);ready.write_text(str(os.getpid()))
    deadline=time.perf_counter()+20
    while not ready.with_name(ready.name+'-release').exists() and time.perf_counter()<deadline:time.sleep(.02)
    try:conn.send({'ok':True,'value':{}})
    except (BrokenPipeError,EOFError,OSError):pass

def abandon_parent(settings,ready):
    owner=Supervisor(settings,target=orphan_worker)
    thread=threading.Thread(target=lambda:owner.call('hold',{'ready':ready}),daemon=True);thread.start()
    deadline=time.perf_counter()+8
    while not Path(ready).exists() and time.perf_counter()<deadline:time.sleep(.02)
    os._exit(17)

class DeploymentLeaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.settings=Settings(self.root/'api',self.root/'models',device='cpu',dtype='float32',
             call_timeout_seconds=5,runtime_lock_dir=self.root/'shared-locks')
    def tearDown(self):self.temp.cleanup()
    def test_different_api_and_benchmark_state_dirs_share_one_device_owner(self):
        api=Supervisor(self.settings,target=fake_worker)
        bench=Supervisor(replace(self.settings,data_dir=self.root/'benchmark'),target=fake_worker)
        try:
            api.call('ping')
            with self.assertRaisesRegex(ModelCallError,'another_model_owner_is_active'):bench.call('ping')
            api.close();self.assertEqual(bench.call('ping')['op'],'ping')
        finally:api.close();bench.close()
    def test_timeout_override_cannot_disable_hard_budget(self):
        owner=Supervisor(self.settings,target=fake_worker)
        try:
            for value in (float('nan'),float('inf'),0,-1,True):
                with self.subTest(value=value),self.assertRaises(ValueError):owner.call('ping',timeout=value)
            self.assertIsNone(owner.process)
        finally:owner.close()
    def test_launch_failure_releases_admission_lease(self):
        owner=Supervisor(self.settings,target=fake_worker)
        try:
            with patch('multiprocessing.process.BaseProcess.start',side_effect=OSError('controlled launch failure')):
                with self.assertRaises(OSError):owner.call('ping')
            self.assertIsNone(owner.process);self.assertIsNone(owner.lock_handle)
            self.assertEqual(owner.call('ping')['op'],'ping')
        finally:owner.close()
    def test_callback_failure_terminates_active_worker(self):
        owner=Supervisor(self.settings,target=fake_worker)
        try:
            owner.call('ping');calls=0
            def fail():
                nonlocal calls
                calls+=1
                if calls>1:raise RuntimeError('controlled callback failure')
                return False
            with self.assertRaisesRegex(RuntimeError,'controlled callback failure'):
                owner.call('wait',{'seconds':2},cancel=fail)
            self.assertIsNone(owner.process)
            self.assertEqual(owner.call('ping')['op'],'ping')
        finally:owner.close()
    def test_orphan_retains_compute_lease_after_scheduler_is_killed(self):
        ready=self.root/'orphan-ready';parent=mp.get_context('spawn').Process(target=abandon_parent,args=(self.settings,str(ready)))
        parent.start();parent.join(timeout=10)
        if parent.is_alive():parent.terminate();parent.join();self.fail('fixture parent did not exit')
        self.assertEqual(parent.exitcode,17);self.assertTrue(ready.is_file())
        contender=Supervisor(replace(self.settings,data_dir=self.root/'new-api'),target=fake_worker)
        try:
            with self.assertRaisesRegex(ModelCallError,'another_model_owner_is_active'):contender.call('ping')
            ready.with_name(ready.name+'-release').touch()
            deadline=time.perf_counter()+5
            while True:
                try:
                    lease=FileLease(self.settings.model_lock_dir/'cpu-compute.lock').acquire();lease.close();break
                except LeaseBusy:
                    if time.perf_counter()>=deadline:raise
                    time.sleep(.02)
            self.assertEqual(contender.call('ping')['op'],'ping')
        finally:
            ready.with_name(ready.name+'-release').touch();contender.close();parent.close()

class HealthProbeTests(unittest.TestCase):
    def response(self,value,status=200):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,limit):return json.dumps(value).encode()
        r=Response();r.status=status;return r
    def test_liveness_or_fixture_never_means_model_readiness(self):
        for value in ({'live':True},{'ready':False,'fixture':False},{'ready':True,'fixture':True}):
            with self.subTest(value=value),patch.object(healthcheck.urllib.request,'urlopen',return_value=self.response(value)):
                self.assertFalse(healthcheck.check_ready(token='controlled-token'))
        with patch.object(healthcheck.urllib.request,'urlopen',return_value=self.response({'ready':True,'fixture':False})) as request:
            self.assertTrue(healthcheck.check_ready(token='controlled-token'))
            self.assertEqual(request.call_args.args[0].headers['Authorization'],'Bearer controlled-token')
    def test_missing_token_and_http_errors_fail_closed(self):
        with patch.dict(os.environ,{},clear=True),patch.object(healthcheck.urllib.request,'urlopen') as request:
            self.assertFalse(healthcheck.check_ready());request.assert_not_called()
        with patch.object(healthcheck.urllib.request,'urlopen',side_effect=OSError('unreachable')):
            self.assertFalse(healthcheck.check_ready(token='controlled-token'))

if __name__=='__main__':unittest.main()
