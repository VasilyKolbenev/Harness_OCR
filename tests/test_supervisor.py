"""Real spawn/OS-lock/deadline integration with a non-neural child process."""
import os
from pathlib import Path
import tempfile
import time
import unittest

from harness_ocr.inference.supervisor import (Supervisor, ModelCallError, ModelCanceled, ModelDeadline)
from harness_ocr.settings import Settings


def fake_worker(conn, settings):
    try:
        while True:
            request = conn.recv()
            if request is None:
                break
            op, args = request['op'], request['args']
            if op == 'crash':
                os._exit(42)
            if op == 'error':
                conn.send({'ok': False, 'error': 'fixture_failure'})
                continue
            if op == 'wait':
                time.sleep(args['seconds'])
            conn.send({'ok': True, 'value': {'pid': os.getpid(), 'op': op}})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        conn.close()


class SupervisorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(Path(self.temp.name) / 'data', Path(self.temp.name) / 'models',
            device='cpu', dtype='float32', call_timeout_seconds=5,
            runtime_lock_dir=Path(self.temp.name)/'shared-locks')
        self.owner = Supervisor(self.settings, target=fake_worker)

    def tearDown(self):
        self.owner.close()
        self.temp.cleanup()

    def test_hard_timeout_kills_owner_and_next_call_spawns_fresh(self):
        first = self.owner.call('ping')['pid']
        started = time.monotonic()
        with self.assertRaises(ModelDeadline):
            self.owner.call('wait', {'seconds': 5}, timeout=.15)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNone(self.owner.process)
        second = self.owner.call('ping')['pid']
        self.assertNotEqual(first, second)
        self.assertGreaterEqual(self.owner.restarts, 1)

    def test_cancellation_terminates_active_worker_and_preserves_reusability(self):
        first = self.owner.call('ping')['pid']
        started = time.monotonic()
        with self.assertRaises(ModelCanceled):
            self.owner.call('wait', {'seconds': 5}, cancel=lambda: time.monotonic() - started > .15)
        self.assertIsNone(self.owner.process)
        self.assertNotEqual(self.owner.call('ping')['pid'], first)

    def test_cancel_before_compute_never_starts_child(self):
        with self.assertRaises(ModelCanceled):
            self.owner.call('ping', cancel=lambda: True)
        self.assertIsNone(self.owner.process)
        self.assertIsNone(self.owner.lock_handle)

    def test_worker_error_and_unexpected_exit_both_restart_cleanly(self):
        self.owner.call('ping')
        for op in ['error', 'crash']:
            with self.assertRaises(ModelCallError):
                self.owner.call(op)
            self.assertIsNone(self.owner.process)
            self.assertEqual(self.owner.call('ping')['op'], 'ping')

    def test_os_lock_prevents_second_owner_and_releases_on_close(self):
        self.owner.call('ping')
        second = Supervisor(self.settings, target=fake_worker)
        try:
            with self.assertRaisesRegex(ModelCallError, 'another_model_owner_is_active'):
                second.call('ping')
            self.owner.close()
            self.assertEqual(second.call('ping')['op'], 'ping')
        finally:
            second.close()


if __name__ == '__main__':
    unittest.main()
