"""Malformed bearer input fails as authentication, never as a server error."""
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from harness_ocr.api import create_app
from harness_ocr.service import Service
from harness_ocr.settings import Settings

class NoModels:
    def close(self):pass
    def call(self,*args,**kwargs):raise AssertionError('No inference allowed')

class AuthAuditTests(unittest.TestCase):
    def test_non_ascii_bearer_is_401_and_invalid_server_tokens_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            service=Service(Settings(root/'state',root/'models',device='cpu',dtype='float32',allow_fixture=True),NoModels(),fixture=True)
            for invalid in ['я'*32,'a'*24+'\t','a'*24+' ']:
                with self.subTest(token=repr(invalid)),self.assertRaises(RuntimeError):
                    create_app(service,invalid,'b'*32)
            with TestClient(create_app(service,'a'*32,'b'*32)) as client:
                result=client.get('/v1/profiles',headers={b'Authorization':b'Bearer \xff'})
                self.assertEqual(result.status_code,401)
                self.assertEqual(result.json(),{'detail':'unauthorized'})

if __name__=='__main__':unittest.main()
