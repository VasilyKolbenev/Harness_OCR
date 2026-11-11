"""Actual MCP SDK stdio JSON-RPC -> fixed local HTTP endpoint; no model work."""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import unittest

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

TOKEN = 'fixture-token-32-characters-long-abc'
JOB_ID = 'a' * 32
ARTIFACT_ID = 'b' * 32


class MCPIntegrationTests(unittest.TestCase):
    def test_stdio_protocol_catalog_schema_validation_and_http_binding(self):
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def respond(self):
                if self.headers.get('Authorization') != 'Bearer ' + TOKEN:
                    self.send_error(401)
                    return
                length = int(self.headers.get('Content-Length', '0'))
                body = json.loads(self.rfile.read(length)) if length else None
                calls.append((self.command, self.path, body))
                if self.path == '/v1/profiles':
                    result = [{'profile_id': 'fixture_profile', 'qualification_status': 'not_qualified'}]
                elif self.path == '/v1/jobs':
                    result = {'id': JOB_ID, 'status': 'queued', 'request': body}
                elif self.path.endswith('/result'):
                    result = {'document_id': JOB_ID, 'decision': 'review', 'service_approval': False}
                else:
                    result = {'id': JOB_ID, 'status': 'canceled' if self.path.endswith('/cancel') else 'completed'}
                data = json.dumps(result).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = respond
            do_POST = respond

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        worker.start()

        async def exchange():
            env = {k: v for k, v in os.environ.items() if k.upper() in {'PATH', 'SYSTEMROOT', 'TEMP', 'TMP', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA'}}
            env.update({'PYTHONPATH': str(Path(__file__).resolve().parents[1] / 'src'),
                'PYTHONIOENCODING': 'utf-8', 'HARNESS_API_TOKEN': TOKEN,
                'HARNESS_SERVICE_URL': f'http://127.0.0.1:{server.server_port}'})
            parameters = StdioServerParameters(command=sys.executable, args=['-m', 'harness_ocr.mcp_server'], env=env)
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=10) as session:
                    initialized = await session.initialize()
                    self.assertEqual(initialized.server_info.name, 'harness-ocr')
                    catalog = await session.list_tools()
                    tools = {tool.name: tool for tool in catalog.tools}
                    self.assertEqual(set(tools), {'ocr_profiles', 'ocr_submit', 'ocr_status', 'ocr_result', 'ocr_cancel'})
                    self.assertEqual(set(tools['ocr_submit'].input_schema['properties']), {'artifact_id', 'profile_id', 'subset'})
                    for tool in tools.values():
                        self.assertFalse(tool.input_schema['additionalProperties'])
                        self.assertFalse({'path', 'base_url', 'token', 'authorized', 'policy_id', 'reviewer', 'auto_accept'} & set(tool.input_schema.get('properties', {})))
                    listed = await session.call_tool('ocr_profiles')
                    self.assertFalse(listed.is_error)
                    self.assertIn('fixture_profile', json.dumps(listed.model_dump()))
                    submitted = await session.call_tool('ocr_submit', {'artifact_id': ARTIFACT_ID,
                        'profile_id': 'fixture_profile', 'subset': ['document.number']})
                    self.assertFalse(submitted.is_error)
                    self.assertIn(JOB_ID, json.dumps(submitted.model_dump()))
                    before = len(calls)
                    invalid = await session.call_tool('ocr_status', {'job_id': '../../private'})
                    self.assertTrue(invalid.is_error)
                    self.assertEqual(len(calls), before)
                    forged = await session.call_tool('ocr_submit', {'artifact_id': ARTIFACT_ID,
                        'profile_id': 'fixture_profile', 'authorized': True, 'path': '/etc/passwd'})
                    self.assertTrue(forged.is_error)
                    self.assertEqual(len(calls), before)
                    for name in ['ocr_status', 'ocr_result', 'ocr_cancel']:
                        result = await session.call_tool(name, {'job_id': JOB_ID})
                        self.assertFalse(result.is_error)

        try:
            asyncio.run(exchange())
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
        self.assertEqual(calls[0][:2], ('GET', '/v1/profiles'))
        self.assertEqual(calls[1], ('POST', '/v1/jobs', {'artifact_id': ARTIFACT_ID,
            'profile_id': 'fixture_profile', 'subset': ['document.number']}))
        self.assertEqual(len(calls), 5)


if __name__ == '__main__':
    unittest.main()
