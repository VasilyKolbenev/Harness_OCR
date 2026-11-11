"""MCP SDK 2 adapter to the same authenticated service, for an existing catalog.

Stdio only: credentials come from process environment; tools cannot choose hosts,
filesystem paths, operator identities, model weights or security policy.
"""
import os
from . import __version__
import httpx
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError


class StrictMCPServer(MCPServer):
    """SDK 2 defaults ignore extra arguments; reject before any HTTP action.

    Override the public SDK methods so the advertised contract and actual
    dispatch agree, without modifying SDK internals or accepting **kwargs.
    """
    async def list_tools(self):
        tools = await super().list_tools()
        for tool in tools:
            tool.input_schema = {**tool.input_schema, 'additionalProperties': False}
        return tools

    async def call_tool(self, name, arguments, context=None):
        tool = next((t for t in await self.list_tools() if t.name == name), None)
        if tool is not None:
            extras = set(arguments) - set(tool.input_schema.get('properties', {}))
            if extras:
                raise ToolError('unknown_tool_arguments:' + ','.join(sorted(extras)))
        return await super().call_tool(name, arguments, context)

def make_server(base_url=None,token=None):
    base_url=base_url or os.environ.get('HARNESS_SERVICE_URL','http://127.0.0.1:8080')
    token=token or os.environ.get('HARNESS_API_TOKEN','')
    if not token:raise RuntimeError('HARNESS_API_TOKEN_required')
    server=StrictMCPServer('harness-ocr',version=__version__,instructions=
        'OCR provides source-backed extraction and review decisions. It never approves a government service.')
    def request(method,path,body=None):
        with httpx.Client(base_url=base_url,headers={'Authorization':'Bearer '+token},timeout=30) as client:
            r=client.request(method,path,json=body);r.raise_for_status();return r.json()
    @server.tool()
    def ocr_profiles()->list[dict]:
        """List versioned extraction profiles and their qualification status."""
        return request('GET','/v1/profiles')
    @server.tool()
    def ocr_submit(artifact_id:str,profile_id:str,subset:list[str]|None=None)->dict:
        """Submit a previously uploaded opaque artifact. Returns a durable job ID."""
        body={'artifact_id':artifact_id,'profile_id':profile_id}
        if subset is not None:body['subset']=subset
        return request('POST','/v1/jobs',body)
    @server.tool()
    def ocr_status(job_id:str)->dict:
        """Read durable job state; terminal errors remain explicit."""
        _id(job_id);return request('GET',f'/v1/jobs/{job_id}')
    @server.tool()
    def ocr_result(job_id:str)->dict:
        """Get literal fields, evidence, validation and review decision."""
        _id(job_id);return request('GET',f'/v1/jobs/{job_id}/result')
    @server.tool()
    def ocr_cancel(job_id:str)->dict:
        """Cancel a queued/running job; partial evidence cannot be accepted."""
        _id(job_id);return request('POST',f'/v1/jobs/{job_id}/cancel')
    return server

def _id(value):
    if len(value)!=32 or any(c not in '0123456789abcdef' for c in value):raise ToolError('invalid_job_id')

def main():make_server().run(transport='stdio')
if __name__=='__main__':main()
