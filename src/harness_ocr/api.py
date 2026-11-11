"""Authenticated, single-tenant service boundary. No paths or authority flags in requests."""
from __future__ import annotations
from contextlib import asynccontextmanager
import os
import hmac
from typing import Literal
from fastapi import FastAPI,Depends,Header,HTTPException,Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel,ConfigDict,Field,StrictBool,field_validator
from .settings import Settings
from . import __version__
from .service import Service
from .core import RevisionConflict

class JobRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    artifact_id:str=Field(pattern=r'^[0-9a-f]{32}$')
    profile_id:str=Field(min_length=1,max_length=128)
    subset:list[str]|None=Field(default=None,max_length=100)
    qwen:Literal['qwen-0.8b','qwen-4b']|None=None
    mode:Literal['hybrid','rules']='hybrid'

class Correction(BaseModel):
    model_config=ConfigDict(extra='forbid')
    field_key:str=Field(min_length=1,max_length=128)
    raw_value:str|dict
    evidence_ids:list[str]=Field(min_length=1,max_length=100)
    expected_revision:int=Field(ge=0,strict=True)

    @field_validator('raw_value')
    @classmethod
    def bounded_literal(cls,value):
        import json
        if not value or len(json.dumps(value,ensure_ascii=False))>250000:raise ValueError('invalid_raw_value_size')
        if isinstance(value,str) and len(value)>10000:raise ValueError('literal_length_limit')
        if isinstance(value,dict):
            from .typed_fields import _structure
            if not _structure(value):raise ValueError('invalid_typed_value_contract')
        return value

class AuditOutcome(BaseModel):
    model_config=ConfigDict(extra='forbid')
    critical_error:StrictBool
    evidence_ids:list[str]=Field(min_length=1,max_length=100)
    correction_revision:int|None=Field(default=None,ge=0,strict=True)

def create_app(service=None,api_token=None,review_token=None):
    service=service or Service(Settings.from_env())
    api_token=api_token or os.getenv('HARNESS_API_TOKEN','')
    review_token=review_token or os.getenv('HARNESS_REVIEW_TOKEN','')
    if (any(not isinstance(t,str) or not t.isascii() or any(ch.isspace() for ch in t)
            for t in (api_token,review_token))
            or min(len(api_token),len(review_token))<24 or api_token==review_token):
        raise RuntimeError('two_distinct_server_tokens_min24_required')
    def scopes(authorization:str|None=Header(default=None)):
        supplied=authorization[7:] if authorization and authorization.startswith('Bearer ') else ''
        if not supplied.isascii():raise HTTPException(401,'unauthorized')
        if hmac.compare_digest(supplied,review_token):return {'read','submit','review'}
        if hmac.compare_digest(supplied,api_token):return {'read','submit'}
        raise HTTPException(401,'unauthorized')
    def reviewer(granted=Depends(scopes)):
        if 'review' not in granted:raise HTTPException(403,'review_scope_required')
    @asynccontextmanager
    async def lifespan(app):
        service.start()
        try:yield
        finally:service.close()
    app=FastAPI(title='Harness OCR',version=__version__,lifespan=lifespan)
    @app.exception_handler(KeyError)
    async def missing(request,exc):return JSONResponse({'detail':'not_found'},status_code=404)
    @app.exception_handler(ValueError)
    async def invalid(request,exc):return JSONResponse({'detail':str(exc)},status_code=422)
    @app.exception_handler(RevisionConflict)
    async def conflict(request,exc):return JSONResponse({'detail':'revision_conflict'},status_code=409)
    @app.get('/health/live')
    def live():
        status=service.liveness_status()
        return JSONResponse(status,status_code=200 if status['live'] else 503)
    @app.get('/health/ready',dependencies=[Depends(scopes)])
    def ready():
        status=service.readiness_status()
        return JSONResponse(status,status_code=200 if status['ready'] else 503)
    @app.get('/v1/profiles',dependencies=[Depends(scopes)])
    def profiles():return service.engine.profiles.list()
    @app.post('/v1/artifacts',dependencies=[Depends(scopes)])
    async def upload(request:Request):
        if request.headers.get('content-type','').split(';')[0]!='application/octet-stream':
            raise HTTPException(415,'use_application_octet_stream')
        data=bytearray()
        async for chunk in request.stream():
            if len(data)+len(chunk)>service.settings.max_upload_bytes:raise HTTPException(413,'upload_size_limit')
            data.extend(chunk)
        return service.artifacts.put(bytes(data))
    @app.post('/v1/jobs',dependencies=[Depends(scopes)],status_code=202)
    def submit(request:JobRequest,idempotency_key:str|None=Header(default=None)):
        if idempotency_key and len(idempotency_key)>192:raise HTTPException(422,'idempotency_key_limit')
        try:return _job_public(service.submit(request.model_dump(exclude_none=True),idempotency_key))
        except RuntimeError as e:raise HTTPException(503,str(e))
    @app.get('/v1/jobs/{jid}',dependencies=[Depends(scopes)])
    def status(jid:str):return _job_public(service.jobs.get(jid))
    @app.get('/v1/jobs/{jid}/result',dependencies=[Depends(scopes)])
    def result(jid:str):
        value=service.result(jid)
        if value is None:raise HTTPException(409,'result_not_available')
        return value
    @app.post('/v1/jobs/{jid}/cancel',dependencies=[Depends(scopes)])
    def cancel(jid:str):
        return _job_public(service.cancel(jid))
    @app.get('/v1/jobs/{jid}/trace',dependencies=[Depends(scopes)])
    def trace(jid:str):
        service.jobs.get(jid);return service.engine.trace.records(document_id=jid)
    @app.post('/v1/jobs/{jid}/review',dependencies=[Depends(reviewer)])
    def review(jid:str,correction:Correction):return service.review(jid,correction.model_dump())
    @app.post('/v1/audits/{task_id}/outcome',dependencies=[Depends(reviewer)])
    def audit_outcome(task_id:str,outcome:AuditOutcome):
        return service.record_audit_outcome(task_id,outcome.model_dump(),authorized=True)
    return app

def _job_public(job):
    return {k:v for k,v in job.items() if k not in {'result','idem','request_sha'}}
