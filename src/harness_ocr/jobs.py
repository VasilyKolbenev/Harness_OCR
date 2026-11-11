from __future__ import annotations
from contextlib import contextmanager
import sqlite3
import json
import uuid
from pathlib import Path
from copy import deepcopy
from .util import utc,digest

TERMINAL={'completed','failed','canceled','budget_exhausted','interrupted'}

class JobStore:
    def __init__(self,path,max_pending=64):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.max_pending=max_pending
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,idem TEXT UNIQUE,request_sha TEXT,request TEXT,status TEXT,cancel INTEGER,result TEXT,created TEXT,updated TEXT)')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=30);db.row_factory=sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        try:
            with db:yield db
        finally:db.close()
    def create(self,request,idempotency_key=None):
        sha=digest(request)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if idempotency_key:
                old=db.execute('SELECT * FROM jobs WHERE idem=?',(idempotency_key,)).fetchone()
                if old:
                    if old['request_sha']!=sha:raise ValueError('idempotency_key_payload_mismatch')
                    return self._decode(old)
            count=db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]
            if count>=self.max_pending:raise ValueError('queue_full')
            jid=uuid.uuid4().hex;now=utc()
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?)',(jid,idempotency_key,sha,json.dumps(request), 'queued',0,None,now,now))
        return self.get(jid)
    def _decode(self,row):
        d=dict(row);d['request']=json.loads(d['request']);d['result']=json.loads(d['result']) if d['result'] else None
        d['cancel_requested']=bool(d.pop('cancel'));return d
    def get(self,jid):
        with self.connect() as db:row=db.execute('SELECT * FROM jobs WHERE id=?',(jid,)).fetchone()
        if not row:raise KeyError('job_not_found')
        return self._decode(row)
    def next(self,expected_id=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if expected_id is None:
                row=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
            else:
                row=db.execute("SELECT * FROM jobs WHERE status='queued' AND id=?",(expected_id,)).fetchone()
            if not row:return None
            db.execute("UPDATE jobs SET status='running',updated=? WHERE id=?",(utc(),row['id']))
        return self.get(row['id'])
    def peek(self):
        """Inspect the next queued request without consuming it for verification."""
        with self.connect() as db:
            row=db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created,id LIMIT 1").fetchone()
        return self._decode(row) if row else None
    def cancel(self,jid):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT status FROM jobs WHERE id=?',(jid,)).fetchone()
            if not row:raise KeyError('job_not_found')
            if row['status'] not in TERMINAL:
                db.execute("UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'canceled' ELSE status END,updated=? WHERE id=?",(utc(),jid))
        return self.get(jid)
    def finish(self,jid,result):
        # The caller's projection and trace remain immutable. Publication may
        # race with cancellation; its accepted branch must not survive anywhere
        # in the durable canceled result.
        result=deepcopy(result)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT cancel,status FROM jobs WHERE id=?',(jid,)).fetchone()
            if not row:raise KeyError('job_not_found')
            if row['status'] in TERMINAL:raise ValueError('terminal_job_is_immutable')
            if row['cancel']:
                result={**result,'terminal_status':'canceled','completed':False,'decision':'review',
                        'needs_review':True,
                        'reason_codes':sorted(set(result.get('reason_codes',[])+['canceled']))}
                for f in result.get('fields',{}).values():f['decision']='review'
                projection=result.get('contract_result')
                if isinstance(projection,dict):
                    projection['needs_review']=True
                    for f in projection.get('fields',{}).values():f['decision']='review'
                result.pop('audit',None)
                result.pop('drift',None)
            status=result['terminal_status']
            if status not in TERMINAL:raise ValueError('invalid_terminal_status')
            db.execute('UPDATE jobs SET status=?,result=?,updated=? WHERE id=?',(status,json.dumps(result,ensure_ascii=False),utc(),jid))
        return self.get(jid)
    def recover(self):
        # An interrupted model call is never silently repeated after restart.
        with self.connect() as db:
            rows=db.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
            db.execute("UPDATE jobs SET status='interrupted',updated=? WHERE status='running'",(utc(),))
        return [r[0] for r in rows]
