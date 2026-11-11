"""Opaque, single-tenant artifact store. No client-controlled filesystem paths."""
from pathlib import Path
from contextlib import contextmanager
import sqlite3
import uuid
from .util import file_sha, utc

class ArtifactStore:
    def __init__(self, root, max_bytes=30*1024*1024):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.max_bytes=max_bytes; self.db=self.root/'artifacts.sqlite'
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY,sha TEXT NOT NULL,size INTEGER,created TEXT)')
    @contextmanager
    def connect(self):
        c=sqlite3.connect(self.db,timeout=30)
        try:
            with c:yield c
        finally:c.close()
    def put(self, content:bytes):
        if not content or len(content)>self.max_bytes: raise ValueError('upload_size_limit')
        aid=uuid.uuid4().hex; p=self.root/aid
        with p.open('xb') as f: f.write(content)
        sha=file_sha(p)
        try:
            with self.connect() as c: c.execute('INSERT INTO artifacts VALUES(?,?,?,?)',(aid,sha,len(content),utc()))
        except BaseException:
            p.unlink(missing_ok=True); raise
        return {'artifact_id':aid,'source_sha256':sha,'bytes':len(content)}
    def get(self, aid):
        if not isinstance(aid,str) or len(aid)!=32 or any(x not in '0123456789abcdef' for x in aid):
            raise ValueError('invalid_artifact_id')
        with self.connect() as c: row=c.execute('SELECT sha,size FROM artifacts WHERE id=?',(aid,)).fetchone()
        if not row: raise KeyError('artifact_not_found')
        p=self.root/aid
        if file_sha(p)!=row[0]: raise ValueError('artifact_integrity_failure')
        return p,{'artifact_id':aid,'source_sha256':row[0],'bytes':row[1]}
