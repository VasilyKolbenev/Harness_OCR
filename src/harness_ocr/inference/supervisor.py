"""One spawned model owner, OS lock, hard deadline/cancel and restart after failure."""
from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
import threading
import time
import os
import traceback
import math
from .locks import FileLease,LeaseBusy

class ModelCallError(RuntimeError): pass
class ModelCanceled(ModelCallError): pass
class ModelDeadline(ModelCallError): pass

def _worker_entry(conn,settings,target):
    # The child owns a second lease for its entire lifetime. If the scheduler is
    # SIGKILLed, its orphan cannot overlap a newly started model owner.
    lease=FileLease(settings.model_lock_dir/(settings.device+'-compute.lock'))
    try:
        lease.acquire();conn.send({'bootstrap':True,'ok':True})
        target(conn,settings)
    except LeaseBusy:
        conn.send({'bootstrap':True,'ok':False,'error':'another_model_owner_is_active'})
    finally:
        lease.close();conn.close()

def _worker(conn,settings):
    hf=None; cheap=None
    try:
        while True:
            request=conn.recv()
            if request is None: break
            try:
                op=request['op']; args=request.get('args',{})
                if op=='probe':
                    import torch
                    if settings.device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('cuda_required_no_cpu_fallback')
                    value={'device':settings.device,'torch':torch.__version__,
                           'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           'capability':list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None}
                    if args.get('warmup'):
                        from .hf import HFReaders
                        hf=HFReaders(settings);hf.load('xiaomi');hf.load(settings.qwen)
                        value['loaded_models']=['xiaomi',settings.qwen]
                elif op=='ingest':
                    from ..ingest import decode
                    value=decode(args['path'],args['output'],settings,profile_id=args.get('profile_id'))
                elif op=='cheap':
                    if cheap is None:
                        from .cheap import CheapReader
                        cheap=CheapReader(settings.models_dir,threads=settings.cpu_threads)
                    value=cheap.read(**args)
                else:
                    if hf is None:
                        from .hf import HFReaders
                        hf=HFReaders(settings)
                    if op=='ocr': value=hf.ocr(**args)
                    elif op=='select': value=hf.select(**args)
                    elif op=='visual_read': value=hf.visual_read(**args)
                    else: raise ValueError('unknown_worker_operation')
                conn.send({'ok':True,'value':value})
            except BaseException as e:
                conn.send({'ok':False,'error':type(e).__name__+':'+str(e)[:500]})
    except (EOFError,BrokenPipeError): pass
    finally: conn.close()

class Supervisor:
    def __init__(self,settings,target=_worker):
        self.settings=settings; self.target=target; self.process=None; self.conn=None
        self.mutex=threading.Lock(); self.lock_handle=None; self.restarts=0
    def _start(self,deadline,cancel=None):
        self.settings.data_dir.mkdir(parents=True,exist_ok=True)
        self.lock_handle=FileLease(self.settings.model_lock_dir/(self.settings.device+'-admission.lock'))
        try:
            self.lock_handle.acquire()
        except BaseException:
            self.lock_handle.close(); self.lock_handle=None; raise ModelCallError('another_model_owner_is_active')
        child=None
        try:
            ctx=mp.get_context('spawn');self.conn,child=ctx.Pipe()
            self.process=ctx.Process(target=_worker_entry,args=(child,self.settings,self.target),daemon=True)
            self.process.start();child.close();child=None
            while True:
                if cancel and cancel():raise ModelCanceled('canceled_during_worker_startup')
                if time.perf_counter()>=deadline:raise ModelDeadline('worker_startup_deadline')
                if self.conn.poll(.05):
                    reply=self.conn.recv()
                    if reply.get('bootstrap') is not True or reply.get('ok') is not True:
                        raise ModelCallError(reply.get('error','invalid_worker_bootstrap'))
                    return
                if not self.process.is_alive():raise ModelCallError('worker_died_during_startup')
        except BaseException:
            if child:child.close()
            self._stop();raise
    def _stop(self):
        if self.process:
            if self.process.pid is not None:
                if self.process.is_alive(): self.process.terminate()
                self.process.join(timeout=3)
                if self.process.is_alive(): self.process.kill();self.process.join(timeout=3)
                if self.process.is_alive(): raise ModelCallError('worker_could_not_be_killed')
            self.process.close();self.process=None;self.restarts+=1
        if self.conn: self.conn.close();self.conn=None
        if self.lock_handle: self.lock_handle.close();self.lock_handle=None
    def close(self):
        with self.mutex:self._stop()
    def call(self,op,args=None,cancel=None,timeout=None):
        chosen=self.settings.call_timeout_seconds if timeout is None else timeout
        if isinstance(chosen,bool) or not isinstance(chosen,(int,float)) or not math.isfinite(chosen) or chosen<=0:
            raise ValueError('call_timeout_must_be_finite_and_positive')
        with self.mutex:
            if cancel and cancel(): raise ModelCanceled('canceled_before_compute')
            deadline=time.perf_counter()+chosen
            try:
                if not self.process or not self.process.is_alive():self._stop();self._start(deadline,cancel)
                self.conn.send({'op':op,'args':args or {}})
                while True:
                    if cancel and cancel():raise ModelCanceled('canceled_worker_terminated')
                    if time.perf_counter()>=deadline:raise ModelDeadline('hard_deadline_worker_terminated')
                    if self.conn.poll(.05):
                        try:reply=self.conn.recv()
                        except EOFError:raise ModelCallError('worker_died')
                        if not reply['ok']:raise ModelCallError(reply['error'])
                        return reply['value']
                    if not self.process.is_alive():raise ModelCallError('worker_died')
            except BaseException:
                self._stop();raise
