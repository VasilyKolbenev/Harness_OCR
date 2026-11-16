"""Bounded cleanup restricted to observed PID + creation-time descendants."""
from __future__ import annotations
from pathlib import Path
import os
import signal
import time
from .plan import utc,write


class VerifiedTree:
    def __init__(self,pid,new_session=False):
        import psutil
        self.psutil=psutil;root=psutil.Process(pid)
        self.pid=pid;self.created=root.create_time();self.records={};self.errors=[]
        self.records[pid]={'pid':pid,'created':self.created,'depth':0}
        self.group=None
        if new_session:
            if os.name!='posix' or os.getpgid(pid)!=pid or os.getsid(pid)!=pid:
                raise RuntimeError('row_private_process_group_not_verified')
            self.group=pid
    def verified(self,record):
        try:
            process=self.psutil.Process(record['pid'])
            if abs(process.create_time()-record['created'])>=.01:
                error='pid_identity_mismatch'
                if error not in self.errors:self.errors.append(error)
                return None
            return process if process.is_running() and process.status()!=self.psutil.STATUS_ZOMBIE else None
        except self.psutil.NoSuchProcess:return None
        except self.psutil.Error as error:
            self.errors.append(type(error).__name__);return None
    def snapshot(self):
        root=self.verified({'pid':self.pid,'created':self.created})
        if root is None:return
        try:children=root.children(recursive=True)
        except self.psutil.NoSuchProcess:return
        except self.psutil.Error as error:self.errors.append(type(error).__name__);return
        for child in children:
            try:
                created=child.create_time();parent=child.ppid()
                if not child.is_running() or child.status()==self.psutil.STATUS_ZOMBIE:continue
                if child.pid in self.records and abs(self.records[child.pid]['created']-created)>=.01:
                    self.errors.append('pid_identity_mismatch');continue
                depth=self.records.get(parent,{}).get('depth',0)+1
                self.records[child.pid]={'pid':child.pid,'created':created,'depth':depth}
            except self.psutil.NoSuchProcess:continue
            except self.psutil.Error as error:self.errors.append(type(error).__name__)
    def alive(self):return [r for r in self.records.values() if self.verified(r) is not None]
    def group_anchor(self):
        if self.group is None:return False
        for record in list(self.records.values()):
            process=self.verified(record)
            if process is not None:
                try:
                    if os.getpgid(process.pid)==self.group and os.getsid(process.pid)==self.group:return True
                except ProcessLookupError:continue
                except PermissionError:self.errors.append('owned_group_inspection_denied')
        return False
    def capture_private_group(self):
        # Only inspect group membership after an observed PID+creation-time
        # anchor proves this is still the row's newly created private session.
        if not self.group_anchor():return False
        for process in self.psutil.process_iter(['pid']):
            matched=False
            try:
                if os.getpgid(process.pid)==self.group and os.getsid(process.pid)==self.group:
                    matched=True
                    created=process.create_time()
                    old=self.records.get(process.pid)
                    if old and abs(old['created']-created)>=.01:
                        self.errors.append('pid_identity_mismatch');continue
                    self.records[process.pid]={'pid':process.pid,'created':created,
                        'depth':0 if process.pid==self.pid else 99}
            except (ProcessLookupError,self.psutil.NoSuchProcess):continue
            except (PermissionError,self.psutil.AccessDenied):
                if matched:self.errors.append('owned_group_member_identity_denied')
        return True
    def stop(self,wait_seconds=3,kill_wait_seconds=1):
        self.snapshot();actions=[]
        if self.capture_private_group():
            try:os.killpg(self.group,signal.SIGTERM);actions.append({'group':self.group,'action':'terminate_verified_private_group'})
            except ProcessLookupError:pass
            except PermissionError:self.errors.append('owned_group_terminate_denied')
        records=sorted(self.records.values(),key=lambda r:(r['pid']==self.pid,-r['depth']))
        for record in records:
            process=self.verified(record)
            if process is not None:
                try:process.terminate();actions.append({'pid':record['pid'],'action':'terminate_verified'})
                except self.psutil.NoSuchProcess:pass
                except self.psutil.Error as error:self.errors.append(type(error).__name__)
        end=time.monotonic()+wait_seconds
        while time.monotonic()<end and self.alive():time.sleep(.05)
        if self.capture_private_group():
            try:os.killpg(self.group,signal.SIGKILL);actions.append({'group':self.group,'action':'kill_verified_private_group'})
            except ProcessLookupError:pass
            except PermissionError:self.errors.append('owned_group_kill_denied')
        for record in records:
            process=self.verified(record)
            if process is not None:
                try:process.kill();actions.append({'pid':record['pid'],'action':'kill_verified'})
                except self.psutil.NoSuchProcess:pass
                except self.psutil.Error as error:self.errors.append(type(error).__name__)
        until=time.monotonic()+kill_wait_seconds
        while time.monotonic()<until and self.alive():time.sleep(.02)
        return actions


def wait_child(process,tree,stop_path,compute_end,kill_start):
    requested=False;forced=[];next_snapshot=0
    while process.poll() is None:
        if time.monotonic()>=next_snapshot:tree.snapshot();next_snapshot=time.monotonic()+.25
        if utc()>=compute_end and not requested:
            write(stop_path,{'reason':'global_administrative_deadline'});requested=True
        if utc()>=kill_start:
            forced=tree.stop();process.wait(timeout=1);break
        time.sleep(.05)
    return {'exit_code':process.returncode,'administrative_stop_requested':requested,
        'forced_cleanup':forced,'verified_processes':list(tree.records.values()),'inspection_errors':tree.errors}
