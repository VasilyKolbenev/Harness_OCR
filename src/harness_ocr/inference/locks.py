"""OS-owned leases. Stale files are harmless; live file descriptors own locks."""
from pathlib import Path
import os

class LeaseBusy(RuntimeError):pass

class FileLease:
    def __init__(self,path):self.path=Path(path);self.handle=None
    def acquire(self):
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        handle=self.path.open('a+b')
        try:
            if os.name=='nt':
                import msvcrt
                if self.path.stat().st_size==0:handle.write(b'0');handle.flush()
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except (OSError,BlockingIOError) as error:
            handle.close();raise LeaseBusy('another_model_owner_is_active') from error
        self.handle=handle;return self
    def close(self):
        if self.handle:self.handle.close();self.handle=None
    def __enter__(self):return self.acquire()
    def __exit__(self,*args):self.close()
