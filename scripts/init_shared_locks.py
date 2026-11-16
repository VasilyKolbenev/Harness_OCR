"""Initialize only the dedicated /locks volume; never replace live lock inodes.

Run as root in an isolated container with no GPU, network or other mounts.
The root-owned files in a sticky directory can be opened by application UIDs,
but an application UID cannot unlink another owner's active lease file.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import stat

FILES = ('cpu-admission.lock', 'cpu-compute.lock', 'cuda-admission.lock', 'cuda-compute.lock',
         'gpu-delivery.lock')


def initialize_locks(lock_dir, *, _test_directory=False):
    directory = Path(lock_dir)
    if os.name != 'posix' or not hasattr(os, 'O_NOFOLLOW'):
        raise ValueError('shared_lock_init_requires_posix_nofollow')
    if not directory.is_absolute() or (str(directory) != '/locks' and not _test_directory):
        raise ValueError('only_explicit_locks_mount_is_allowed')
    if not _test_directory and os.geteuid() != 0:
        raise ValueError('shared_lock_init_requires_root')
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        directory.mkdir(mode=0o1777)
        metadata = directory.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ValueError('lock_mount_must_be_real_directory')
    folder = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = []
    try:
        actual = os.fstat(folder)
        if (actual.st_dev, actual.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise ValueError('lock_directory_changed')
        # Open and validate every existing inode before changing permissions.
        # O_NONBLOCK prevents a planted FIFO from blocking before fstat.
        for name in FILES:
            descriptor = os.open(name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                                 0o666, dir_fd=folder)
            opened.append((name, descriptor))
            entry = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=folder, follow_symlinks=False)
            if (not stat.S_ISREG(entry.st_mode) or entry.st_nlink != 1
                    or (entry.st_dev, entry.st_ino) != (linked.st_dev, linked.st_ino)):
                raise ValueError('lease_file_must_be_single_regular_inode')
        os.fchmod(folder, 0o1777)
        if not _test_directory:
            os.fchown(folder, 0, 0)
        rows = []
        for name, descriptor in opened:
            if not _test_directory:
                os.fchown(descriptor, 0, 0)
            os.fchmod(descriptor, 0o666)
            entry = os.fstat(descriptor)
            linked = os.stat(name, dir_fd=folder, follow_symlinks=False)
            if (entry.st_dev, entry.st_ino) != (linked.st_dev, linked.st_ino):
                raise ValueError('lease_inode_changed')
            rows.append({'name':name, 'inode':entry.st_ino, 'device':entry.st_dev,
                         'mode':stat.S_IMODE(entry.st_mode), 'bytes':entry.st_size,
                         'uid':entry.st_uid, 'gid':entry.st_gid})
        return {'schema_version':'harness-shared-lock-init/1', 'status':'pass',
                'lock_dir':str(directory), 'directory_mode':stat.S_IMODE(os.fstat(folder).st_mode),
                'files':rows, 'live_inodes_replaced':False, 'contents_truncated':False,
                'test_directory':_test_directory, 'production_initialized':not _test_directory}
    finally:
        for _, descriptor in opened:
            os.close(descriptor)
        os.close(folder)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-dir', required=True, help='The isolated /locks volume only')
    args = parser.parse_args(argv)
    try:
        result = initialize_locks(args.lock_dir)
    except Exception as exc:
        result = {'schema_version':'harness-shared-lock-init/1', 'status':'fail',
                  'error_type':type(exc).__name__, 'production_initialized':False,
                  'error_code':str(exc) if isinstance(exc,ValueError) else 'filesystem_lock_initialization_failed',
                  'errno':getattr(exc,'errno',None)}
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
