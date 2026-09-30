"""Conservative cross-process HDD admission with batched write leases."""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sqlite3
import threading

GIB = 2**30
_guards = {}
_lock = threading.RLock()


def scan_bytes(root):
    """Concurrent rename/unlink is benign; permission and I/O errors propagate."""
    seen, total = set(), 0
    def walk(directory):
        nonlocal total
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_symlink():
                            if not Path(entry.path).resolve().is_relative_to(Path(root).resolve()):
                                raise RuntimeError('Study path escapes HDD root: '+entry.path)
                            continue
                        stat = entry.stat(follow_symlinks=False)
                        inode = (stat.st_dev,stat.st_ino)
                        if inode in seen: continue
                        seen.add(inode)
                        total += max(stat.st_size,getattr(stat,'st_blocks',0)*512)
                        if entry.is_dir(follow_symlinks=False): walk(entry.path)
                    except FileNotFoundError: pass
        except FileNotFoundError: pass
    walk(root)
    return total


class ProjectSpace:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)
        self.path = self.root/'campaign'/'space.sqlite'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.limit = int(os.environ.get('UAV_PROJECT_LIMIT_BYTES',256*GIB))
        self.reserve = int(os.environ.get('UAV_CHECKPOINT_RESERVE_BYTES',2*GIB))
        self.free_reserve = 100*GIB
        self.local_remaining = 0
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS space (id INTEGER PRIMARY KEY CHECK(id=1), baseline INTEGER NOT NULL, charged INTEGER NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS leases (pid INTEGER PRIMARY KEY, remaining_upper INTEGER NOT NULL)')
            if db.execute('SELECT 1 FROM space').fetchone() is None:
                db.execute('INSERT INTO space VALUES(1,?,0)',(scan_bytes(self.root)+16*2**20,))

    def connect(self):
        db=sqlite3.connect(self.path,timeout=10)
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
        return db

    def bound(self):
        with self.connect() as db:
            baseline,charged=db.execute('SELECT baseline,charged FROM space WHERE id=1').fetchone()
        return baseline+charged

    def check(self, additional=0, shutdown=False):
        maximum=self.limit if shutdown else self.limit-self.reserve
        bound=self.bound()
        if bound+additional > maximum or shutil.disk_usage(self.root).free-additional < self.free_reserve:
            raise RuntimeError('Project/HDD capacity admission exhausted; checkpoint and stop')
        return bound

    def charge(self, ceiling, shutdown=False):
        if ceiling < 0: raise ValueError('Negative storage reservation')
        with _lock:
            if not shutdown and self.local_remaining >= ceiling:
                self.local_remaining-=ceiling; return
            lease=ceiling if shutdown else max(64*2**20,ceiling)
            with self.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                baseline,charged=db.execute('SELECT baseline,charged FROM space WHERE id=1').fetchone()
                maximum=self.limit if shutdown else self.limit-self.reserve
                if baseline+charged+lease > maximum or shutil.disk_usage(self.root).free-lease < self.free_reserve:
                    raise RuntimeError('Project/HDD capacity reservation refused')
                db.execute('UPDATE space SET charged=charged+? WHERE id=1',(lease,))
                db.execute('INSERT OR REPLACE INTO leases VALUES(?,?)',(os.getpid(),0 if shutdown else lease-ceiling))
            self.local_remaining=0 if shutdown else lease-ceiling

    def observe(self):
        # A live scan never lowers the bound, so an ENOENT cannot free capacity.
        size=scan_bytes(self.root)
        with self.connect() as db:
            db.execute('UPDATE space SET baseline=MAX(baseline,?-charged) WHERE id=1',(size+16*2**20,))
        return self.check()

    def reconcile_quiescent(self):
        """Only the trainer's idle/checkpoint boundary may reclaim conservative charges."""
        with self.connect() as db:
            before=db.execute('SELECT charged FROM space WHERE id=1').fetchone()[0]
        size=scan_bytes(self.root)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            after=db.execute('SELECT charged FROM space WHERE id=1').fetchone()[0]
            outstanding=db.execute('SELECT COALESCE(SUM(remaining_upper),0) FROM leases').fetchone()[0]
            db.execute('UPDATE space SET baseline=?,charged=? WHERE id=1',
                       (size+outstanding+16*2**20,max(0,after-before)))
        return self.check()


def guard():
    root=os.environ.get('UAV_PROJECT_ROOT')
    if not root: return None
    with _lock:
        key=(os.getpid(),root)
        if key not in _guards: _guards[key]=ProjectSpace(root)
        return _guards[key]


def reserve_write(path, ceiling, shutdown=False):
    space=guard()
    if space:
        if not Path(path).resolve().is_relative_to(space.root):
            raise ValueError('Study write escapes verified project root: '+str(path))
        space.charge(int(ceiling),shutdown)


@contextmanager
def shutdown_writes():
    old=os.environ.get('UAV_SHUTDOWN_WRITES')
    os.environ['UAV_SHUTDOWN_WRITES']='1'
    try: yield
    finally:
        if old is None: os.environ.pop('UAV_SHUTDOWN_WRITES',None)
        else: os.environ['UAV_SHUTDOWN_WRITES']=old
