"""Collection-only curriculum, provenance and transactional attempt admission."""
import json
import math
import os
from pathlib import Path
import sqlite3
import time

from .common import config

SCENES = 'photo-map-scenes/v3'
MISSIONS = 'photo-map-missions/v5'
LABELS = 'privileged-photo-map-labels/v5'
DATASET = 'photo-map-dataset/v6'


def curriculum_slot(index):
    # A deterministic twenty-slot block has exact 20/30/35/15 proportions.
    slots = [0]*4 + [1]*6 + [2]*7 + [3]*3
    return config()['episodes']['curriculum'][slots[index % len(slots)]]


def source_slot(index):
    # Rotate sources between curriculum blocks so both collection streams
    # receive every length band, including the 1–2 km missions.
    cycle=config()['collection']['source_cycle']
    return cycle[(index+(index//20)*(len(cycle)//2))%len(cycle)]


def route_complexity(points):
    """Conservative geometry proxy, distinct from annotated semantic decisions."""
    import numpy as np
    p = np.asarray(points, float)
    d = np.diff(p, axis=0)
    turns = sum(float(np.dot(a[:2], b[:2]) / max(1e-9, np.linalg.norm(a[:2])*np.linalg.norm(b[:2])))
                < math.cos(math.radians(30)) for a, b in zip(d, d[1:])
                if min(np.linalg.norm(a[:2]), np.linalg.norm(b[:2])) > 1)
    previous = 0; altitude_choices = 0
    for edge in d:
        sign = int(np.sign(edge[2])) if abs(edge[2]) >= 4 else 0
        if sign and sign != previous: altitude_choices += 1
        if sign: previous = sign
    length = float(np.linalg.norm(d, axis=1).sum())
    # Reject artificial revisits except the immediate neighbouring route segment.
    loop = any(np.linalg.norm(p[i]-p[j]) < 5 for i in range(len(p)) for j in range(i+3, len(p)))
    return dict(turns=int(turns), altitude_choices=altitude_choices,
                decision_proxy=int(turns+altitude_choices), reference_length_m=length, artificial_loop=loop)


class AttemptQueue:
    """Atomic reservations count failed/interrupted attempts; never auto-retry."""
    def __init__(self, path, identity):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, stream TEXT, split TEXT, '
                       'status TEXT, pid INTEGER, started REAL, result TEXT)')
            if 'reserved_disk_bytes' not in {r[1] for r in db.execute('PRAGMA table_info(attempts)')}:
                db.execute('ALTER TABLE attempts ADD COLUMN reserved_disk_bytes INTEGER NOT NULL DEFAULT 0')
            db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('registry', identity))
            if db.execute('SELECT value FROM metadata WHERE key=?', ('registry',)).fetchone()[0] != identity:
                raise ValueError('Queue registry identity changed')

    def connect(self):
        return sqlite3.connect(self.path, timeout=30, isolation_level='IMMEDIATE')

    def reserve(self, ident, stream, split, limit=None, required_disk_bytes=0, output=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT id FROM attempts WHERE id=?', (ident,)).fetchone(): return False
            if split == 'train':
                caps = config()['collection']['stream_budgets']
                if stream not in caps: raise ValueError('Unrecognized collection stream')
                total = db.execute("SELECT COUNT(*) FROM attempts WHERE split='train'").fetchone()[0]
                used = db.execute("SELECT COUNT(*) FROM attempts WHERE split='train' AND stream=?", (stream,)).fetchone()[0]
                if total >= min(config()['training']['budgets']['collection_episodes'], limit or math.inf) or used >= caps[stream]:
                    return False
            if required_disk_bytes:
                import shutil
                pending=db.execute("SELECT COALESCE(SUM(reserved_disk_bytes),0) FROM attempts WHERE status='running'").fetchone()[0]
                if required_disk_bytes<0 or shutil.disk_usage(output).free<pending+required_disk_bytes:
                    raise RuntimeError('Insufficient aggregate disk for concurrent complete episodes')
            db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?)',
                       (ident, stream, split, 'running', os.getpid(), time.time(), None,int(required_disk_bytes)))
            return True

    def finish(self, ident, result):
        with self.connect() as db:
            db.execute('UPDATE attempts SET status=?,result=? WHERE id=?',
                       ('finished', json.dumps(result, allow_nan=False), ident))

    def summary(self):
        with self.connect() as db:
            return [dict(split=s, stream=t, status=k, attempts=n) for s,t,k,n in
                    db.execute('SELECT split,stream,status,COUNT(*) FROM attempts GROUP BY split,stream,status')]


def admission(profile, output):
    """No fixed RAM percentage/reserve. A measured workload peak is mandatory."""
    from .common import available_memory
    import shutil
    needed = int(profile['incremental_peak_bytes']) + int(profile['transient_peak_bytes'])
    disk = int(profile['recording_peak_bytes']) + int(profile.get('checkpoint_bytes', 0))
    if min(int(profile['incremental_peak_bytes']),int(profile['transient_peak_bytes']))<0 or needed <= 0 or disk <= 0 or not profile.get('measured'):
        raise ValueError('Admission requires measured positive memory and disk peaks')
    if available_memory() < needed: return False
    return shutil.disk_usage(output).free >= disk


def workload_identity(manifests,split,packages,perception_package,teacher_spec,variants):
    import hashlib
    from .common import digest
    root=Path(__file__).resolve().parent
    source={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*.py'))}
    assets={'missions':digest(Path(manifests)/(split+'.json')),'campaign':digest(Path(__file__).with_name('campaign.json'))}
    if packages:
        from .common import read
        assets['packages']=digest(packages)
        for seed,path in read(packages)['seeds'].items():assets['package-'+str(seed)]=digest(Path(path)/'package.json')
    if perception_package:assets['perception']=digest(Path(perception_package)/'package.json')
    if teacher_spec:assets['teacher']=digest(teacher_spec)
    return hashlib.sha256(json.dumps(dict(source=source,assets=assets,variants=variants),sort_keys=True).encode()).hexdigest()


class ResourceMonitor:
    """Observed aggregate growth, including transients; no guessed RAM reserve."""
    def __init__(self,output,registry,identity,workers):
        import threading
        import uuid
        from .common import digest
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/('resources-'+uuid.uuid4().hex+'.json')
        self.record=dict(schema='photo-map-resources/v1',registry_sha256=digest(registry),workload_identity=identity,workers=workers)
        self.stop=threading.Event();self.errors=[];self.samples=[]

    def sample(self):
        import shutil
        from .common import available_memory
        self.samples.append((time.monotonic(),available_memory(),shutil.disk_usage(self.output).free))

    def __enter__(self):
        import threading
        self.before={str(p.resolve()) for p in self.output.glob('*/episode/result.json')}
        self.sample()
        def loop():
            while not self.stop.wait(.25):
                try:self.sample()
                except Exception as error:self.errors.append(type(error).__name__);break
        self.thread=threading.Thread(target=loop,daemon=True);self.thread.start();return self

    def __exit__(self,kind,*args):
        from .common import write,digest
        self.stop.set();self.thread.join();self.sample()
        first=self.samples[0];last=self.samples[-1]
        paths=[p for p in self.output.glob('*/episode/result.json') if str(p.resolve()) not in self.before]
        self.record.update(measured=not self.errors,qualified=False,errors=self.errors,
            interrupted=kind is not None,wall_seconds=last[0]-first[0],samples=len(self.samples),
            incremental_peak_bytes=max(0,first[1]-min(r[1] for r in self.samples)),transient_peak_bytes=0,
            transient_accounting='included in observed aggregate peak; 250ms sampling',
            recording_peak_bytes=max(0,first[2]-min(r[2] for r in self.samples)),
            disk_accounting='filesystem growth including any concurrent acquisitions; conservative upper bound',
            results=[dict(path=str(p.resolve()),sha256=digest(p)) for p in paths])
        write(self.path,self.record)


def select_concurrency(rows):
    """Select by valid flight-seconds/wall-second; reject latency/physics failures."""
    eligible = [r for r in rows if r['qualified'] and r['clock_speed'] == 1
                and r['p95_source_to_dispatch_s'] <= .05 and r['p99_source_to_dispatch_s'] <= .25
                and r['timing_valid_fraction'] >= .99 and r['wall_seconds'] > 0]
    if not eligible: raise ValueError('No qualified concurrency measurement')
    return max(eligible, key=lambda r: (r['valid_flight_seconds']/r['wall_seconds'], -r['workers']))
