"""Persistent RGB keyframe index and fixed-width mission context, without pose."""
import json
import math
from pathlib import Path
import sqlite3
import numpy as np
from .mission_contracts import CONTEXT_WIDTH


class FlightMemory:
    def __init__(self, path, mission_id):
        self.path, self.mission_id = Path(path), mission_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        # This derived RGB index is reconstructible from recorded contexts.
        # Flush between missions, never block a live decision on HDD fsync.
        self.db.execute('PRAGMA synchronous=NORMAL')
        self.db.execute('PRAGMA wal_autocheckpoint=0')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS keyframes
            (id TEXT PRIMARY KEY, mission TEXT, frame INTEGER, stamp REAL, path TEXT, descriptor TEXT);
          CREATE TABLE IF NOT EXISTS targets
            (id TEXT, mission TEXT, status TEXT, stamp REAL, PRIMARY KEY(id,mission));
        ''')
        self.revision = self.db.execute('SELECT COUNT(*) FROM keyframes WHERE mission=?', (mission_id,)).fetchone()[0]
        self.last = None

    @staticmethod
    def normalize(descriptor):
        d = np.asarray(descriptor, dtype=np.float32)
        if d.shape != (960,) or not np.isfinite(d).all():
            raise ValueError('Expected finite frozen MobileNet descriptor')
        return d/max(float(np.linalg.norm(d)), 1e-12)

    def observe(self, ident, stamp, descriptor, path):
        d = self.normalize(descriptor)
        if self.last and stamp < self.last['stamp']:
            raise ValueError('Mission memory clock went backwards')
        if (self.last is None or stamp-self.last['stamp'] >= 5 or
                float(d @ self.last['descriptor']) < .9):
            row = dict(id=f'{self.mission_id}:f{ident}', frame_id=ident, stamp=stamp,
                       rgb_path=str(path), descriptor=d, source='keyframe')
            with self.db:
                self.db.execute('INSERT OR IGNORE INTO keyframes VALUES (?,?,?,?,?,?)',
                                (row['id'], self.mission_id, ident, stamp, str(path), json.dumps(d.tolist())))
            self.last = row
            self.revision += 1

    def begin(self,mission_id):
        # Reuse the live WAL connection across resets. Closing its last
        # connection recreates/fsyncs a new WAL header on the next first frame.
        self.mission_id=mission_id
        self.revision=self.db.execute('SELECT COUNT(*) FROM keyframes WHERE mission=?',(mission_id,)).fetchone()[0]
        self.last=None

    def flush(self):
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA wal_checkpoint(PASSIVE)')
        self.db.execute('PRAGMA synchronous=NORMAL')

    def retrieve(self, goal_descriptor, current_frame, limit=3):
        goal = self.normalize(goal_descriptor)
        rows = self.db.execute('SELECT id,frame,stamp,path,descriptor FROM keyframes WHERE mission=? AND frame!=?',
                               (self.mission_id, current_frame)).fetchall()
        selected = []
        candidates = [dict(id=r[0], frame_id=r[1], stamp=r[2], rgb_path=r[3],
                           descriptor=np.asarray(json.loads(r[4])), source='keyframe') for r in rows]
        while candidates and len(selected) < limit:
            row = max(candidates, key=lambda r: float(r['descriptor'] @ goal)-.5*max(
                [float(r['descriptor'] @ s['descriptor']) for s in selected] or [0.]))
            selected.append(row)
            candidates = [r for r in candidates if r['id'] != row['id']]
        return selected

    def target_status(self, ident, status, stamp):
        if status not in ('tried', 'completed', 'failed', 'expired'):
            raise ValueError('Invalid target status')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO targets VALUES (?,?,?,?)',
                            (ident, self.mission_id, status, stamp))

    def tried(self):
        return [dict(id=r[0], status=r[1], stamp=r[2]) for r in self.db.execute(
            'SELECT id,status,stamp FROM targets WHERE mission=? ORDER BY stamp DESC LIMIT 32', (self.mission_id,))]

    def vector(self, obs, current_descriptor, goal_descriptor, target=None):
        current, goal = self.normalize(current_descriptor), self.normalize(goal_descriptor)
        tried = self.tried()
        values = [obs['remaining_s']/obs['deadline_s'], min(1., obs['deadline_s']/600),
                  float(current @ goal), min(1., self.revision/100),
                  min(1., len(tried)/32), float(bool(self.last)),
                  float(target is not None), target.confidence if target else 0.,
                  min(1., max(0., obs['sim_s']-target.generation_s)/30) if target else 0.,
                  # Localization is presently unknown; never substitute true pose.
                  1., 0., 0., 0., 0., 0., 0.]
        if len(values) != CONTEXT_WIDTH or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid mission/memory input')
        return values

    def close(self):
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        self.db.close()
