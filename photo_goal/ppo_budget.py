"""One campaign owner, durable reservations before actions, conservative recovery."""
import sqlite3
import threading
import os
from pathlib import Path
from .common import read,write


class Budget:
    def __init__(self,workspace,cfg):
        self.path=Path(workspace)/'budget.json';self.cfg=cfg;self.lock=threading.Lock()
        self.campaign=read(self.path)
        if cfg.get('profile')=='full':
            missing=[key for key in ('physical_transitions','world_updates','grounding_updates','preference_updates') if key not in self.campaign]
            if missing:raise ValueError('Import preserved full campaign counters before training: '+', '.join(missing))
        self.campaign.setdefault('physical_transitions',self.campaign['ppo_transitions'])
        self.db=sqlite3.connect(Path(workspace)/'overnight.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        if cfg.get('schema')=='photo-goal-city-run/v1':self.db.execute('PRAGMA wal_autocheckpoint=0')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS batches
          (id TEXT PRIMARY KEY, requested INTEGER, issued INTEGER DEFAULT 0, actual INTEGER DEFAULT 0, optimized INTEGER DEFAULT 0, status TEXT);
          CREATE TABLE IF NOT EXISTS attempts (id TEXT PRIMARY KEY, kind TEXT, worker INTEGER);
          CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, batch TEXT, worker INTEGER, kind TEXT);''')
        if 'unobserved_discarded' not in {r[1] for r in self.db.execute('PRAGMA table_info(batches)')}:
            self.db.execute('ALTER TABLE batches ADD COLUMN unobserved_discarded INTEGER DEFAULT 0')
        self.db.commit()

    def collection_journal(self):
        # The full physical/PPO batch is reserved before flight. WAL NORMAL
        # retains process-crash recovery without HDD fsync on every action.
        # Machine-crash losses require explicit checkpoint/ledger reconciliation;
        # the durable whole-batch charge is never silently refunded.
        mode='NORMAL' if self.cfg.get('schema')=='photo-goal-city-run/v1' else 'FULL'
        self.db.execute('PRAGMA synchronous='+mode)

    def durable_campaign(self):
        write(self.path,self.campaign)
        with self.path.open('rb') as stream:os.fsync(stream.fileno())
        if os.name!='nt':
            directory=os.open(str(self.path.parent),os.O_RDONLY)
            try:os.fsync(directory)
            finally:os.close(directory)

    def reserve_batch(self,ident,ceiling):
        with self.lock:
            self.db.execute('PRAGMA synchronous=FULL')
            amount=self.cfg['rollout_steps']
            if self.campaign['ppo_transitions']+amount>min(ceiling,self.cfg['campaign_transitions']):return False
            if self.campaign['physical_transitions']+amount>self.cfg['campaign_transitions']:return False
            # The shared JSON includes the entire reservation before any dispatch.
            # If interrupted, unconsumed reservations remain conservatively charged
            # until explicit reconciliation; they are never silently reused.
            self.campaign['ppo_transitions']+=amount;self.campaign['physical_transitions']+=amount;self.durable_campaign()
            with self.db:self.db.execute('INSERT INTO batches(id,requested,status) VALUES (?,?,?)',(ident,amount,'collecting'))
            return True

    def transition(self,batch,worker):
        with self.lock:
            self.collection_journal()
            with self.db:
                count=self.db.execute('SELECT requested,issued FROM batches WHERE id=?',(batch,)).fetchone()
                if count is None or count[1]>=count[0]:raise RuntimeError('No remaining reserved transitions')
                self.db.execute('UPDATE batches SET issued=issued+1 WHERE id=?',(batch,))
                self.db.execute('INSERT INTO events(batch,worker,kind) VALUES (?,?,?)',(batch,worker,'dispatch_reserved'))

    def confirm(self,batch,worker):
        with self.lock,self.db:
            self.db.execute('UPDATE batches SET actual=actual+1 WHERE id=? AND actual<issued',(batch,))
            self.db.execute('INSERT INTO events(batch,worker,kind) VALUES (?,?,?)',(batch,worker,'step_observed'))

    def snapshot(self):
        with self.lock:
            row=self.db.execute('SELECT SUM(requested),SUM(issued),SUM(actual),SUM(optimized),SUM(unobserved_discarded) FROM batches').fetchone()
            return dict(campaign=dict(self.campaign),overnight_ledger=dict(zip(('reserved','dispatch_reserved','observed','optimized','unobserved_discarded'),[v or 0 for v in row])))

    def discard_unobserved(self,batch,worker):
        with self.lock,self.db:
            issued,actual,discarded=self.db.execute('SELECT issued,actual,unobserved_discarded FROM batches WHERE id=?',(batch,)).fetchone()
            if issued<=actual+discarded:raise RuntimeError('No unconfirmed dispatch to discard')
            self.db.execute('UPDATE batches SET unobserved_discarded=unobserved_discarded+1 WHERE id=?',(batch,))
            self.db.execute('INSERT INTO events(batch,worker,kind) VALUES (?,?,?)',(batch,worker,'unconfirmed_dispatch_discarded'))
        self.replace_invalid(batch)

    def replace_invalid(self,batch):
        with self.lock:
            ceiling=self.cfg['campaign_transitions'] if self.cfg.get('profile')=='full' else self.cfg['smoke_transitions']
            if (self.campaign['ppo_transitions']>=min(ceiling,self.cfg['campaign_transitions']) or
                    self.campaign['physical_transitions']>=self.cfg['campaign_transitions']):
                raise RuntimeError('No budget to replace unsupported transition')
            self.campaign['ppo_transitions']+=1;self.campaign['physical_transitions']+=1;write(self.path,self.campaign)
            with self.db:self.db.execute('UPDATE batches SET requested=requested+1 WHERE id=?',(batch,))

    def attempt(self,ident,worker,smoke=True,kind='learner'):
        with self.lock:
            cap=self.cfg['learner_attempts'] if self.cfg.get('profile')=='full' or not smoke else self.cfg['smoke_attempts']
            if self.campaign['training_attempts']>=self.cfg['campaign_attempts']:raise RuntimeError('Campaign attempts exhausted')
            if kind=='learner' and self.campaign['learner_attempts']>=cap:raise RuntimeError('Learner attempts exhausted')
            self.campaign['training_attempts']+=1
            if kind=='learner':self.campaign['learner_attempts']+=1
            write(self.path,self.campaign)
            with self.db:self.db.execute('INSERT INTO attempts VALUES (?,?,?)',(ident,kind,worker))

    def finish(self,batch,optimized):
        with self.lock:
            self.db.execute('PRAGMA synchronous=FULL')
            with self.db:
                row=self.db.execute('SELECT requested,actual,unobserved_discarded FROM batches WHERE id=?',(batch,)).fetchone()
                if optimized and row[0]!=row[1]+row[2]:raise ValueError('Cannot optimize an incomplete reservation')
                self.db.execute('UPDATE batches SET optimized=?,status=? WHERE id=?',
                    (self.cfg['rollout_steps'] if optimized else 0,'optimized' if optimized else 'archived',batch))

    def batch_info(self,batch):
        with self.lock:
            row=self.db.execute('SELECT requested,issued,actual,unobserved_discarded,status FROM batches WHERE id=?',(batch,)).fetchone()
            if row is None:raise ValueError('Missing pending batch reservation')
            return dict(zip(('requested','issued','actual','unobserved_discarded','status'),row))

    def reserve_physical(self,amount):
        """Non-PPO learner pairing/grounding flights have a conservative physical charge."""
        if not isinstance(amount,int) or amount<=0:raise ValueError('Invalid physical reservation')
        with self.lock:
            if self.campaign['physical_transitions']+amount>self.cfg['campaign_transitions']:
                raise RuntimeError('Physical campaign transitions exhausted')
            self.campaign['physical_transitions']+=amount;write(self.path,self.campaign)

    def reserve_updates(self,kind,amount,ceiling):
        if kind not in ('world','grounding','preference') or not isinstance(amount,int) or amount<=0:
            raise ValueError('Invalid supervised update reservation')
        with self.lock:
            key=kind+'_updates'
            used=self.campaign[key]
            if used+amount>ceiling:raise RuntimeError(kind+' update budget exhausted')
            self.campaign[key]=used+amount;write(self.path,self.campaign)

    def close(self):self.db.close()
