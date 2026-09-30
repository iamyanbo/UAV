"""Archive the observed zero-row failed startup without refunding its budget."""
from pathlib import Path
import time
import torch
from photo_goal.common import read,write,digest
from photo_goal.ppo_budget import Budget
from photo_goal.mission_contracts import city_config

root=Path('/mnt/hdd2/yanbocheng/photo-goal-native')
path=root/'city-training/latest.pt'
saved=torch.load(path,map_location='cpu',weights_only=False)
pending=saved.get('pending')
if saved['counts']['accepted_transitions'] or saved['counts']['world_updates'] or not pending or pending['rows']:
    raise RuntimeError('This recovery applies only to the observed zero-row startup failure')
budget=Budget(root/'campaign',city_config());batch=pending['batch_id'];info=budget.batch_info(batch)
if info['actual']!=0 or info['issued']!=1 or info['status']!='collecting':
    raise RuntimeError('Failed startup ledger differs from the observed evidence')
before=budget.snapshot()
with budget.lock,budget.db:
    budget.db.execute('UPDATE batches SET unobserved_discarded=issued,status=? WHERE id=?',('archived',batch))
    budget.db.execute('INSERT INTO events(batch,worker,kind) VALUES (?,?,?)',(batch,0,'startup_no_observation_archived_no_refund'))
archive=root/'runs'/('startup-recovery-'+str(time.time_ns()));archive.mkdir()
checkpoint_sha=digest(path);path.rename(archive/'latest.pt')
for name in ('latest.json','status.json'):
    source=root/'city-training'/name
    if source.exists():source.rename(archive/name)
write(archive/'recovery.json',dict(reason='Zero-row freshness failure; original checkpoint and all charges retained',
    batch=batch,checkpoint_sha256=checkpoint_sha,ledger_before=before,ledger_after=budget.snapshot(),
    next_initialization=str(root/'checkpoints/city-initialized.pt')))
budget.close();print(str(archive))
