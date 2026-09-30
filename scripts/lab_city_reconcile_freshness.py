"""Preserve the recorded on-policy batch and charge its missing stale dispatch."""
from pathlib import Path
import os
import time
import torch
from photo_goal.common import read,write,digest
from photo_goal.ppo_budget import Budget
from photo_goal.mission_contracts import city_config

root=Path('/mnt/hdd2/yanbocheng/photo-goal-native')
path=root/'city-training/latest.pt';saved=torch.load(path,map_location='cpu',weights_only=False)
pending=saved.get('pending');status=read(root/'city-training/status.json')
if not pending or status.get('error')!='RuntimeError: Stale policy decision; watchdog brakes':
    raise RuntimeError('Only the recorded stale-action failure is admitted by this recovery')
budget=Budget(root/'campaign',city_config());batch=pending['batch_id'];info=budget.batch_info(batch)
if info['actual']!=len(pending['rows']) or info['issued']!=info['actual']+info['unobserved_discarded']+1:
    raise RuntimeError('Pending batch/one missing dispatch disagree')
archive=root/'runs'/('freshness-recovery-'+str(time.time_ns()));archive.mkdir()
os.link(path,archive/'previous.pt');before=budget.snapshot()
budget.discard_unobserved(batch,0)
write(archive/'recovery.json',dict(reason='One unobserved stale dispatch conservatively charged and replaced',
    batch=batch,checkpoint_sha256=digest(path),preserved_rows=len(pending['rows']),
    ledger_before=before,ledger_after=budget.snapshot()))
budget.close();print(str(archive))
