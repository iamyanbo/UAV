"""Read the actual active window, process state and training ledger."""
import json
import sqlite3
import argparse
from pathlib import Path
from photo_goal.common import read

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/mnt/hdd2/yanbocheng/photo-goal-native'))
parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
root=args.root.resolve()
active=read(root/'runs/active-city-window.json');job=Path(active['job'])
plan=read(job/'launch.json')
run_root=args.run_dir or Path(plan.get('run_dir',str(root/'city-training')))
checkpoint_path=run_root/'latest.json'
checkpoint=read(checkpoint_path) if checkpoint_path.exists() else None
with sqlite3.connect('file:'+str(root/'campaign/overnight.sqlite')+'?mode=ro',uri=True) as db:
    rows=db.execute("SELECT COALESCE(SUM(actual),0),COALESCE(SUM(optimized),0) FROM batches WHERE id NOT LIKE 'qualification-%'").fetchone()
    qualification=db.execute("SELECT COALESCE(SUM(actual),0) FROM batches WHERE id LIKE 'qualification-%'").fetchone()[0]
def alive(pid):
    status=Path('/proc')/str(pid)/'status'
    return status.exists() and '\nState:\tZ' not in status.read_text()
guidance=run_root/'guidance.jsonl';selected=0
if guidance.exists():
    for line in guidance.read_text().splitlines():
        try:selected+=bool(json.loads(line).get('selected'))
        except json.JSONDecodeError:pass
print(json.dumps(dict(job=str(job),status=plan['status'],
    operator_alive=alive(plan['pid']),trainer_alive=alive(plan.get('trainer_pid',0)),
    qwen_alive=alive(plan.get('qwen_pid',0)),observed_training_rows=rows[0],
    metrics_alive=alive(plan.get('metrics_pid',0)),metrics_dashboard=plan.get('metrics_dashboard'),
    accepted_guidance_responses=selected,
    accepted_ppo_rows=rows[1],qualification_rows_excluded=qualification,
    checkpoint_counts=checkpoint['counts'] if checkpoint else None,
    phase_id=plan.get('phase_id'),phase_accepted_batches=checkpoint.get('phase_accepted_batches') if checkpoint else None,
    started_utc=plan.get('started_at_utc',plan.get('started_utc')),ended_utc=plan.get('ended_at_utc'),
    exit_reason=plan.get('exit_reason'),maximum_hours=plan['maximum_hours']),indent=2))
