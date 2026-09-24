"""Carry new perception demonstrations through the complete learning flow."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from program_scheduler import atomic, digest, run
from run_connected_cycle import specification


def main():
    parser=argparse.ArgumentParser()
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--perception-cycle',type=Path)
    source.add_argument('--goal-cycle',type=Path,help='Attempt-aware goal correction round with separate qualification')
    parser.add_argument('--base-pack',type=Path,help='Immutable perception pack whose goal checkpoint is replaced')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--recollect-demonstrations',action='store_true',
        help='Collect a new ten-flight batch when runtime evidence contracts changed')
    args=parser.parse_args()
    if not 0<args.hours<=8:parser.error('At most eight hours including waiting')
    if args.goal_cycle and not args.base_pack:parser.error('--goal-cycle requires --base-pack')
    args.output.mkdir(parents=True,exist_ok=True)
    receipt=args.output/'continuation.json';deadline=time.monotonic()+args.hours*3600
    while time.monotonic()<deadline-150:
        if (args.output/'STOP').exists() or (Path.home()/'uav-rgb-flight/STOP').exists():
            atomic(receipt,dict(status='operator_stopped',accepted=False));return 2
        state_path=args.goal_cycle/'state.json' if args.goal_cycle else args.perception_cycle/'execution/state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else None
        if state:
            rows=state['stages']
            failures={k:v['reason'] for k,v in rows.items() if v['status'] in ('failed','blocked')}
            if failures:
                atomic(receipt,dict(status='blocked',failures=failures,accepted=False));return 2
            dependency='corrected-goal-qualification' if args.goal_cycle else 'new-perception-demonstrations'
            if rows[dependency]['status']=='completed':break
        atomic(receipt,dict(status='waiting_for_perception_and_demonstrations',accepted=False))
        time.sleep(5)
    else:
        atomic(receipt,dict(status='window_expired_waiting',accepted=False));return 2
    dependencies=('goal-corrective-training','corrected-goal-qualification') if args.goal_cycle else ('selected-perception-pack','new-perception-demonstrations')
    for name in dependencies:
        row=rows[name]
        for ref in [row['worker_receipt'],*row.get('artifacts',[])]:
            if digest(ref['path'])!=ref['sha256']:raise ValueError('Changed dependency: '+ref['path'])
    if args.goal_cycle:
        pack=args.output/'perception-checkpoints';base=json.loads((args.base_pack/'checkpoints.json').read_text())
        goal=Path(rows['goal-corrective-training']['job'])/'visual-training/best.pt'
        qualification=json.loads(Path(rows['corrected-goal-qualification']['worker_receipt']['path']).read_text())
        if qualification['checkpoint_sha256']!=digest(goal):raise ValueError('Qualification does not match selected goal checkpoint')
        command=[sys.executable,str(Path(__file__).with_name('package_navigation.py')),
            '--goal',str(goal),'--goal-match-threshold',str(base['goal_match_threshold']),'--output',str(pack)]
        expected={'goal':digest(goal)}
        for role in ('odometry','projection','safety','vision'):
            if role not in base['artifacts']:continue
            ref=base['artifacts'][role];path=args.base_pack/ref['path']
            if digest(path)!=ref['sha256']:raise ValueError('Changed base perception artifact: '+role)
            expected[role]=ref['sha256'];command+=['--'+role,str(path)]
        if not pack.exists():subprocess.run(command,check=True)
        actual=json.loads((pack/'checkpoints.json').read_text())
        if {k:v['sha256'] for k,v in actual['artifacts'].items()}!=expected:
            raise ValueError('Changed perception inputs require a new output round')
        if actual['goal_match_threshold']!=base['goal_match_threshold']:raise ValueError('Changed stopping threshold requires a new round')
        collection=None
    else:
        pack=Path(rows['selected-perception-pack']['artifacts'][0]['path']).parent
        collection=Path(rows['new-perception-demonstrations']['job'])/'collection/flights.json'
    spec=specification(pack,pack/'safety.json','train-00000',None if args.recollect_demonstrations else collection)
    spec['perception_cycle']=str((args.goal_cycle or args.perception_cycle).resolve())
    spec['recollect_demonstrations']=args.recollect_demonstrations
    path=args.output/'programme-spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Changed inputs require a new round')
    if not path.exists():atomic(path,spec)
    atomic(receipt,dict(status='full_flow_running',accepted=False))
    code=run(path,args.output/'execution',(deadline-time.monotonic())/3600)
    final=json.loads((args.output/'execution/state.json').read_text())['stages']
    independent=all(final[name]['status']=='completed' for name in
        ('world-update','policy-update','qwen-supervised-update','dagger-update','ppo-update',
         'updated-policy-flight','final-reload-flight','development-flights'))
    preference=final['qwen-preference-update']['status']
    continuation=None
    if independent and deadline-time.monotonic()>150:
        continuation=args.output/'cumulative-window'
        atomic(receipt,dict(status='cumulative_training_running',independent_flow_completed=True,
            preference_status=preference,accepted=False))
        with (args.output/'cumulative.log').open('a') as log:
            subprocess.run([sys.executable,str(Path(__file__).with_name('run_continuation_window.py')),
                '--cycle',str(args.output),'--output',str(continuation),
                '--hours',str((deadline-time.monotonic())/3600)],stdout=log,stderr=subprocess.STDOUT,check=False)
    atomic(receipt,dict(status='window_ended',independent_flow_completed=independent,
        preference_status=preference,cumulative_window=str(continuation) if continuation else None,
        scheduler_return_code=code,accepted=False,
        reason='Movement, map handover and navigation are measured separately; timing optimization is deferred'))
    return code


if __name__=='__main__':raise SystemExit(main())
