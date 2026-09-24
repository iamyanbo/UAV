"""Continue eligible learning and physical collection in a guarded window.

Existing integration receipts bind all reused datasets and checkpoints. New
data/objectives require a new round; incomplete preference supervision does
not prevent independent world, policy, or collection work.
"""
import argparse
import hashlib
import json
from pathlib import Path

from program_scheduler import run
from resume_connected_cycle import connected_state


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def specification(cycle, increment, previous=None):
    state=connected_state(cycle)
    rows=state['stages'];stages=[]
    def completed(name):
        row=rows[name]
        if row['status']!='completed':raise ValueError('Integration stage incomplete: '+name)
        for ref in [row['worker_receipt'],*row.get('artifacts',[])]:
            if digest(ref['path'])!=ref['sha256']:raise ValueError('Changed integration artifact: '+ref['path'])
        return row
    flights=completed('development-flights')
    collection=Path(flights['job'])/'collection/flights.json'
    development=json.loads(collection.read_text())
    if len(development)!=10 or not any(r['result'].get('actual_path_length_m',0)>.5 for r in development):
        raise ValueError('Ten internal development flights with actual movement are required')
    completed('updated-policy-flight');completed('final-reload-flight')
    for module,prior,data_stage,data_suffix,cap in (
        ('world','world-update','world-data','world-data',300000),
        ('policy','dagger-update','dagger-data','dataset',200000)):
        row=completed(prior);completed(data_stage)
        dataset=Path(next(p for p in row['input_hashes'] if p.endswith('/manifest.json'))).parent
        checkpoint=Path(row['job'])/'training/final.pt'
        if previous:
            prior_state=json.loads((previous/'execution/state.json').read_text())
            row=prior_state['stages'][module+'-continuation']
            if row['status']!='completed':raise ValueError('Resume the unfinished previous window first')
            for ref in [row['worker_receipt'],*row['artifacts']]:
                if digest(ref['path'])!=ref['sha256']:raise ValueError('Changed continuation artifact')
            checkpoint=Path(row['job'])/'training/final.pt'
        # The initial imitation update precedes the DAgger optimizer round.
        prior_updates=int(rows['policy-update']['progress']['updates']) if module=='policy' else 0
        target=min(cap-prior_updates,int(row['progress']['updates'])+increment)
        if target<=int(row['progress']['updates']):continue
        command=['python3','{source}/stage_worker.py','train-'+module,'--dataset',str(dataset),
            '--updates',str(target),'--resume',str(checkpoint)]
        if module=='policy':command+=['--goal-checkpoint','goal.pt']
        resume=list(command);resume[resume.index('--resume')+1]='{checkpoint}'
        stages.append(dict(id=module+'-continuation',kind='gpu',peak_gib=36,seconds=7200,
            depends_on=[],inputs=[str(dataset/'manifest.json'),str(checkpoint)],command=command,
            resume_command=resume,checkpoint='{job}/training/latest.pt',
            receipt='{job}/training/result.json',outputs=['{job}/training/final.pt']))
    # These new learner flights remain raw causal data for the next explicit
    # dataset round. Do not silently mix them into an optimizer's old dataset.
    pack=Path(completed('ppo-pack')['artifacts'][0]['path']).parent
    root=Path.home()/'uav-rgb-flight'
    labels=root/'launches/20260922T002848Z/manifests/evaluator_labels/train.json'
    seen=set()
    for receipt in (root/'launches').glob('*/episode/result.json'):
        value=json.loads(receipt.read_text())
        if value.get('split')=='train':seen.add(value['episode_id'])
    candidates=[r['episode_id'] for r in json.loads(labels.read_text())['episodes']
        if r['split']=='train' and r['episode_id'] not in seen and
        int(hashlib.sha256(r['goal_region_id'].encode()).hexdigest()[:8],16)%10!=0][:10]
    if candidates:
        command=['python3','{source}/collect_learning_round.py','--controller-checkpoints',str(pack)]
        for identifier in candidates:command+=['--episode-id',identifier]
        stages.insert(0,dict(id='next-training-collection',kind='flight',peak_gib=72,seconds=1800,
            depends_on=[],inputs=[str(pack/'checkpoints.json'),str(labels)],command=command,
            receipt='{job}/collection/result.json',outputs=['{job}/collection/flights.json']))
    reload_collection=Path(rows['updated-policy-flight']['job'])/'collection/flights.json'
    episode=json.loads(reload_collection.read_text())[0]['result']['episode_id']
    stages.insert(0,dict(id='source-reload-flight',kind='flight',peak_gib=72,seconds=1200,
        depends_on=[],inputs=[str(pack/'checkpoints.json'),str(reload_collection)],
        command=['python3','{source}/collect_learning_round.py','--episode-id',episode,
            '--controller-checkpoints',str(pack),'--goal-collection',str(reload_collection),'--with-deliberation'],
        receipt='{job}/collection/result.json',outputs=['{job}/collection/flights.json']))
    return dict(schema='training-dependencies/v1',integration_state_sha256=digest(cycle/'execution/state.json'),
        scope='Cumulative budget continuation; new raw flights require a subsequent dataset round',
        remaining_supervision='New grounding examples and strict matched preferences remain collection-dependent',stages=stages)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cycle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous-window',type=Path)
    parser.add_argument('--updates-per-window',type=int,default=2000)
    parser.add_argument('--hours',type=float,default=8);args=parser.parse_args()
    if not 0<args.hours<=8 or args.updates_per_window<1:parser.error('Positive updates and at most eight hours required')
    args.output.mkdir(parents=True,exist_ok=True);path=args.output/'programme-spec.json'
    # Resume the exact schedule; do not reselect episodes after collection.
    if not path.exists():path.write_text(json.dumps(specification(args.cycle.resolve(),args.updates_per_window,
        args.previous_window.resolve() if args.previous_window else None),indent=2))
    raise SystemExit(run(path,args.output/'execution',args.hours))
