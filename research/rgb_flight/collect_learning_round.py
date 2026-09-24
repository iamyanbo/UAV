"""Collect complete physical training flights and preserve unsuccessful outcomes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--episode-id',action='append')
    parser.add_argument('--demonstration-batch',action='store_true')
    parser.add_argument('--import-collection',type=Path)
    parser.add_argument('--development-batch',action='store_true')
    parser.add_argument('--controller-checkpoints',type=Path)
    parser.add_argument('--sample-policy',action='store_true')
    parser.add_argument('--with-deliberation',action='store_true')
    parser.add_argument('--demonstrate',action='store_true')
    parser.add_argument('--goal',type=Path)
    parser.add_argument('--goal-collection',type=Path,help='Reuse exact goal pixels from this completed collection')
    parser.add_argument('--collection-name',default='collection')
    parser.add_argument('--bootstrap-hold-seconds',type=float,default=0.)
    args=parser.parse_args()
    root=Path.home()/'uav-rgb-flight'
    job=Path(os.environ['RGB_JOB_DIR']).resolve();output=(job/args.collection_name).resolve()
    if not output.is_relative_to(job) or output==job:raise ValueError('Collection output must remain inside its guarded job')
    output.mkdir()
    labels=root/'launches/20260922T002848Z/manifests/evaluator_labels/train.json'
    if args.demonstration_batch or args.development_batch:
        candidates=json.loads(labels.read_text())['episodes'];selected={'train':[],'validation':[]}
        for row in candidates:
            if row['split']!='train':raise ValueError('Only campaign training manifest may supply development')
            split='validation' if int(hashlib.sha256(row['goal_region_id'].encode()).hexdigest()[:8],16)%10==0 else 'train'
            selected[split].append(row['episode_id'])
        args.episode_id=selected['validation'][:10] if args.development_batch else selected['train'][:8]+selected['validation'][:2]
        if len(args.episode_id)!=10:raise ValueError('Ten disjoint manifest episodes required')
        if args.demonstration_batch:args.demonstrate=True
    if not args.episode_id:parser.error('Specify episode IDs or a ten-flight batch')
    flights=[];retained_failures=[]
    if args.import_collection:
        for item in json.loads(args.import_collection.read_text()):
            saved=json.loads((Path(item['episode_path'])/'result.json').read_text())
            if saved!=item['result']:raise ValueError('Imported flight receipt changed')
            if saved['episode_id'] not in args.episode_id:continue
            if saved.get('learned_controller',{}).get('status')!='completed':
                retained_failures.append(item);continue
            if args.controller_checkpoints and saved.get('controller_checkpoint_sha256')!=hashlib.sha256((args.controller_checkpoints/'checkpoints.json').read_bytes()).hexdigest():
                raise ValueError('Imported demonstration checkpoint differs')
            flights.append(item)
        (output/'imported-failures.json').write_text(json.dumps(retained_failures,indent=2))
    if args.goal and args.goal_collection:raise ValueError('Choose one exact goal source')
    goal_flights=json.loads(args.goal_collection.read_text()) if args.goal_collection else None
    for identifier in args.episode_id:
        if any(r['result']['episode_id']==identifier for r in flights):continue
        command=[sys.executable,str(Path(__file__).with_name('spark_launch.py')),
            '--probe','visual-goal-flight','--evaluator-labels',str(labels),
            '--episode-id',identifier,'--obstacle-field',str(root/'launches/20260922T002848Z/obstacle-field.npz'),
            '--bootstrap-hold-seconds',str(args.bootstrap_hold_seconds)]
        if args.controller_checkpoints:
            command+=['--controller-checkpoints',str(args.controller_checkpoints),'--integration-only']
            if args.sample_policy:command+=['--sample-policy']
            if args.with_deliberation:command+=['--with-deliberation']
            if args.demonstrate:command+=['--demonstrate']
        goal=args.goal
        if goal_flights is not None:
            matches=[item for item in goal_flights if item['result']['episode_id']==identifier]
            if len(matches)!=1:raise ValueError('Goal collection requires exactly one matching episode')
            goal=Path(matches[0]['episode_path'])/'goal'
        if goal:command+=['--goal',str(goal)]
        for attempt in range(2):
            process=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
            (output/f'{identifier}-attempt-{attempt}.log').write_text(process.stdout)
            launches=[]
            for line in process.stdout.splitlines():
                try:value=json.loads(line)
                except ValueError:continue
                if isinstance(value,dict) and 'launch' in value:launches.append(Path(value['launch']))
            if len(launches)!=1:raise RuntimeError('Flight launch identity unavailable; inspect retained log')
            episode=launches[0]/'episode';path=episode/'result.json'
            receipt=json.loads(path.read_text()) if path.exists() else None
            item=dict(episode_path=str(episode),result=receipt,launcher_return_code=process.returncode)
            complete=bool(receipt and receipt['status'] in ('expert_flight_finished','learned_flight_finished')
                and not receipt.get('broker_errors') and (not args.controller_checkpoints or
                receipt.get('learned_controller',{}).get('status')=='completed'))
            if complete:
                # Collision and navigation timeouts are complete outcomes;
                # never retry them to manufacture a successful episode.
                flights.append(item);(output/'flights.json').write_text(json.dumps(flights,indent=2));break
            launch_receipt=launches[0]/'result.json'
            item['launcher_result']=json.loads(launch_receipt.read_text()) if launch_receipt.exists() else None
            retained_failures.append(item)
            (output/'infrastructure-failures.json').write_text(json.dumps(retained_failures,indent=2))
        else:
            (output/'result.json').write_text(json.dumps(dict(status='failed',accepted=False,
                reason='Two infrastructure attempts failed; all receipts retained',episode_id=identifier,
                complete_flights=len(flights)),indent=2))
            raise RuntimeError('Physical collection infrastructure failed twice; outcomes retained')
    result=dict(status='completed',accepted=False,complete_flights=len(flights),
        successes=sum(bool(f['result'].get('success')) for f in flights),all_failures_retained=True,
        demonstration_batch=args.demonstration_batch,development_batch=args.development_batch,
        any_movement=any(f['result'].get('actual_path_length_m',0)>.5 for f in flights),
        controller_frozen_during_collection=True,flights=str(output/'flights.json'))
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
