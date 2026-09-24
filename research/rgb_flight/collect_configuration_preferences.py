"""Matched complete-flight configuration outcomes under a frozen controller.

Failed flights are censored for completion time. Two timeouts therefore tie;
millisecond scheduler jitter must never manufacture a preference winner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoints','qwen'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--goal',type=Path);parser.add_argument('--goal-collection',type=Path)
    parser.add_argument('--episode-id',required=True)
    parser.add_argument('--max-pairs',type=int,default=20)
    parser.add_argument('--pair-index',type=int,default=-1)
    args=parser.parse_args()
    if not 1<=args.max_pairs<=20:parser.error('One to twenty matched pairs')
    if args.goal is None and args.goal_collection:
        flights=json.loads(args.goal_collection.read_text())
        flights=[r for r in flights if r['result']['episode_id']==args.episode_id]
        if len(flights)!=1:raise ValueError('Matched goal requires one reference attempt')
        args.goal=Path(flights[0]['episode_path'])/'goal'
    if args.goal is None:parser.error('Exact four-view goal reference required')
    job=Path(os.environ['RGB_JOB_DIR'])
    output=job/('configuration-preferences' if args.pair_index<0 else f'configuration-pair-{args.pair_index:03d}')
    output.mkdir()
    if args.pair_index<0:
        pairs=[]
        for index in range(args.max_pairs):
            subprocess.run([sys.executable,str(Path(__file__)), '--checkpoints',str(args.checkpoints),
                '--qwen',str(args.qwen),'--goal',str(args.goal),'--episode-id',args.episode_id,
                '--max-pairs','1','--pair-index',str(index)],check=True)
            pair=json.loads((job/f'configuration-pair-{index:03d}'/'result.json').read_text());pairs.append(pair)
            (output/'pairs.json').write_text(json.dumps(pairs,indent=2))
            if pair['preference_update_ready'] or pair.get('control_exposure_blocked'):break
        result=dict(pairs[-1],pairs=pairs,matched_pairs=len(pairs),ties=sum(p['ambiguous'] for p in pairs),
            preference_status='ready' if pairs[-1]['preference_update_ready'] else 'blocked')
        (output/'outcomes.json').write_text(json.dumps(result['outcomes'],indent=2))
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    manifest=json.loads((args.checkpoints/'checkpoints.json').read_text())
    controller={role:item['sha256'] for role,item in manifest['artifacts'].items() if role!='configuration'}
    controller['qwen']=digest(args.qwen)
    controller_hash=hashlib.sha256(json.dumps(controller,sort_keys=True).encode()).hexdigest()
    configurations=[dict(target_id=None,grounded_kind='none',intention='search',goal_weight=1.,time_weight=1.,
        information_weight=1.,additional_caution=caution,confidence=0.,validity_horizon_seconds=4.,deliberate_immediately=True)
        for caution in (1.,2.+(args.pair_index%5)*.4)]
    conditions=dict(episode_id=args.episode_id,goal_panorama_sha256=json.loads((args.goal/'goal.json').read_text())['panorama_sha256'],
        clock_speed=1.,maximum_speed_mps=3.,rollout_seed=0,scene='env_airsim_16',controller_sha256=controller_hash)
    conditions['target_rule']='same deterministic causal frontier teacher, refreshed against currently observed IDs; compare bounded scoring multipliers'
    outcomes=[]
    for index,configuration in enumerate(configurations):
        config=output/f'configuration-{index}.json';config.write_text(json.dumps(configuration,indent=2))
        pack=output/f'checkpoints-{index}'
        command=[sys.executable,str(Path(__file__).with_name('package_navigation.py'))]
        for role in ('goal','odometry','world','policy','projection','safety'):
            command+=['--'+role,str(args.checkpoints/manifest['artifacts'][role]['path'])]
        if 'vision' in manifest['artifacts']:
            command+=['--vision',str(args.checkpoints/manifest['artifacts']['vision']['path'])]
        command+=['--qwen',str(args.qwen),'--configuration',str(config),'--goal-match-threshold',str(manifest['goal_match_threshold']),
                  '--output',str(pack)]
        subprocess.run(command,check=True)
        name=f'matched-configuration-{args.pair_index}-{index}'
        subprocess.run([sys.executable,str(Path(__file__).with_name('collect_learning_round.py')),
            '--controller-checkpoints',str(pack),'--episode-id',args.episode_id,'--goal',str(args.goal),
            '--with-deliberation','--collection-name',name],check=True)
        flight=json.loads((job/name/'flights.json').read_text())[0]
        result=flight['result'];episode=Path(flight['episode_path'])
        deliberation=json.loads((episode/'learned-controller/runtime/deliberation/result.json').read_text())
        if deliberation['status']!='completed' or deliberation['actual_planner_calls']<1:
            raise RuntimeError('Matched flight did not execute the trained predictive/configuration path')
        if result['runtime_goal_sha256']!=conditions['goal_panorama_sha256']:raise ValueError('Matched exact goal pixels changed')
        proposals=[json.loads(line) for line in (episode/'learned-controller/runtime/proposals.jsonl').read_text().splitlines()]
        exposed=sum(p['mode']=='predictive_then_independent_safety' and not p['safety']['overridden'] and p.get('broker_acceptance',{}).get('accepted',False) for p in proposals)
        outcomes.append(dict(applied_predictive_commands=exposed,complete_flight=True,episode_path=str(episode),result_sha256=digest(episode/'result.json'),
            configuration=configuration,collision=bool(result.get('airsim_collision') or result.get('geometry_collision')),
            success=bool(result['success']),elapsed_sim_seconds=result['elapsed_sim_seconds'],termination=result['termination']))
        (output/'outcomes.json').write_text(json.dumps(outcomes,indent=2))
    a,b=outcomes
    rank=lambda outcome:(not outcome['collision'],outcome['success'])
    winner=None
    if rank(a)!=rank(b):winner=0 if rank(a)>rank(b) else 1
    elif a['success'] and abs(a['elapsed_sim_seconds']-b['elapsed_sim_seconds'])>1.:
        winner=0 if a['elapsed_sim_seconds']<b['elapsed_sim_seconds'] else 1
    no_exposure=not any(x['applied_predictive_commands'] for x in outcomes)
    if no_exposure:winner=None
    result=dict(status='completed',accepted=False,control_exposure_blocked=no_exposure,matched_conditions=conditions,
        matched_conditions_sha256=hashlib.sha256(json.dumps(conditions,sort_keys=True).encode()).hexdigest(),
        controller_sha256=controller_hash,outcomes=outcomes,winner=winner,ambiguous=winner is None,
        ranking='collision avoidance, then success, then completion time among successes; <=1 s is a timing tie',
        preference_update_ready=winner is not None,
        reason=('Neither configuration affected dispatched controls; preference learning blocked' if no_exposure else 'Matched outcome tie; no preferred configuration label exists') if winner is None else None)
    (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
