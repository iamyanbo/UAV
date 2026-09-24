"""Evidence-driven expert/DAgger/constrained-PPO curriculum scheduler.

This reads completed physical episode receipts.  It does not fabricate rollouts
or mark a stage complete from optimizer iteration counts alone.
"""
import argparse
import hashlib
import json
from pathlib import Path


MILESTONES=(250,1000,2500,5000,10000)
SPEEDS=(3.,4.5,6.)
BUDGETS=dict(world_updates=300_000,imitation_updates=200_000,ppo_physics_transitions=10_000_000,
             qwen_configuration_examples=25_000,matched_configuration_pairs=2_000)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_receipts(root):
    rows=[]
    seen=set()
    for path in sorted(Path(root).rglob('episode/result.json')):
        result=json.loads(path.read_text())
        if result.get('status')=='expert_flight_finished' and result.get('success'):
            labels=path.parent/'training_labels/frames.jsonl'
            commands=path.parent/'training_labels/commands.jsonl'
            goal=path.parent/'goal/goal.json'
            observations=path.parent/'observations/storage.json'
            if (labels.exists() and commands.exists() and goal.exists() and observations.exists()
                    and json.loads(observations.read_text()).get('complete')):
                if result['episode_id'] in seen:
                    continue
                seen.add(result['episode_id'])
                rows.append(dict(episode_id=result['episode_id'],result=str(path),result_sha256=digest(path),
                                 labels_sha256=digest(labels),commands_sha256=digest(commands),
                                 goal_sha256=digest(goal),observations_sha256=digest(observations),
                                 speed_mps=result['maximum_speed_mps']))
    return rows


def validation_gate(paths, speed):
    results=[json.loads(Path(path).read_text()) for path in paths]
    results=[row for row in results if row.get('maximum_speed_mps')==speed and row.get('split')=='validation']
    if any(row.get('clock_speed')!=1.0 or not row.get('controller_checkpoint_sha256') for row in results):
        return dict(passed=False,reason='missing_clock_or_frozen_controller_provenance',episodes=len(results))
    if len({row.get('controller_checkpoint_sha256') for row in results})>1:
        return dict(passed=False,reason='mixed_controller_checkpoints',episodes=len(results))
    if len({row.get('episode_id') for row in results})!=len(results):
        return dict(passed=False,reason='repeated_validation_episode_ids',episodes=len(results))
    if len(results)<150:
        return dict(passed=False,reason='fewer_than_150_validation_episodes',episodes=len(results))
    collisions=sum(row.get('termination') in ('collision','geometry_collision') for row in results)
    success=sum(bool(row.get('success')) and row.get('termination')=='success' for row in results)
    success_rate=success/len(results); collision_rate=collisions/len(results)
    return dict(passed=success_rate>=.9 and collision_rate<=.01,episodes=len(results),
                collision_free_success=success_rate,collision_rate=collision_rate)


def state(collection_root, validation_results=(), training_manifest=None):
    episodes=episode_receipts(collection_root)
    count=len(episodes)
    reached=[value for value in MILESTONES if count>=value]
    readiness=json.loads(Path(training_manifest).read_text()) if training_manifest else {}
    start_training=bool(readiness.get('readiness',{}).get('training_ready') and
        {'train','validation'}<={row['split'] for row in readiness.get('episodes',[])})
    gates={str(speed):validation_gate(validation_results,speed) for speed in SPEEDS}
    active_speed=3.
    if gates['3.0']['passed']:
        active_speed=4.5
    if gates['3.0']['passed'] and gates['4.5']['passed']:
        active_speed=6.
    return dict(schema='visual-goal-training-state/v1',valid_expert_episodes=count,
        training_authorized=start_training,training_ready=start_training,deployment_accepted=False,next_collection_milestone=next((x for x in MILESTONES if x>count),None),
        reached_milestones=reached,programme=['fast_visual_odometry','goal_matcher','jepa_world_model',
          'critic_and_mode1_imitation','qwen_lora','dagger','constrained_ppo','qwen_outcome_preferences'],
        budgets=BUDGETS,speed_gates=gates,active_speed_mps=active_speed,
        collection_and_training_overlap=start_training and count<10000,
        lexicographic_objective=['avoid_collision','reach_and_stop','minimize_simulated_completion_time'],
        episodes=episodes)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',type=Path,help='Dependency specification for a remote execution round')
    parser.add_argument('--round',type=Path)
    parser.add_argument('--hours',type=float,default=8)
    parser.add_argument('--collection-root',type=Path)
    parser.add_argument('--training-manifest',type=Path)
    parser.add_argument('--validation-result',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.execute:
        if not args.round or not 0<args.hours<=8:parser.error('--round and a window in (0,8] hours required')
        from program_scheduler import run
        raise SystemExit(run(args.execute,args.round,args.hours))
    if not args.collection_root or not args.output:parser.error('Report requires --collection-root and --output')
    value=state(args.collection_root,args.validation_result,args.training_manifest)
    temporary=args.output.with_suffix('.pending'); temporary.write_text(json.dumps(value,indent=2)); temporary.replace(args.output)
    print(json.dumps({k:value[k] for k in ('valid_expert_episodes','training_authorized','next_collection_milestone','active_speed_mps')}))


if __name__=='__main__':
    main()
