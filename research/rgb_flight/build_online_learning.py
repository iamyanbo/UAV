"""Build DAgger/PPO batches from preserved live-controller physics attempts."""
import argparse
import json
import math
from pathlib import Path
import shutil

import numpy as np
import torch
from action_intervals import executed_slots
from build_policy_sequences import build as build_policy
from dagger import aggregate
from episode_store import verified_rgb_storage
from navigation_state import CheckpointSet,checksum
from trajectory_bundle import aligned_labels,read_rows


def live_attempt(collection,pack):
    flights=json.loads(collection.read_text())
    if len(flights)!=1:raise ValueError('One fixed-policy physical attempt per integration batch')
    episode=Path(flights[0]['episode_path']);result=json.loads((episode/'result.json').read_text())
    if result.get('split')!='train':raise ValueError('Sealed validation/test flights cannot enter online learning')
    if result.get('controller_kind')!='learned_mode_1' or result.get('status')!='learned_flight_finished':
        raise ValueError('A completed physical learner rollout is required')
    if result.get('controller_checkpoint_sha256')!=pack.identity:raise ValueError('Collected checkpoint identity differs')
    runtime=episode/'learned-controller/runtime';receipt=json.loads((runtime/'result.json').read_text())
    if receipt['status']!='completed':raise ValueError('Live inference infrastructure did not complete')
    rows=[]
    for item in receipt['shards']:
        path=runtime/item['path']
        if checksum(path)!=item['sha256']:raise ValueError('Changed learner-state shard')
        rows.extend(torch.load(path,weights_only=True)['samples'])
    if not rows:raise ValueError('No physical learner states')
    return episode,result,runtime,receipt,rows


def dagger_data(args,pack,episode,result,runtime,receipt,rows):
    args.output.mkdir(parents=True,exist_ok=False)
    aggregate([episode.parent],args.output/'aggregation.json')
    corrections={r['frame_id']:r for r in read_rows(episode/'training_labels/dagger.jsonl')}
    public=[]
    for row in rows:
        if not row['metric_geometry_available']:
            correction=corrections.get(row['frame_id'])
            if not correction or correction.get('teacher')!='observed-depth-motion-demonstrations/v6' or not correction['expert_observation_conditioned']:
                raise ValueError('Missing audited correction at a learner-visited recovery state')
        public.append(dict(row['runtime'],frame_id=row['frame_id'],sim_ns=row['sim_ns'],
            latest_observation_ns=row['latest_observation_ns'],visual_available=row['visual_available'],
            metric_geometry_available=row['metric_geometry_available'],
            **{k:row[k] for k in ('map_status','initialization_elapsed_seconds','input_validity','goal_probability','goal_match_threshold','target_available')}))
    replay=args.output/'causal-live';replay.mkdir()
    shard=replay/'runtime.pt';torch.save(dict(samples=public),shard)
    shutil.copyfile(runtime/'goal-tokens.pt',replay/'goal-tokens.pt')
    manifest=dict(checkpoint_set_sha256=pack.identity,episode_id=result['episode_id'],
        source_rgb_sha256=verified_rgb_storage(episode/'observations')['stream_sha256'],
        shards=[dict(path=shard.name,sha256=checksum(shard))],
        goal_tokens=dict(path='goal-tokens.pt',sha256=checksum(replay/'goal-tokens.pt')),
        slow_feature_mode='actual online publication at physical learner-visited states')
    (replay/'manifest.json').write_text(json.dumps(manifest,indent=2))
    dataset=args.output/'dataset'
    build_policy(args.dataset,replay,args.navigation_pack,args.world_checkpoint,dataset)
    current=json.loads((dataset/'manifest.json').read_text())
    old=json.loads((args.base_policy_data/'manifest.json').read_text())
    if old['perception_artifacts_sha256']!=current['perception_artifacts_sha256']:raise ValueError('DAgger perception differs')
    for i,item in enumerate(old['windows']):
        origin=args.base_policy_data/item['path']
        if checksum(origin)!=item['sha256']:raise ValueError('Changed prior DAgger sequence')
        destination=dataset/'windows'/f'aggregate-{i:06d}.pt';shutil.copyfile(origin,destination)
        current['windows'].append(dict(item,path=str(destination.relative_to(dataset))))
    episodes={e['episode_id']:e for e in old['episodes']}
    for e in current['episodes']:
        if e['episode_id'] in episodes and episodes[e['episode_id']]['split']!=e['split']:raise ValueError('DAgger split changed')
        episodes[e['episode_id']]=e
    current['episodes']=list(episodes.values())
    current['attempts']=list({a['attempt_id']:a for a in old.get('attempts',[])+current.get('attempts',[])}.values())
    current['dagger']=dict(actual_learner_states=True,corrections=len(corrections),
        aggregation_sha256=checksum(args.output/'aggregation.json'),prior_dataset_sha256=checksum(args.base_policy_data/'manifest.json'))
    (dataset/'manifest.json').write_text(json.dumps(current,indent=2))
    shutil.copyfile(pack.paths['policy'],dataset/'initial-policy.pt')
    value=dict(status='completed',accepted=False,learner_states=len(rows),corrections=len(corrections),
        aggregated_windows=len(current['windows']),dataset=str(dataset),scope='Recovery DAgger; no privileged hidden-goal direction labels')
    (args.output/'result.json').write_text(json.dumps(value,indent=2));return value


def ppo_data(args,pack,episode,result,runtime,receipt,rows):
    if not receipt['sampled_policy'] or any(not r['proposal']['stochastic'] for r in rows):
        raise ValueError('PPO requires sampled behavior proposals with stored probabilities')
    args.output.mkdir(parents=True,exist_ok=False);(args.output/'shards').mkdir()
    behavior=pack.spec['artifacts']['policy']['sha256']
    if any(r['proposal']['behavior_policy_sha256']!=behavior for r in rows):raise ValueError('Behavior changed during collection')
    labels=read_rows(episode/'training_labels/frames.jsonl')
    aligned=aligned_labels([dict(frame_id=r['frame_id'],sim_ns=r['sim_ns']) for r in rows],labels)
    evaluator=json.loads((episode/'evaluator_labels/episode.json').read_text())
    commands=read_rows(episode/'training_labels/commands.jsonl')
    goals=torch.load(runtime/'goal-tokens.pt',weights_only=True)['tokens']
    transition=[]
    for index,(row,next_row) in enumerate(zip(rows,rows[1:])):
        dt=(next_row['sim_ns']-row['sim_ns'])/1e9
        # Gaps that reset the live policy's recurrence start a new segment.
        valid=0<dt<=.25 and aligned[index]['motion_valid'] and aligned[index+1]['motion_valid']
        dispatched=[r for r in commands if row['sim_ns']<=r.get('dispatch_sim_ns',r['sim_ns'])<next_row['sim_ns']]
        past=[r for r in commands if r.get('dispatch_sim_ns',r['sim_ns'])<=row['sim_ns']]
        valid=valid and bool(past) and commands[-1]['sim_ns']>=next_row['sim_ns']
        if past:valid=valid and row['sim_ns']-past[-1]['sim_ns']<=250000000
        executed=np.zeros(4);cursor=row['sim_ns'];current=past[-1]['values'] if past else [0.]*4
        for item in dispatched:
            executed+=(item['sim_ns']-cursor)*np.asarray(current);cursor=item['sim_ns'];current=item['values']
        executed+=(next_row['sim_ns']-cursor)*np.asarray(current)
        executed=executed/max(1,next_row['sim_ns']-row['sim_ns'])
        if valid:
            d0=math.dist(aligned[index]['true_position_ned_m'],evaluator['goal_ned_m'])
            d1=math.dist(aligned[index+1]['true_position_ned_m'],evaluator['goal_ned_m'])
            reward=(d0-d1)-10*dt/180
            interval=[r for r in labels if row['sim_ns']<r['state_sim_ns']<=next_row['sim_ns']]
            collision=any(r.get('airsim_collision') or r.get('geometry_collision') for r in interval)
        else:reward=0.;collision=False
        transition.append(dict(valid=valid,dt=dt,reward=reward,collision=collision,
            executed=executed.tolist()))
    # Each learning suffix is unique. Its preceding 20 states are context only.
    shards=[];used=set();skipped=0
    for start in range(0,len(transition)-20,40):
        stop=min(start+60,len(transition));selected=rows[start:stop];events=transition[start:stop]
        if len(selected)<=20:continue
        # A recurrent window cannot traverse a live hidden-state reset.
        if any(e['dt']>.25 or e['dt']<=0 for e in events):skipped+=1;continue
        valid=torch.tensor([[False]*20+[e['valid'] for e in events[20:]]])
        if not bool(valid.any()):continue
        features={key:torch.stack([r['runtime'][key] for r in selected])[None] for key in
            ('state','task','previous_command','target_context','current_tokens','goal_context','depth_tokens')}
        context=[]
        for row in selected:
            mask=row['runtime']['memory_valid'][:,None]
            context.append((row['runtime']['memory'][:,:256]*mask).sum(0)/mask.sum().clamp_min(1))
        features.update(memory_context=torch.stack(context)[None],goal_tokens=goals[None])
        proposal=[r['proposal'] for r in selected]
        next_proposal=[r['proposal'] for r in rows[start+1:stop+1]]
        ids=[episode.parent.name+':'+str(selected[i]['frame_id']) for i in range(20,len(selected)) if events[i]['valid']]
        if used.intersection(ids):raise ValueError('Repeated physical learning transition')
        used.update(ids)
        tensor=lambda values:torch.tensor([values],dtype=torch.float32)
        truncated=torch.zeros_like(valid);truncated[:,-1]=True
        data=dict(runtime=features,initial_hidden=selected[0]['initial_hidden'][None],valid=valid,
            behavior_policy_sha256=behavior,transition_ids=ids,
            latent_action=tensor([p['latent_action'] for p in proposal]),sampled_stop=tensor([p['sampled_stop'] for p in proposal]),
            old_logprob=tensor([p['proposal_log_probability'] for p in proposal]),
            executed_command=tensor([e['executed'] for e in events]),delta_seconds=tensor([e['dt'] for e in events]),
            reward=tensor([e['reward'] for e in events]),collision_cost=tensor([e['collision'] for e in events]),
            value=tensor([p['reward_value'] for p in proposal]),next_value=tensor([p['reward_value'] for p in next_proposal]),
            cost_value=tensor([p['collision_cost_value'] for p in proposal]),next_cost_value=tensor([p['collision_cost_value'] for p in next_proposal]),
            terminated=torch.zeros_like(valid),truncated=truncated)
        path=args.output/'shards'/f'ppo-{len(shards):06d}.pt';torch.save(data,path)
        shards.append(dict(path=str(path.relative_to(args.output)),sha256=checksum(path)))
    if not shards:raise ValueError('No valid fresh recurrent physics transitions')
    for role,name in [('policy','initial-policy.pt'),('goal','goal.pt')]:shutil.copyfile(pack.paths[role],args.output/name)
    termination={'boundary_exit':'out_of_bounds','tracking_loss':'failure'}.get(result['termination'],result['termination'])
    manifest=dict(actual_continuous_physics=True,all_failures_retained=True,
        reward_privileged_distance_progress_training_only=True,behavior_policy_sha256=behavior,
        behavior_frozen_during_collection=True,safety_filter_active=True,slow_planner_disabled=True,
        initial_policy_checkpoint='initial-policy.pt',goal_encoder_checkpoint_sha256=pack.spec['artifacts']['goal']['sha256'],
        perception_artifacts_sha256={role:pack.spec['artifacts'][role]['sha256'] for role in ('goal','odometry','projection','vision') if role in pack.spec['artifacts']},
        completed_episodes=[dict(attempt_id=episode.parent.name,termination=termination,
            collision=bool(result.get('airsim_collision') or result.get('geometry_collision')))],shards=shards,
        collected_proposals=len(rows),unique_learning_transitions=len(used),skipped_gap_windows=skipped,
        reward_definition='metric distance progress minus elapsed time cost 10/180 per second; only observed endpoints; incomplete tails bootstrap',
        termination_semantics='shards end at observed RGB states before evaluator termination; collection-truncated tails bootstrap; completed episode collision uses separate evaluator receipt',
        source_outcome_sha256=checksum(episode/'result.json'),source_collection_sha256=checksum(args.collection))
    manifest['source_hashes']={str(path.relative_to(episode)):checksum(path) for path in
        [episode/'training_labels/frames.jsonl',episode/'training_labels/commands.jsonl',
         episode/'training_labels/constant_commands.json',runtime/'result.json',episode/'observations/frames.jsonl']}
    (args.output/'ppo.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='completed',accepted=False,unique_physics_transitions=len(used),shards=len(shards),collected_proposals=len(rows))
    (args.output/'result.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('dagger','ppo'),required=True)
    for name in ('collection','dataset','navigation-pack','world-checkpoint','base-policy-data','output'):
        parser.add_argument('--'+name,type=Path,required=name in ('collection','dataset','navigation-pack','output'))
    args=parser.parse_args();torch.set_num_threads(4)
    pack=CheckpointSet(args.navigation_pack/'checkpoints.json',integration_only=True)
    data=live_attempt(args.collection,pack)
    print(json.dumps((dagger_data if args.phase=='dagger' else ppo_data)(args,pack,*data)))
