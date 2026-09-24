"""Join causal replay with separately stored labels on the world-model grid.

Missing dispatch coverage is retained as an invalid action mask. Application
delay remains part of the dynamics; no application timestamp is fabricated.
"""
import argparse
import bisect
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from action_intervals import executed_slots
from navigation_state import CheckpointSet,checksum
from trajectory_bundle import aligned_labels,read_rows

DT=200000000


def checked_json(root,reference):
    path=(root/reference['path']).resolve()
    if not path.is_relative_to(root.resolve()) or checksum(path)!=reference['sha256']:
        raise ValueError('Changed or escaping trajectory reference')
    return json.loads(path.read_text())


def build(bundle,checkpoints,replays,output):
    source=json.loads((bundle/'manifest.json').read_text())
    pack=CheckpointSet(checkpoints,integration_only=True)
    output.mkdir(parents=True,exist_ok=False);(output/'windows').mkdir()
    windows=[];episodes={};normalization=[];motion_samples=[];features=[];provenance=[]
    for directory in replays:
        replay=json.loads((directory/'manifest.json').read_text())
        if replay['checkpoint_set_sha256']!=pack.identity:
            perception={k:pack.spec['artifacts'][k]['sha256'] for k in ('goal','odometry','projection','vision') if k in pack.spec['artifacts']}
            if replay.get('perception_artifacts_sha256')!=perception:
                raise ValueError('Mixed perception checkpoints require separate causal data rounds')
        if not (replay['slow_feature_mode'].startswith('measured paced replay') or replay['slow_feature_mode']=='actual online publication at physical learner-visited states'):
            raise ValueError('World inputs require measured asynchronous publication, not retrospective instant features')
        matches=[]
        for attempt in source['attempts']:
            if attempt['episode_id']!=replay['episode_id']:continue
            runtime=checked_json(bundle,attempt['runtime'])
            if runtime['rgb_stream_sha256']==replay['source_rgb_sha256']:matches.append((attempt,runtime))
        if len(matches)!=1:raise ValueError('Replay must identify exactly one original physical attempt')
        attempt,record=matches[0];identifier=attempt['episode_id'];physical=Path(attempt['episode_path'])
        label_record=checked_json(bundle,attempt['training_labels'])
        evaluator=json.loads(Path(label_record['evaluator_path']).read_text())
        label_samples=read_rows(physical/'training_labels/frames.jsonl')
        traces=read_rows(directory/'timing.jsonl');timing={r['frame_id']:r for r in traces}
        samples=[]
        for ref in replay['shards']:
            path=directory/ref['path']
            if checksum(path)!=ref['sha256']:raise ValueError('Changed runtime shard')
            value=torch.load(path,map_location='cpu',weights_only=True)
            if value['episode_id']!=identifier:raise ValueError('Cross-episode runtime shard')
            samples.extend(value['samples'])
        if not samples:raise ValueError('Empty runtime replay')
        if 'goal_tokens' in replay:
            ref=replay['goal_tokens'];path=directory/ref['path']
            if checksum(path)!=ref['sha256']:raise ValueError('Changed goal-token cache')
            goal_tokens=torch.load(path,weights_only=True)['tokens']
        else:goal_tokens=samples[0]['goal_tokens']
        if goal_tokens.shape!=(4,300,256):raise ValueError('Four complete spatial goal grids required')
        anchor_wall=traces[0]['available_monotonic'];anchor_ns=traces[0]['sim_ns']
        availability=[timing[r['frame_id']]['available_monotonic']+timing[r['frame_id']]['processing_seconds'] for r in samples]
        if any(a>b for a,b in zip(availability,availability[1:])):raise ValueError('Noncausal runtime publication order')
        begin=((samples[0]['sim_ns']+DT-1)//DT)*DT;end=samples[-1]['sim_ns']
        ticks=list(range(begin,end+1,DT));selected=[]
        for tick in ticks:
            index=bisect.bisect_right(availability,anchor_wall+(tick-anchor_ns)/1e9)-1
            while index>=0 and samples[index]['sim_ns']>tick:index-=1
            selected.append(index)
        labels=aligned_labels([dict(frame_id=i,sim_ns=t) for i,t in enumerate(ticks)],label_samples)
        episodes[identifier]=dict(episode_id=identifier,split=attempt['split'],goal_region_id=attempt['goal_region_id'],
            start_goal_pair_id=evaluator.get('start_goal_pair_id',identifier),
            simulated_seconds=(end-samples[0]['sim_ns'])/1e9)
        provenance.append(dict(attempt_id=attempt['attempt_id'],source_checkpoint_set_sha256=replay['checkpoint_set_sha256'],replay_manifest_sha256=checksum(directory/'manifest.json'),
                               result_sha256=attempt['result_sha256']))
        for start in range(0,max(0,len(ticks)-30),10):
            indices=selected[start:start+31]
            if len(indices)!=31 or any(i<0 for i in indices):continue
            rows=[samples[i] for i in indices];grid=ticks[start:start+31]
            values={name:torch.stack([r[name] for r in rows]) for name in
                    ('z','state','memory','memory_valid','task','target_context','previous_command')}
            values.update(goal_tokens=goal_tokens,sim_ns=torch.tensor(grid,dtype=torch.int64),
                latest_observation_ns=torch.tensor([r['latest_observation_ns'] for r in rows],dtype=torch.int64),
                visual_available=torch.tensor([r['visual_available'] for r in rows]))
            targets={name:[] for name in ('motion','collision','visibility','information','goal_match','time_to_goal',
                'z_valid','state_valid','motion_valid','collision_valid','visibility_valid','information_valid',
                'goal_match_valid','time_to_goal_valid','time_censored','action_interval_valid')}
            actions=[]
            for index,tick in enumerate(grid):
                slots=executed_slots(record['commands'],tick,record.get('constant_command_epochs',[]))
                actions.append(slots['values'] if slots['values'] is not None else [[0.]*4 for _ in range(4)])
                targets['action_interval_valid'].append(all(slots['valid']))
                current=labels[start+index];future=labels[min(start+index+1,len(labels)-1)]
                motion_valid=index<30 and current['motion_valid'] and future['motion_valid']
                motion=np.zeros(6)
                if motion_valid:
                    q0,q1=[Rotation.from_quat(r['true_quaternion_xyzw']) for r in (current,future)]
                    motion=np.r_[q0.inv().apply(np.asarray(future['true_position_ned_m'])-current['true_position_ned_m']),
                                 (q0.inv()*q1).as_rotvec()]
                targets['motion'].append(motion.tolist());targets['motion_valid'].append(motion_valid)
                interval=[r for r in label_samples if tick<int(r.get('state_sim_ns',round(r['sim_seconds']*1e9)))<=tick+DT]
                targets['collision'].append(float(any(r.get('airsim_collision') or r.get('geometry_collision') for r in interval)))
                targets['collision_valid'].append(bool(interval) and index<30)
                match_valid=future['motion_valid'] and index<30
                near=False
                if match_valid:
                    difference=np.asarray(future['true_position_ned_m'])-evaluator['goal_ned_m']
                    near=np.linalg.norm(difference[:2])<=3 and abs(difference[2])<=2
                targets['goal_match'].append(float(near));targets['goal_match_valid'].append(match_valid)
                # Exact completion is observed only on successful demonstrations.
                remaining=max(0.,(record['frames'][-1]['sim_ns']-tick-DT)/1e9)
                targets['time_to_goal'].append(remaining);targets['time_to_goal_valid'].append(index<30)
                targets['time_censored'].append(not label_record['success'])
                for name in ('visibility','information'):
                    targets[name].append(0.);targets[name+'_valid'].append(False)
                next_row=rows[min(index+1,30)]
                targets['z_valid'].append(index<30 and next_row['visual_available'] and next_row['latest_observation_ns']>rows[index]['latest_observation_ns'])
                targets['state_valid'].append(index<30 and next_row['tracking_confidence']>0 and next_row['frame_id']!=rows[index]['frame_id'])
            values['action']=torch.tensor(actions,dtype=torch.float32)
            supervision={k:torch.tensor(v,dtype=torch.bool if k.endswith('_valid') or k=='time_censored' else torch.float32) for k,v in targets.items()}
            path=output/'windows'/f'world-{len(windows):06d}.pt'
            torch.save(dict(episode_id=identifier,attempt_id=attempt['attempt_id'],runtime=values,training_labels=supervision,
                source_frame_ids=[r['frame_id'] for r in rows],
                action_missing_value='zeros are masked placeholders; never accepted as executed commands'),path)
            windows.append(dict(module='world',episode_id=identifier,path=str(path.relative_to(output)),sha256=checksum(path),
                                action_timing_valid=bool(supervision['action_interval_valid'][:30].all())))
            if attempt['split']=='train':
                normalization.append(values['state']);motion_samples.append(supervision['motion'][supervision['motion_valid']])
                features.append(values['z'][values['visual_available']].reshape(-1,256))
    normalizer=None
    if normalization and any(len(x) for x in motion_samples) and any(len(x) for x in features):
        state=torch.cat(normalization);motion=torch.cat(motion_samples);feature=torch.cat(features)
        path=output/'normalization.pt'
        torch.save(dict(fit_split='train',episode_ids=[k for k,v in episodes.items() if v['split']=='train'],
            state_mean=state.mean(0),state_scale=state.std(0).clamp_min(.01),
            motion_scale=motion.std(0).clamp_min(.001),feature_scale=feature.std(0).clamp_min(.01)),path)
        normalizer=dict(path=path.name,sha256=checksum(path))
    reasons=[]
    if not windows or not any(w['action_timing_valid'] for w in windows):reasons.append('dispatch_coverage_missing')
    if not any(v['split']=='validation' for v in episodes.values()):reasons.append('development_replay_missing')
    if normalizer is None:reasons.append('insufficient_training_normalization')
    result=dict(status='completed',accepted=False,candidate_windows=len(windows),
        timing_valid_windows=sum(w['action_timing_valid'] for w in windows),training_ready=not reasons,blocking_reasons=reasons)
    manifest=dict(schema='world-sequence-views/v2',action_semantics='post-safety-dispatch/50ms-v3',belief_version='shared-droid-local-depth/v4',foundation=dict(accepted=False,training_ready=not reasons,
        valid_expert_episodes=source.get('unique_successful_expert_episodes',source['unique_successful_episodes']),visual_goal_runtime=True),
        perception_artifacts_sha256={role:pack.spec['artifacts'][role]['sha256'] for role in ('goal','odometry','projection','vision') if role in pack.spec['artifacts']},
        goal_encoder_checkpoint_sha256=pack.spec['artifacts']['goal']['sha256'],
        projection_sha256=pack.spec['artifacts'].get('projection',{}).get('sha256'),
        navigation_checkpoint_set_sha256=pack.identity,source_bundle_sha256=checksum(bundle/'manifest.json'),
        episodes=list(episodes.values()),windows=windows,normalization=normalizer,provenance=provenance,
        readiness=result,clock_alignment='ClockSpeed 1 paced-replay wall intervals anchored at first RGB availability; no original-flight feature availability claimed')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));(output/'result.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--checkpoints',type=Path,required=True);parser.add_argument('--replay',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(build(args.bundle,args.checkpoints,args.replay,args.output)))
