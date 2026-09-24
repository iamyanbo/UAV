"""Causal world-belief replay and observation-conditioned recovery imitation.

The bootstrap teacher supplies bounded translating startup demonstrations.
Search targets use observed evidence, without privileged destination direction.
DAgger rounds use the same teacher at states actually visited by the learner.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from action_intervals import executed_slots
from startup import demonstration
from build_world_sequences import checked_json
from learning_models import WorldModel
from navigation_state import CheckpointSet,checksum


def build(bundle,replay,checkpoints,world_checkpoint,output):
    source=json.loads((bundle/'manifest.json').read_text())
    pack=CheckpointSet(checkpoints/'checkpoints.json',integration_only=True)
    manifest=json.loads((replay/'manifest.json').read_text())
    if manifest['checkpoint_set_sha256']!=pack.identity:raise ValueError('Replay perception differs')
    matches=[]
    for attempt in source['attempts']:
        if attempt['episode_id']!=manifest['episode_id']:continue
        record=checked_json(bundle,attempt['runtime'])
        if record['rgb_stream_sha256']==manifest['source_rgb_sha256']:matches.append((attempt,record))
    if len(matches)!=1:raise ValueError('Ambiguous physical attempt')
    attempt,record=matches[0]
    labels=checked_json(bundle,attempt['training_labels'])
    correction_path=Path(attempt['episode_path'])/'training_labels/dagger.jsonl'
    corrections={r['frame_id']:r for r in (json.loads(line) for line in correction_path.read_text().splitlines())} if correction_path.exists() else {}
    # A classifier's false stop is a failed action, not an expert target. The
    # evaluator is used only here, after flight, to mask invalid supervision;
    # it supplies neither a search direction nor a runtime goal coordinate.
    evaluator=json.loads(Path(labels['evaluator_path']).read_text())
    goal=torch.tensor(evaluator['goal_ned_m'])
    aligned={r['frame_id']:r for r in labels['frames']}
    rejected_stops=set()
    for frame,teacher in corrections.items():
        if not teacher.get('explicit_stop'):continue
        label=aligned.get(frame)
        valid=bool(label and label.get('motion_valid'))
        if valid:
            delta=torch.tensor(label['true_position_ned_m'])-goal
            valid=bool(delta[:2].norm()<=3. and delta[2].abs()<=2.)
        if not valid:rejected_stops.add(frame)
    rows=[]
    for reference in manifest['shards']:
        path=replay/reference['path']
        if checksum(path)!=reference['sha256']:raise ValueError('Modified causal replay')
        rows.extend(torch.load(path,weights_only=True)['samples'])
    goal_ref=manifest['goal_tokens'];goal_path=replay/goal_ref['path']
    if checksum(goal_path)!=goal_ref['sha256']:raise ValueError('Modified goal tokens')
    goals=torch.load(goal_path,weights_only=True)['tokens'].cuda()
    saved=torch.load(world_checkpoint,map_location='cpu',weights_only=True)
    if saved.get('module')!='world' or saved.get('update',0)<1:raise ValueError('Trained world checkpoint required')
    required={role:pack.spec['artifacts'][role]['sha256'] for role in ('goal','odometry','projection','vision') if role in pack.spec['artifacts']}
    if saved.get('perception_artifacts_sha256')!=required:raise ValueError('World perception differs')
    model=WorldModel().cuda().eval().requires_grad_(False);model.load_state_dict(saved['model'])
    hidden=torch.zeros(1,256,device='cuda');last_tick=None;belief_valid=True;origin=None
    # Belief advances only over action intervals whose command is known.
    with torch.inference_mode():
        for row in rows:
            now=row['sim_ns']
            if last_tick is None:last_tick=now;origin=row
            if now-last_tick>=200000000:
                slots=executed_slots(record['commands'],last_tick,record.get('constant_command_epochs',[]))
                belief_valid=belief_valid and all(slots['valid'])
                if belief_valid:
                    prediction=model(origin['z'][None].cuda(),origin['state'][None].cuda(),hidden,
                        origin['memory'][None].cuda(),origin['memory_valid'][None].cuda(),
                        torch.tensor(slots['values'],device='cuda',dtype=torch.float32)[None],
                        origin['task'][None].cuda(),goals[None],origin['target_context'][None].cuda())
                    hidden=prediction['belief'].mean(0)
                last_tick=now;origin=row
            row['belief']=hidden[0].cpu().clone();row['belief_valid']=belief_valid
            previous=row
    output.mkdir(parents=True,exist_ok=False);(output/'windows').mkdir()
    shutil.copyfile(pack.paths['goal'],output/'goal.pt')
    end_ns=record['frames'][-1]['sim_ns'];collision=any(r.get('collision_event_observed') for r in labels['frames'])
    windows=[];primitive_values=[]
    # Include the final complete window: the last stopping/failure observation
    # must not disappear merely because its index is between stride boundaries.
    offsets=sorted(set(range(0,len(rows)-39,20)) | ({len(rows)-40} if len(rows)>=40 else set()))
    for offset in offsets:
        selected=rows[offset:offset+40]
        times=torch.tensor([r['sim_ns'] for r in selected],dtype=torch.int64)
        if not bool(((times.diff()>0)&(times.diff()<=250000000)).all()):continue
        if not all(r['belief_valid'] for r in selected):continue
        teachers=[corrections.get(r['frame_id']) for r in selected]
        recovery=torch.tensor([bool(t and t.get('teacher')=='observed-depth-exploration/v5' and t.get('expert_observation_conditioned')
            and r['frame_id'] not in rejected_stops) for r,t in zip(selected,teachers)])
        if not bool(recovery[20:].any()):continue
        runtime={key:torch.stack([r[key] for r in selected]) for key in
            ('z','state','belief','memory','memory_valid','task','target_context','previous_command','current_tokens','goal_context','depth_tokens')}
        runtime.update(goal_tokens=goals.cpu(),sim_ns=times,
            latest_observation_ns=torch.tensor([r['latest_observation_ns'] for r in selected],dtype=torch.int64),
            visual_available=torch.tensor([r['visual_available'] for r in selected]),
            maximum_speed_mps=torch.full((40,),3.))
        seconds=(end_ns-times).float()/1e9
        # These are demonstrated returns, not optimal completion times. Missing
        # clearance/information/command targets are masked, never made zero labels.
        primitive=torch.zeros(40,8);valid=torch.zeros(40,8,dtype=torch.bool)
        primitive[:,1]=float(collision);valid[:,1]=True
        primitive[:,3]=seconds;valid[:,3]=bool(labels['success'])
        primitive[:,7]=float(not labels['success']);valid[:,7]=True
        supervision=dict(expert_command=torch.tensor([t['expert_command'] if t else [0.]*4 for t in teachers]),imitation_valid=recovery,
            explicit_stop=torch.tensor([float(t.get('explicit_stop',False)) if t else 0. for t in teachers]),fixed_return=-seconds-100*float(collision)+100*float(labels['success']),
            collision_return=torch.full((40,),float(collision)),primitive_return=primitive,primitive_valid=valid)
        path=output/'windows'/f'policy-{len(windows):06d}.pt'
        torch.save(dict(episode_id=attempt['episode_id'],attempt_id=attempt['attempt_id'],runtime=runtime,
            training_labels=supervision,teacher_source='observed-depth-exploration/v5',teacher_reasons=[t.get('reason') if t else 'missing_supervision' for t in teachers]),path)
        windows.append(dict(module='policy',episode_id=attempt['episode_id'],attempt_id=attempt['attempt_id'],path=str(path.relative_to(output)),sha256=checksum(path),
                            timing_qualified=bool((times.diff()<=75000000).all())))
        if attempt['split']=='train':primitive_values.append(primitive)
    if not windows:raise ValueError('No fresh contiguous training windows with causal world beliefs')
    normalization=output/'normalization.pt'
    torch.save(dict(fit_split='train',episode_ids=[attempt['episode_id']] if primitive_values else [],
        primitive_scale=torch.cat(primitive_values).abs().mean(0).clamp_min(1.) if primitive_values else torch.ones(8)),normalization)
    evaluator=json.loads(Path(labels['evaluator_path']).read_text())
    result=dict(status='completed',accepted=False,windows=len(windows),
        timing_qualified_windows=sum(w['timing_qualified'] for w in windows),
        observation_grid='all original timestamps; no duplication; gaps <=250 ms match live recurrent reset; 75 ms acceptance unchanged',
        scope='Executed observation-conditioned exploration and recovery; deployment remains unqualified',
        training_ready=bool(windows),deployment_accepted=False)
    audited=[corrections[r['frame_id']] for r in rows if r['frame_id'] in corrections and
        corrections[r['frame_id']].get('teacher')=='observed-depth-exploration/v5' and
        corrections[r['frame_id']].get('expert_observation_conditioned')]
    from collections import Counter
    result['demonstration_coverage']=dict(unique_audited_observations=len(audited),
        rejected_false_or_unverifiable_stops=len(rejected_stops),
        explicit_stops=sum(bool(t.get('explicit_stop')) for t in audited),
        reasons=dict(Counter(t.get('reason','unspecified') for t in audited)))
    spec=dict(schema='policy-sequence-views/v1',action_semantics='post-safety-dispatch/50ms-v3',belief_version='shared-droid-local-depth/v4',teacher_version='observed-depth-exploration/v5',foundation=dict(accepted=False,training_ready=True,visual_goal_runtime=True,
        valid_expert_episodes=source.get('unique_successful_expert_episodes',0)),
        episodes=[dict(episode_id=attempt['episode_id'],split=attempt['split'],goal_region_id=attempt['goal_region_id'],
            start_goal_pair_id=evaluator.get('start_goal_pair_id',attempt['episode_id']),
            simulated_seconds=(record['frames'][-1]['sim_ns']-record['frames'][0]['sim_ns'])/1e9)],
        attempts=[dict(attempt_id=attempt['attempt_id'],split=attempt['split'],
            simulated_seconds=(record['frames'][-1]['sim_ns']-record['frames'][0]['sim_ns'])/1e9)],
        windows=windows,normalization=dict(path=normalization.name,sha256=checksum(normalization)),
        perception_artifacts_sha256=required,goal_encoder_checkpoint_sha256=required['goal'],
        projection_sha256=required['projection'],world_checkpoint_sha256=checksum(world_checkpoint),
        navigation_checkpoint_set_sha256=pack.identity,source_bundle_sha256=checksum(bundle/'manifest.json'),
        source_replay_sha256=checksum(replay/'manifest.json'),readiness=result)
    (output/'manifest.json').write_text(json.dumps(spec,indent=2));(output/'result.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('bundle','replay','checkpoints','world-checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    print(json.dumps(build(args.bundle,args.replay,args.checkpoints,args.world_checkpoint,args.output)))
