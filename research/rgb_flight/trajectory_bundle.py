"""Immutable trajectory index and timestamp-aligned, separate training labels.

No inference features are synthesized here. Slow features and memory require
an independently versioned causal replay before this bundle can train control.
"""
import argparse
import bisect
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from episode_store import verified_rgb_storage
from goal_io import load_goal


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def aligned_labels(frames, samples, max_gap_seconds=.25):
    # Old files associated a nearby state with a frame ID. The association is
    # not synchronization: interpolate from actual state timestamps instead.
    unique = {int(row.get('state_sim_ns', round(row['sim_seconds']*1e9))): row for row in samples}
    times = sorted(unique)
    output = []
    for frame in frames:
        ns = frame['sim_ns']; right = bisect.bisect_left(times, ns)
        row = dict(frame_id=frame['frame_id'], observation_sim_ns=ns, motion_valid=False)
        if right < len(times) and times[right] == ns:
            before = after = times[right]
        elif 0 < right < len(times):
            before, after = times[right-1:right+1]
        else:
            output.append(row); continue
        if (after-before)/1e9 > max_gap_seconds:
            output.append(row); continue
        a, b = unique[before], unique[after]
        fraction = (ns-before)/max(1, after-before)
        quat = (a['true_quaternion_xyzw'] if before == after else
                Slerp([0., 1.], Rotation.from_quat([a['true_quaternion_xyzw'], b['true_quaternion_xyzw']]))([fraction]).as_quat()[0].tolist())
        row.update(motion_valid=True, interpolation_source_sim_ns=[before, after],
            interpolation_span_seconds=(after-before)/1e9,
            true_position_ned_m=((1-fraction)*np.asarray(a['true_position_ned_m'])+
                                 fraction*np.asarray(b['true_position_ned_m'])).tolist(),
            true_quaternion_xyzw=quat,
            collision_event_observed=bool(a.get('airsim_collision') or b.get('airsim_collision') or
                                          a.get('geometry_collision') or b.get('geometry_collision')),
            airsim_contact_observed=bool(a.get('airsim_collision') or b.get('airsim_collision')),
            privileged_geometry_intersection=bool(a.get('geometry_collision') or b.get('geometry_collision')),
            collision_interval_sim_ns=[before, after])
        output.append(row)
    return output


def timing(frames, commands):
    dt = np.diff([row['sim_ns'] for row in frames])/1e9
    issued = np.diff([row['issued_monotonic'] for row in commands])
    def metrics(values, threshold):
        return dict(count=len(values), mean_seconds=float(np.mean(values)) if len(values) else None,
            p95_seconds=float(np.quantile(values,.95)) if len(values) else None,
            max_seconds=float(np.max(values)) if len(values) else None,
            missed_deadlines=int(np.sum(values>threshold)))
    rgb=metrics(dt,.075); control=metrics(issued,.05)
    hz=1/float(np.mean(dt)) if len(dt) and np.all(dt>0) else 0.
    return dict(fresh_rgb_hz=hz, rgb=rgb, command=control,
        accepted=bool(hz>=19 and rgb['p95_seconds']<=.075 and
                      control['p95_seconds'] is not None and control['p95_seconds']<=.05),
        application_timestamps_available=bool(commands) and all(row.get('application_sim_ns') is not None for row in commands))


def build(root, output, initialization=None):
    root=root.resolve(); output.mkdir(parents=True,exist_ok=False)
    (output/'runtime').mkdir(); (output/'training_labels').mkdir()
    attempts=[]; excluded=[]; split_groups={}
    for receipt in sorted(root.rglob('episode/result.json')):
        result=json.loads(receipt.read_text()); episode=receipt.parent
        if result.get('split')!='train':
            continue  # sealed campaign validation and test are never opened
        attempt=episode.parent.name+'-'+result['episode_id']
        try:
            storage=verified_rgb_storage(episode/'observations')
            frames=read_rows(episode/'observations/frames.jsonl')
            commands=read_rows(episode/'training_labels/commands.jsonl')
            samples=read_rows(episode/'training_labels/frames.jsonl')
            if not frames or len(frames)!=storage['frames']:
                raise ValueError('empty or incomplete recorded stream index')
            if any(b['sim_ns']<=a['sim_ns'] for a,b in zip(frames,frames[1:])):
                raise ValueError('duplicate/nonmonotonic image exposure timestamps')
            goal=load_goal(episode/'goal',result['episode_id'])
            evaluator=json.loads((episode/'evaluator_labels/episode.json').read_text())
            region=evaluator['goal_region_id']
            split='validation' if int(hashlib.sha256(region.encode()).hexdigest()[:8],16)%10==0 else 'train'
            if result['episode_id'] in split_groups and split_groups[result['episode_id']]!=split:
                raise ValueError('episode crosses internal split')
            split_groups[result['episode_id']]=split
            aligned=aligned_labels(frames,samples)
            runtime_path=output/'runtime'/f'{attempt}.json'
            label_path=output/'training_labels'/f'{attempt}.json'
            runtime=dict(episode_id=result['episode_id'],attempt_id=attempt,goal_path=str(episode/'goal'),
                goal_sha256=goal.content_sha256,rgb_stream=str(episode/'observations/rgb.zlib'),
                rgb_stream_sha256=storage['stream_sha256'],frames=frames,commands=commands,
                constant_command_epochs=json.loads((episode/'training_labels/constant_commands.json').read_text())
                    if (episode/'training_labels/constant_commands.json').exists() else [],
                causal_replay=None,provenance=dict(encoder=None,mapper=None,projection=None,normalization=None))
            runtime_path.write_text(json.dumps(runtime,allow_nan=False))
            # Remaining time is censored for failed trajectories; a failed
            # command is never promoted to an imitation target.
            expert=result.get('controller_kind','privileged_shortest_path_expert')=='privileged_shortest_path_expert'
            label_path.write_text(json.dumps(dict(frames=aligned,termination=result.get('termination'),
                success=bool(result.get('success')),remaining_time_censored=not bool(result.get('success')),
                search_imitation_valid=False,local_expert_motion_valid=expert and bool(result.get('success')),
                teacher='privileged_shortest_path_upper_bound' if expert else result.get('teacher_provenance') or 'learned_policy_rollout',
                controller_checkpoint_sha256=result.get('controller_checkpoint_sha256'),
                evaluator_path=str(episode/'evaluator_labels/episode.json')),
                allow_nan=False))
            attempts.append(dict(episode_id=result['episode_id'],attempt_id=attempt,split=split,campaign_split='train',
                goal_region_id=region,episode_path=str(episode),success=bool(result.get('success')),
                controller_kind=result.get('controller_kind','privileged_shortest_path_expert'),
                teacher_provenance=result.get('teacher_provenance'),
                runtime=dict(path=str(runtime_path.relative_to(output)),sha256=digest(runtime_path)),
                training_labels=dict(path=str(label_path.relative_to(output)),sha256=digest(label_path)),
                result_sha256=digest(receipt),frames=len(frames),aligned_frames=sum(r['motion_valid'] for r in aligned),
                timing=timing(frames,commands)))
        except (ValueError,KeyError,FileNotFoundError) as error:
            excluded.append(dict(attempt_id=attempt,result=str(receipt),reason=str(error)))
    if not attempts:
        raise ValueError('No usable recorded training attempts')
    initial=None
    if initialization:
        destination=output/'initialization.pt'
        shutil.copyfile(initialization,destination)
        initial=dict(source=str(initialization),path=destination.name,sha256=digest(destination))
    infrastructure=[]
    for path in sorted(root.glob('*/result.json')):
        if not (path.parent/'episode/result.json').exists():
            receipt=json.loads(path.read_text())
            if receipt.get('status') in ('failed','runtime_dependency_failed'):
                infrastructure.append(dict(path=str(path),sha256=digest(path),episode_id=receipt.get('episode_id'),
                    probe=receipt.get('probe'),status=receipt['status'],
                    identity_status='recorded' if receipt.get('episode_id') else 'legacy_launch_identity_requires_campaign_receipt'))
    manifest=dict(schema='visual-trajectory-bundle/v1',action_semantics='post-safety-dispatch/50ms-v3',teacher_contract='observed-exploration/v4',teacher_versions=sorted({x['teacher_provenance'] for x in attempts if x['teacher_provenance']}),collection_root=str(root),attempts=attempts,
        infrastructure_failures=infrastructure,
        initialization=initial,
        excluded_attempts=excluded,source_sha256=digest(__file__),
        accepted=False,readiness=dict(visual_sequences=True,causal_replay=False,world=False,policy=False),
        unique_successful_episodes=len({r['episode_id'] for r in attempts if r['success']}),
        unique_successful_expert_episodes=len({r['episode_id'] for r in attempts if r['success'] and r['controller_kind']=='privileged_shortest_path_expert'}),
        timing_passed_attempts=sum(r['timing']['accepted'] for r in attempts),
        reason='Causal runtime replay and disjoint development supervision required; dispatch timing uncertainty is explicit')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    result=dict(status='completed',accepted=False,attempts=len(attempts),excluded=len(excluded),
        unique_successful_episodes=manifest['unique_successful_episodes'],
        timing_passed_attempts=manifest['timing_passed_attempts'],manifest_sha256=digest(output/'manifest.json'))
    (output/'result.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--collection-root',type=Path,required=True)
    parser.add_argument('--initialization',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(build(args.collection_root,args.output,args.initialization)))
