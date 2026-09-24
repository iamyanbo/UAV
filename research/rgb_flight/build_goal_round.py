"""Build a new goal-supervision round from timestamp-aligned trajectories.

All four original views remain immutable. Failed flights contribute negatives;
their remaining time/value labels are censored. The source manifests are never
overwritten and campaign validation/test data are not opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from build_visual_dataset import spaced
from trajectory_bundle import digest


def build(bundle,output,initialization):
    manifest_path=bundle/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['schema']!='visual-trajectory-bundle/v1':raise ValueError('Aligned trajectory bundle required')
    root=Path(manifest['collection_root']);episodes=[];windows=[];seen={}
    # One attempt per task for visual sampling; keep failures if no success is
    # available. All attempts remain available in the trajectory bundle.
    for item in manifest['attempts']:
        old=seen.get(item['episode_id'])
        if old is None or (item['success'],item['aligned_frames'])>(old['success'],old['aligned_frames']):
            seen[item['episode_id']]=item
    for item in seen.values():
        if item['campaign_split']!='train':raise ValueError('Sealed campaign split in training bundle')
        runtime_path=bundle/item['runtime']['path'];label_path=bundle/item['training_labels']['path']
        if digest(runtime_path)!=item['runtime']['sha256'] or digest(label_path)!=item['training_labels']['sha256']:
            raise ValueError('Changed source trajectory')
        runtime=json.loads(runtime_path.read_text());labels=json.loads(label_path.read_text())
        source=Path(item['episode_path']);evaluator=json.loads((source/'evaluator_labels/episode.json').read_text())
        goal=np.asarray(evaluator['goal_ned_m']);frames={r['frame_id']:r for r in runtime['frames']}
        end_ns=runtime['frames'][-1]['sim_ns']
        positive=[];boundary=[];far=[]
        for row in labels['frames']:
            if not row['motion_valid']:continue
            delta=np.asarray(row['true_position_ned_m'])-goal
            horizontal=np.linalg.norm(delta[:2]);vertical=abs(delta[2]);distance=np.linalg.norm(delta)
            # Keep a declared 10 cm ambiguity band around the evaluator's
            # 3 m / 2 m limits; the evaluator itself is unchanged.
            if horizontal<=2.9 and vertical<=1.9:positive.append(row['frame_id'])
            elif horizontal>=3.1 or vertical>=2.1:
                (far if distance>=12 else boundary).append(row['frame_id'])
        chosen=[(frame,'positive') for frame in spaced(positive,24)]
        chosen += [(frame,'boundary_negative') for frame in spaced(boundary,32)]
        chosen += [(frame,'far_negative') for frame in spaced(far,64)]
        if not chosen:continue
        episode_id=item['episode_id'];relative=str(source.relative_to(root))
        episodes.append(dict(episode_id=episode_id,split=item['split'],campaign_split='train',
            goal_region_id=item['goal_region_id'],episode_path=relative,goal_sha256=runtime['goal_sha256'],
            attempt_id=item['attempt_id'],success=item['success'],source_labels_sha256=item['training_labels']['sha256']))
        for frame,category in chosen:
            remaining=max(0.,(end_ns-frames[frame]['sim_ns'])/1e9)
            windows.append(dict(module='goal',episode_id=episode_id,episode_path=relative,frame_id=frame,
                category=category,training_labels=dict(near_goal=category=='positive',match_valid=True,
                time_to_goal_seconds=remaining,time_valid=item['success'],
                terminal_return=1.-min(1.,remaining/180.) if item['success'] else 0.,value_valid=item['success'])))
    if manifest['unique_successful_episodes']<250:
        raise ValueError('Fewer than 250 successful tasks in source collection')
    output.mkdir(parents=True,exist_ok=False)
    if initialization:shutil.copyfile(initialization,output/'initialization.pt')
    value=dict(schema='visual-goal-supervision/v3',collection_root=str(root),episodes=episodes,windows=windows,
        valid_expert_episodes=manifest['unique_successful_episodes'],
        supervised_successful_episodes=sum(e['success'] for e in episodes),
        source_manifest_sha256=digest(manifest_path),initialization_sha256=digest(initialization) if initialization else None,
        label_alignment='interpolated at exposure; <=250 ms state bracket; 10 cm goal-boundary ambiguity band',
        time_supervision='successful demonstration remaining time only; unsuccessful outcomes censored',
        objective_version='goal-recognition-calibration/v3')
    (output/'visual-training.json').write_text(json.dumps(value,allow_nan=False))
    categories={split:{kind:sum(r['category']==kind and next(e['split'] for e in episodes if e['episode_id']==r['episode_id'])==split for r in windows)
        for kind in ('positive','boundary_negative','far_negative')} for split in ('train','validation')}
    result=dict(status='completed',accepted=False,episodes=len(episodes),windows=len(windows),categories=categories,
        manifest_sha256=digest(output/'visual-training.json'))
    (output/'result.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--initialization',type=Path)
    args=parser.parse_args();print(json.dumps(build(args.bundle,args.output,args.initialization)))
