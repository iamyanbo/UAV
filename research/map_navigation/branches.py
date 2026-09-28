"""Offline counterfactual mission preparation from recorded settled states.

Each alternative starts a NEW physical simulator flight from the same settled
pose. No pretending replayed frames are alternate-action execution.
"""
from pathlib import Path
import math
import numpy as np
from .common import read,write,digest
from .data import lines,yaw


def prepare_branches(registry,flights,output):
    scenes={r['scene_id']:r for r in read(registry)['scenes']}
    public=[];private=[]
    for result_path in sorted(Path(flights).rglob('episode/result.json')):
        result=read(result_path)
        if result.get('split') not in ('train','validation'):continue
        episode=result_path.parent;scene=scenes[result['scene_id']]
        task=read(episode/'evaluator_labels/episode.json')
        samples=lines(episode/'training_labels/frames.jsonl')
        # Restrict branching to settled states: velocity/angular dynamics are
        # not restored by AirSim pose placement. Moving-state clones are invalid.
        settled=[r for r in samples if np.linalg.norm(r['true_velocity_ned_mps'])<.1 and
                 not r['airsim_collision'] and not r['geometry_collision'] and abs(r['observation_label_skew_seconds'])<=.1]
        if not settled:continue
        sample=settled[len(settled)//2]
        group=digest(episode/'training_labels/frames.jsonl')[:16]+'-'+str(sample['frame_id'])
        for altitude in ('maintain','gain','lose'):
            ident=group+'-'+altitude
            proposal=dict(intention='inspect',target_reference=None,target_source='none',
                          altitude=altitude,confidence=1.,horizon_s=5.)
            public.append(dict(episode_id=ident,scene_id=scene['scene_id'],split=scene['split'],map_sha256=scene['map_sha256'],
                goal_views=1,goal_record='goals/'+ident,timeout_s=task['timeout_s'],collection_source='manoeuvre',
                branch_group=group,initial_subgoal=proposal))
            private.append(dict(task,episode_id=ident,start_ned_m=sample['true_position_ned_m'],
                start_yaw_degrees=math.degrees(yaw(sample['true_quaternion_xyzw'])),
                branch_source_frame=sample['frame_id'],branch_source_sha256=digest(episode/'training_labels/frames.jsonl'),
                branch_reset_semantics='settled pose; new flight, not moving-state restoration'))
    if not public:raise ValueError('No eligible settled training/validation branch states')
    for split in ('train','validation'):
        write(Path(output)/(split+'.json'),dict(schema='photo-map-missions/v4',registry_sha256=digest(registry),
            episodes=[r for r in public if r['split']==split]))
        write(Path(output)/'evaluator_labels'/(split+'.json'),dict(schema='privileged-photo-map-labels/v4',
            episodes=[r for r in private if r['split']==split]))
