"""Post-flight labels. Simulator facts and visual hypotheses stay distinguishable."""
from bisect import bisect_right
import json
import math
from pathlib import Path
import numpy as np
from .common import digest,read,write


def lines(path):
    if not Path(path).exists():return []
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def annotate(episode,field,result,conditions):
    from scipy.spatial import cKDTree
    episode=Path(episode);frames=lines(episode/'training_labels/frames.jsonl')
    if not frames:return None
    task=read(episode/'evaluator_labels/episode.json')
    decisions=lines(episode/'photo-controller/runtime/decisions.jsonl')
    if not decisions:decisions=lines(episode/'perception/runtime/decisions.jsonl')
    by_frame={r['frame_id']:r for r in decisions}
    tree=cKDTree(field.occupied);collision_times=[r['state_sim_ns'] for r in frames if r['airsim_collision'] or r['geometry_collision']]
    total_climb=0.;previous=None;future_collision=False;rows=[]
    for frame in reversed(frames):
        future_collision=future_collision or frame['airsim_collision'] or frame['geometry_collision']
        p=np.asarray(frame['true_position_ned_m']);stamp=frame['state_sim_ns'];d=by_frame.get(frame['frame_id'],{})
        teacher=d.get('teacher');visible=frame.get('expert_label',{}).get('goal_visibility')
        margins=np.minimum(p-np.asarray(field.bounds)[0],np.asarray(field.bounds)[1]-p)
        q=frame['true_quaternion_xyzw'];x,y,z,w=q
        roll=math.atan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=math.asin(float(np.clip(2*(w*y-z*x),-1,1)))
        horizon={str(seconds):any(stamp<=t<=stamp+seconds*1e9 for t in collision_times) for seconds in (1,4,10,30)}
        reason=d.get('safety_reason')
        if reason is None and frame.get('safety_modified_command')!=frame.get('proposed_command'):reason='privileged_geometry_veto'
        rows.append(dict(schema='photo-map-step-label/v1',frame_id=frame['frame_id'],state_sim_ns=stamp,
            source='simulator-state-and-survey',geometric_goal_visible=visible,visual_recognizability=None,
            visual_hypothesis=teacher.get('recognizability_hypothesis') if teacher else None,
            teacher=teacher,phase=teacher['phase'] if teacher else d.get('flight_phase',frame.get('flight_phase','unknown')),
            events=teacher.get('events',[]) if teacher else [],
            sampled_surface_distance_m=float(tree.query(p)[0]),clearance_certified=False,
            envelope_margin_m=float(margins.min()),geofence_violation=bool((margins<0).any()),
            roll_rad=roll,pitch_rad=pitch,attitude_limit_violation=None,energy_joules=None,
            safety_reason=reason,collision_within_s=horizon,future_collision=bool(future_collision),
            remaining_flight_s=max(0.,(frames[-1]['state_sim_ns']-stamp)/1e9),
            outcome_success=bool(result.get('success')),conditions=conditions,
            uncertainty_conditions=['occluded_goal'] if visible==0 else [],
            label_valid=abs(frame.get('observation_label_skew_seconds',1))<=.0125))
    rows.reverse()
    for row,frame in zip(rows,frames):
        z=frame['true_position_ned_m'][2]
        if previous is not None:total_climb+=max(0.,previous-z)
        previous=z;row['cumulative_climb_distance_m']=total_climb
    path=episode/'training_labels/annotations.jsonl'
    with path.open('w') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    receipt=dict(schema='photo-map-label-receipt/v1',source_frames_sha256=digest(episode/'training_labels/frames.jsonl'),
        annotations_sha256=digest(path),frames=len(rows),energy_measured=False,
        imitation_eligible=sum(bool(r['teacher'] and r['teacher']['policy_valid'] and r['label_valid']) for r in rows))
    write(episode/'training_labels/annotation-receipt.json',receipt)
    return receipt
