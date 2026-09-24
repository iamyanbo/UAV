"""Held-out recorded-episode motion, accumulated drift and interval coverage."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from episode_store import frames
from goal_io import load_goal
from navigation_state import CheckpointSet,CausalNavigationState,checksum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',type=Path,default=Path('/dataset'))
    parser.add_argument('--checkpoints',type=Path,default=Path('/navigation/checkpoints.json'))
    parser.add_argument('--output',type=Path,default=Path('/output/odometry-qualification'))
    args=parser.parse_args();manifest=json.loads((args.dataset/'manifest.json').read_text())
    pack=CheckpointSet(args.checkpoints,integration_only=True);torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    summary=[];errors=[];coverage=[];horizons={str(h):[] for h in (1,2,4,10)}
    stopped=False
    for attempt in manifest['attempts']:
        if attempt['split']!='validation':continue
        if Path('/output/CHECKPOINT_REQUEST').exists():stopped=True;break
        path=args.dataset/attempt['training_labels']['path']
        if checksum(path)!=attempt['training_labels']['sha256']:raise ValueError('Changed held-out labels')
        labels={r['frame_id']:r for r in json.loads(path.read_text())['frames']}
        physical=Path(attempt['episode_path']);goal=load_goal(physical/'goal')
        goal_rgb=torch.from_numpy(np.stack([np.frombuffer(raw,np.uint8).reshape(480,640,3).copy() for raw in goal.rgb_views])).permute(0,3,1,2)
        core=CausalNavigationState(pack,attempt['episode_id'],goal_rgb)
        previous_label=None;anchors=[];segments=[];segment_anchor=None;last_drift=None;count=paired=0;gaps=0
        try:
            for row,rgb in frames(physical/'observations'):
                value=core.observe(row,rgb);count+=1;label=labels.get(row['frame_id'])
                if core.warmup==0:
                    if last_drift:segments.append(last_drift)
                    anchors=[];segment_anchor=None;last_drift=None;gaps+=count>1
                if not label or not label['motion_valid']:
                    previous_label=None;continue
                true_rotation=Rotation.from_quat(label['true_quaternion_xyzw'])
                true_position=np.asarray(label['true_position_ned_m'])
                if previous_label is not None and core.warmup>core.required_warmup and core.last_body_motion is not None:
                    q0=Rotation.from_quat(previous_label['true_quaternion_xyzw'])
                    truth=np.r_[q0.inv().apply(true_position-np.asarray(previous_label['true_position_ned_m'])),(q0.inv()*true_rotation).as_rotvec()]
                    error=core.last_body_motion.cpu().numpy()-truth
                    std=core.last_motion_std.cpu().numpy();errors.append(error);coverage.append(np.abs(error)<=1.96*std);paired+=1
                previous_label=label
                if core.warmup<=core.required_warmup:continue
                now=row['sim_ns'];position=core.position.cpu().numpy().copy();rotation=core.rotation.cpu().numpy().copy()
                current=dict(ns=now,position=position,rotation=rotation,true_position=true_position,true_rotation=true_rotation,done=set())
                if segment_anchor is None:segment_anchor=current
                def drift(anchor):
                    expected=anchor['true_rotation'].inv().apply(true_position-anchor['true_position'])
                    estimated=anchor['rotation'].T@(position-anchor['position'])
                    estimated_rotation=Rotation.from_matrix(anchor['rotation'].T@rotation)
                    expected_rotation=anchor['true_rotation'].inv()*true_rotation
                    return dict(seconds=(now-anchor['ns'])/1e9,translation_error_m=float(np.linalg.norm(estimated-expected)),
                        rotation_error_rad=float((expected_rotation.inv()*estimated_rotation).magnitude()),
                        true_displacement_m=float(np.linalg.norm(expected)))
                last_drift=drift(segment_anchor)
                for anchor in anchors:
                    age=(now-anchor['ns'])/1e9
                    for horizon in (1,2,4,10):
                        if horizon<=age<=horizon+.25 and horizon not in anchor['done']:
                            horizons[str(horizon)].append(dict(attempt_id=attempt['attempt_id'],**drift(anchor)))
                            anchor['done'].add(horizon)
                anchors=[a for a in anchors if now-a['ns']<=10250000000]
                if not anchors or now-anchors[-1]['ns']>=1000000000:anchors.append(current)
            if last_drift:segments.append(last_drift)
        finally:core.close()
        result=dict(attempt_id=attempt['attempt_id'],episode_id=attempt['episode_id'],frames=count,
            paired_motion_labels=paired,observation_gaps_over_250ms=gaps,segments=segments)
        summary.append(result)
        with (args.output/'episodes.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    error=np.asarray(errors);covered=np.asarray(coverage)
    details={k:dict(windows=len(v),translation_rmse_m=float(np.sqrt(np.mean([x['translation_error_m']**2 for x in v]))) if v else None,
        rotation_rmse_rad=float(np.sqrt(np.mean([x['rotation_error_rad']**2 for x in v]))) if v else None) for k,v in horizons.items()}
    result=dict(status='checkpointed' if stopped else 'completed',accepted=False,
        checkpoint_sha256=pack.spec['artifacts']['odometry']['sha256'],source_bundle_sha256=checksum(args.dataset/'manifest.json'),
        development_attempts=len(summary),development_episode_ids=sorted({r['episode_id'] for r in summary}),
        motion_pairs=len(error),translation_rmse_m=float(np.sqrt(np.mean(error[:,:3]**2))) if len(error) else None,
        rotation_rmse_rad=float(np.sqrt(np.mean(error[:,3:]**2))) if len(error) else None,
        marginal_95pct_coverage=float(covered.mean()) if len(covered) else None,horizons=details,
        limitations=['Repeated attempts are not independent episodes','Unaligned labels and RGB gaps are reported, not interpolated across missing coverage',
                     'Long-horizon covariance uses first-order independent increment propagation; empirical calibration remains required'])
    (args.output/'horizons.json').write_text(json.dumps(horizons));(args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
