"""Privileged post-flight analysis. Never imported or mounted as runtime data."""
import argparse
import json
from pathlib import Path
import numpy as np
from calibration import camera_extrinsics


def align_similarity(estimate, truth):
    x, y = estimate - estimate.mean(0), truth - truth.mean(0)
    variance = np.square(x).sum() / len(x)
    if variance < 1e-8:
        raise ValueError('Insufficient translation for monocular similarity alignment')
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag(signs) @ vt
    scale = float((singular * signs).sum() / variance)
    translated = scale * estimate @ rotation.T
    translated += truth.mean(0) - translated.mean(0)
    return translated, scale


def quaternion_rotation(value):
    x,y,z,w = np.asarray(value) / np.linalg.norm(value)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def analyze(episode, reconstruction=None, output=None, final_memory=False):
    episode=Path(episode)
    states=json.loads((episode/'engineering_only/states.json').read_text())
    frames={row['frame_id']:row for row in map(json.loads,(episode/'observations/frames.jsonl').read_text().splitlines())}
    reconstruction=Path(reconstruction) if reconstruction else episode/'runtime_perception/reconstruction'
    reconstruction_result=json.loads((reconstruction/'result.json').read_text())
    trajectory=[json.loads(line) for line in (reconstruction/'tracking.jsonl').read_text().splitlines()]
    if final_memory:
        from memory_snapshot import MemorySnapshot
        now=max(row['sim_ns'] for row in frames.values())
        memory=MemorySnapshot(reconstruction/'memory',now,next(iter(frames.values()))['episode_id'],offline_prefix=True)
        trajectory=[]
        for camera in memory.observed_cameras().values():
            source=camera['source']
            trajectory.append(dict(frame_id=source['frame_id'],sim_ns=source['sim_ns'],
                estimated_pose_source_frame=source['frame_id'], initialized=True,
                estimated_c2w_arbitrary_scale=camera['w2c'].float().inverse().tolist(),
                processed_monotonic_seconds=0.))
    ext=camera_extrinsics(json.loads((episode.parent/'settings.json').read_text()))
    offset=np.asarray(ext['camera_origin_body_m'])
    origin=states[0]['sim_ns']
    times=np.array([(s['sim_ns']-origin)/1e9 for s in states])
    true_camera=np.array([np.asarray(s['position'])+quaternion_rotation(s['quaternion'])@offset for s in states])
    estimate, truth, source_ages, processing_ages = [], [], [], []
    for row in trajectory:
        source=frames.get(row['estimated_pose_source_frame'])
        received=frames.get(row['frame_id'])
        if not row['initialized'] or source is None or received is None:
            continue
        t=(source['sim_ns']-origin)/1e9
        if not times[0]<=t<=times[-1]:
            continue
        value=np.asarray(row['estimated_c2w_arbitrary_scale'])[:3,3]
        if not np.isfinite(value).all():
            raise ValueError('Nonfinite recorded RGB pose')
        estimate.append(value)
        truth.append([np.interp(t,times,true_camera[:,axis]) for axis in range(3)])
        source_ages.append((row['sim_ns']-source['sim_ns'])/1e9)
        processing_ages.append(row['processed_monotonic_seconds']-received['received_monotonic'])
    result=dict(scope='privileged post-flight similarity-aligned camera trajectory; not runtime metric-scale accuracy',
                matched_estimates=len(estimate), runtime_ground_truth_used=False, metric_scale_established=False)
    if final_memory:
        result['scope']='retrospective final-memory camera alignment; contains later observations and is forbidden as historical policy input'
    if len(estimate)>=4:
        aligned,scale=align_similarity(np.asarray(estimate),np.asarray(truth))
        error=np.linalg.norm(aligned-np.asarray(truth),axis=1)
        result.update(retrospective_alignment_scale=scale, aligned_camera_position_rmse_m=float(np.sqrt(np.mean(error**2))),
                      aligned_camera_position_p95_m=float(np.quantile(error,.95)),
                      estimated_pose_source_age_p95_sim_seconds=float(np.quantile(source_ages,.95)),
                      coordinate_gauge=reconstruction_result.get('coordinate_gauge','changing_upstream_gauge; global_alignment_is_not_reliable'))
        if reconstruction_result.get('live_async_execution') and not final_memory:
            result['received_rgb_to_tracking_output_p95_wall_seconds']=float(np.quantile(processing_ages,.95))
    else:
        result['failure']='Too few initialized, time-matched estimates'
    target=Path(output) if output else episode/'engineering_only/perception_metrics.json'
    with target.open('x') as stream:
        json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('episode')
    parser.add_argument('--reconstruction')
    parser.add_argument('--output')
    parser.add_argument('--final-memory',action='store_true',help='Privileged retrospective analysis only; never a training/runtime feature')
    args=parser.parse_args()
    print(json.dumps(analyze(args.episode,args.reconstruction,args.output,args.final_memory)),flush=True)
