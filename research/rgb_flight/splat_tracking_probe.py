"""Causal observation-only released Splat-SLAM tracker diagnostic.

No final BA, ground-truth alignment, mapper or metric-scale claim. This exercises
the actual pretrained frontend and DSPO kernels, beyond backbone-only checks.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

from episode_store import frames


class Printer:
    def print(self,*args):
        print(*args,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--frames',type=int,default=256,help='Zero processes the entire retained episode')
    args=parser.parse_args()
    if args.frames<0:
        parser.error('Frame limit cannot be negative')
    sys.path.insert(0,'/upstream/splat')
    from thirdparty.glorie_slam.modules.droid_net import DroidNet
    from thirdparty.glorie_slam.depth_video import DepthVideo
    from thirdparty.glorie_slam.motion_filter import MotionFilter
    from thirdparty.glorie_slam.frontend import Frontend
    from thirdparty.glorie_slam.backend import Backend
    torch.set_num_threads(4)
    torch.manual_seed(43)
    cfg=yaml.safe_load(Path('/upstream/splat/configs/splat_slam.yaml').read_text())
    cfg.update(data={'output':'/output'},scene='causal-tracking',only_tracking=True)
    cfg['cam'].update(H=480,W=640,H_out=480,W_out=640,fx=320.,fy=320.,cx=320.,cy=240.)
    cfg['tracking']['pretrained']='/models/splat/droid.pth'
    cfg['mono_prior']['depth_pretrained']='/models/splat/omnidata_dpt_depth_v2.ckpt'
    cfg['tracking']['backend']['final_ba']=False
    output=Path('/output/causal-tracking')
    (output/'mono_priors/depths').mkdir(parents=True)
    (output/'config.json').write_text(json.dumps(cfg,indent=2))
    net=DroidNet()
    state={k.removeprefix('module.'):v for k,v in torch.load(cfg['tracking']['pretrained'],map_location='cpu',weights_only=True).items()}
    for name in ['update.weight.2.weight','update.weight.2.bias','update.delta.2.weight','update.delta.2.bias']:
        state[name]=state[name][:2]
    net.load_state_dict(state,strict=True)
    net=net.eval().cuda()
    video=DepthVideo(cfg,Printer())
    motion=MotionFilter(net,video,cfg,thresh=cfg['tracking']['motion_filter']['thresh'],device='cuda:0')
    frontend=Frontend(net,video,cfg)
    backend=Backend(net,video,cfg)
    last_ba=0
    durations=[]
    initialized_frames=0
    received_timestamps=[]
    started=time.monotonic()
    with (output/'causal_estimates.jsonl').open('x') as estimates:
        for index,(record,rgb) in enumerate(frames('/observations')):
            if args.frames and index>=args.frames:
                break
            if video.counter.value>=cfg['tracking']['buffer']-1:
                raise RuntimeError('Tracking buffer full; no silent history deletion')
            if index==0:
                first_ns=record['sim_ns']
            received_timestamps.append(record['sim_ns'])
            cal=record['calibration']
            intrinsic=torch.tensor([cal[k] for k in ['fx','fy','cx','cy']])
            # CPU input avoids aliasing the RGB tensor with the frontend's
            # in-place GPU normalization before the mono-depth prediction.
            image=torch.from_numpy(np.frombuffer(rgb,np.uint8).copy().reshape(480,640,3)).permute(2,0,1)[None].float()/255
            begin=time.monotonic()
            with torch.no_grad():
                motion.track(index,image,intrinsic)
                frontend()
                current=video.counter.value-1
                if frontend.is_initialized and current>=last_ba+cfg['tracking']['backend']['ba_freq']:
                    backend.dense_ba(2)
                    last_ba=current
            torch.cuda.synchronize()
            durations.append(time.monotonic()-begin)
            initialized_frames+=int(frontend.is_initialized)
            pose=video.poses[current].detach().cpu().tolist()
            pose_frame=int(video.timestamp[current].item())
            if not all(np.isfinite(pose)):
                raise RuntimeError('Nonfinite RGB-estimated pose')
            # This immutable row contains ONLY the estimate available after
            # this received prefix. Future BA never rewrites it.
            estimates.write(json.dumps(dict(frame_id=record['frame_id'],source_rgb_sha256=record['rgb_sha256'],
                latest_observation_sim_ns=record['sim_ns'],keyframes=video.counter.value,
                estimated_pose_source_frame=pose_frame,estimated_pose_sim_ns=received_timestamps[pose_frame],
                estimated_pose_age_seconds=(record['sim_ns']-received_timestamps[pose_frame])/1e9,
                initialized=bool(frontend.is_initialized),estimated_camera_pose_arbitrary_scale=pose,
                processing_seconds=durations[-1]))+'\n')
            estimates.flush()
            if index%20==0:
                print(json.dumps(dict(frame=index,keyframes=video.counter.value,initialized=bool(frontend.is_initialized),
                                      processing_seconds=durations[-1])),flush=True)
    result=dict(status='causal_tracker_diagnostic_completed',frames=len(durations),initialized_frames=initialized_frames,
                keyframes=video.counter.value,elapsed_wall_seconds=time.monotonic()-started,
                observed_sim_seconds=(received_timestamps[-1]-first_ns)/1e9,
                requested_frame_limit=args.frames,
                frame_processing_max_seconds=float(max(durations)),
                frame_processing_p50_seconds=float(np.median(durations)),frame_processing_p95_seconds=float(np.percentile(durations,95)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                metric_scale_established=False,mapping_exercised=False,foundation_passed=False)
    (output/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
