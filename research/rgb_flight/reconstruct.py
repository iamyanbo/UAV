"""Causal physical-episode reconstruction using released RGB tracker + mapper."""
import argparse
import hashlib
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml
from episode_store import frames, verified_rgb_storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--observations', default='/observations')
    parser.add_argument('--output', default='/output/reconstruction')
    parser.add_argument('--frames', type=int, default=0)
    parser.add_argument('--asynchronous-map', action='store_true')
    parser.add_argument('--tracking-optimizer', choices=['DSPO','DBA'], default='DSPO',
                        help='Released Splat-SLAM pose optimizer; diagnostic choice is recorded')
    parser.add_argument('--broker-socket')
    parser.add_argument('--episode-id')
    parser.add_argument('--vision', help='Immutable RGB metric-depth configuration')
    args = parser.parse_args()
    if not args.broker_socket:
        verified_rgb_storage(args.observations)
    sys.path.insert(0, '/upstream/splat')
    from thirdparty.glorie_slam.modules.droid_net import DroidNet
    from thirdparty.glorie_slam.depth_video import DepthVideo
    from thirdparty.glorie_slam.motion_filter import MotionFilter
    from thirdparty.glorie_slam.frontend import Frontend
    from thirdparty.glorie_slam.backend import Backend
    from causal_mapper import CausalMapper
    from mapping_worker import AsyncMapper
    from causal_pose import CausalPoseEstimator, CausalMapGauge, gauge_tracked_video
    from types import SimpleNamespace
    torch.set_num_threads(4)
    torch.manual_seed(43)
    cfg = yaml.safe_load(Path('/upstream/splat/configs/splat_slam.yaml').read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'episode/mono_priors/depths').mkdir(parents=True)
    cfg.update(data={'output': str(output)}, scene='episode', only_tracking=True)
    cfg['cam'].update(H=480, W=640, H_out=480, W_out=640, fx=320., fy=320., cx=320., cy=240.)
    cfg['tracking']['pretrained'] = '/models/splat/droid.pth'
    cfg['mono_prior']['depth_pretrained'] = '/models/splat/omnidata_dpt_depth_v2.ckpt'
    cfg['tracking']['backend']['final_ba'] = False
    cfg['tracking']['backend']['BA_type'] = args.tracking_optimizer
    net = DroidNet()
    weights = {k.removeprefix('module.'): v for k, v in torch.load(cfg['tracking']['pretrained'], map_location='cpu', weights_only=True).items()}
    for name in ['update.weight.2.weight', 'update.weight.2.bias', 'update.delta.2.weight', 'update.delta.2.bias']:
        weights[name] = weights[name][:2]
    net.load_state_dict(weights, strict=True)
    net = net.eval().cuda()
    video = gauge_tracked_video(DepthVideo)(cfg, SimpleNamespace(print=lambda *args: None))
    pose_estimator = CausalPoseEstimator(net, video)
    map_gauge = CausalMapGauge()
    metric_depth = None
    metric_frames = OrderedDict()
    metric_scale = None
    metric_usable_count = 0
    if args.vision:
        from metric_depth import MetricDepth, fit_depth_scale
        metric_depth = MetricDepth(args.vision)
        (output/'metric').mkdir()
    motion = MotionFilter(net, video, cfg, thresh=cfg['tracking']['motion_filter']['thresh'], device='cuda:0')
    frontend, backend = Frontend(net, video, cfg), Backend(net, video, cfg)
    mapper = None
    received = OrderedDict()
    sources = []
    last_ba = 0
    last_mapped_frame = -1
    count = 0
    initialized_count = 0
    gauge_rejections = 0
    empty_graph_frames = 0
    previous_observation_ns = None
    sampled_out = 0
    begin = time.monotonic()
    def observations():
        if not args.broker_socket:
            yield from frames(args.observations)
            return
        from wire import BrokerClient
        channel = BrokerClient(args.broker_socket, args.episode_id)
        last = -1
        try:
            while not Path('/output/PERCEPTION_STOP').exists():
                if not Path(args.broker_socket).exists():
                    time.sleep(.1)
                    continue
                try:
                    record, rgb = channel.observe(last)
                except (EOFError, OSError, RuntimeError):
                    if Path('/output/PERCEPTION_STOP').exists():
                        return
                    raise
                last = record['frame_id']
                record['rgb_sha256'] = hashlib.sha256(rgb).hexdigest()
                yield record, rgb
        finally:
            channel.close()
    if args.broker_socket:
        Path('/output/tracker.ready').write_text(json.dumps(dict(episode_id=args.episode_id, loaded='Droid and Omnidata', monotonic=time.monotonic())))
    try:
        with (output / 'tracking.jsonl').open('x') as log:
            for record, rgb in observations():
                if args.frames and count >= args.frames:
                    break
                # Tracker cadence is simulation-time based. Slowing physics must
                # not multiply redundant tracking work or change the input rate.
                if previous_observation_ns is not None and record['sim_ns'] - previous_observation_ns < 50000000:
                    sampled_out += 1
                    continue
                previous_observation_ns = record['sim_ns']
                index = len(sources)
                if Path('/output/CHECKPOINT_REQUEST').exists():
                    break  # Prefix artifacts are immutable; resumption replays the episode.
                if video.counter.value >= cfg['tracking']['buffer'] - 1:
                    raise RuntimeError('Tracker keyframe buffer exhausted; retain prefix and report tracking failure')
                if mapper is None:
                    mapper_type = AsyncMapper if args.asynchronous_map else CausalMapper
                    mapper = mapper_type(cfg, output / 'memory', record['episode_id'])
                if args.asynchronous_map:
                    mapper.poll()
                received[index] = rgb
                sources.append(record)
                image = torch.from_numpy(np.frombuffer(rgb, np.uint8).copy().reshape(480, 640, 3)).permute(2, 0, 1)[None].float() / 255
                if metric_depth is not None:
                    metric_frames[index] = metric_depth(np.frombuffer(rgb,np.uint8).reshape(480,640,3),record['calibration'])
                intrinsic = torch.tensor([record['calibration'][k] for k in ('fx', 'fy', 'cx', 'cy')])
                with torch.no_grad():
                    motion.track(index, image, intrinsic)
                    frontend()
                    current = video.counter.value - 1
                    if frontend.is_initialized and frontend.graph.ii.numel() and current >= last_ba + cfg['tracking']['backend']['ba_freq']:
                        backend.dense_ba(2)
                        last_ba = current
                    video.update_valid_depth_mask()
                published = None
                if frontend.is_initialized and not frontend.graph.ii.numel():
                    empty_graph_frames += 1
                    log.write(json.dumps(dict(frame_id=record['frame_id'],sim_ns=record['sim_ns'],
                        processed_monotonic_seconds=time.monotonic(),estimated_c2w_arbitrary_scale=None,
                        initialized=False,memory_version=mapper.version,gauge_changes=video.gauge_changes,
                        tracking_failure='empty_active_factor_graph'))+'\n');log.flush();count+=1
                    continue
                if frontend.is_initialized:
                    try:map_gauge.update(video)
                    except RuntimeError as error:
                        if str(error)!='Causal camera anchors cannot establish a consistent similarity gauge':raise
                        # An inconsistent prefix is tracking loss, not permission
                        # to reflect/force its scale or use its geometry. Preserve
                        # the failure and let later observed anchors recover.
                        gauge_rejections+=1
                        log.write(json.dumps(dict(frame_id=record['frame_id'],sim_ns=record['sim_ns'],
                            processed_monotonic_seconds=time.monotonic(),estimated_c2w_arbitrary_scale=None,
                            initialized=False,memory_version=mapper.version,gauge_changes=video.gauge_changes,
                            tracking_failure=str(error)))+'\n');log.flush();count+=1
                        continue
                    # Avoid the newest, still-removable frontend keyframe.
                    if metric_depth is not None:
                        pairs=[]
                        for video_index in range(max(0,current-7),current+1):
                            key=int(video.timestamp[video_index].item())
                            if key not in metric_frames:continue
                            d,v,_=video.get_depth_and_pose(video_index,'cuda:0')
                            pairs.append((metric_frames[key],d.cpu()*map_gauge.scale,v.cpu()))
                        metric_scale=fit_depth_scale(pairs)
                        metric_usable_count+=int(metric_scale['usable'])
                    stable = max(0, current - 2)
                    source_index = int(video.timestamp[stable].item())
                    if source_index > last_mapped_frame and source_index in received:
                        with torch.no_grad():
                            depth, valid, c2w = video.get_depth_and_pose(stable, 'cuda:0')
                            depth *= map_gauge.scale
                            c2w = map_gauge.pose(c2w)
                            corrections = []
                            for video_index in range(current + 1):
                                key = int(video.timestamp[video_index].item())
                                if key in mapper.sources:
                                    old_depth, old_valid, old_c2w = video.get_depth_and_pose(video_index, 'cuda:0')
                                    old_depth *= map_gauge.scale
                                    old_c2w = map_gauge.pose(old_c2w)
                                    old_depth[~old_valid] = 0
                                    corrections.append((key, old_c2w.inverse(), old_depth))
                        source_image = torch.from_numpy(np.frombuffer(received[source_index], np.uint8).copy().reshape(480, 640, 3)).permute(2, 0, 1).float() / 255
                        published = mapper.update(source_index, sources[source_index], source_image, depth, valid,
                                                  c2w.inverse(), record['calibration'], record['sim_ns'], corrections)
                        if published:
                            last_mapped_frame = source_index
                            print(json.dumps(published), flush=True)
                    oldest_needed = int(video.timestamp[max(0, current - 3)].item())
                    while received and next(iter(received)) < oldest_needed:
                        received.popitem(last=False)
                    retained={int(video.timestamp[k].item()) for k in range(max(0,current-8),current+1)}
                    for key in list(metric_frames):
                        if key not in retained and key!=index:del metric_frames[key]
                if frontend.is_initialized:
                    pose, pose_quality = pose_estimator.estimate(index, image, intrinsic)
                    pose_index = index
                else:
                    pose = video.get_pose(current, 'cpu')
                    pose_index = int(video.timestamp[current].item())
                    pose_quality = dict(method='uninitialized_keyframe', correspondence_weight=None, reprojection_error_pixels=None)
                pose = map_gauge.pose(pose).tolist()
                metric_file=None
                if metric_depth is not None:
                    metric_file='metric/'+str(record['frame_id'])+'.pt'
                    path=output/metric_file
                    torch.save(dict(depth=metric_frames[index][::8,::8],frame_id=record['frame_id'],
                        observation_ns=record['sim_ns'],scale=metric_scale,calibration=record['calibration']),path.with_suffix('.tmp'))
                    path.with_suffix('.tmp').replace(path)
                initialized_count += int(frontend.is_initialized)
                log.write(json.dumps(dict(frame_id=record['frame_id'], sim_ns=record['sim_ns'],
                           estimated_pose_source_frame=sources[pose_index]['frame_id'],
                           estimated_pose_age_sim_seconds=(record['sim_ns']-sources[pose_index]['sim_ns'])/1e9,
                           processed_monotonic_seconds=time.monotonic(),
                           estimated_c2w_arbitrary_scale=pose, initialized=bool(frontend.is_initialized),
                           metric_depth_file=metric_file,metric_scale=metric_scale,
                           memory_version=mapper.version, gauge_scale=video.gauge_scale,
                           causal_map_gauge_scale=map_gauge.scale, gauge_alignment_residual=map_gauge.alignment_residual,
                           gauge_changes=video.gauge_changes, pose_quality=pose_quality)) + '\n')
                log.flush()
                count += 1
    finally:
        if args.asynchronous_map and mapper:
            mapper.close()
    result = dict(status='live_observed_stream_reconstructed' if args.broker_socket else 'causal_reconstruction_complete' if not args.frames and count == len(sources) and not Path('/output/CHECKPOINT_REQUEST').exists() else 'causal_prefix_reconstructed',
                  frames=count, memory_versions=mapper.version if mapper else 0,
                  elapsed_wall_seconds=time.monotonic() - begin,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  metric_scale_established=metric_usable_count>0, metric_scale_usable_frames=metric_usable_count,
                  metric_uncertainty_calibrated=False,live_async_execution=bool(args.broker_socket), foundation_passed=False,
                  tracking_initialized_frames=initialized_count, sampling_skipped_frames=sampled_out,
                  current_pose_method='past_anchor_pose_only', coordinate_gauge='causal_common_camera_similarity; arbitrary_scale',
                  gauge_changes=video.gauge_changes)
    result['tracking_optimizer'] = args.tracking_optimizer
    result['gauge_rejected_frames'] = gauge_rejections
    result['empty_active_graph_frames'] = empty_graph_frames
    result.update(mapping_execution='asynchronous_bounded_worker' if args.asynchronous_map else 'sequential_causal_replay',
                  mapping_requests_superseded=getattr(mapper, 'superseded', 0))
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
