"""Replay recorded RGB through the exact online causal navigation state.

Only observations, the four-view goal and model snapshots are inputs. Labels
are joined by downstream training code after this observation-only replay.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from episode_store import frames,verified_rgb_storage
from goal_io import load_goal
from navigation_state import CheckpointSet,CausalNavigationState,checksum


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--observations',type=Path,default=Path('/observations'))
    parser.add_argument('--goal',type=Path,default=Path('/goal'))
    parser.add_argument('--checkpoints',type=Path,default=Path('/navigation/checkpoints.json'))
    parser.add_argument('--output',type=Path,default=Path('/output/navigation-replay'))
    parser.add_argument('--integration-only',action='store_true');parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--video',action='store_true',help='Pace original frame availability and run the actual frozen video encoder asynchronously')
    parser.add_argument('--mapping',action='store_true',help='Run the native RGB tracker and mapper against the paced observation-only broker')
    args=parser.parse_args();storage=verified_rgb_storage(args.observations);goal=load_goal(args.goal)
    checkpoints=CheckpointSet(args.checkpoints,integration_only=args.integration_only)
    goal_rgb=torch.from_numpy(np.stack([np.frombuffer(raw,np.uint8).reshape(480,640,3).copy() for raw in goal.rgb_views])).permute(0,3,1,2)
    torch.set_num_threads(4);core=CausalNavigationState(checkpoints,goal.episode_id,goal_rgb)
    args.output.mkdir(parents=True,exist_ok=False)
    goal_cache=args.output/'goal-tokens.pt'
    torch.save(dict(tokens=core.goal_tokens[0].cpu(),goal_sha256=goal.content_sha256,
                    encoder_checkpoint_sha256=checkpoints.spec['artifacts']['goal']['sha256']),goal_cache)
    video=None
    if args.video:
        from async_video import AsyncVideoFeatures
        video=AsyncVideoFeatures(goal.episode_id,args.output/'video')
    mapping=broker=None;mapping_receipt=None
    if args.mapping:
        if not args.video:raise ValueError('Mapping replay requires the paced asynchronous video path')
        from recorded_broker import RecordedRGBBroker
        from async_mapping import AsyncMapping
        broker=RecordedRGBBroker(args.output/'ipc/rgb.sock',goal)
        mapping=AsyncMapping(goal.episode_id,args.output/'reconstruction',broker.path,checkpoints.paths.get('vision'))
    replay_start=time.monotonic();original_start=None;publication_lateness=[];video_receipt=None
    count=0;chunks=[];shards=[];timings=[];last_ns=-1;missing_video=missing_geometry=0
    goal_conditioned_frames=0;alignment_reasons={};last_grounding_ns=None
    def flush():
        nonlocal chunks
        if not chunks:return
        path=args.output/f'runtime-{len(shards):06d}.pt';torch.save(dict(episode_id=goal.episode_id,samples=chunks),path)
        shards.append(dict(path=path.name,sha256=checksum(path),samples=len(chunks),
                           first_ns=chunks[0]['sim_ns'],last_ns=chunks[-1]['sim_ns']))
        chunks=[]
    failure=None;cleanup_errors=[]
    try:
        for row,rgb in frames(args.observations):
            if args.frames and count>=args.frames:break
            if Path('/output/CHECKPOINT_REQUEST').exists():break
            if video:
                if original_start is None:original_start=row['received_monotonic']
                scheduled=replay_start+row['received_monotonic']-original_start
                remaining=scheduled-time.monotonic()
                if remaining>0:time.sleep(remaining)
                row=dict(row,received_monotonic=time.monotonic())
                publication_lateness.append(row['received_monotonic']-scheduled)
                if broker:
                    broker.publish(row,rgb);mapping.poll(core,row)
                result=video.poll()
                if result is not None:core.enqueue('video',result)
                video.observe(row,rgb)
            started=time.monotonic();value=core.observe(row,rgb);torch.cuda.current_stream().synchronize();timings.append(time.monotonic()-started)
            if not torch.isclose(value['task'][0],value['task'].new_tensor(value['goal_probability'])):
                raise ValueError('Goal evidence was lost from policy/world conditioning')
            goal_conditioned_frames+=1
            if core.diagnostics:
                with (args.output/'alignment.jsonl').open('a') as log:
                    for event in core.diagnostics:
                        log.write(json.dumps(event)+'\n')
                        reason=event.get('reason','unknown');alignment_reasons[reason]=alignment_reasons.get(reason,0)+1
                core.diagnostics.clear()
            with (args.output/'timing.jsonl').open('a') as timing:
                timing.write(json.dumps(dict(frame_id=row['frame_id'],sim_ns=row['sim_ns'],
                    available_monotonic=row['received_monotonic'],processing_seconds=timings[-1],
                    processing_deadline_missed=timings[-1]>.05,
                    publication_lateness_seconds=publication_lateness[-1] if video else None))+'\n')
            if value['sim_ns']<=last_ns or value['latest_observation_ns']>value['sim_ns']:
                raise ValueError('Replay violated historical observation order')
            if not torch.isfinite(value['state']).all():raise ValueError('Nonfinite causal state')
            last_ns=value['sim_ns'];missing_video+=not value['visual_available'];missing_geometry+=not value['metric_geometry_available']
            # Original RGB is referenced, never copied into a fabricated grid.
            value.pop('image');value.pop('goal_tokens')
            # Cache only against the immutable encoder in this replay manifest.
            # Keep exact original RGB references alongside all spatial tokens.
            value['current_tokens']=value['current_tokens'][0]
            value['goal_context']=value['goal_context'][0]
            if last_grounding_ns is None or row['sim_ns']-last_grounding_ns>=3_000_000_000:
                value['configuration_evidence']=core.grounding_context(row['sim_ns'],value['config'])
                last_grounding_ns=row['sim_ns']
            value['config']=asdict(value['config']) if value['config'] else None
            value['rgb_reference']=dict(frame_id=row['frame_id'],sha256=row['rgb_sha256'])
            value={k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in value.items()}
            chunks.append(value);count+=1
            if len(chunks)>=64:flush()
        flush()
    except Exception as error:
        import traceback
        failure=dict(type=type(error).__name__,message=str(error))
        (args.output/'failure.log').write_text(traceback.format_exc())
    finally:
        flush()
        core.close()
        for name,worker in (('video',video),('mapping',mapping),('broker',broker)):
            if worker is None:continue
            try:
                receipt=worker.close()
                if name=='video':video_receipt=receipt
                elif name=='mapping':mapping_receipt=receipt
            except Exception as error:cleanup_errors.append(dict(component=name,type=type(error).__name__,message=str(error)))
    provenance=dict(schema='causal-navigation-replay/v1',episode_id=goal.episode_id,
        checkpoint_set_sha256=checkpoints.identity,source_rgb_sha256=storage['stream_sha256'],
        goal_sha256=goal.content_sha256,runtime_source_sha256=checksum(Path(__file__).with_name('navigation_state.py')),
        observation_time_grid='original exposure timestamps; no duplicated images',
        slow_feature_mode='measured paced replay with actual asynchronous video completion' if video else 'unavailable; no retrospective output inserted into historical inputs',
        availability_clock='replay monotonic; original image exposure timestamps preserved' if video else 'original recorded monotonic',
        goal_tokens=dict(path=goal_cache.name,sha256=checksum(goal_cache)),
        video=video_receipt,mapping=mapping_receipt,shards=shards)
    (args.output/'manifest.json').write_text(json.dumps(provenance,indent=2))
    result=dict(status='failed' if failure or cleanup_errors else 'checkpointed' if Path('/output/CHECKPOINT_REQUEST').exists() else 'completed',accepted=False,
        failure=failure,cleanup_errors=cleanup_errors,
        frames=count,missing_video_frames=missing_video,missing_metric_geometry_frames=missing_geometry,
        goal_conditioned_frames=goal_conditioned_frames,alignment_reasons=alignment_reasons,
        processing_p95_seconds=float(np.quantile(timings,.95)) if timings else None,
        processing_max_seconds=max(timings) if timings else None,
        processing_deadlines_missed=sum(t>.05 for t in timings),
        publication_lateness_p95_seconds=float(np.quantile(publication_lateness,.95)) if publication_lateness else None,
        scope='shared causal fast path; slow-feature publication replay and flight acceptance remain separate')
    (args.output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    return 2 if failure or cleanup_errors else 0


if __name__=='__main__':raise SystemExit(main())
