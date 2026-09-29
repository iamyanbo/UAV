"""Pilot/concurrency receipts from actual recorded flights, never GPU occupancy."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from .common import read,write,digest
from .annotations import lines


def report(flights,output,wall_seconds,workers,registry=None,resources=None,project_workers=None):
    if wall_seconds<=0 or workers<1:raise ValueError('Measured wall duration and worker count required')
    resource=read(resources) if resources else None
    if resource and (resource['workers']!=workers or not registry or resource['registry_sha256']!=digest(registry)):
        raise ValueError('Resource receipt has different workers or registry')
    paths=sorted(Path(flights).rglob('episode/result.json'))
    if resource:
        identities={r['path']:r['sha256'] for r in resource['results']}
        paths=[p for p in paths if str(p.resolve()) in identities]
        if len(paths)!=len(identities) or any(digest(p)!=identities[str(p.resolve())] for p in paths):raise ValueError('Resource receipt results missing or changed')
        if abs(resource['wall_seconds']-wall_seconds)>1:raise ValueError('Use wall duration from resource receipt')
    attempts=[];latencies=[];good_seconds=0.;recorded_seconds=0.;sizes=[];skews=[]
    for path in paths:
        r=read(path);episode=path.parent
        frames=lines(episode/'training_labels/frames.jsonl');commands=lines(episode/'training_labels/commands.jsonl')
        annotations=lines(episode/'training_labels/annotations.jsonl')
        span=(frames[-1]['state_sim_ns']-frames[0]['state_sim_ns'])/1e9 if len(frames)>1 else 0
        valid=[abs(f['observation_label_skew_seconds'])<=.0125 for f in frames]
        actual=[c['source_received_to_command_wall_seconds'] for c in commands if c.get('source_received_to_command_wall_seconds') is not None]
        wall_span=commands[-1]['submitted_monotonic']-commands[0]['submitted_monotonic'] if len(commands)>1 else 0
        ratio=(commands[-1]['dispatch_sim_ns']-commands[0]['dispatch_sim_ns'])/1e9/wall_span if wall_span>0 else 0
        latencies.extend(actual);skews.extend(valid)
        accepted=bool(span>0 and actual and np.quantile(actual,.95)<=.05 and np.quantile(actual,.99)<=.25
                      and valid and sum(valid)/len(valid)>=.99 and r.get('clock_speed')==1.
                      and .95<=ratio<=1.05 and r.get('timing_gate',{}).get('capture_control_passed'))
        good_seconds+=span if accepted else 0;recorded_seconds+=span
        size=sum(p.stat().st_size for p in episode.rglob('*') if p.is_file());sizes.append(size)
        attempts.append(dict(path=str(path),sha256=digest(path),scene=r.get('scene_id'),stream=r.get('collection_stream'),
            curriculum=r.get('curriculum'),termination=r.get('termination',r.get('status')),seconds=span,
            bytes=size,timing_accepted=accepted,sim_wall_ratio=ratio,annotated_frames=len(annotations),success=bool(r.get('success'))))
    throughput=good_seconds/wall_seconds
    mean_seconds=recorded_seconds/len(attempts) if attempts else None
    result=dict(schema='photo-map-pilot/v1',workers=workers,wall_seconds=wall_seconds,
        attempts=attempts,by_scene=dict(Counter(r['scene'] for r in attempts)),by_curriculum=dict(Counter(r['curriculum'] for r in attempts)),
        failure_taxonomy=dict(Counter(r['termination'] for r in attempts)),valid_flight_seconds=good_seconds,
        valid_flight_seconds_per_wall_second=throughput,recorded_seconds=recorded_seconds,
        recording_bytes=sum(sizes),bytes_per_sim_hour=sum(sizes)/recorded_seconds*3600 if recorded_seconds else None,
        estimated_10000_flight_wall_hours=(10000*mean_seconds/throughput/3600) if throughput and mean_seconds else None,
        estimates_exclude_training_and_evaluation=True,
        p95_source_to_dispatch_s=float(np.quantile(latencies,.95)) if latencies else None,
        p99_source_to_dispatch_s=float(np.quantile(latencies,.99)) if latencies else None,
        timing_valid_fraction=sum(skews)/len(skews) if skews else 0.,clock_speed=1.,
        qualified=bool(attempts and all(r['timing_accepted'] for r in attempts)),
        registry_sha256=digest(registry) if registry else None)
    write(output,result)
    if resource:
        profile=dict(resource,qualified=bool(result['qualified'] and resource['measured'] and not resource['interrupted']),
                     measurement_workers=workers,report_sha256=digest(output))
        if project_workers is not None:
            if not workers<project_workers<=2*workers:raise ValueError('Project at most a doubling for the next engineering benchmark')
            factor=project_workers/workers
            for key in ('incremental_peak_bytes','transient_peak_bytes','recording_peak_bytes'):profile[key]=int(profile[key]*factor)
            profile.update(workers=project_workers,qualified=False,projected_from_measured=True)
        write(Path(output).with_suffix('.profile.json'),profile)


def main():
    p=argparse.ArgumentParser();p.add_argument('--flights',required=True);p.add_argument('--output',required=True)
    p.add_argument('--wall-seconds',type=float,required=True);p.add_argument('--workers',type=int,required=True);p.add_argument('--registry')
    p.add_argument('--resources');p.add_argument('--project-workers',type=int)
    a=p.parse_args();report(a.flights,a.output,a.wall_seconds,a.workers,a.registry,a.resources,a.project_workers)

if __name__=='__main__':main()
