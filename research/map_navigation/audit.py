"""Deferred data coverage reporting, not fabricated flight acceptance."""
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from .common import read,write


def altitude_plot(path, profiles, title):
    """Dependency-free SVG artifact from actual supplied trajectory samples."""
    from html import escape
    usable=[(name,np.asarray(points,float)) for name,points in profiles if len(points)>1]
    if not usable:return
    distances=[np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))] for _,p in usable]
    xmax=max(float(d[-1]) for d in distances) or 1.
    heights=np.concatenate([-p[:,2] for _,p in usable]);low=float(heights.min());high=max(low+1,float(heights.max()))
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="960" height="460" viewBox="0 0 960 460">',
           '<rect width="960" height="460" fill="white"/>',
           f'<text x="60" y="28" font-family="sans-serif" font-size="18">{escape(title)}</text>',
           '<path d="M60 60V390H900" fill="none" stroke="black"/>',
           '<text x="360" y="440" font-family="sans-serif">Traveled 3D distance (m)</text>',
           '<text x="65" y="52" font-family="sans-serif">Height above NED origin (m), not AGL</text>']
    colors=('#2563eb','#c2410c','#15803d','#7e22ce')
    for i,((name,points),distance) in enumerate(zip(usable,distances)):
        xy=np.c_[60+840*distance/xmax,390-320*(-points[:,2]-low)/(high-low)]
        line=' '.join(f'{x:.1f},{y:.1f}' for x,y in xy)
        parts.append(f'<polyline points="{line}" fill="none" stroke="{colors[i%4]}" stroke-width="2"/>')
        parts.append(f'<text x="{80+200*(i%4)}" y="{410-16*(i//4)}" fill="{colors[i%4]}" font-family="sans-serif" font-size="12">{escape(name)}</text>')
    parts.extend([f'<text x="65" y="385">{low:.1f}</text>',f'<text x="65" y="78">{high:.1f}</text>',
                  f'<text x="855" y="407">{xmax:.1f}</text>','</svg>'])
    Path(path).write_text('\n'.join(parts),encoding='utf-8')


def audit_dataset(dataset, output):
    spec=read(dataset);by_scene=defaultdict(lambda:dict(attempt_sources=Counter(),windows_by_phase=Counter(),
        camera_pitch_bins=Counter(),height_bins=Counter(),world_windows=0,collision_windows=0,positive_episodes=set()))
    for ep in spec['episodes']:
        by_scene[ep['scene_id']]['attempt_sources'][ep['collection_source']]+=1
    for row in spec['windows']:
        ep=spec['episodes'][row['episode']];entry=by_scene[ep['scene_id']]
        entry['windows_by_phase'][row.get('flight_phase','unlabelled')]+=1
        entry['camera_pitch_bins'][str(round(row['camera_pitch_deg']/15)*15)]+=1
        entry['height_bins'][str(int(np.floor(row['labels']['above_surface']/10)*10))]+=1
        entry['world_windows']+=int(bool(row['future']) and row.get('observation_schema')=='observation-context/v2')
        entry['collision_windows']+=int(any(row['labels']['future_collision']))
        if row['labels']['arrival']:entry['positive_episodes'].add(ep['path'])
    for entry in by_scene.values():entry['positive_episodes']=len(entry['positive_episodes'])
    banks=Counter(r['scene_id'] for r in spec.get('bank_records',[]))
    missing=[]
    for scene,entry in by_scene.items():
        for phase in ('climb','descent','scan','settle'):
            if not entry['windows_by_phase'][phase]:missing.append(dict(scene=scene,missing=phase))
        if not entry['world_windows']:missing.append(dict(scene=scene,missing='recorded_or_replayed_runtime_estimates'))
        if not entry['positive_episodes']:missing.append(dict(scene=scene,missing='positive_arrival_episodes'))
        if not entry['collision_windows']:missing.append(dict(scene=scene,missing='collision_supervision; risk calibration unestablished'))
    write(output,dict(schema='aerial-data-audit/v1',scenes=dict(by_scene),bank_images=dict(banks),missing=missing,
        connection_quality=spec.get('data_quality',{}),
        excluded_attempts=spec.get('audit',[]),accepted=False,
        caveat='Coverage counts are not navigation effectiveness or statistical independence.'))


def flight_evidence(results, output):
    """Aggregate real attempt records, preserving failures and missing telemetry."""
    rows=[];plot_root=Path(output).parent/(Path(output).stem+'-profiles');plot_root.mkdir(parents=True,exist_ok=True)
    for path in Path(results).rglob('episode/result.json'):
        result=read(path);decision_path=path.parent/'photo-controller/runtime/decisions.jsonl'
        decisions=[__import__('json').loads(x) for x in decision_path.read_text().splitlines()] if decision_path.exists() else []
        commands_path=path.parent/'training_labels/commands.jsonl'
        commands=[__import__('json').loads(x) for x in commands_path.read_text().splitlines()] if commands_path.exists() else []
        labels_path=path.parent/'training_labels/frames.jsonl'
        labels=[__import__('json').loads(x) for x in labels_path.read_text().splitlines()] if labels_path.exists() else []
        positions=np.asarray([r['true_position_ned_m'] for r in labels])
        travel=np.diff(positions,axis=0) if len(positions)>1 else np.zeros((0,3))
        profile=plot_root/(path.parent.parent.name+'.svg')
        if len(positions)>1:altitude_plot(profile,[('recorded flight',positions)],'Recorded altitude profile: '+result['episode_id'])
        latencies=[r['source_received_to_command_wall_seconds'] for r in commands if r.get('source_received_to_command_wall_seconds') is not None]
        phases=defaultdict(float)
        for a,b in zip(decisions,decisions[1:]):
            phases[a.get('flight_phase','unlabelled')]+=max(0.,b['observed_s']-a['observed_s'])
        rows.append(dict(episode_id=result['episode_id'],variant=result.get('variant'),seed=result.get('seed'),
            source=result.get('collection_source'),success=bool(result.get('success')),termination=result.get('termination',result.get('status')),
            horizontal_travel_m=float(np.linalg.norm(travel[:,:2],axis=1).sum()),vertical_travel_m=float(np.abs(travel[:,2]).sum()),
            altitude_profile=str(profile) if len(positions)>1 else None,
            route_advantage=result.get('route_advantage'),phase_seconds=dict(phases),
            selected_candidates=dict(Counter(r['selected_candidate'] for r in decisions if 'selected_candidate' in r)),
            dispatched_candidates=dict(Counter(r['candidate_id'] for r in commands if r.get('candidate_id'))),
            prediction_uses=sum(bool(r.get('prediction_used')) for r in decisions),
            prediction_discards=dict(Counter(r['last_prediction_discard'] for r in decisions if r.get('last_prediction_discard'))),
            camera_to_dispatch_seconds=None if not latencies else dict(zip(('p50','p95','p99'),np.quantile(latencies,[.5,.95,.99]).tolist())),
            timing_scope='source RGB receipt to dispatch; not actuator application',
            missing_decisions=not bool(decisions),missing_commands=not bool(commands)))
    write(output,dict(schema='aerial-flight-evidence/v1',attempts=rows,accepted=False))
