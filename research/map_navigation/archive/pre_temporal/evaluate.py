"""Summaries of real recorded outcomes, including incomplete/failed matrices."""
import math
from pathlib import Path
import numpy as np
from .common import config,read,write
from .data import lines,yaw


def wilson(success,total):
    if not total:return None
    z=1.96;p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [center-radius,center+radius]


def report(manifest,results,output):
    missions=read(manifest)['episodes'];by_id={r['episode_id']:r for r in missions}
    cfg=config();rows={};localization={}
    for path in Path(results).rglob('episode/result.json'):
        row=read(path);key=(row.get('variant'),row.get('seed'),row['episode_id'])
        if row['episode_id'] not in by_id or row.get('variant') not in cfg['variants']:continue
        if row.get('seed') not in cfg['evaluation']['seeds']:raise ValueError('Unexpected evaluation seed')
        if key in rows:raise ValueError('Duplicate trial '+str(key))
        if row.get('clock_speed')!=1.:raise ValueError('Evaluation requires real-time simulator clock')
        if row['scene_id']!=by_id[row['episode_id']]['scene_id']:raise ValueError('Wrong evaluation environment')
        rows[key]=row
        decision_path=path.parent/'photo-controller/runtime/decisions.jsonl'
        label_path=path.parent/'training_labels/frames.jsonl'
        if decision_path.exists() and label_path.exists():
            labels=lines(label_path);errors=[];false_stops=0
            goal=np.asarray(read(path.parent/'evaluator_labels/episode.json')['goal_ned_m'])
            by_frame={r['frame_id']:r for r in labels if abs(r['observation_label_skew_seconds'])<=.1}
            for decision in lines(decision_path):
                label=by_frame.get(decision['frame_id'])
                if label is None:continue
                hypotheses=decision['belief']['hypotheses']
                if hypotheses and decision['belief']['aligned']:
                    errors.append(float(np.linalg.norm(np.asarray(hypotheses[0]['position'])-label['true_position_ned_m'])))
                position=np.asarray(label['true_position_ned_m'])
                false_stops+=int(decision['stop'] and (np.linalg.norm(position[:2]-goal[:2])>3 or abs(position[2]-goal[2])>2))
            localization[str(key)]=dict(aligned_position_errors_m=errors,false_stop_decisions=false_stops,
                false_stop_scope='planned decisions; includes decisions rejected as stale before dispatch')
    expected={(v,s,e) for v in cfg['variants'] for s in cfg['evaluation']['seeds'] for e in by_id}
    summary={}
    for variant in cfg['variants']:
        group=[r for (v,_,_),r in rows.items() if v==variant];n=len(group)
        successes=[r for r in group if r.get('success')];collisions=sum(r.get('termination') in ('collision','geometry_collision') for r in group)
        times=[r['elapsed_sim_seconds'] for r in successes if 'elapsed_sim_seconds' in r]
        errors=[r['final_navigation_error_m'] for r in group if 'final_navigation_error_m' in r]
        summary[variant]=dict(trials=n,success_rate=len(successes)/n if n else None,success_ci95=wilson(len(successes),n),
            collision_rate=collisions/n if n else None,collision_ci95=wilson(collisions,n),
            median_success_time_s=float(np.median(times)) if times else None,median_final_error_m=float(np.median(errors)) if errors else None,
            meets_operational_targets=bool(n==len(by_id)*len(cfg['evaluation']['seeds']) and n and len(successes)/n>=.9 and collisions/n<=.01),
            terminations={reason:sum(r.get('termination')==reason for r in group) for reason in sorted({r.get('termination','unknown') for r in group})})
    strata={}
    for variant in cfg['variants']:
        for category in ('overflight_better','low_better','comparable'):
            group=[r for (v,_,_),r in rows.items() if v==variant and r.get('route_advantage')==category]
            strata[variant+'/'+category]=dict(trials=len(group),successes=sum(bool(r.get('success')) for r in group))
    paired={};rng=np.random.default_rng(0)
    for reference,variant in [('geometry','geometric_candidates'),('geometric_candidates','predictive_candidates')]:
        differences=[]
        for seed in cfg['evaluation']['seeds']:
            for ident in by_id:
                a=rows.get((reference,seed,ident));b=rows.get((variant,seed,ident))
                if a and b:differences.append(int(bool(b.get('success')))-int(bool(a.get('success'))))
        if differences:
            x=np.asarray(differences);bootstrap=[float(rng.choice(x,len(x),replace=True).mean()) for _ in range(2000)]
            paired[reference+'->'+variant]=dict(pairs=len(x),success_difference=float(x.mean()),ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
                                caveat='paired trial bootstrap; does not estimate uncertainty across all possible cities')
    missing=sorted(expected-set(rows))
    write(output,dict(schema='photo-map-evaluation/v2',complete=not missing,missing=missing,variants=summary,paired=paired,
         localization=localization,strata=strata,all_failures_retained=True,energy={'compute':'not measured by this evaluator','propulsion':'not measured; time is not energy'},
         generalization_scope='registered held-out scenes only; no real-satellite or physical-flight claim'))

