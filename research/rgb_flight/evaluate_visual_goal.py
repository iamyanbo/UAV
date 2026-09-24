"""Summarize sealed complete-flight results without dropping failures."""
import argparse
import json
import math
from pathlib import Path
import statistics


VARIANTS=('mode_1_only','mode_1_plus_geometry_planner','jepa_without_gaussian_history',
          'jepa_with_episode_gaussian_memory','fixed_qwen_configuration',
          'trained_qwen_configuration','privileged_expert_upper_bound')
SEEDS=(0,1,2)


def summarize(result_files, sealed_manifest, mechanism_evidence):
    manifest=json.loads(Path(sealed_manifest).read_text())
    episodes={row['episode_id'] for row in manifest['episodes']}
    if len(episodes)!=200 or any(row['split']!='test' for row in manifest['episodes']):
        raise ValueError('Expected the sealed 200-episode runtime manifest')
    rows=[json.loads(Path(path).read_text()) for path in result_files]
    keys=[]
    for row in rows:
        if row.get('clock_speed')!=1. or row.get('episode_id') not in episodes or row.get('variant') not in VARIANTS or row.get('seed') not in SEEDS:
            raise ValueError('Result is outside the sealed ClockSpeed=1 evaluation')
        if row.get('termination') not in ('success','collision','geometry_collision','timeout','boundary_exit','tracking_loss'):
            raise ValueError('Every result needs an explicit retained termination')
        keys.append((row['variant'],row['seed'],row['episode_id']))
    if len(keys)!=len(set(keys)):
        raise ValueError('Repeated sealed trial')
    expected={(variant,seed,episode) for variant in VARIANTS for seed in SEEDS for episode in episodes}
    missing=expected-set(keys)
    metrics={}
    for variant in VARIANTS:
        group=[row for row in rows if row['variant']==variant]
        count=len(group)
        success=[row for row in group if row['termination']=='success' and row.get('success')]
        collisions=[row for row in group if row['termination'] in ('collision','geometry_collision')]
        metrics[variant]=dict(trials=count,collision_free_success=len(success)/count if count else None,
            collision_rate=len(collisions)/count if count else None,
            median_completion_time_seconds=statistics.median(row['elapsed_sim_seconds'] for row in success) if success else None,
            median_path_efficiency=statistics.median(row['path_efficiency'] for row in success) if success else None,
            median_final_error_m=statistics.median(row['final_navigation_error_m'] for row in group) if group else None,
            median_traveled_distance_m=statistics.median(row['traveled_distance_m'] for row in group) if group else None,
            p95_control_latency_seconds=(sorted(row['control_latency_p95_seconds'] for row in group)[math.ceil(.95*count)-1] if group else None),
            mean_actual_velocity_mps=(statistics.fmean(row['mean_actual_velocity_mps'] for row in group) if group else None))
        value=metrics[variant]
        value['qualifies_for_time_score']=bool(value['collision_free_success'] is not None and
                                               value['collision_free_success']>=.9 and value['collision_rate']<=.01)
    mechanisms=json.loads(Path(mechanism_evidence).read_text())
    required=('candidate_actions_different_futures','jepa_changes_action_ranking','qwen_changes_plan',
              'episode_memory_helps_revisits','runtime_privilege_isolation')
    if any(not mechanisms.get(key,{}).get('raw_evidence') for key in required):
        raise ValueError('Mechanism claims require raw evidence paths')
    scored={key:value['median_completion_time_seconds'] for key,value in metrics.items() if value['qualifies_for_time_score']}
    return dict(status='complete' if not missing else 'incomplete',missing_trials=len(missing),
                primary_score='median simulated completion time among threshold-qualified variants',
                ranking=sorted(scored,key=scored.get),metrics=metrics,mechanisms=mechanisms,
                all_failures_retained=len(rows)==len(keys))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,nargs='+',required=True)
    parser.add_argument('--sealed-manifest',type=Path,required=True)
    parser.add_argument('--mechanism-evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(args.results,args.sealed_manifest,args.mechanism_evidence)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(status=result['status'],missing_trials=result['missing_trials'],ranking=result['ranking'])))


if __name__=='__main__':
    main()
