"""Capture checked A/B photo-goal tasks without exhaustive route certification.

This explicitly scoped Mode 1 pilot does not admit the two-mode experiment.
Positions are evaluator/reset data; actor inputs remain RGB and past commands.
"""
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read,write,digest,FlightLock,Window


def candidates(rows,scene_id,seed):
    rng=np.random.default_rng(seed)
    if rows and rows[0].get('kind')=='mission':
        buckets={}
        for t in rows:buckets.setdefault((t.get('distance_bin'),t.get('difficulty')),[]).append(t)
        result=[]
        for i in range(max(map(len,buckets.values()))):
            for cell in sorted(buckets,key=str):
                if i<len(buckets[cell]):result.append(dict(buckets[cell][i]))
        return result
    # Previously checked endpoint positions are paired anew; connectivity is
    # unknown, not claimed from their old short-corridor certificates.
    poses=sorted({tuple(t[k]) for t in rows for k in ('start','goal')})
    bounds=[(np.min(poses,axis=0)-20).tolist(),(np.max(poses,axis=0)+20).tolist()]
    result=[];seen=set()
    for i in range(2000):
        a,b=rng.choice(len(poses),2,replace=False);start=list(poses[a]);goal=list(poses[b]);d=math.dist(start,goal)
        if not 40<=d<=300 or (a,b) in seen:continue
        seen.add((a,b))
        result.append(dict(id=f'{scene_id}-endpoint-{i:05d}',scene_id=scene_id,split='train',kind='mission',
            start=start,goal=goal,start_yaw_deg=float(rng.uniform(-180,180)),goal_yaw_deg=float(rng.uniform(-180,180)),
            distance_bin='40-100' if d<100 else '100-200' if d<200 else '200-300',difficulty='unclassified',
            bounds=bounds,route_feasibility='unknown',field=''))
    return result


def prepare(args):
    from .ppo_env import PilotEnvironment
    cfg=read(args.config);spec=read(args.inventory);root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    window=Window(args.hours);tasks=[];scenes=[];rejected=[]
    manifest=dict(schema='photo-map-endpoint-tasks/v1',config_sha256=digest(args.config),tasks=tasks,scenes=scenes,
                  experiment='endpoint-mode1-pilot',route_coverage_qualified=False,rejected=rejected)
    with FlightLock(Path(args.workspace)/'ppo-campaign','endpoint-capture'):
        for entry in spec['scenes']:
            scene_id=entry['scene_id'];source=read(entry['candidates']);rows=candidates(source['tasks'],scene_id,cfg['seed'])
            seed=root/(scene_id+'-seed.json');write(seed,dict(tasks=rows))
            scene={k:entry[k] for k in ('scene_id','descriptor','qualification')}
            scene.update(split='train',qualification_sha256=digest(scene['qualification']),seed_sha256=digest(seed))
            scenes.append(scene);accepted=[]
            with PilotEnvironment(read(scene['descriptor']),root/(scene_id+'-worker'),cfg) as env:
                env.calibrate()
                def capture(raw):
                    t=dict(raw);directory=root/t['id'];directory.mkdir();t['camera']=cfg['camera'];t.setdefault('field','')
                    try:
                        goal_reset=env.reset_pose(t['goal'],t['goal_yaw_deg']);rgb,stamp,_=env.image()
                        if float(rgb.std())<8:raise ValueError('Goal photograph lacks texture')
                        goal_image=directory/'goal.png';Image.fromarray(rgb).save(goal_image)
                        start_reset=env.reset_pose(t['start'],t['start_yaw_deg']);rgb,_,_=env.image()
                        start_image=directory/'start.png';Image.fromarray(rgb).save(start_image)
                        endpoint=directory/'endpoints.json'
                        write(endpoint,dict(passed=True,start=t['start'],goal=t['goal'],start_yaw_deg=t['start_yaw_deg'],
                            goal_yaw_deg=t['goal_yaw_deg'],start_reset=start_reset,goal_reset=goal_reset,
                            goal_capture_sim_ns=stamp,camera=cfg['camera'],scope='endpoints only; no route acceptance'))
                        t.update(goal_image=str(goal_image.resolve()),goal_sha256=digest(goal_image),
                            start_image=str(start_image.resolve()),endpoint_evidence=str(endpoint.resolve()),endpoint_sha256=digest(endpoint),
                            timeout_s=20 if t['kind']=='arrival' else min(300,max(120,2*math.dist(t['start'],t['goal'])/1.5+30)),
                            start_yaw_sampling='uniform_360_independent' if t['kind']=='mission' else 'arrival_positive_and_heading_nearmiss')
                        tasks.append(t);write(root/'tasks.json',manifest);return t
                    except Exception as error:
                        rejected.append(dict(id=t['id'],error=str(error)));write(directory/'failure.json',rejected[-1])
                        write(root/'tasks.json',manifest);return None
                for raw in rows[:80]:
                    if not window.remaining() or len(accepted)>=16:break
                    result=capture(raw)
                    if result:accepted.append(result)
                for i,t in enumerate(accepted[:4]):
                    if not window.remaining():break
                    arrival=dict(t,id=t['id']+'-arrival',kind='arrival',start=list(t['goal']),
                        start_yaw_deg=(t['goal_yaw_deg']+(0 if i%2==0 else 90)+180)%360-180,
                        difficulty='arrival_positive' if i%2==0 else 'arrival_heading_nearmiss',distance_bin='arrival')
                    capture(arrival)
    write(root/'tasks.json',manifest)
    validate_endpoint_manifest(manifest,cfg,args.config)
    write(root/'ready.json',dict(endpoint_pilot_ready=True,task_count=len(tasks),mode2_enabled=False,
        navigation_accepted=False,manifest_sha256=digest(root/'tasks.json')))


def validate_endpoint_manifest(manifest,cfg,config_path):
    if manifest.get('schema')!='photo-map-endpoint-tasks/v1' or manifest.get('config_sha256')!=digest(config_path):
        raise ValueError('Explicit endpoint pilot manifest/config required')
    if not cfg.get('endpoint_pilot') or cfg.get('mode2_enabled') or cfg.get('training_pause'):
        raise ValueError('Endpoint pilot is Mode 1 with explicit boundary truncations')
    scenes={s['scene_id']:s for s in manifest['scenes']}
    if len(scenes)<2:raise ValueError('Two previously qualified training scenes required')
    for s in scenes.values():
        q=read(s['qualification'])
        if digest(s['qualification'])!=s['qualification_sha256'] or not q.get('qualified'):
            raise ValueError('Changed/incomplete scene qualification')
        if not all(q.get('checks',{}).get(k) for k in ('camera','geometry','motion','collision','stop','stale_frame','timing')):
            raise ValueError('Physical simulator semantics must already be qualified')
        for key in ('camera','reset','arrival','limits','freshness_s','step_s'):
            if q['config'][key]!=cfg[key]:raise ValueError('Physical settings differ: '+key)
        if q['scene_sha256']!=digest(s['descriptor']):raise ValueError('Scene descriptor changed')
        for asset in read(s['descriptor'])['assets']:
            if digest(asset['path'])!=asset['sha256']:raise ValueError('Scene assets changed')
        selected=[t for t in manifest['tasks'] if t['scene_id']==s['scene_id']]
        if sum(t['kind']=='mission' for t in selected)<8 or sum(t['kind']=='arrival' for t in selected)<2:
            raise ValueError('Too few checked mission/arrival tasks: '+s['scene_id'])
    ids=set()
    for t in manifest['tasks']:
        if t['id'] in ids or t['scene_id'] not in scenes or t['split']!='train':raise ValueError('Invalid task identity/split')
        ids.add(t['id'])
        if t['camera']!=cfg['camera'] or digest(t['goal_image'])!=t['goal_sha256'] or digest(t['endpoint_evidence'])!=t['endpoint_sha256']:
            raise ValueError('Endpoint evidence changed')
        e=read(t['endpoint_evidence'])
        if not e['passed'] or any(e[k]!=t[k] for k in ('start','goal','start_yaw_deg','goal_yaw_deg','camera')):
            raise ValueError('Endpoint mismatch')
        for k in ('start_reset','goal_reset'):
            if not e[k][-1]['passed']:raise ValueError('Endpoint reset failed')
        if t['kind']=='mission' and not 40<=math.dist(t['start'],t['goal'])<=300:raise ValueError('Mission distance outside pilot range')
        if t['kind'] not in ('mission','arrival'):raise ValueError('Invalid endpoint pilot stream')
        if not 0<t['timeout_s']<=cfg['episode_s']:raise ValueError('Task timeout outside configuration')
    return scenes


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','inventory','output','workspace'):p.add_argument('--'+name,required=True)
    p.add_argument('--hours',type=float,default=.25)
    prepare(p.parse_args())
