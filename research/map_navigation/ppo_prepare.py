"""Offline PPO task construction from observed geometry, never an action teacher.

Input inventory binds reviewed scene geography and existing depth-survey fields.
Goal images are independently captured with the final mission camera calibration.
"""
import argparse
import json
import math
from pathlib import Path
import sqlite3
import zipfile
import numpy as np
from PIL import Image
from ..rgb_flight.obstacle_field import PrivilegedObstacleField
from .common import read,write,digest,Window
from .ppo_env import PilotEnvironment


class CameraEnvelopeField(PrivilegedObstacleField):
    """Conservative orientation-independent bound including forward camera."""
    mount_radius=.75

    @property
    def collision_radii(self):return np.full(3,max(.75,self.mount_radius)+.25)

    def swept_collision(self,start,end):
        if super().swept_collision(start,end):return True
        # Require observed support around the body too, not just along its centre.
        radius=float(self.collision_radii[0]);spacing=min(.5,self.free_resolution/2)
        axis=np.linspace(-radius,radius,int(math.ceil(2*radius/spacing))+1)
        offsets=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
        centres=np.linspace(start,end,max(2,int(math.ceil(math.dist(start,end)/spacing))+1))
        samples=(centres[:,None,:]+offsets[None,:,:]).reshape(-1,3)
        return bool(np.any(samples<self.bounds[0]) or np.any(samples>self.bounds[1]) or not self._known_free(samples).all())


def candidates(episodes,scene_id,field,limit):
    seen=set();buckets={k:[] for k in ('level','climb','descent')}
    for row in sorted(episodes,key=lambda x:str(x['trajectory_id'])):
        if str(row['scene_id'])!=str(scene_id).replace('env_','') or row['trajectory_id'] in seen:continue
        seen.add(row['trajectory_id']);path=row['reference_path']
        for i in range(0,len(path),3):
            start=np.asarray(path[i][:3],float)
            for j in range(i+1,min(i+30,len(path))):
                goal=np.asarray(path[j][:3],float);delta=goal-start;distance=float(np.linalg.norm(delta))
                if not 10<=distance<=30 or np.linalg.norm(delta[:2])<8:continue
                behavior='level' if abs(delta[2])<1 else 'climb' if delta[2]<-1 else 'descent'
                if len(buckets[behavior])>=limit:continue
                if field.swept_collision(start,goal):continue
                yaw=math.degrees(math.atan2(delta[1],delta[0]))
                ident=f'env{scene_id}-{row["trajectory_id"]}-{i}-{j}'
                buckets[behavior].append(dict(id=ident,start=start.tolist(),goal=goal.tolist(),
                    behavior=behavior,start_yaw_deg=yaw,goal_yaw_deg=yaw,reference_id=row['trajectory_id']))
                break
        if all(len(v)>=limit for v in buckets.values()):break
    return buckets


def reconcile(ledger,output,engineering_roots=()):
    """Read-only reconciliation of the existing engineering attempt database."""
    uri=Path(ledger).resolve().as_uri()+'?mode=ro'
    with sqlite3.connect(uri,uri=True) as db:
        rows=db.execute("SELECT stream,status,COUNT(*) FROM attempts WHERE split='train' GROUP BY stream,status").fetchall()
    # This helper only recognizes the pre-PPO reference ledger, not arbitrary
    # training sources. Refuse to silently treat another stream as zero PPO use.
    if any(stream!='reference' for stream,_,_ in rows):raise ValueError('Mixed ledger requires broader reconciliation')
    engineering={}
    for root in engineering_roots:
        for path in Path(root).rglob('telemetry.jsonl'):
            # Each recording starts only after reset qualification, including
            # failed physical checks. Keep the source evidence and count once.
            first=path.open(encoding='utf-8-sig').readline()
            if first and json.loads(first).get('kind')=='reset':engineering[str(path.resolve())]=digest(path)
    write(output,dict(schema='photo-map-pilot-budget/v1',source=str(Path(ledger).resolve()),
        source_sha256=digest(ledger),rows=rows,engineering_recordings=engineering,
        training_attempts=sum(n for _,_,n in rows)+len(engineering),
        learner_attempts=0,ppo_transitions=0))


def assemble(inventory,budget,output):
    """Seal reviewed measured-task receipts without rewriting raw qualification."""
    cfgpath=Path(__file__).with_name('ppo_pilot.json');cfg=read(cfgpath)
    root=Path(output);root.mkdir(parents=True,exist_ok=False);tasks=[];scenes=[]
    for entry in read(inventory)['scenes']:
        q=read(entry['qualification']);review=read(entry['review']);candidate=read(entry['tasks'])
        if review['qualification_sha256']!=digest(entry['qualification']):raise ValueError('Review is for different qualification')
        if not review.get('camera_clear') or not review.get('geography_reviewed') or not review.get('notes'):
            raise ValueError('Explicit camera/geography review required')
        if q.get('semantic_error') or not all(q.get('checks',{}).get(k) for k in ('camera','geometry','motion','collision','stop','stale_frame','timing')):
            raise ValueError('Live qualification checks failed')
        if q['tasks_sha256']!=digest(entry['tasks']) or q['config']!=cfg:raise ValueError('Task/config provenance differs')
        if q['scene_sha256']!=digest(entry['descriptor']):raise ValueError('Descriptor differs')
        scene_id=read(entry['descriptor'])['scene_id'];qualified=root/(scene_id+'-qualified.json')
        q.update(qualified=True,camera_reviewed=True,geometry_reviewed=True,pending=None,
            review=str(Path(entry['review']).resolve()),review_sha256=digest(entry['review']),
            raw_qualification=str(Path(entry['qualification']).resolve()),raw_qualification_sha256=digest(entry['qualification']))
        write(qualified,q)
        scenes.append(dict(scene_id=scene_id,split=entry['split'],geography_id=review['geography_id'],
            descriptor=str(Path(entry['descriptor']).resolve()),qualification=str(qualified.resolve()),qualification_sha256=digest(qualified)))
        for task in candidate['tasks']:
            if task['scene_id']!=scene_id or task['split']!=entry['split']:raise ValueError('Candidate split mismatch')
            tasks.append(task)
    write(root/'tasks.json',dict(schema='photo-map-ppo-tasks/v1',config_sha256=digest(cfgpath),
        scenes=scenes,tasks=tasks,inventory_sha256=digest(inventory),
        prior_budget_usage=dict(receipt=str(Path(budget).resolve()),sha256=digest(budget)),
        scope='small integration curriculum; not sufficient for generalization claims'))


def prepare(inventory,annotations,output,budget,limit,hours):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    cfgpath=Path(__file__).with_name('ppo_pilot.json');cfg=read(cfgpath);window=Window(hours)
    spec=read(inventory)
    if spec.get('schema')!='photo-map-ppo-inventory/v1':raise ValueError('Reviewed PPO inventory required')
    tasks=[];scene_rows=[]
    with zipfile.ZipFile(annotations) as archive:
        by_split={split:json.loads(archive.read(name))['episodes'] for split,name in [('train','train.json'),('validation','val_unseen.json')]}
    try:
        for scene in spec['scenes']:
            if not window.remaining():break
            if not scene.get('geography_reviewed') or not scene.get('geography_id'):raise ValueError('Review geography before task construction')
            field=CameraEnvelopeField.load(scene['field'])
            field.mount_radius=float(np.linalg.norm([cfg['camera'][k] for k in ('x','y','z')]))
            if field.observed_free is None:raise ValueError('Endpoint-only depth field cannot establish free corridors')
            split=scene['split']
            if split not in by_split:raise ValueError('Sealed test scenes cannot prepare pilot tasks')
            groups=candidates(by_split[split],scene['scene_id'],field,limit)
            if not all(groups.values()):raise ValueError('Missing observed level/climb/descent coverage: '+str(scene['scene_id']))
            descriptor=read(scene['descriptor'])
            if str(descriptor['scene_id'])!=str(scene['scene_id']):raise ValueError('Inventory scene identity differs')
            scene_rows.append({k:scene[k] for k in ('scene_id','split','geography_id','descriptor','qualification','qualification_sha256')})
            with PilotEnvironment(descriptor,root/('capture-'+str(scene['scene_id'])),cfg) as env:
                for behavior in ('level','climb','descent'):
                    for task in groups[behavior]:
                        if not window.remaining():break
                        # Independent initialization failures are recorded, not
                        # converted into successful sample labels.
                        try:
                            reset=env.reset_pose(task['goal'],task['goal_yaw_deg']);env.calibrate()
                            rgb,stamp,_=env.image()
                            goal_path=root/(task['id']+'-goal.png');Image.fromarray(rgb).save(goal_path)
                            evidence=root/(task['id']+'-geometry.json')
                            write(evidence,dict(task_id=task['id'],start=task['start'],goal=task['goal'],camera=cfg['camera'],swept_volume_free=True,
                                source_field=str(Path(scene['field']).resolve()),source_field_sha256=digest(scene['field']),
                                clearance_model='observed sparse field, conservative body-and-camera sphere',
                                radius_m=float(field.collision_radii[0]),resolution_m=field.resolution,
                                goal_capture_sim_ns=stamp,goal_reset=reset))
                            task.update(scene_id=scene['scene_id'],split=split,bounds=field.bounds.tolist(),
                                camera=cfg['camera'],unobstructed=True,goal_image=str(goal_path.resolve()),
                                goal_sha256=digest(goal_path),geometry_evidence=str(evidence.resolve()),geometry_evidence_sha256=digest(evidence))
                            tasks.append(task)
                        except Exception as error:
                            write(root/(task['id']+'-failure.json'),dict(error=str(error),training_eligible=False))
    finally:
        write(root/'tasks.json',dict(schema='photo-map-ppo-tasks/v1',config_sha256=digest(cfgpath),
            inventory_sha256=digest(inventory),annotations_sha256=digest(annotations),scenes=scene_rows,tasks=tasks,
            prior_budget_usage=dict(receipt=str(Path(budget).resolve()),sha256=digest(budget)),
            generated_task_count=len(tasks),training_qualified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('budget');a.add_argument('--ledger',required=True);a.add_argument('--output',required=True)
    a.add_argument('--engineering-root',action='append',default=[],help='Prior training-scene engineering recordings to count conservatively')
    a=sub.add_parser('assemble');a.add_argument('--inventory',required=True);a.add_argument('--budget',required=True);a.add_argument('--output',required=True)
    a=sub.add_parser('tasks');a.add_argument('--inventory',required=True);a.add_argument('--annotations',required=True)
    a.add_argument('--output',required=True);a.add_argument('--budget',required=True);a.add_argument('--per-behavior',type=int,default=20);a.add_argument('--hours',type=float,default=8)
    a=p.parse_args()
    if a.stage=='budget':reconcile(a.ledger,a.output,a.engineering_root)
    elif a.stage=='assemble':assemble(a.inventory,a.budget,a.output)
    else:
        if a.per_behavior<1:p.error('Positive per-behavior count required')
        prepare(a.inventory,a.annotations,a.output,a.budget,a.per_behavior,a.hours)
