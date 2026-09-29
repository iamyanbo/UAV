"""Offline survey, endpoint capture, and representative physical qualification.

This privileged controller is ONLY an engineering check. Its commands cannot
be imported as learner rollouts or an observation-grounded search teacher.
"""
import argparse
from collections import Counter
import math
from pathlib import Path
import uuid
import numpy as np
from PIL import Image
from .common import read,write,digest,Window,FlightLock
from .ppo_env import PilotEnvironment
from .ppo_bootstrap import capture_depth
from .ppo_actions import command_from_latent
from .ppo_tasks import ObservedVolume,nominal_time
from .ppo_geometry import save_costs
from .ppo_budget import Budget


def survey(args):
    cfg=read(args.config);spec=read(args.plan);root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    window=Window(args.hours);receipts=[]
    with FlightLock(Path(args.workspace)/'ppo-campaign','ppo-survey'),PilotEnvironment(read(args.scene),root/'worker',cfg,args.port) as env:
        env.calibrate()
        for i,capture in enumerate(spec['captures']):
            if not window.remaining():break
            try:
                _,record=capture_depth(env,capture['position'],capture['yaw_deg'],root,f'capture-{i:05d}')
                receipts.append(str((root/f'capture-{i:05d}.json').resolve()))
            except Exception as error:write(root/f'capture-{i:05d}-failure.json',dict(error=str(error)))
            write(root/'receipts.json',dict(receipts=receipts,scene_sha256=digest(args.scene),plan_sha256=digest(args.plan),
                complete=len(receipts)==len(spec['captures']),fixed_camera=True,offline_privileged_only=True))


def route_flight(env,task,root,budget):
    ident='engineering-'+uuid.uuid4().hex
    if task['split']=='train':budget.attempt(ident,-1,kind='engineering')
    obs=env.reset(task,ident);route=task['reference_path'];index=1;rows=[];passed=False
    while obs['elapsed_s']<task['timeout_s']:
        state=obs['state'];target=np.asarray(route[index]);delta=target-state['position']
        if np.linalg.norm(delta)<1.5 and index<len(route)-1:index+=1;target=np.asarray(route[index]);delta=target-state['position']
        yaw=math.radians(state['attitude_deg'][2]);c,s=math.cos(yaw),math.sin(yaw)
        world=.7*delta;world[:2]/=max(1,np.linalg.norm(world[:2])/1.5);world[2]=np.clip(world[2],-.5,.5)
        final=index==len(route)-1 and np.linalg.norm(delta)<2
        desired_yaw=task['goal_yaw_deg'] if final else math.degrees(math.atan2(delta[1],delta[0]))
        error=(desired_yaw-state['attitude_deg'][2]+180)%360-180
        target=np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1],world[2],np.clip(error,-30,30)])
        latent=np.arctanh(np.clip(target/np.asarray(env.cfg['limits']),-.999,.999))
        command=command_from_latent(latent,obs['preceding_command'],env.cfg['step_s'],env.cfg['limits'],env.cfg['acceleration'])
        stop=final and env.in_goal(state)
        response=env.step(command,stop,obs['frame']);obs=response['observation']
        rows.append(dict(state=state,command=command,event=response['event'],dt=response['dt']))
        if response['terminated'] or response['truncated']:
            passed=response['event']=='success';break
    result=dict(schema='ppo-route-qualification/v1',task_id=task['id'],passed=passed,
        distance_bin=task['distance_bin'],difficulty=task['difficulty'],rows=rows,
        purpose='privileged physical engineering qualification, never policy imitation',task=task)
    path=root/(task['id']+'-physical.json');write(path,result)
    return dict(path=str(path.resolve()),sha256=digest(path))


def auxiliary_candidates(volume,missions,seed,count=40):
    rng=np.random.default_rng(seed);result=[];centres=volume.centres.keys
    for i in range(count*30):
        if len(result)>=count*2:break
        start=np.asarray(volume.position(centres[int(rng.integers(len(centres)))]));yaw=float(rng.uniform(-180,180))
        kind='arrival' if len(result)%2 else 'execution'
        if kind=='arrival':
            # Balanced valid-stop and near-miss poses; labels remain privileged.
            radius=float(rng.uniform(0,2) if (len(result)//2)%2 else rng.uniform(3.5,6))
            angle=rng.uniform(-math.pi,math.pi);delta=np.array([radius*math.cos(angle),radius*math.sin(angle),0.])
            goal_yaw=yaw+float(rng.uniform(-20,20))
        else:
            angle=math.radians(yaw+float(rng.uniform(-25,25)));distance=float(rng.uniform(3,6))
            delta=np.array([distance*math.cos(angle),distance*math.sin(angle),float(rng.choice([-1.,0.,1.]))]);goal_yaw=yaw
        goal=np.asarray(volume.position(volume.key(start+delta)))
        if not volume.clear(start,goal):continue
        template=missions[i%len(missions)];route=[start.tolist(),goal.tolist()]
        row={k:template[k] for k in ('scene_id','split','field','field_sha256','bounds')}
        row.update(id=f'{template["scene_id"]}-aux-{seed}-{i}',kind=kind,start=route[0],goal=route[1],
            start_yaw_deg=yaw,goal_yaw_deg=goal_yaw,reference_path=route,reference_s=max(1,nominal_time(route)),
            timeout_s=10 if kind=='execution' else 20,behavior='level',camera=template.get('camera'))
        if kind=='execution':row['exercise']=dict(intention='approach',altitude='gain' if delta[2]<0 else 'lose' if delta[2]>0 else 'maintain',roi=[.33,.25,.67,.75])
        result.append(row)
    if Counter(r['kind'] for r in result)!=Counter(execution=count,arrival=count):raise ValueError('Insufficient observed auxiliary coverage')
    return result


def capture(args):
    cfg=read(args.config);candidate=read(args.candidates);root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    tasks=[];routes=[];rejected=[];covered=set();window=Window(args.hours)
    missions=candidate['tasks']
    if not missions:raise ValueError('No generated missions')
    volume=ObservedVolume.load(missions[0]['field']);rows=list(missions)
    if args.auxiliary:rows+=auxiliary_candidates(volume,missions,args.seed)
    with FlightLock(Path(args.workspace)/'ppo-campaign','ppo-task-capture'):
        budget=Budget(Path(args.workspace)/'ppo-campaign',cfg)
        try:
            with PilotEnvironment(read(args.scene),root/'worker',cfg,args.port) as env:
                env.calibrate()
                for raw in rows:
                    if not window.remaining():break
                    t=dict(raw);directory=root/t['id'];directory.mkdir();t['camera']=cfg['camera']
                    try:
                        goal_reset=env.reset_pose(t['goal'],t['goal_yaw_deg']);rgb,stamp,_=env.image()
                        if float(rgb.std())<8:raise ValueError('Goal photograph lacks visual texture')
                        image=directory/'goal.png';Image.fromarray(rgb).save(image)
                        start_reset=env.reset_pose(t['start'],t['start_yaw_deg']);start_rgb,_,_=env.image()
                        start_image=directory/'start.png';Image.fromarray(start_rgb).save(start_image)
                        endpoint=dict(passed=True,start=t['start'],goal=t['goal'],start_yaw_deg=t['start_yaw_deg'],goal_yaw_deg=t['goal_yaw_deg'],
                            start_reset=start_reset,goal_reset=goal_reset,camera=cfg['camera'],goal_capture_sim_ns=stamp)
                        if t['kind']=='execution':
                            # A visible, finite depth patch beyond the free-space
                            # target grounds approach; free air is not a landmark.
                            depth,receipt=capture_depth(env,t['start'],t['start_yaw_deg'],directory,'execution-target')
                            transform=np.asarray(receipt['camera_to_ned']);optical=(np.asarray(t['goal'])-transform[:3,3])@transform[:3,:3]
                            if optical[2]<=.2:raise ValueError('Execution target behind fixed camera')
                            u=receipt['fx']*optical[0]/optical[2]+receipt['cx'];v=receipt['fx']*optical[1]/optical[2]+receipt['cy']
                            x=u/receipt['width'];y=v/receipt['height']
                            if not .15<x<.85 or not .15<y<.85:raise ValueError('Execution target outside central visible camera support')
                            patch=depth[int(v)-2:int(v)+3,int(u)-2:int(u)+3]
                            distance=float(np.linalg.norm(optical))
                            if not np.isfinite(patch).all() or not distance+1<float(np.median(patch))<100:raise ValueError('No visible supported approach reference')
                            t['exercise']['roi']=[x-.12,y-.12,x+.12,y+.12]
                            endpoint['reference_depth']=receipt
                        endpoint_path=directory/'endpoints.json';write(endpoint_path,endpoint)
                        t.update(goal_image=str(image.resolve()),goal_sha256=digest(image),start_image=str(start_image.resolve()),
                            start_sha256=digest(start_image),endpoint_evidence=str(endpoint_path.resolve()),endpoint_sha256=digest(endpoint_path))
                        if 'cost_field' not in t:
                            costs,_=volume.costs(t['goal']);path=directory/'costs.npz'
                            save_costs(path,costs)
                            t.update(cost_field=str(path.resolve()),cost_sha256=digest(path))
                        if t['kind']=='mission':
                            cell=(t['distance_bin'],t['difficulty'])
                            if cell not in covered:
                                receipt=route_flight(env,t,directory,budget);routes.append(receipt)
                                if not read(receipt['path'])['passed']:raise ValueError('Representative physical route failed')
                                covered.add(cell)
                        tasks.append(t)
                    except Exception as error:
                        rejected.append(dict(id=t['id'],error=str(error)));write(directory/'rejection.json',rejected[-1])
                    write(root/'captured.json',dict(schema='ppo-task-capture/v2',tasks=tasks,physical_routes=routes,rejected=rejected,
                        scene_descriptor_sha256=digest(args.scene),training_qualified=False))
        finally:budget.close()


def assemble(args):
    cfg=read(args.config);inventory=read(args.inventory);tasks=[];scenes=[]
    for entry in inventory['scenes']:
        captured=read(entry['capture']);scene={k:entry[k] for k in ('scene_id','split','descriptor','qualification','geography_review')}
        if captured['scene_descriptor_sha256']!=digest(scene['descriptor']):raise ValueError('Capture scene differs')
        for name in ('qualification','geography_review'):scene[name+'_sha256']=digest(scene[name])
        scene['physical_routes']=captured['physical_routes'];scenes.append(scene);tasks.extend(captured['tasks'])
    selected=set(inventory['sanity_validation_ids'])
    for t in tasks:t['sanity_validation']=t['id'] in selected
    manifest=dict(schema='photo-map-ppo-tasks/v2',config_sha256=digest(args.config),scenes=scenes,tasks=tasks)
    from .ppo_tasks import validate_tasks
    validate_tasks(manifest,cfg);write(args.output,manifest)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(Path(__file__).with_name('ppo_overnight.json')))
    sub=p.add_subparsers(dest='stage',required=True)
    for name in ('survey','capture'):
        a=sub.add_parser(name)
        for key in ('scene','output','workspace'):a.add_argument('--'+key,required=True)
        a.add_argument('--hours',type=float,default=8);a.add_argument('--port',type=int,default=43551)
        if name=='survey':a.add_argument('--plan',required=True)
        else:
            a.add_argument('--candidates',required=True);a.add_argument('--auxiliary',action='store_true');a.add_argument('--seed',type=int,default=0)
    a=sub.add_parser('assemble');a.add_argument('--inventory',required=True);a.add_argument('--output',required=True)
    args=p.parse_args();globals()[args.stage](args)
