"""Measured short-task preparation; privileged geometry never enters the actor.

Released paths only propose endpoints. Two opposing metric-depth observations
must support the entire sampled body/camera corridor, followed by a physical
flight. Rejected candidates and all sensor evidence remain on disk. This small
bootstrap is an integration curriculum, not a representative research dataset.
"""
import argparse
import json
import math
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image
from .common import read,write,digest,Window
from .ppo_env import PilotEnvironment


def proposals(annotations,scene_id,split,height_offset=2.):
    with zipfile.ZipFile(annotations) as archive:
        rows=json.loads(archive.read('train.json' if split=='train' else 'val_unseen.json'))['episodes']
    seen=set();endpoints=set()
    for row in sorted(rows,key=lambda r:str(r['trajectory_id'])):
        if int(row['scene_id'])!=int(str(scene_id).replace('env_','')) or row['trajectory_id'] in seen:continue
        seen.add(row['trajectory_id']);path=row['reference_path'];proposed=False
        for i in range(2,len(path)-2,5):
            start=np.asarray(path[i][:3],float)
            for j in range(i+1,min(len(path),i+20)):
                goal=np.asarray(path[j][:3],float)
                if not 12<=np.linalg.norm((goal-start)[:2])<=18 or abs(goal[2]-start[2])>2:continue
                # Extra height is only a candidate. Depth and physical flight
                # must qualify it; a source path supplies no clearance labels.
                start[2]-=height_offset;goal[2]=start[2]
                key=tuple(np.round(np.r_[start,goal],2))
                if key in endpoints:break
                endpoints.add(key)
                yaw=math.degrees(math.atan2(*(goal-start)[[1,0]]))
                for behavior,dz in [('level',0),('climb',-3),('descent',3)]:
                    begin=start.copy();end=goal.copy();heading=yaw
                    if behavior=='descent':
                        begin=goal.copy();begin[2]-=dz;end=start.copy();heading=(yaw+360)%360-180
                    else:end[2]+=dz
                    yield dict(id=f'{scene_id}-{row["trajectory_id"]}-{i}-{j}-{behavior}-h{height_offset:g}',
                        start=begin.tolist(),goal=end.tolist(),start_yaw_deg=heading,goal_yaw_deg=heading,
                        behavior=behavior,reference_id=row['trajectory_id'],scene_id=scene_id,split=split)
                proposed=True
                break
            # Spread a bounded preparation batch across distinct source routes
            # instead of exhausting it on many nearby legs of the first route.
            if proposed:break


def capture_depth(env,position,yaw,root,name):
    from ..rgb_flight.capture_obstacle_field import matrix
    resets=env.reset_pose(position,yaw)
    a=env.airsim;started=time.monotonic()
    response=env.client.simGetImages([a.ImageRequest('front_custom',a.ImageType.DepthPerspective,True,False)],env.vehicle)[0]
    width,height=env.cfg['qualification_depth_size']
    if (response.width,response.height)!=(width,height):raise ValueError('Depth calibration mismatch')
    depth=np.asarray(response.image_data_float,np.float32).reshape(height,width)
    path=root/(name+'.npy');np.save(path,depth,allow_pickle=False)
    transform=matrix(response.camera_position,response.camera_orientation)
    rgb,_,_=env.image();Image.fromarray(rgb).save(root/(name+'.png'))
    fx=width/(2*math.tan(math.radians(env.cfg['camera']['fov']/2)))
    record=dict(path=str(path.resolve()),sha256=digest(path),depth_type='DepthPerspective',
        camera_to_ned=transform.tolist(),width=width,height=height,fx=fx,cx=width/2,cy=height/2,
        sim_ns=response.time_stamp,reset=resets,capture_wall_s=time.monotonic()-started)
    write(root/(name+'.json'),record)
    return depth,record


def corridor(start,goal,captures):
    # A one-metre cube half-extent encloses the body, forward mount and margin.
    # Dense samples plus a 3x3 minimum-depth patch conservatively erode edges.
    spacing=.25;radius=1.
    axis=np.arange(-radius,radius+spacing/2,spacing)
    offsets=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
    centres=np.linspace(start,goal,int(math.ceil(math.dist(start,goal)/spacing))+1)
    points=(centres[:,None,:]+offsets[None,:,:]).reshape(-1,3)
    supported=np.zeros(len(points),bool)
    for depth,record in captures:
        t=np.asarray(record['camera_to_ned']);optical=(points-t[:3,3])@t[:3,:3]
        z=optical[:,2];safe=np.maximum(z,.001)
        u=np.rint(record['fx']*optical[:,0]/safe+record['cx']).astype(int)
        v=np.rint(record['fx']*optical[:,1]/safe+record['cy']).astype(int)
        visible=(z>.2)&(u>=1)&(u<record['width']-1)&(v>=1)&(v<record['height']-1)
        ids=np.flatnonzero(visible)
        local=np.stack([depth[v[ids]+dy,u[ids]+dx] for dy in (-1,0,1) for dx in (-1,0,1)])
        measured=local.min(0)
        # No-hit/invalid ranges cannot certify observed free space.
        supported[ids]|=(np.isfinite(local).all(0)&(measured>np.linalg.norm(optical[ids],axis=1)+.35)&(local.max(0)<200))
    return dict(swept_volume_free=bool(supported.all()),supported_fraction=float(supported.mean()),
        samples=len(points),spacing_m=spacing,half_extent_m=radius,depth_margin_m=.35,
        centre_support=supported.reshape(len(centres),-1).all(1).tolist(),
        method='opposed measured radial-depth views; 3x3 minimum range; finite observed support')


def supported_crop(task,captures):
    """Select a >=10 m contiguous supported portion, then check it afresh."""
    evidence=corridor(task['start'],task['goal'],captures)
    if evidence['swept_volume_free']:return evidence
    support=evidence['centre_support'];centres=np.linspace(task['start'],task['goal'],len(support))
    runs=[];begin=None
    for i,good in enumerate(support+[False]):
        if good and begin is None:begin=i
        if not good and begin is not None:
            if math.dist(centres[begin],centres[i-1])>=10:runs.append((begin,i-1))
            begin=None
    for begin,end in sorted(runs,key=lambda r:r[1]-r[0],reverse=True):
        candidate=corridor(centres[begin],centres[end],captures)
        if candidate['swept_volume_free']:
            task['start']=centres[begin].tolist();task['goal']=centres[end].tolist();return candidate
    return evidence


def physical_flight(env,task,root):
    """Offline qualification controller only; not imitation or PPO data."""
    obs=env.reset(task,'physical-'+task['id']);rows=[];started=time.monotonic()
    previous=np.zeros(4);last=time.monotonic()
    while time.monotonic()-started<60:
        state=obs['state'];delta=np.asarray(task['goal'])-state['position']
        yaw=math.radians(state['attitude_deg'][2]);c,s=math.cos(yaw),math.sin(yaw)
        world=delta*.7;world[:2]/=max(1,np.linalg.norm(world[:2]));world[2]=np.clip(world[2],-.5,.5)
        target=np.array([c*world[0]+s*world[1],-s*world[0]+c*world[1],world[2],0.])
        now=time.monotonic();dt=min(.1,max(.01,now-last));last=now
        previous+=np.clip(target-previous,-np.array([1,1,.5,45])*dt,np.array([1,1,.5,45])*dt)
        stop=np.linalg.norm(delta)<.5 and state['speed']<.3
        result=env.step(previous.tolist(),stop,obs['frame']);obs=result['observation']
        rows.append(dict(state=obs['state'],dt=result['dt'],event=result['event'],command=previous.tolist()))
        if result['terminated'] or result['truncated']:break
    with env.lock:env.command=[0.]*4
    record=dict(passed=bool(rows and rows[-1]['event']=='success'),steps=rows,scope='offline task qualification only')
    write(root/'physical-flight.json',record)
    if not record['passed']:raise RuntimeError('Physical corridor qualification failed')
    return record


def candidate_inputs(args,descriptor,cfg):
    if not args.observed_candidates:
        for task in proposals(args.annotations,descriptor['scene_id'],args.split,args.height_offset):yield task,None,None
        return
    root=Path(args.observed_candidates);saved=read(root/'tasks-candidate.json')
    if Path(saved['descriptor']).resolve()!=Path(args.scene).resolve():raise ValueError('Observed candidate scene differs')
    for row in saved['rejected']:
        task=dict(row['task']);path=root/task['id']/'geometry.json'
        if not path.exists():continue
        geometry=read(path)
        if task['scene_id']!=descriptor['scene_id'] or task['split']!=args.split or geometry['camera']!=cfg['camera']:
            raise ValueError('Observed candidate provenance differs')
        captures=[]
        for record in geometry['captures']:
            if digest(record['path'])!=record['sha256']:raise ValueError('Observed depth changed')
            captures.append((np.load(record['path'],allow_pickle=False),record))
        yield task,captures,dict(path=str(path.resolve()),sha256=digest(path))


def prepare(args):
    cfg=read(Path(__file__).with_name('ppo_pilot.json'));descriptor=read(args.scene)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);window=Window(args.hours)
    tasks=read(args.seed_tasks)['tasks'] if args.seed_tasks else [];rejected=[]
    for task in tasks:
        if task['scene_id']!=descriptor['scene_id'] or task['split']!=args.split or task['camera']!=cfg['camera']:
            raise ValueError('Seed tasks must retain scene, split and calibration')
        if digest(task['geometry_evidence'])!=task['geometry_evidence_sha256']:raise ValueError('Seed geometry changed')
    counts={k:sum(t['behavior']==k for t in tasks) for k in ('level','climb','descent')}
    try:
        with PilotEnvironment(descriptor,out/'worker',cfg,port=args.port) as env:
            env.calibrate()
            for index,(task,captures,source_geometry) in enumerate(candidate_inputs(args,descriptor,cfg)):
                if not window.remaining() or all(n>=args.per_behavior for n in counts.values()):break
                if counts[task['behavior']]>=args.per_behavior:continue
                if index>=args.max_candidates:break
                root=out/task['id'];root.mkdir();print('candidate',task['id'],flush=True)
                try:
                    if captures is None:
                        captures=[capture_depth(env,task['start'],task['start_yaw_deg'],root,'from-start'),
                                  capture_depth(env,task['goal'],task['goal_yaw_deg']+180,root,'from-goal')]
                    evidence=supported_crop(task,captures)
                    if not evidence['swept_volume_free'] and evidence['supported_fraction']>.95 and len(captures)==2:
                        delta=np.asarray(task['goal'])-task['start'];side=np.array([-delta[1],delta[0],0.])
                        side*=4/np.linalg.norm(side);mid=(np.asarray(task['start'])+task['goal'])/2
                        for sign in (-1,1):
                            position=mid+sign*side;direction=mid-position
                            yaw=math.degrees(math.atan2(direction[1],direction[0]))
                            try:captures.append(capture_depth(env,position.tolist(),yaw,root,'side-'+str(sign)))
                            except RuntimeError as error:write(root/('side-'+str(sign)+'-failure.json'),dict(error=str(error)))
                        evidence=supported_crop(task,captures)
                    evidence.update(task_id=task['id'],start=task['start'],goal=task['goal'],camera=cfg['camera'],captures=[r for _,r in captures])
                    if source_geometry:evidence['reused_observations']=source_geometry
                    write(root/'geometry.json',evidence)
                    if not evidence['swept_volume_free']:raise RuntimeError('Unsupported corridor: '+str(evidence['supported_fraction']))
                    env.reset_pose(task['goal'],task['goal_yaw_deg']);rgb,stamp,_=env.image()
                    image=root/'goal.png';Image.fromarray(rgb).save(image)
                    lo=np.minimum(task['start'],task['goal'])-np.array([8,8,5]);hi=np.maximum(task['start'],task['goal'])+np.array([8,8,5])
                    task.update(bounds=[lo.tolist(),hi.tolist()],camera=cfg['camera'],unobstructed=True,
                        goal_image=str(image.resolve()),goal_sha256=digest(image),goal_capture_sim_ns=stamp)
                    physical_flight(env,task,root)
                    evidence['physical_evidence']=str((root/'physical-flight.json').resolve())
                    evidence['physical_sha256']=digest(root/'physical-flight.json');write(root/'geometry.json',evidence)
                    task.update(geometry_evidence=str((root/'geometry.json').resolve()),geometry_evidence_sha256=digest(root/'geometry.json'))
                    tasks.append(task);counts[task['behavior']]+=1;print('accepted',task['id'],flush=True)
                except Exception as error:
                    row=dict(task=task,error=str(error));rejected.append(row);write(root/'rejected.json',row)
                    print('rejected',task['id'],str(error)[:300],flush=True)
                write(out/'progress.json',dict(tasks=tasks,rejected=rejected,counts=counts))
    finally:
        write(out/'tasks-candidate.json',dict(config_sha256=digest(Path(__file__).with_name('ppo_pilot.json')),
            descriptor=str(Path(args.scene).resolve()),tasks=tasks,rejected=rejected,counts=counts,training_qualified=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--split',choices=['train','validation'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--per-behavior',type=int,default=2);p.add_argument('--max-candidates',type=int,default=30)
    p.add_argument('--hours',type=float,default=2)
    p.add_argument('--port',type=int,default=43551)
    p.add_argument('--seed-tasks',help='Reuse immutable accepted task receipts from an earlier preparation run')
    p.add_argument('--height-offset',type=float,default=2.,help='Candidate height above reference, still subject to identical clearance checks')
    p.add_argument('--observed-candidates',help='Reassess preserved depth receipts; goal capture and physical execution remain fresh')
    args=p.parse_args()
    if args.per_behavior<1 or args.max_candidates<1:p.error('Positive task/candidate limits required')
    if not math.isfinite(args.height_offset) or not 0<=args.height_offset<=20:p.error('Height offset must be between 0 and 20 m')
    prepare(args)
