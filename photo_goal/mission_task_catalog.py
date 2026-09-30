"""Private route-qualified 3D reset catalog and checkpointed mission quotas."""
import math
import uuid
from pathlib import Path
import numpy as np
from PIL import Image
from .common import read, write, digest, contained, FlightLock
from .mission_contracts import city_config, identity
from .mission_geometry import Geometry, RouteUnavailable
from .mission_resources import Resources, RunWindow
from .mission_environment import CityEnvironment
from .mission_space import reserve_write


class TaskSampler:
    def __init__(self,tasks,state=None):
        self.tasks=tasks;self.rng=np.random.default_rng(20260930)
        self.slots=[];self.bands=[];self.support_slots=[];self.mid_slots=[]
        if state:
            self.rng.bit_generator.state=state['rng']
            for name in ('slots','bands','support_slots','mid_slots'):setattr(self,name,list(state[name]))
        required={(kind,band) for kind,band in [('regular',0),('regular',1),('regular',2),('intermediate',0),('intermediate',1),('support',0),('support',1)]}
        if not required.issubset({(t['task_class'],t['sampling_band']) for t in tasks}):
            raise ValueError('Task catalog lacks a required class/band')

    def state(self):
        return dict(rng=self.rng.bit_generator.state,**{n:list(getattr(self,n)) for n in ('slots','bands','support_slots','mid_slots')})

    def draw_slot(self):
        if not self.slots:
            self.slots=['regular']*10+['intermediate']*6+['support']*4;self.rng.shuffle(self.slots)
        kind=self.slots.pop()
        if kind=='regular':
            if not self.bands:self.bands=[0]*12+[1]*14+[2]*14;self.rng.shuffle(self.bands)
            band=self.bands.pop()
        else:
            name='support_slots' if kind=='support' else 'mid_slots'
            slots=getattr(self,name)
            if not slots:slots.extend([0,1]);self.rng.shuffle(slots)
            band=slots.pop()
        candidates=[t for t in self.tasks if t['task_class']==kind and t['sampling_band']==band]
        return dict(candidates[int(self.rng.integers(len(candidates)))],support=kind=='support')


def load_catalog(path,scene_sha256,project_root=None):
    path=Path(path).resolve();data=read(path)
    if data.get('schema')!='photo-goal-taskset/v2' or data['scene_sha256']!=scene_sha256:
        raise ValueError('Task catalog/scene mismatch')
    side=contained(path.parent,data['private']['path'])
    if digest(side)!=data['private']['sha256']:raise ValueError('Task private sidecar changed')
    private=read(side);labels={t['id']:t for t in private['tasks']}
    tasks=[]
    for public in data['tasks']:
        task=dict(labels[public['id']],**public)
        for key in ('goal_image','start_image'):
            if digest(task[key])!=task[key+'_sha256']:raise ValueError('Task RGB changed')
        if not task.get('endpoint_qualified') or not task['route']['swept_clear']:
            raise ValueError('Incomplete physical/route task qualification')
        if abs(math.dist(task['start'],task['goal'])-task['distance_m'])>.1:raise ValueError('Incorrect 3D separation')
        tasks.append(task)
    import os
    project_root=project_root or os.environ.get('UAV_PROJECT_ROOT')
    if not project_root:raise ValueError('Explicit project root required for private geometry resolution')
    geometry=contained(Path(project_root),private['geometry']['path'])
    if digest(geometry)!=private['geometry']['sha256']:raise ValueError('Task geometry changed')
    return tasks,geometry


def capture(args):
    root=Path(args.root).resolve();cfg=city_config(args.config)
    output=Path(args.output).resolve()
    if not output.is_relative_to(root):raise ValueError('Task output escapes project')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'tasks.json').exists():raise ValueError('Published task catalog is immutable')
    geometry=Geometry(args.geometry);scene_path=Path(args.scene or root/'scene.json')
    if geometry.spec['scene_sha256']!=digest(scene_path):raise ValueError('Geometry belongs to another scene')
    scene=read(scene_path);window=RunWindow(args.hours);resources=Resources(root,cfg,args.device)
    progress=output/'capture-progress.json'
    rng=np.random.default_rng(20260930)
    saved=read(progress) if progress.exists() else dict(goals=[],tasks=[],rejections=[])
    if progress.exists():
        if saved['geometry_sha256']!=digest(args.geometry) or saved['config_sha256']!=identity(cfg):raise ValueError('Capture resume identity mismatch')
        rng.bit_generator.state=saved['rng']
    saved.update(geometry_sha256=digest(args.geometry),config_sha256=identity(cfg))
    historical=[]
    if (root/'city-tasks.json').exists():
        for task in read(root/'city-tasks.json')['tasks']:historical.extend([task['start'],task['goal']])
    # Rank regions using free-space geometry only, before any policy outcomes.
    regions=[]
    for i in range(4):
        for j in range(4):
            lo=np.array([-350+i*175,-350+j*175]);hi=lo+175
            a=geometry.index([*lo,-120]);b=geometry.index([*hi,-2])
            volume=geometry.clear[tuple(slice(x,y) for x,y in zip(a,b))]
            if volume.any():regions.append((float(volume.mean()),i,j,lo,hi))
    regions.sort(key=lambda r:(-r[0],r[1],r[2]))
    if len(regions)<3:raise RouteUnavailable('Insufficient viable spatial split regions')
    development=regions[0]
    sealed=next((r for r in regions[1:] if np.linalg.norm(r[3]-development[3])>=100),None)
    if sealed is None:raise RouteUnavailable('No separated sealed goal region')
    reserved=[development,sealed]
    def inside(point,region,margin=0):
        return bool(np.all(np.asarray(point)[:2]>=region[3]+margin) and np.all(np.asarray(point)[:2]<=region[4]-margin))
    def save_progress():
        saved['rng']=rng.bit_generator.state;write(progress,saved)
    with FlightLock(root,'city-v2-task-capture'):
        with CityEnvironment(scene,output/'native-capture',cfg) as env:
            env.calibrate()
            if scene.get('backend')!='projectairsim':raise RuntimeError('Native surface adapter not qualified for this backend')
            world=env.owned.session.world
            def surface(xy):
                raw=float(world.get_surface_elevation_at_point(*map(float,xy)))
                if not math.isfinite(raw):raise RouteUnavailable('Missing supporting surface')
                return raw if geometry.spec['surface_units']=='ned_z' else -raw
            def photo(name,pose,yaw):
                evidence=env.reset_pose(pose,yaw);rgb,stamp,_=env.image()
                if float(rgb.std())<8:raise RouteUnavailable('Unusable endpoint RGB')
                path=output/(name+'-'+uuid.uuid4().hex[:12]+'.png');reserve_write(path,rgb.nbytes+65536)
                Image.fromarray(rgb).save(path)
                from .mission_rgb_store import pixel_hash
                return dict(path=str(path),sha256=digest(path),pixel_sha256=pixel_hash(rgb),stamp_ns=stamp,evidence=evidence)
            requested=['train']*64+['development']*12+['sealed']*12
            for index in range(len(saved['goals']),len(requested)):
                split=requested[index];region=development if split=='development' else sealed
                for attempt in range(200):
                    if not window.admits(120):save_progress();raise RuntimeError('Task capture window ended; resume progress retained')
                    xy=rng.uniform(-330,330,size=2) if split=='train' else rng.uniform(region[3]+20,region[4]-20)
                    if split=='train' and any(inside(xy,r) for r in reserved):continue
                    if any(np.linalg.norm(xy-np.asarray(p)[:2])<20 for p in historical):continue
                    try:
                        z_surface=surface(xy);pose=[*map(float,xy),z_surface-float(rng.uniform(3,15))]
                        if not geometry.safe(pose) or not -140<=pose[2]<=-2:continue
                        yaw=float(rng.uniform(0,360));image=photo('goal-'+str(index),pose,yaw)
                        if any(g['goal_capture']['pixel_sha256']==image['pixel_sha256'] for g in saved['goals']):
                            raise RouteUnavailable('Duplicate goal pixels; raw capture retained')
                        saved['goals'].append(dict(id='goal-'+str(index),split=split,goal=pose,goal_yaw_deg=yaw,
                            goal_image=image['path'],goal_image_sha256=image['sha256'],goal_capture=image,
                            supporting_surface_z=z_surface))
                        save_progress();break
                    except (RouteUnavailable,RuntimeError) as error:saved['rejections'].append(dict(goal=index,error=str(error)))
                else:save_progress();raise RouteUnavailable('Could not fill goal stratum '+str(index))
            train=[g for g in saved['goals'] if g['split']=='train']
            inventory=[('train','regular',i%3,(i//3)%3) for i in range(256)]
            inventory += [('train','intermediate',i%2,i%3) for i in range(154)]
            inventory += [('train','support',i%2,0) for i in range(102)]
            for split in ('development','sealed'):inventory += [(split,'regular',i%3,(i//3+i%3)%3) for i in range(12)]
            for index in range(len(saved['tasks']),len(inventory)):
                split,kind,band,vertical=inventory[index]
                pool=train if split=='train' else [g for g in saved['goals'] if g['split']==split]
                goal=pool[index%len(pool)];b=np.asarray(goal['goal']);yaw_b=goal['goal_yaw_deg']
                for attempt in range(200):
                    if not window.admits(120):save_progress();raise RuntimeError('Task-pair window ended; progress retained')
                    resources.check()
                    if kind=='support':
                        yaw=yaw_b+float(rng.uniform(-15,15));dz=float(rng.uniform(-.5,.5));radius=float(rng.uniform(.5,2.5))
                        if band:
                            violation=(index//2)%3
                            if violation==0:radius=float(rng.uniform(3.5,12))
                            elif violation==1:dz=float(rng.choice([-1,1])*rng.uniform(2.5,5))
                            else:yaw=yaw_b+float(rng.choice([-1,1])*rng.uniform(45,180))
                        angle=float(rng.uniform(0,2*math.pi));a=b+np.array([radius*math.cos(angle),radius*math.sin(angle),dz]);deadline=60
                    else:
                        lo,hi=([50,100],[100,200],[200,300])[band] if kind=='regular' else ([10,25],[25,50])[band]
                        # Guarantee genuine high-end coverage, not only a nominal 300 m cap.
                        if kind=='regular' and band==2 and index<48:lo,hi=280,300
                        distance=float(rng.uniform(lo,hi));limit=30 if kind=='regular' else 10
                        dz=float(rng.uniform(-3 if kind=='regular' else -2,3 if kind=='regular' else 2)) if vertical==0 else float(rng.uniform(5 if kind=='regular' else 2,min(limit,distance*.8)))*(1 if vertical==1 else -1)
                        radius=math.sqrt(max(0,distance**2-dz**2));angle=float(rng.uniform(0,2*math.pi))
                        a=b+np.array([radius*math.cos(angle),radius*math.sin(angle),dz])
                        bearing=math.degrees(math.atan2(b[1]-a[1],b[0]-a[0]))
                        yaw=bearing+(0,90,180)[index%3]+float(rng.uniform(-15,15))
                        deadline=(180,300,600)[band] if kind=='regular' else 120
                    try:
                        sa=surface(a[:2]);clearance=sa-a[2]
                        if not 3<=clearance<=15 or not geometry.safe(a):continue
                        ceiling=max(-a[2],-b[2])+20
                        bounds=[[*np.minimum(a[:2],b[:2])-60,-min(ceiling,140)],
                                [*np.maximum(a[:2],b[:2])+60,-2]]
                        bounds[0][:2]=np.maximum(bounds[0][:2],[-350,-350]).tolist()
                        bounds[1][:2]=np.minimum(bounds[1][:2],[350,350]).tolist()
                        if split=='train' and any(np.all(np.asarray(bounds[0])[:2]<r[4]) and np.all(np.asarray(bounds[1])[:2]>r[3]) for r in reserved):continue
                        task=dict(id='task-'+str(index),scene_id=scene['scene_id'],split=split,task_class=kind,
                            sampling_band=band,start=a.tolist(),goal=b.tolist(),start_yaw_deg=yaw,goal_yaw_deg=yaw_b,
                            distance_m=float(np.linalg.norm(a-b)),timeout_s=deadline,bounds=bounds,
                            vertical_category=('level','ascent','descent')[vertical],supporting_surface_z=sa,
                            goal_id=goal['id'],goal_image=goal['goal_image'],goal_image_sha256=goal['goal_image_sha256'])
                        task['route']=geometry.validate_pair(task)
                        start=photo(task['id']+'-start',task['start'],yaw)
                        if split=='train' and any(g['split']!='train' and g['goal_capture']['pixel_sha256']==start['pixel_sha256'] for g in saved['goals']):
                            raise RouteUnavailable('Held goal pixels would enter training observations')
                        task.update(start_image=start['path'],start_image_sha256=start['sha256'],endpoint_qualified=True,
                                    reset_evidence=start['evidence'],support=kind=='support')
                        saved['tasks'].append(task);save_progress();break
                    except (RouteUnavailable,RuntimeError) as error:saved['rejections'].append(dict(task=index,error=str(error)))
                else:save_progress();raise RouteUnavailable('Unfilled route/height/task stratum '+str(index))
    if sum(t['split']=='train' and 280<=t['distance_m']<=300 for t in saved['tasks'])<16:
        raise RouteUnavailable('Missing sixteen qualified 280–300 m pairs')
    private=output/'private.json'
    write(private,dict(schema='photo-goal-private-tasks/v2',tasks=saved['tasks'],goals=saved['goals'],
        reserved_regions=[dict(lo=r[3].tolist(),hi=r[4].tolist()) for r in reserved],
        geometry=dict(path=str(Path(args.geometry).resolve()),sha256=digest(args.geometry))))
    public=[{k:t[k] for k in ('id','split','goal_id','goal_image','goal_image_sha256','timeout_s')} for t in saved['tasks']]
    # A virtual RGB view atlas: tile locations are pixel layout, never geography.
    tiles=[];held={g['goal_capture']['pixel_sha256'] for g in saved['goals'] if g['split']!='train'}
    selected=[g['goal_image'] for g in saved['goals'] if g['split']=='train']
    for goal in [g for g in saved['goals'] if g['split']=='train']:
        selected.append(next(t['start_image'] for t in saved['tasks'] if t['split']=='train' and t['goal_id']==goal['id']))
    from .mission_rgb_store import pixel_hash
    for image in selected:
        with Image.open(image) as source:sha=pixel_hash(np.asarray(source))
        if sha in held:raise ValueError('Held goal pixels would enter training survey')
        index=len(tiles);x=(index%8)*640;y=(index//8)*480
        tiles.append(dict(id='view-'+str(index),image=Path(image).name,sha256=digest(image),pixel_bounds=[x,y,x+640,y+480]))
    write(output/'survey.json',dict(schema='rgb-survey/v1',scene_id=scene['scene_id'],tiles=tiles,
        calibration=dict(units='pixels',provenance=dict(scope='Unordered known-city RGB views; no geographic calibration',camera=cfg['camera']))))
    write(output/'tasks.json',dict(schema='photo-goal-taskset/v2',scene_sha256=digest(scene_path),
        config_sha256=identity(cfg),calibration_sha256=identity(cfg['camera']),tasks=public,
        private=dict(path='private.json',sha256=digest(private))))
