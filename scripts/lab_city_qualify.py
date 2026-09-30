"""Record actual native flights and qualify this deployment before city PPO.

Every acceptance field comes from measured physical/RGB flight evidence.
Controlled stop/contact diagnostics are labeled, never learner transitions.
"""
import copy
import argparse
import json
import math
import os
from pathlib import Path
import signal
import time
import uuid
import numpy as np
import torch
from photo_goal.common import read,write,digest,FlightLock
from photo_goal.mission_contracts import city_config,identity
from photo_goal.mission_environment import CityEnvironment
from photo_goal.mission_checkpoint import load_components
from photo_goal.mission_resources import Resources
from photo_goal.mission_resources import RunWindow
from photo_goal.mission_scheduler import CityFeatureBank,warm_city_policy
from photo_goal.mission_mode2 import MissionGuidance
from photo_goal.native_full_training import implementation_identity,runtime,motor_command
from photo_goal.ppo_scheduler import Inference
from photo_goal.ppo_core import command_from_latent
from photo_goal.compute import ComputeLane
from photo_goal.rgb_survey import RGBSurvey
from photo_goal.ppo_budget import Budget

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,default=Path('/mnt/hdd2/yanbocheng/photo-goal-native'))
for name in ('checkpoint','config','scene','taskset','qualification','survey','run-dir','backbone'):
    parser.add_argument('--'+name,type=Path)
parser.add_argument('--hours',type=float,default=8)
parser.add_argument('--geometry-samples',type=Path)
parser.add_argument('--geometry-qualification',type=Path)
args=parser.parse_args();root=args.root.resolve();cfg=city_config(args.config)
window=RunWindow(args.hours)
from photo_goal.mission_storage import configure
configure(root)
if cfg.get('task_sampling')=='quota_v2':
    catalog=root/'rgb/catalog.sqlite';catalog.parent.mkdir(parents=True,exist_ok=True)
    os.environ['UAV_RGB_CATALOG']=str(catalog)
checkpoint=args.checkpoint or root/'checkpoints/city-initialized.pt'
scene_path=args.scene or root/'scene.json';taskset=args.taskset or root/'city-tasks.json'
scene=read(scene_path);task_data=read(taskset)
if task_data.get('schema')=='photo-goal-taskset/v2':
    from photo_goal.mission_task_catalog import load_catalog
    tasks,_=load_catalog(taskset,digest(scene_path))
else:tasks=task_data['tasks']
task=next(t for t in tasks if t['split']=='train')
survey_path=args.survey or root/'data/visual-bootstrap/rgb-atlas.json';survey=RGBSurvey(survey_path)
out=args.run_dir or root/'runs'/('city-qualification-'+uuid.uuid4().hex[:12]);out.mkdir(parents=True)
resources=Resources(root,cfg,'cuda');torch.set_num_threads(8)
receipt=dict(schema='photo-goal-city-qualification/v1',passed=False,
    scene_sha256=digest(scene_path),tasks_sha256=digest(taskset),
    survey_sha256=digest(survey_path),config_sha256=identity(cfg),
    implementation_sha256=implementation_identity(),complete_flight_receipts=[],
    actor_checkpoint_sha256=digest(checkpoint),
    scope='Native physical flight integration; no navigation learning or held-out success claim')
def check(condition,message):
    if not condition:raise RuntimeError(message)
def stop(signum,frame):raise KeyboardInterrupt('Bounded qualification shutdown')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
actor,optimizer,world,world_optimizer,meta=load_components(checkpoint,
    args.backbone or root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt',cfg,'cuda')
bank=CityFeatureBank(actor,out,survey)
warm_city_policy(actor,bank,root,task)
def log(row):
    with (out/'guidance.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
guidance=MissionGuidance(('127.0.0.1',48005),root/'campaign/qwen-auth.bin',survey,log,digest(checkpoint))
scheduler=Inference(actor,ComputeLane(),{0:bank},guidance,wait_s=0.)
budget=None;batch='qualification-'+uuid.uuid4().hex
def reserve_qualification_step():
    global batch
    if not window.admits(10):raise RuntimeError('Qualification window ended; no passing receipt')
    info=budget.batch_info(batch)
    if info['issued']>=info['requested']:
        budget.finish(batch,False);batch='qualification-'+uuid.uuid4().hex
        check(budget.reserve_batch(batch,cfg['campaign_transitions']),'Qualification budget exhausted')
    budget.transition(batch,0)

def qualify_geometry(session):
    import itertools
    from projectairsim.types import Pose
    data=read(args.geometry_samples)
    samples=data.get('samples',[]);surfaces=data.get('surface_samples',[])
    check(len(samples)>=6 and len(surfaces)>=2,'Geometry needs independent free/contact and street/roof samples')
    check(sum(s['occupied'] for s in samples)>=2 and sum(not s['occupied'] for s in samples)>=2,'Geometry needs both occupancy classes')
    evidence=[]
    for sample in samples+surfaces:
        ref=sample['evidence'];check(digest(ref['path'])==ref['sha256'],'Changed native geometry evidence');evidence.append(ref)
    layouts=list(itertools.permutations(('x','y','z')));observations=[]
    for sample in samples:
        center=np.asarray(sample['position'])+np.array([7.,13.,19.])
        pose=Pose(dict(translation=dict(zip(('x','y','z'),center.tolist())),rotation=dict(w=1,x=0,y=0,z=0),frame_id='DEFAULT_ID'))
        raw=np.asarray(session.world.create_voxel_grid(pose,64,64,64,1,actors_to_ignore=['drone_1'],write_file=False))
        check(raw.shape==(64**3,) and raw.dtype==np.bool_,'Malformed native voxel response')
        index=np.floor(np.asarray(sample['position'])-(center-32)).astype(int)
        layouts=[axes for axes in layouts if bool(raw.reshape((64,)*3).transpose(tuple(axes.index(a) for a in ('x','y','z')))[tuple(index)])==bool(sample['occupied'])]
        observations.append(dict(position=sample['position'],occupied=sample['occupied'],voxel_center=center.tolist()))
    check(len(layouts)==1,'Occupancy layout ambiguous or inconsistent; more independent native evidence required')
    units=['ned_z','world_height'];surface_observations=[]
    for sample in surfaces:
        raw=float(session.world.get_surface_elevation_at_point(*sample['xy']))
        check(math.isfinite(raw),'Missing surface elevation')
        units=[u for u in units if abs((-raw if u=='ned_z' else raw)-sample['world_height_m'])<=1.]
        surface_observations.append(dict(xy=sample['xy'],native_value=raw,measured_world_height_m=sample['world_height_m']))
    check(len(units)==1,'Surface sign/layer ambiguous; include a measured elevated surface')
    result=dict(schema='photo-goal-native-geometry-qualification/v1',scene_sha256=digest(scene_path),
        axis_verified=True,scale_verified=True,occupancy_verified=True,surface_verified=True,
        contact_verified=receipt['collision_verified'],array_axes=list(layouts[0]),array_order='C',
        surface_units=units[0],evidence=evidence,observations=observations,surface_observations=surface_observations,
        sample_manifest_sha256=digest(args.geometry_samples),qualification_script_sha256=digest(__file__))
    write(args.geometry_qualification,result)
try:
    with FlightLock(root,'city-deployment-qualification'):
        budget=Budget(root/'campaign',cfg)
        check(budget.reserve_batch(batch,cfg['campaign_transitions']),'Qualification physical budget exhausted')
        with CityEnvironment(scene,out/'worker',cfg) as env:
            env.phase='training';env.calibrate()
            receipt['reset_cycles']=[env.reset_pose(task['start'],task['start_yaw_deg']) for _ in range(20)]
            check(all(any(r['passed'] for r in rows) for rows in receipt['reset_cycles']),'Reset stability failed')
            # Complete deterministic stop diagnostics; no action-space gate.
            for name,near,expected in [('arrival-stop',True,'success'),('false-stop',False,'false_stop')]:
                diagnostic=copy.deepcopy(task)
                if near:diagnostic['start']=list(task['goal']);diagnostic['start_yaw_deg']=task['goal_yaw_deg']
                obs=env.reset(diagnostic,name)
                result=env.step([0.]*4,True,obs['frame'])
                check(result['event']==expected,'Incorrect stop outcome: '+str(result['event']))
                env.recorder.flush()
                path=out/(name+'.json')
                write(path,dict(controller='controlled-stop-diagnostic',event=result['event'],terminated=result['terminated'],
                               recording=str(env.recorder.root),dt=result['dt'],command_intervals=result['command_intervals']))
                receipt['complete_flight_receipts'].append(dict(path=str(path),sha256=digest(path)))
            # Deliberately withhold a decision and observe the independent brake.
            env.reset(task,'freshness-brake');time.sleep(.4)
            with env.lock:dispatch=list(env.dispatches);fault=env.fault
            receipt['freshness_brake']=dict(fault=fault,dispatches=dispatch)
            receipt['freshness_verified']=bool(fault and any(r['stale'] and r['command']==[0.]*4 for r in dispatch))
            check(receipt['freshness_verified'],'Independent stale-source brake did not fire')
            receipt['command_axes']=[]
            for axis,command in enumerate(([.5,0,0,0],[0,.5,0,0],[0,0,-.5,0],[0,0,0,10])):
                diagnostic=dict(task,timeout_s=60)
                obs=env.reset(diagnostic,'command-axis-'+str(axis));before=obs['state']
                for _ in range(math.ceil(1/cfg['step_s'])):
                    reserve_qualification_step();result=env.step(command,False,obs['frame']);budget.confirm(batch,0)
                    check(not result['terminated'],'Physical failure during command-axis diagnosis')
                    obs=result['observation']
                yaw=math.radians(before['attitude_deg'][2]);delta=np.asarray(obs['state']['position'])-before['position']
                body=[delta[0]*math.cos(yaw)+delta[1]*math.sin(yaw),-delta[0]*math.sin(yaw)+delta[1]*math.cos(yaw),delta[2]]
                dyaw=(obs['state']['attitude_deg'][2]-before['attitude_deg'][2]+180)%360-180
                passed=(body[axis]*(1 if axis<2 else -1)>.05) if axis<3 else dyaw>2
                receipt['command_axes'].append(dict(axis=axis,command=command,body_delta_m=body,yaw_delta_deg=dyaw,passed=bool(passed),controller='controlled-axis-diagnostic'))
                check(passed,'Native body-axis/units response failed')
                with env.lock:env.active=False;env.command=[0.]*4;env.done=True
                env.client.cancelLastTask(env.vehicle);env.client.moveByVelocityAsync(0.,0.,0.,.2,vehicle_name=env.vehicle)
            receipt['command_axes_verified']=all(r['passed'] for r in receipt['command_axes'])
            # Complete flights use the actual frozen encoder/recurrent actor and
            # asynchronous Qwen service. No privileged state enters inference.
            timing=[];clock=[];intervals=[];boundary=False
            receipt['timing']=dict(rows=timing,clock=clock,intervals=intervals)
            training=[t for t in tasks if t['split']=='train']
            selected=[min((t for t in training if not t.get('support')),key=lambda t:t['distance_m']),
                      max((t for t in training if not t.get('support')),key=lambda t:t['distance_m'])]
            if task_data.get('schema')=='photo-goal-taskset/v2':
                selected=[next(t for t in training if t['task_class']=='support'),
                          next(t for t in training if t['task_class']=='intermediate')]+selected
            else:
                support=copy.deepcopy(task);start=support['support_starts'][0]
                support.update(start=start['position'],start_yaw_deg=start['yaw_deg'],timeout_s=60)
                selected=[support]+selected
            flight=0;receipt['infrastructure_cuts']=[]
            while flight<len(selected):
                if not window.admits(30):raise RuntimeError('Qualification window ended without complete flights')
                task=dict(selected[flight])
                if not task.get('timeout_s'):
                    from photo_goal.mission_contracts import task_band
                    task['timeout_s']=cfg['mission_deadlines_s'][task_band(task['distance_m'],cfg)]
                mission='actual-learner-'+str(flight)+'-'+uuid.uuid4().hex[:8]
                budget.attempt(mission,0,smoke=False)
                from PIL import Image
                with Image.open(task['goal_image']) as image:
                    scheduler.call('initialize',worker=0,image=image.convert('RGB'),path=task['goal_image'],mission_id=mission)
                obs=env.reset(task,mission);steps=0
                decision=scheduler.call('decision',worker=0,obs=runtime(obs,task,mission,survey,cfg),execution=False)
                while True:
                    issued=False
                    measured=dict(decision_s=decision['decision_s'],source_age_s=time.perf_counter()-obs['source_wall'],
                        encode_s=decision['encode_s'],context_s=decision['context_s'],policy_s=decision['policy_s'])
                    try:
                        resources.check()
                        command=motor_command(decision,obs,cfg)
                        reservation_started=time.perf_counter();reserve_qualification_step();issued=True
                        measured['reservation_s']=time.perf_counter()-reservation_started
                        measured['source_age_s']=time.perf_counter()-obs['source_wall']
                        result=env.step(command,bool(decision['stop']),obs['frame'],policy_sha256=receipt['actor_checkpoint_sha256'])
                        budget.confirm(batch,0);issued=False
                    except RuntimeError as error:
                        if str(error) not in ('Stale policy decision; watchdog brakes',
                            'Dispatcher failed: Active control watchdog exceeded source freshness'):raise
                        with env.lock:env.active=False;env.command=[0.]*4;env.done=True
                        env.client.cancelLastTask(env.vehicle)
                        env.client.moveByVelocityAsync(0.,0.,0.,.2,vehicle_name=env.vehicle)
                        if issued:budget.discard_unobserved(batch,0)
                        env.stop_camera();env.recorder.flush()
                        cut=dict(attempt=mission,error=str(error),timing=measured,infrastructure_cut=True,
                            complete_flight=False,used_as_terminal_reward=False,discarded_dispatch=issued,
                            recording=str(env.recorder.root))
                        receipt['infrastructure_cuts'].append(cut);write(out/(mission+'-cut.json'),cut)
                        print(json.dumps(dict(event='qualification_freshness_cut',**cut)),flush=True)
                        guidance.invalidate(receipt['actor_checkpoint_sha256']);bank.trim()
                        time.sleep(.2)
                        break # Reset and retry this same required full-flight slot.
                    timing.append(measured)
                    following=result['observation'];clock.append([obs['sim_s'],following['sim_s']])
                    intervals.append(dict(dt=result['dt'],segments=result['command_intervals']))
                    steps+=1;obs=following
                    if result['terminated']:
                        env.stop_camera();env.recorder.flush();path=out/(mission+'.json')
                        write(path,dict(controller='city-actor-from-qualified-checkpoint',event=result['event'],terminated=True,
                                       steps=steps,recording=str(env.recorder.root),navigation_success=result['event']=='success'))
                        receipt['complete_flight_receipts'].append(dict(path=str(path),sha256=digest(path)))
                        env.pause_terminal_boundary()
                        before=env.state();guidance.residency('cpu');time.sleep(.5)
                        after=env.state();check(before==after,'Physical state advanced at optimizer boundary')
                        torch.cuda.empty_cache()
                        guidance.residency('cuda');env.release_terminal();time.sleep(.1)
                        check(env.state()['sim_ns']>before['sim_ns'],'Physics did not resume after the terminal boundary')
                        boundary=True;flight+=1
                        bank.trim()
                        print(json.dumps(dict(event='qualification_complete_flight',slot=flight,receipt=str(path))),flush=True)
                        break
                    decision=scheduler.call('decision',worker=0,obs=runtime(obs,task,mission,survey,cfg),execution=False)
            receipt['continuous_physics']=all(b>a for a,b in clock)
            receipt['command_intervals_verified']=all(abs(sum(s[4] for s in r['segments'])-r['dt'])<1e-6 for r in intervals)
            receipt['boundary_resume_verified']=boundary
            receipt['camera_reviewed']=len(obs['rgb'])==640*480*3 and float(np.frombuffer(obs['rgb'],np.uint8).std())>=8
            receipt['camera_review_scope']='Actual native RGB shape/encoding/variance; no human-review claim'
            receipt['timing']=dict(rows=timing,clock=clock,intervals=intervals)
            check(max(r['source_age_s'] for r in timing)<=cfg['freshness_s'],'Actor workload exceeded freshness')
            # Contact is measured through the same physical vehicle and native
            # collision topic. This diagnostic does not enter PPO/replay.
            with env.lock:env.active=False
            check(env.dispatch_idle.wait(3),'Control dispatch did not quiesce')
            env.fault=None;session=env.owned.session;baseline=len(session.collision_events)
            session.reset([task['start'][0],task['start'][1],-1.],task['start_yaw_deg'])
            until=time.monotonic()+8
            while time.monotonic()<until and len(session.collision_events)==baseline:
                session.command([0.,0.,1.,0.]);time.sleep(.05)
            contacts=session.collision_events[baseline:]
            receipt['contact_events']=contacts
            receipt['collision_verified']=any(r['event'].get('time_stamp',0)>0 and 'impact_point' in r['event'] and 'object_name' in r['event'] for r in contacts)
            check(receipt['collision_verified'],'No native physical contact report; inspect event schema')
            if args.geometry_samples:
                check(args.geometry_qualification is not None,'Specify geometry receipt output')
                qualify_geometry(session)
            receipt['resources']=resources.check()
            receipt['passed']=all(receipt[k] for k in ('continuous_physics','camera_reviewed','collision_verified',
                'freshness_verified','command_intervals_verified','boundary_resume_verified','command_axes_verified'))
except BaseException as error:
    receipt['error']=type(error).__name__+': '+str(error)
    raise
finally:
    if budget:
        budget.finish(batch,False);receipt['budget']=budget.snapshot();budget.close()
    write(out/'receipt.json',receipt);write(args.qualification or root/'city-qualification.json',receipt)
    scheduler.close();guidance.close();bank.close()
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('timing','reset_cycles','freshness_brake','contact_events')}),flush=True)
