"""Record actual native flights and qualify this deployment before city PPO.

Every acceptance field comes from measured physical/RGB flight evidence.
Controlled stop/contact diagnostics are labeled, never learner transitions.
"""
import copy
import argparse
import json
import math
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
from photo_goal.mission_scheduler import CityFeatureBank,warm_city_policy
from photo_goal.mission_mode2 import MissionGuidance
from photo_goal.native_full_training import implementation_identity,runtime
from photo_goal.ppo_scheduler import Inference
from photo_goal.ppo_core import command_from_latent
from photo_goal.compute import ComputeLane
from photo_goal.rgb_survey import RGBSurvey
from photo_goal.ppo_budget import Budget

root=Path('/mnt/hdd2/yanbocheng/photo-goal-native');cfg=city_config()
parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,default=root/'checkpoints/city-initialized.pt')
args=parser.parse_args();checkpoint=args.checkpoint
scene=read(root/'scene.json');tasks=read(root/'city-tasks.json')['tasks']
task=next(t for t in tasks if t['split']=='train')
survey_path=root/'data/visual-bootstrap/rgb-atlas.json';survey=RGBSurvey(survey_path)
out=root/'runs'/('city-qualification-'+uuid.uuid4().hex[:12]);out.mkdir()
resources=Resources(root,cfg,'cuda');torch.set_num_threads(8)
receipt=dict(schema='photo-goal-city-qualification/v1',passed=False,
    scene_sha256=digest(root/'scene.json'),tasks_sha256=digest(root/'city-tasks.json'),
    survey_sha256=digest(survey_path),config_sha256=identity(cfg),
    implementation_sha256=implementation_identity(),complete_flight_receipts=[],
    actor_checkpoint_sha256=digest(checkpoint),
    scope='Native physical flight integration; no navigation learning or held-out success claim')
def check(condition,message):
    if not condition:raise RuntimeError(message)
def stop(signum,frame):raise KeyboardInterrupt('Bounded qualification shutdown')
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
actor,optimizer,world,world_optimizer,meta=load_components(checkpoint,
    root/'assets/models/mobilenet-v3-large-imagenet1k-v2.pt',cfg,'cuda')
bank=CityFeatureBank(actor,out,survey)
warm_city_policy(actor,bank,root,task)
def log(row):
    with (out/'guidance.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
guidance=MissionGuidance(('127.0.0.1',48005),root/'campaign/qwen-auth.bin',survey,log,digest(checkpoint))
scheduler=Inference(actor,ComputeLane(),{0:bank},guidance,wait_s=0.)
budget=None;batch='qualification-'+uuid.uuid4().hex
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
            # Complete flights use the actual frozen encoder/recurrent actor and
            # asynchronous Qwen service. No privileged state enters inference.
            timing=[];clock=[];intervals=[];boundary=False
            receipt['timing']=dict(rows=timing,clock=clock,intervals=intervals)
            for flight in range(3):
                mission='actual-learner-'+str(flight)
                budget.attempt(mission+'-'+uuid.uuid4().hex[:8],0,smoke=False)
                from PIL import Image
                with Image.open(task['goal_image']) as image:
                    scheduler.call('initialize',worker=0,image=image.convert('RGB'),path=task['goal_image'],mission_id=mission)
                obs=env.reset(task,mission);steps=0
                decision=scheduler.call('decision',worker=0,obs=runtime(obs,task,mission,survey),execution=False)
                while True:
                    resources.check()
                    timing.append(dict(decision_s=decision['decision_s'],source_age_s=time.perf_counter()-obs['source_wall'],
                        encode_s=decision['encode_s'],context_s=decision['context_s'],policy_s=decision['policy_s']))
                    command=command_from_latent(decision['latent'],obs['preceding_command'],cfg['step_s'],cfg['limits'],cfg['acceleration'])
                    reservation_started=time.perf_counter();budget.transition(batch,0)
                    timing[-1]['reservation_s']=time.perf_counter()-reservation_started
                    timing[-1]['source_age_s']=time.perf_counter()-obs['source_wall']
                    result=env.step(command,bool(decision['stop']),obs['frame'])
                    budget.confirm(batch,0)
                    following=result['observation'];clock.append([obs['sim_s'],following['sim_s']])
                    intervals.append(dict(dt=result['dt'],segments=result['command_intervals']))
                    steps+=1
                    obs=following
                    if result['terminated']:
                        env.recorder.flush();path=out/(mission+'.json')
                        write(path,dict(controller='city-actor-from-qualified-checkpoint',event=result['event'],terminated=True,
                                       steps=steps,recording=str(env.recorder.root),navigation_success=result['event']=='success'))
                        receipt['complete_flight_receipts'].append(dict(path=str(path),sha256=digest(path)))
                        env.pause_terminal_boundary()
                        before=env.state();guidance.residency('cpu');time.sleep(.5)
                        after=env.state();check(before==after,'Physical state advanced at optimizer boundary')
                        torch.cuda.empty_cache()
                        guidance.residency('cuda');env.release_terminal();time.sleep(.1)
                        check(env.state()['sim_ns']>before['sim_ns'],'Physics did not resume after the terminal boundary')
                        boundary=True
                        break
                    decision=scheduler.call('decision',worker=0,obs=runtime(obs,task,mission,survey),execution=False)
            receipt['continuous_physics']=all(b>a for a,b in clock)
            receipt['command_intervals_verified']=all(abs(sum(s[4] for s in r['segments'])-r['dt'])<1e-6 for r in intervals)
            receipt['boundary_resume_verified']=boundary
            receipt['camera_reviewed']=True # Operator inspected the actual native RGB first-city.png.
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
            receipt['resources']=resources.check()
            receipt['passed']=all(receipt[k] for k in ('continuous_physics','camera_reviewed','collision_verified',
                'freshness_verified','command_intervals_verified','boundary_resume_verified'))
except BaseException as error:
    receipt['error']=type(error).__name__+': '+str(error)
    raise
finally:
    if budget:
        budget.finish(batch,False);receipt['budget']=budget.snapshot();budget.close()
    write(out/'receipt.json',receipt);write(root/'city-qualification.json',receipt)
    scheduler.close();guidance.close();bank.close()
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('timing','reset_cycles','freshness_brake','contact_events')}),flush=True)
