"""Live pilot task/worker qualification, with explicit pending visual review."""
import argparse
import copy
from pathlib import Path
import time
import numpy as np
from .common import read,write,digest
from .ppo_env import PilotEnvironment,qualify


def run(args):
    cfg=read(Path(__file__).with_name('ppo_pilot.json'));candidates=read(args.tasks)
    tasks=candidates['tasks']
    if {t['behavior'] for t in tasks}!={'level','climb','descent'}:raise ValueError('Three physically verified altitude categories required')
    task=next(t for t in tasks if t['behavior']=='level');out=Path(args.output)
    qualify(args.scene,out,cfg,task['start'],task['start_yaw_deg'])
    q=read(out/'qualification.json');q['checks']={};q['tasks_sha256']=digest(args.tasks)
    if not q.get('reset_passed'):raise RuntimeError('Reset qualification failed')
    try:
        with PilotEnvironment(read(args.scene),out/'semantics-worker',cfg) as env:
            q['checks']['geometry']=all(read(t['geometry_evidence'])['swept_volume_free'] for t in tasks)
            q['checks']['motion']=all(read(read(t['geometry_evidence'])['physical_evidence'])['passed'] for t in tasks)
            obs=env.reset(task,'false-stop');result=env.step([0.]*4,True,obs['frame'])
            q['false_stop']=dict(event=result['event'],state=result['observation']['state'])
            arrived=copy.deepcopy(task);arrived['start']=task['goal'];arrived['start_yaw_deg']=task['goal_yaw_deg']
            obs=env.reset(arrived,'correct-stop');result=env.step([0.]*4,True,obs['frame'])
            q['correct_stop']=dict(event=result['event'],dt=result['dt'],state=result['observation']['state'])
            q['checks']['stop']=q['false_stop']['event']=='false_stop' and q['correct_stop']['event']=='success' and result['dt']>=cfg['arrival']['dwell_s']*.9
            obs=env.reset(task,'stale-frame');env.step([.1,0,0,0],False,obs['frame'])
            time.sleep(cfg['freshness_s']+.15)
            with env.lock:dispatches=list(env.dispatches)
            q['stale_dispatches']=dispatches
            q['checks']['stale_frame']=bool(dispatches and dispatches[-1]['stale'] and dispatches[-1]['command']==[0.]*4)
            obs=env.reset(task,'timing');times=[];durations=[]
            for _ in range(200):
                started=time.monotonic();result=env.step([0.]*4,False,obs['frame']);times.append(time.monotonic()-started)
                durations.append(result['dt']);obs=result['observation']
                if result['terminated']:raise RuntimeError('Unexpected contact during stationary timing check')
            q['timing']=dict(host_step_wall_s=times,sim_durations_s=durations,p95_s=float(np.percentile(times,95)),
                p99_s=float(np.percentile(times,99)),scope='host RGB/state/dispatch path; GPU inference measured separately')
            q['checks']['timing']=q['timing']['p99_s']<cfg['freshness_s']
            # Deliberate slow ground contact in a disposable engineering episode.
            # Exercises the very same collision-to-terminal path as learning.
            contact=copy.deepcopy(task);contact['bounds']=[(np.asarray(task['start'])-100).tolist(),(np.asarray(task['start'])+100).tolist()]
            obs=env.reset(contact,'collision');began=time.monotonic();steps=[]
            while time.monotonic()-began<40:
                result=env.step([0.,0.,.5,0.],False,obs['frame']);obs=result['observation']
                steps.append(dict(state=obs['state'],event=result['event']))
                if result['terminated'] or result['truncated']:break
            q['collision_steps']=steps;q['checks']['collision']=bool(steps and steps[-1]['event']=='collision' and steps[-1]['state']['collision_object'])
            q['checks']['camera']=len(q.get('camera_attitudes',[]))==9 and all(r['pose_agreement'] for r in q['camera_attitudes'])
    except Exception as error:q['semantic_error']=str(error)
    finally:
        q['qualified']=False;q['pending']='Independent camera/geography review and complete check results'
        write(out/'qualification.json',q)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--tasks',required=True);p.add_argument('--output',required=True)
    run(p.parse_args())
