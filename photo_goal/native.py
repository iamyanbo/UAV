"""Native Windows scene capture and measured qualification using the existing worker."""
from pathlib import Path
import json
import math
import os
import time
import uuid
import numpy as np
from PIL import Image, ImageDraw
from .common import read, write, digest
from .ppo_env import PilotEnvironment


def configure(root):
    root = Path(root).resolve()
    scene_file = root / 'scene.json'
    if scene_file.exists():
        return read(scene_file), read(root / 'config.json')
    if os.name != 'nt':
        binary=root/'assets/cityenviron-linux-1.0.1/extracted/CityEnviron/Binaries/Linux/CityEnviron-Linux-Shipping'
        if not binary.is_file():raise RuntimeError('Download the official Linux CityEnviron first')
        cfg=read(Path(__file__).with_name('ppo_endpoint.json'))
        cfg.update(experiment='project-city-native-qualification',minimum_train_scenes=1,
                   minimum_validation_scenes=0,mode2_enabled=False,training_pause=False)
        settings=dict(SettingsVersion=1.2,SimMode='Multirotor',ClockSpeed=1,ViewMode='NoDisplay',ApiServerPort=43551,
            Vehicles={'drone_1':dict(VehicleType='SimpleFlight',AutoCreate=True,
                Cameras={'front_custom':dict(CaptureSettings=[
                    dict(ImageType=0,Width=640,Height=480,FOV_Degrees=90),
                    dict(ImageType=2,Width=160,Height=120,FOV_Degrees=90)])})})
        write(root/'settings.json',settings)
        scene=dict(scene_id='cityenviron-projectairsim-linux-1.0.1',backend='projectairsim',project_root=str(root),
            simulator_root=str(binary.parents[3]),simulator_binary=str(binary),settings=str(root/'settings.json'),
            launch_argv=[str(binary),'-RenderOffscreen','-graphicsadapter=0'],binary_sha256=digest(binary),
            geography_id='cityenviron-projectairsim-1.0.1',historical_windows_equivalence=False)
        write(scene_file,scene);write(root/'config.json',cfg)
        return scene,cfg
    binaries = [p for p in (root/'scenes').rglob('*.exe') if p.parent.name=='Win64' and p.parent.parent.name=='Binaries']
    if len(binaries) != 1:
        raise RuntimeError('Extract the pinned CityEnviron Windows package first; expected one Win64 executable')
    binary = binaries[0]
    runtime = binary.parents[3]
    cfg = read(Path(__file__).with_name('ppo_endpoint.json'))
    cfg.update(experiment='native-city-mode1-qualification', minimum_train_scenes=1,
               minimum_validation_scenes=0, mode2_enabled=False, training_pause=False)
    settings = dict(SettingsVersion=1.2, SimMode='Multirotor', ClockSpeed=1,
        ViewMode='NoDisplay', ApiServerPort=43551,
        Vehicles={'drone_1':dict(VehicleType='SimpleFlight', AutoCreate=True,
            Cameras={'front_custom':dict(CaptureSettings=[
                dict(ImageType=0, Width=640, Height=480, FOV_Degrees=90),
                dict(ImageType=2, Width=160, Height=120, FOV_Degrees=90)])})})
    write(root/'settings.json', settings)
    scene = dict(scene_id='cityenviron-windows-v1.8.1', simulator_root=str(runtime),
        simulator_binary=str(binary), settings=str(root/'settings.json'),
        launch_argv=[str(binary), '-windowed', '-ResX=640', '-ResY=480', '-nosound'],
        binary_sha256=digest(binary), geography_id='airsim-cityenviron')
    write(scene_file, scene); write(root/'config.json', cfg)
    return scene, cfg


def brake(env):
    with env.lock:
        env.active=False
        env.command=[0.]*4
    if not env.dispatch_idle.wait(3):
        raise RuntimeError('Control dispatch did not quiesce')
    env.client.moveByVelocityAsync(0,0,0,.5,vehicle_name=env.vehicle).join()


def overview(root, tasks):
    sheet = Image.new('RGB',(640, len(tasks)*270),'white')
    draw = ImageDraw.Draw(sheet)
    for i,t in enumerate(tasks):
        for j,key in enumerate(('start_image','goal_image')):
            im=Image.open(t[key]).convert('RGB');im.thumbnail((320,240))
            sheet.paste(im,(j*320,i*270+28))
        draw.text((6,i*270+6),f"{t['id']} | A -> B | {t['distance_m']:.1f} m | height change {t['altitude_change_m']:.1f} m",fill='black')
    sheet.save(root/'ab-pictures.jpg',quality=92)


def capture(root, count=12):
    root=Path(root).resolve();scene,cfg=configure(root)
    output=root/'tasks';output.mkdir(exist_ok=True)
    manifest=dict(schema='photo-goal-native-tasks/v1', scene=scene,
        config_sha256=digest(root/'config.json'), tasks=[], rejected=[],
        purpose='engineering qualification; not a frozen research split')
    if (root/'tasks.json').exists():
        previous=read(root/'tasks.json')
        if previous['scene']!=scene or previous['config_sha256']!=manifest['config_sha256']:
            raise RuntimeError('Existing endpoint evidence belongs to another scene/configuration')
        for task in previous['tasks']:
            if digest(task['goal_image'])!=task['goal_sha256']:
                raise RuntimeError('Existing endpoint photograph changed')
        manifest=previous
    if len(manifest['tasks'])>=count:return manifest
    with PilotEnvironment(scene,root/'capture-worker',cfg) as env:
        env.calibrate()
        # Multiple candidate headings/heights; acceptance is based on actual hover,
        # camera payload and observed local clearance, never presumed route feasibility.
        for index in range(120):
            if len(manifest['tasks'])>=count: break
            accepted=len(manifest['tasks']); band=accepted//4
            distance=[60,140,240][min(band,2)]
            offset=(index//4)*15.; azimuth=math.radians((index%4)*90)
            # Higher candidate poses avoid repeatedly resetting inside the
            # native city buildings. Actual hover and RGB/depth still decide
            # acceptance; this does not qualify routes or obstacle clearance.
            a=[offset,0.,-50.-10*(index%3)]
            b=[a[0]+distance*math.cos(azimuth),a[1]+distance*math.sin(azimuth),
               a[2]+[0,-10,10,0][index%4]]
            yaw=(index%4)*90
            start_yaw=yaw+([0,90,-90,180][accepted%4])
            task_id=f'city-{accepted:03d}'
            directory=output/task_id;directory.mkdir(exist_ok=True)
            try:
                reports={}
                for label,position,heading in [('goal',b,yaw),('start',a,start_yaw)]:
                    reports[label]=env.reset_pose(position,heading)
                    rgb,stamp,_=env.image()
                    if rgb.std()<8: raise RuntimeError('Endpoint image lacks texture')
                    depth=env.client.simGetImages([env.airsim.ImageRequest('front_custom',
                        env.airsim.ImageType.DepthPerspective,True,False)],env.vehicle)[0]
                    d=np.asarray(depth.image_data_float)
                    if not len(d) or np.any(np.isfinite(d)&(d>0)&(d<2)):
                        raise RuntimeError('Endpoint has insufficient observed forward clearance')
                    Image.fromarray(rgb).save(directory/(label+'.png'))
                    reports[label+'_camera']=dict(env.capture_pose)
                task=dict(id=task_id,scene_id=scene['scene_id'],split='train',kind='mission',
                    start=a,goal=b,start_yaw_deg=start_yaw,goal_yaw_deg=yaw,
                    start_image=str(directory/'start.png'),goal_image=str(directory/'goal.png'),
                    distance_m=math.dist(a,b),altitude_change_m=a[2]-b[2],
                    initial_heading_offset_deg=(start_yaw-yaw+180)%360-180,
                    goal_initially_visible=None,difficulty='unclassified',route_feasibility='unknown',
                    camera=cfg['camera'],bounds=[[-1000,-1000,-200],[1000,1000,-2]],
                    reset_evidence=reports,timeout_s=300)
                task['goal_sha256']=digest(task['goal_image'])
                manifest['tasks'].append(task)
                write(root/'tasks.json',manifest);overview(root,manifest['tasks'])
                print(json.dumps(dict(captured=task_id,distance_m=task['distance_m'],pictures=str(root/'ab-pictures.jpg'))),flush=True)
            except RuntimeError as error:
                manifest['rejected'].append(dict(index=index,reason=str(error)))
                write(root/'tasks.json',manifest)
        brake(env)
    if len(manifest['tasks'])<count:
        raise RuntimeError(f"Only {len(manifest['tasks'])}/{count} endpoints qualified; no fabricated replacements")
    return manifest


def qualify(root, seconds=1800):
    root=Path(root).resolve();scene,cfg=configure(root)
    tasks=read(root/'tasks.json')['tasks']
    if not tasks:raise RuntimeError('Capture A/B photographs first')
    report=dict(schema='native-scene-qualification/v1',passed=False,resets=[],motion=[],
        scene_sha256=digest(root/'scene.json'),config_sha256=digest(root/'config.json'),
        task_manifest_sha256=digest(root/'tasks.json'))
    samples=[];scheduler=None
    try:
        with PilotEnvironment(scene,root/'qualification-workers'/uuid.uuid4().hex[:12],cfg) as env:
            env.calibrate();report['color_order']=env.color
            for i in range(20):
                task=tasks[i%len(tasks)]
                try:
                    attempts=env.reset_pose(task['start'],task['start_yaw_deg'])
                    report['resets'].append(dict(passed=True,attempts=attempts))
                except RuntimeError as error:
                    report['resets'].append(dict(passed=False,error=str(error)))
                write(root/'qualification.json',report)
            if sum(r['passed'] for r in report['resets'])<19:
                raise RuntimeError('Fewer than 19/20 reset cycles qualified')
            task=tasks[0]
            for command in ([1,0,0,0],[0,1,0,0],[0,0,-.5,0],[0,0,0,15]):
                obs=env.reset(task,'motion-'+str(len(report['motion'])))
                before=obs['state'];end=time.perf_counter()+2
                while time.perf_counter()<end:
                    result=env.step(command,False,obs['frame']);obs=result['observation']
                    if result['terminated']:raise RuntimeError('Motion probe terminated: '+str(result['event']))
                after=obs['state'];brake(env)
                delta=np.array(after['position'])-before['position']
                yaw=math.radians(before['attitude_deg'][2])
                body=[delta[0]*math.cos(yaw)+delta[1]*math.sin(yaw),-delta[0]*math.sin(yaw)+delta[1]*math.cos(yaw),delta[2]]
                yaw_delta=(after['attitude_deg'][2]-before['attitude_deg'][2]+180)%360-180
                index=next(i for i,v in enumerate(command) if v)
                measured=body[index] if index<3 else yaw_delta
                passed=bool(measured*command[index]>0 and abs(measured)>(.15 if index<3 else 3))
                report['motion'].append(dict(command=command,body_delta=body,yaw_delta=yaw_delta,passed=passed))
                if not passed:raise RuntimeError('Command axis or yaw response disagrees')
            # Exercise the unchanged independent freshness brake deliberately.
            obs=env.reset(task,'stale-brake');env.step([0,0,0,0],False,obs['frame'])
            time.sleep(.4)
            report['stale_brake']=bool(env.fault and env.sent==[0.]*4)
            brake(env)
            if not report['stale_brake']:raise RuntimeError('Freshness brake was not observed')
            # Deliberate ground contact is diagnostic control, never PPO data.
            env.fault=None;env.reset_pose(task['start'],task['start_yaw_deg'])
            baseline=env.state()['collision_ns'];contact_started=time.perf_counter()
            report['collision_reporting']=False
            while time.perf_counter()-contact_started<90:
                env.client.moveByVelocityAsync(0,0,1,.2,vehicle_name=env.vehicle)
                state=env.state()
                if state['collision'] and state['collision_ns']>baseline:
                    report['collision_reporting']=True;report['contact']=state;break
                time.sleep(.1)
            brake(env)
            if not report['collision_reporting']:raise RuntimeError('Ground contact not reported within diagnostic limit')
            # Include the actual temporal actor in the measured GPU workload.
            # Its outputs are logged for timing; diagnostic hover owns commands.
            import torch
            from .ppo_core import ActorCritic
            from .ppo_scheduler import FeatureBank, Inference
            from .compute import ComputeLane
            torch.set_num_threads(2)
            checkpoint_path=root/'weights/previous-update-000002.pt'
            model=ActorCritic(root/'weights/mobilenet-v3-large-imagenet1k-v2.pt').cuda()
            model.load_state_dict(torch.load(checkpoint_path,map_location='cpu',weights_only=False)['model'],strict=True)
            model.eval();bank=FeatureBank();scheduler=Inference(model,ComputeLane(),{0:bank},None,wait_s=0.)
            goal=Image.open(task['goal_image']).convert('RGB')
            scheduler.call('initialize',worker=0,image=goal,path=task['goal_image'])
            warm=dict(rgb=goal.tobytes(),rgb_path=task['goal_image'],sim_s=0.,preceding_command=[0.]*4)
            scheduler.call('decision',worker=0,obs=warm,execution=False)
            bank.history.clear();bank.memory.frames.clear()
            report['actor_checkpoint_sha256']=digest(checkpoint_path)
            # Continuous camera/control/recording, with ordinary episode boundaries.
            started=time.perf_counter();episode=0
            while time.perf_counter()-started<seconds:
                obs=env.reset(task,f'continuous-{episode:03d}');episode+=1
                bank.history.clear();bank.memory.frames.clear()
                until=min(started+seconds,time.perf_counter()+60)
                while time.perf_counter()<until:
                    t=time.perf_counter();decision=scheduler.call('decision',worker=0,obs=obs,execution=False)
                    source_age=time.perf_counter()-obs['source_wall']
                    result=env.step([0,0,0,0],False,obs['frame']);obs=result['observation']
                    samples.append(dict(step_s=time.perf_counter()-t,image_rpc_s=obs['timing']['image_rpc_s'],source_age_s=source_age,decision_s=decision['decision_s'],
                        encode_s=decision['encode_s'],context_s=decision['context_s'],policy_s=decision['policy_s']))
                    bank.trim()
                    if result['terminated']:raise RuntimeError('Continuous run terminated: '+str(result['event']))
                brake(env)
                report['continuous_wall_s']=time.perf_counter()-started
                write(root/'qualification.json',report)
            report['timing']={key:{str(q):float(np.percentile([r[key] for r in samples],q)) for q in (50,95,99,100)} for key in samples[0]}
            report['continuous_frames']=len(samples)
            report['passed']=seconds>=1800
    except BaseException as error:
        report['failure']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        if scheduler:scheduler.close()
        if samples:
            report['timing']={key:{str(q):float(np.percentile([r[key] for r in samples],q)) for q in (50,95,99,100)} for key in samples[0]}
        write(root/'qualification.json',report)
    return report


def reference(root):
    """Bounded, privileged straight-line reference proof; stops on observed obstruction."""
    from .video import render
    from .common import FlightLock
    root=Path(root).resolve();scene,cfg=configure(root)
    task=read(root/'tasks.json')['tasks'][0]
    output=root/'reference'/uuid.uuid4().hex[:12];output.mkdir(parents=True)
    event='time_limit';steps=0;error=None
    with FlightLock(root,'reference-proof'):
        try:
            with PilotEnvironment(scene,output,cfg) as env:
                env.calibrate();env.reference_capture=True
                obs=env.reset(task,'reference');started=time.perf_counter()
                while time.perf_counter()-started<90:
                    delta=np.asarray(task['goal'])-obs['state']['position']
                    if np.linalg.norm(delta)<1.5:
                        event='within_1.5m';break
                    values=env.reference_depth;h,w=values.shape
                    view=values[h//3:2*h//3,w//3:2*w//3]
                    if not view.size or not np.isfinite(view).all() or float(view.min())<5:
                        event='observed_obstruction';break
                    yaw=math.radians(obs['state']['attitude_deg'][2])
                    world=delta/max(np.linalg.norm(delta),1.)
                    command=[float(world[0]*math.cos(yaw)+world[1]*math.sin(yaw)),
                        float(-world[0]*math.sin(yaw)+world[1]*math.cos(yaw)),0.,0.]
                    # This proof does not choose blind climbs or descents.
                    result=env.step(command,False,obs['frame']);obs=result['observation'];steps+=1
                    if result['terminated']:event=result['event'];break
                brake(env)
                if env.recorder:env.recorder.flush()
                write(output/'result.json',dict(kind='privileged_reference_only',training_data=False,
                    task=task['id'],event=event,steps=steps,final_state=obs['state'],goal_distance_m=float(np.linalg.norm(np.asarray(task['goal'])-obs['state']['position']))))
        except Exception as exception:
            error=str(exception);event='infrastructure_failure'
            write(output/'result.json',dict(kind='privileged_reference_only',training_data=False,event=event,error=error,steps=steps))
        if (output/'reference/telemetry.jsonl').exists():
            render(output/'reference',task,root/'reference-proof.mp4','Privileged reference, not PPO: '+event)
    if error:raise RuntimeError(error)
    return str(root/'reference-proof.mp4')
