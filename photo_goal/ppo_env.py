"""Host AirSim worker for the simulation-only PPO pilot.

Run separately from the GPU trainer. Each RPC thread owns its own AirSim client.
Only initialization may place poses. Training may freeze physics at an update
boundary; evaluation never pauses. Privileged info is a reward sidecar.
"""
import argparse
from collections import deque
import json
import math
from multiprocessing.connection import Listener
from pathlib import Path
import queue
import shutil
import threading
import time
import traceback
import numpy as np
from .common import read,write,digest
from .collect import SceneProcess


def xyz(v):return [v.x_val,v.y_val,v.z_val]


def heading_error(a,b):return abs((a-b+180)%360-180)


def camera_settings(settings,cfg,port):
    import copy
    settings=copy.deepcopy(settings);settings['ApiServerPort']=port
    settings['ClockSpeed']=1.;settings['ViewMode']='NoDisplay'
    vehicle=settings['Vehicles']['drone_1'];vehicle['DefaultVehicleState']='Disarmed'
    vehicle['EnableCollisions']=True;vehicle['EnableCollisionPassthrogh']=False
    c=cfg['camera'];camera=vehicle['Cameras']['front_custom']
    camera.update(X=c['x'],Y=c['y'],Z=c['z'],Pitch=c['pitch'],Roll=c['roll'],Yaw=c['yaw'])
    for capture in camera['CaptureSettings']:
        capture.update(Width=c['width'],Height=c['height'],FOV_Degrees=c['fov'],MotionBlurAmount=0)
        if capture['ImageType']==2:
            capture['Width'],capture['Height']=cfg['qualification_depth_size']
    return settings


class Recorder:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.queue=queue.Queue(maxsize=64);self.error=None
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()

    def put(self,name,value):
        if self.error:raise RuntimeError('Recorder failed: '+self.error)
        try:self.queue.put_nowait((name,value))
        except queue.Full:raise RuntimeError('Recording queue overflow; episode invalid')

    def run(self):
        from PIL import Image
        try:
            with (self.root/'telemetry.jsonl').open('w') as stream:
                while True:
                    item=self.queue.get()
                    if item is None:break
                    name,value=item
                    if name=='barrier':stream.flush();value.set()
                    elif name.endswith('.png'):Image.fromarray(value).save(self.root/name,compress_level=1)
                    else:stream.write(json.dumps(value,allow_nan=False)+'\n')
        except Exception as error:self.error=str(error)

    def close(self):
        while self.thread.is_alive():
            try:self.queue.put(None,timeout=.2);break
            except queue.Full:continue
        self.thread.join(timeout=30)
        if self.thread.is_alive() or self.error:raise RuntimeError('Recorder did not finish cleanly')

    def flush(self):
        done=threading.Event();self.put('barrier',done)
        if not done.wait(30) or self.error:raise RuntimeError('Recorder flush failed')


class PilotEnvironment:
    def __init__(self,scene,output,cfg,port=43551):
        import airsim
        self.airsim=airsim;self.cfg=cfg;self.vehicle='drone_1';self.port=port
        self.root=Path(output);self.root.mkdir(parents=True,exist_ok=True)
        self.scene=dict(scene)
        settings=camera_settings(read(scene['settings']),cfg,port)
        write(self.root/'settings.json',settings)
        self.scene.update(settings=str((self.root/'settings.json').resolve()),worker_root=str(self.root.resolve()))
        self.owned=SceneProcess(self.scene);self.thread=None;self.recorder=None
        self.lock=threading.Lock();self.shutdown=threading.Event();self.active=False
        self.dispatch_idle=threading.Event();self.dispatch_idle.set()
        self.command=[0.]*4;self.sent=[0.]*4;self.source_wall=0.;self.fault=None
        self.dispatches=deque(maxlen=256);self.sequence=0;self.command_frame=0
        self.phase='engineering';self.paused=False;self.pause_wall_s=0.;self.settling=False

    def __enter__(self):
        self.owned.__enter__()
        try:
            self.client=self.airsim.MultirotorClient(ip='127.0.0.1',port=self.port,timeout_value=5)
            self.client.enableApiControl(True,self.vehicle)
            self.thread=threading.Thread(target=self.dispatch,daemon=True);self.thread.start()
            return self
        except BaseException:
            self.owned.__exit__(None,None,None);raise

    def __exit__(self,*args):
        self.active=False;self.shutdown.set()
        if self.thread:self.thread.join(timeout=6)
        try:
            if self.recorder:self.recorder.close();self.recorder=None
        finally:self.owned.__exit__(*args)

    def dispatch(self):
        a=self.airsim;client=a.MultirotorClient(ip='127.0.0.1',port=self.port,timeout_value=2)
        pending=deque()
        while not self.shutdown.is_set():
            start=time.perf_counter()
            try:
                with self.lock:
                    active=self.active;age=start-self.source_wall;stale=age>self.cfg['freshness_s']
                    command=[0.]*4 if stale else list(self.command);frame=self.command_frame
                    if active:self.dispatch_idle.clear()
                if active:
                    future=client.moveByVelocityBodyFrameAsync(*command[:3],.2,
                        drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=a.YawMode(True,command[3]),vehicle_name=self.vehicle)
                    pending.append(future)
                    # msgpackrpc's asynchronous call queues work on this client's
                    # IOLoop. A synchronous ping services it without waiting for
                    # the 200 ms motion duration. Another thread's client cannot
                    # service this connection. Check completed futures for errors.
                    if not client.ping():raise RuntimeError('Command RPC ping failed')
                    while pending and pending[0]._set_flag:pending.popleft().get()
                    if len(pending)>16:raise RuntimeError('Command acknowledgements stalled')
                    with self.lock:
                        self.sent=command;self.sequence+=1
                        row=dict(kind='dispatch',id=self.sequence,wall=start,
                            source_frame=frame,age_s=age,command=command,stale=stale,
                            active_policy=not getattr(self,'done',True) and not self.settling,
                            rpc_wall_s=time.perf_counter()-start)
                        self.dispatches.append(row)
                        if self.recorder:self.recorder.put('dispatch',row)
                        if stale and row['active_policy'] and self.cfg.get('schema')=='photo-map-ppo/v2':
                            self.fault='Active control watchdog exceeded source freshness'
            except Exception as error:self.fault=str(error)
            finally:self.dispatch_idle.set()
            self.shutdown.wait(max(0,self.cfg['step_s']-(time.perf_counter()-start)))

    def state(self):
        a=self.airsim;s=self.client.getMultirotorState(self.vehicle);k=s.kinematics_estimated
        pitch,roll,yaw=a.to_eularian_angles(k.orientation)
        collision=self.client.simGetCollisionInfo(self.vehicle)
        return dict(sim_ns=s.timestamp,position=xyz(k.position),velocity=xyz(k.linear_velocity),
            attitude_deg=[math.degrees(roll),math.degrees(pitch),math.degrees(yaw)],
            speed=float(np.linalg.norm(xyz(k.linear_velocity))),
            collision=bool(collision.has_collided),collision_ns=collision.time_stamp,
            contact=xyz(collision.impact_point),collision_object=collision.object_name)

    def reset_pose(self,position,yaw):
        a=self.airsim;cfg=self.cfg['reset'];reports=[]
        for attempt in range(cfg['attempts']):
            started=time.perf_counter()
            with self.lock:self.active=False
            if not self.dispatch_idle.wait(3):raise RuntimeError('Command thread did not quiesce before reset')
            # Arming after placement can reset SimpleFlight state in some forks.
            # Arm first, then place while paused and issue hover before release.
            self.client.enableApiControl(True,self.vehicle);self.client.armDisarm(True,self.vehicle)
            self.client.simPause(True)
            try:
                self.client.simSetVehiclePose(a.Pose(a.Vector3r(*position),a.to_quaternion(0,0,math.radians(yaw))),True,self.vehicle)
                self.client.moveByVelocityAsync(0.,0.,0.,.2,yaw_mode=a.YawMode(False,yaw),vehicle_name=self.vehicle)
            finally:self.client.simPause(False)
            baseline=self.state()['collision_ns'];stable=None;passed=False
            while time.perf_counter()-started<cfg['timeout_s']:
                self.client.moveToPositionAsync(*position,1.,timeout_sec=1.,
                    drivetrain=a.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=a.YawMode(False,yaw),vehicle_name=self.vehicle)
                s=self.state();now=time.perf_counter()
                # A newly reported contact is evidence against this endpoint;
                # do not spend the full hover timeout pushing into its surface.
                if s['collision'] and s['collision_ns']>baseline:
                    break
                good=(math.dist(position,s['position'])<=cfg['position_m'] and
                    heading_error(s['attitude_deg'][2],yaw)<=cfg['heading_deg'] and s['speed']<cfg['speed_mps'] and
                    not (s['collision'] and s['collision_ns']>baseline))
                stable=(stable or now) if good else None
                if stable and now-stable>=cfg['dwell_s']:passed=True;break
                time.sleep(.05)
            reports.append(dict(attempt=attempt,passed=passed,duration_s=time.perf_counter()-started,requested=position,actual=s))
            if passed:
                self.client.cancelLastTask(self.vehicle)
                self.client.moveByVelocityAsync(0.,0.,0.,.2,vehicle_name=self.vehicle)
                return reports
        raise RuntimeError('Reset qualification failed: '+json.dumps(reports))

    def image(self):
        a=self.airsim;started=time.perf_counter()
        try:
            requests=[a.ImageRequest('front_custom',a.ImageType.Scene,False,False)]
            if getattr(self,'reference_capture',False):
                requests.append(a.ImageRequest('front_custom',a.ImageType.DepthPerspective,True,False))
            responses=self.client.simGetImages(requests,self.vehicle);rgb=responses[0]
            if getattr(self,'reference_capture',False):
                depth=responses[1]
                self.reference_depth=np.asarray(depth.image_data_float).reshape(depth.height,depth.width)
        except Exception:
            if self.recorder:self.recorder.put('image_error',dict(kind='image_error',request_wall=started,
                failed_wall=time.perf_counter(),traceback=traceback.format_exc()))
            raise
        self.capture_timing=dict(request_wall=started,response_wall=time.perf_counter())
        if (rgb.width,rgb.height)!=(640,480) or len(rgb.image_data_uint8)!=640*480*3:
            raise ValueError('Invalid camera payload')
        raw=np.frombuffer(rgb.image_data_uint8,np.uint8).reshape(480,640,3)
        pitch,roll,yaw=a.to_eularian_angles(rgb.camera_orientation)
        self.capture_pose=dict(position=xyz(rgb.camera_position),
            attitude_deg=[math.degrees(roll),math.degrees(pitch),math.degrees(yaw)],sim_ns=rgb.time_stamp)
        if self.color=='BGR':raw=raw[:,:,::-1]
        return raw.copy(),rgb.time_stamp,started

    def calibrate(self):
        # Existing actual-binary comparison, imported only on the AirSim host.
        from .vision.calibration import measure_color_order
        for attempt in range(5):
            try:self.color=measure_color_order(self.client)['raw_channel_order'];return
            except RuntimeError:
                if attempt==4:raise
                time.sleep(2)

    def reset(self,task,attempt_id):
        if self.paused:raise RuntimeError('Cannot reset a suspended training episode')
        if str(task['scene_id'])!=str(self.scene['scene_id']):raise ValueError('Task/scene mismatch')
        if Path(attempt_id).name!=attempt_id:raise ValueError('Invalid attempt identifier')
        self.episode_limit=float(task.get('timeout_s',self.cfg['episode_s']))
        if not 0<self.episode_limit<=self.cfg['episode_s']:raise ValueError('Invalid episode limit')
        required=int((self.episode_limit+self.cfg['stop_grace_s'])/self.cfg['step_s']*640*480*3)+64*1024**2
        if shutil.disk_usage(self.root).free<required:raise RuntimeError('Insufficient recording disk for a full episode')
        with self.lock:self.active=False
        if self.recorder:self.recorder.close();self.recorder=None
        self.task=task;self.frame=0;self.fault=None
        reset_started=time.perf_counter()
        reports=self.reset_pose(task['start'],task['start_yaw_deg'])
        if not hasattr(self,'color'):self.calibrate()
        self.reset_wall_s=time.perf_counter()-reset_started
        self.recorder=Recorder(self.root/attempt_id)
        self.recorder.put('reset',dict(kind='reset',attempts=reports,task_id=task['id']))
        self.start=time.perf_counter();self.initial_distance=math.dist(task['start'],task['goal'])
        initial=self.state();self.start_sim_ns=initial['sim_ns']
        self.collision_baseline=initial['collision_ns'];self.previous_state=None
        self.done=False;self.settling=False
        with self.lock:
            self.command=self.sent=[0.]*4;self.source_wall=time.perf_counter();self.active=True;self.dispatches.clear()
        return self.observe()

    def observe(self,frozen=False):
        if self.fault:raise RuntimeError('Dispatcher failed: '+self.fault)
        rgb,stamp,source=self.image();state_started=time.perf_counter();s=self.state();state_finished=time.perf_counter()
        timing=dict(self.capture_timing,state_started_wall=state_started,state_finished_wall=state_finished,
            image_rpc_s=self.capture_timing['response_wall']-source,state_rpc_s=state_finished-state_started)
        if not frozen and self.previous_state is not None and s['sim_ns']<=self.previous_state['sim_ns']:
            raise RuntimeError('Nonadvancing simulator clock')
        with self.lock:
            preceding=list(self.sent)
        self.frame+=1
        self.last_source=source;self.previous_state=s
        self.recorder.put(f'{self.frame:06d}.png',rgb)
        self.recorder.put('observation',dict(kind='observation',frame=self.frame,capture_sim_ns=stamp,
            source_wall=source,preceding_command=preceding,state=s,camera_pose=self.capture_pose,timing=timing))
        return dict(rgb=rgb.tobytes(),frame=self.frame,sim_s=stamp/1e9,preceding_command=preceding,
            state=s,source_wall=source,timing=timing,elapsed_s=(s['sim_ns']-self.start_sim_ns)/1e9,
            reset_wall_s=self.reset_wall_s,pause_wall_s=self.pause_wall_s,
            rgb_path=str((self.recorder.root/f'{self.frame:06d}.png').resolve()))

    def event(self,s):
        if s['collision'] and s['collision_ns']>self.collision_baseline:return 'collision'
        lo,hi=self.task['bounds']
        if any(p<a or p>b for p,a,b in zip(s['position'],lo,hi)):return 'envelope'
        return None

    def in_goal(self,s,speed=True):
        c=self.cfg['arrival'];d=np.asarray(s['position'])-self.task['goal']
        return (np.linalg.norm(d[:2])<=c['horizontal_m'] and abs(d[2])<=c['vertical_m'] and
            heading_error(s['attitude_deg'][2],self.task['goal_yaw_deg'])<=c['heading_deg'] and
            (not speed or s['speed']<=c['speed_mps']))

    def step(self,command,stop,frame,freeze_after=False):
        if self.paused:raise RuntimeError('Resume with a fresh decision before stepping')
        if freeze_after and (self.phase!='training' or not self.cfg.get('training_pause')):
            raise RuntimeError('Physics pause forbidden outside admitted training')
        if self.done:raise RuntimeError('Reset required after terminal/truncated episode')
        if frame!=self.frame:raise ValueError('Action observation mismatch')
        if len(command)!=4 or not np.isfinite(command).all():raise ValueError('Invalid command')
        limits=self.cfg['limits']
        if np.linalg.norm(command[:2])>limits[0]+1e-5 or abs(command[2])>limits[2] or abs(command[3])>limits[3]:
            raise ValueError('Command exceeds envelope')
        started=time.perf_counter();before=self.previous_state
        if started-self.last_source>self.cfg['freshness_s']:raise RuntimeError('Stale policy decision; watchdog brakes')
        self.settling=bool(stop)
        with self.lock:
            self.command=[0.]*4 if stop else list(command);self.source_wall=self.last_source;self.command_frame=frame
        event=self.event(before)
        if not event and stop:
            if not self.in_goal(before,speed=False):event='false_stop'
            else:
                stable=None
                while time.perf_counter()-started<self.cfg['stop_grace_s']:
                    s=self.state();event=self.event(s)
                    if event:break
                    now=time.perf_counter();stable=(stable or now) if self.in_goal(s) else None
                    if stable and now-stable>=self.cfg['arrival']['dwell_s']:event='success';break
                    time.sleep(self.cfg['step_s'])
                event=event or 'false_stop'
        if not stop and not event:time.sleep(max(0,self.cfg['step_s']-(time.perf_counter()-self.last_source)))
        if freeze_after:
            # Freeze before the boundary capture, so there is no unrecorded
            # physical interval between the final transition and optimization.
            self.client.simPause(True)
            with self.lock:self.active=False
            if not self.dispatch_idle.wait(3):raise RuntimeError('Dispatcher failed to suspend')
            self.paused=True;self.pause_started=time.perf_counter()
        observation=self.observe()
        # The watchdog can fire while a camera RPC is blocked. Never return
        # that safety-modified interval as an ordinary valid policy transition.
        if self.fault:raise RuntimeError('Dispatcher failed: '+self.fault)
        event=self.event(observation['state']) or event
        if not event and self.task.get('kind')=='execution':
            delta=np.asarray(observation['state']['position'])-self.task['goal']
            if np.linalg.norm(delta[:2])<=1.5 and abs(delta[2])<=1.:event='subgoal_success'
        truncated=not event and observation['elapsed_s']>=self.episode_limit
        self.done=bool(event or truncated)
        if self.done:
            with self.lock:self.command=[0.]*4
        dt=(observation['state']['sim_ns']-before['sim_ns'])/1e9
        return dict(observation=observation,event=event,terminated=bool(event),truncated=truncated,dt=dt)

    def refresh_boundary(self):
        if not self.paused or self.phase!='training':raise RuntimeError('No training boundary to refresh')
        before=self.previous_state
        obs=self.observe(frozen=True)
        if obs['state']!=before:raise RuntimeError('Physics/state advanced during optimization pause')
        return obs

    def resume_boundary(self,command,stop,frame):
        if not self.paused or self.phase!='training' or self.done:raise RuntimeError('Invalid boundary resume')
        if frame!=self.frame or time.perf_counter()-self.last_source>self.cfg['freshness_s']:
            raise RuntimeError('Fresh boundary observation required')
        if len(command)!=4 or not np.isfinite(command).all():raise ValueError('Invalid resume command')
        limits=self.cfg['limits']
        if np.linalg.norm(command[:2])>limits[0]+1e-5 or abs(command[2])>limits[2] or abs(command[3])>limits[3]:
            raise ValueError('Resume command exceeds limits')
        # Service the new command RPC while still frozen; never unpause onto
        # an old command or inject a brake into a continuing transition.
        actual=[0.]*4 if stop else command
        self.client.moveByVelocityBodyFrameAsync(*actual[:3],.2,
            drivetrain=self.airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=self.airsim.YawMode(True,actual[3]),vehicle_name=self.vehicle)
        self.client.ping()
        self.pause_wall_s+=time.perf_counter()-self.pause_started
        with self.lock:
            self.command=list(actual);self.source_wall=self.last_source;self.command_frame=frame;self.active=True
        self.paused=False;self.client.simPause(False)
        return self.step(command,stop,frame)

    def release_terminal(self):
        if not self.paused or not self.done:raise RuntimeError('Only terminal boundaries can be released')
        self.client.moveByVelocityAsync(0.,0.,0.,.2,vehicle_name=self.vehicle);self.client.ping()
        self.pause_wall_s+=time.perf_counter()-self.pause_started
        self.paused=False;self.client.simPause(False)
        return dict(released=True)


def serve(scene,output,cfg,address,authfile,sim_port=43551):
    host,port=address.rsplit(':',1)
    if host!='127.0.0.1':raise ValueError('Pilot worker must bind loopback only')
    env=None
    with Listener((host,int(port)),authkey=Path(authfile).read_bytes()) as listener:
        with listener.accept() as connection:
            try:
                while True:
                    request=connection.recv();op=request['op']
                    if op=='close':break
                    try:
                        if op=='select':
                            if Path(request['session']).name!=request['session']:raise ValueError('Invalid worker session')
                            if env:env.__exit__(None,None,None);env=None
                            descriptor=request.get('scene') or read(scene)
                            env=PilotEnvironment(descriptor,Path(output)/request['session'],cfg,port=sim_port)
                            env.phase=request.get('phase','engineering')
                            env.__enter__();answer=dict(scene_id=descriptor['scene_id'],config=cfg,sim_port=sim_port,
                                source_sha256=digest(__file__))
                        elif op=='reset':answer=env.reset(request['task'],request['attempt_id'])
                        elif op=='step':answer=env.step(request['command'],request['stop'],request['frame'],request.get('freeze_after',False))
                        elif op=='refresh_boundary':answer=env.refresh_boundary()
                        elif op=='flush':
                            if env.recorder:env.recorder.flush()
                            answer=dict(flushed=True)
                        elif op=='finish_session':
                            if env:env.__exit__(None,None,None);env=None
                            answer=dict(finished=True)
                        elif op=='resume_boundary':answer=env.resume_boundary(request['command'],request['stop'],request['frame'])
                        elif op=='release_terminal':answer=env.release_terminal()
                        elif op=='end_episode':
                            env.done=True;env.settling=True
                            with env.lock:env.command=[0.]*4
                            answer=env.release_terminal() if env.paused else dict(ended=True)
                        elif op=='brake':
                            with env.lock:env.command=[0.]*4;env.settling=True
                            answer=dict(braking=True)
                        else:raise ValueError('Unknown worker operation')
                        connection.send(dict(ok=True,result=answer))
                    except Exception as error:
                        detail=dict(operation=op,error=type(error).__name__+': '+str(error),
                            traceback=traceback.format_exc(),wall=time.perf_counter())
                        if env:
                            with env.lock:env.command=[0.]*4
                            process=env.owned.process
                            detail['simulator_exit_code']=process.poll() if process else None
                            if env.owned.log:
                                log=Path(env.owned.log.name);detail['simulator_log']=str(log)
                                with log.open('rb') as stream:
                                    stream.seek(max(0,log.stat().st_size-8192))
                                    tail=stream.read().decode('utf-8',errors='replace')
                                detail['simulator_log_tail']=tail
                                detail['native_crash_evidence']=('Signal 11 caught' in tail or
                                    'CommonUnixCrashHandler: Signal=' in tail)
                            write(env.root/'worker-failure.json',detail)
                        message=detail['error']
                        if detail.get('native_crash_evidence') or detail.get('simulator_exit_code') is not None:
                            message='Native simulator failure during '+op+': '+message
                        connection.send(dict(ok=False,error=message,diagnostic=detail))
            except EOFError:pass
            finally:
                if env:env.__exit__(None,None,None)


def qualify(scene,output,cfg,position,yaw):
    """Engineering evidence, never automatic camera/geography approval."""
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    record=dict(schema='photo-map-ppo-qualification/v1',scene_sha256=digest(scene),implementation_sha256=digest(__file__),
        config=cfg,qualified=False,camera_reviewed=False,geometry_reviewed=False,resets=[])
    try:
        with PilotEnvironment(read(scene),out/'worker',cfg) as env:
            from PIL import Image
            for i in range(cfg['reset']['qualification_repetitions']):
                rows=env.reset_pose(position,yaw);record['resets'].append(rows)
                if i==0:
                    env.calibrate();rgb,_,_=env.image();Image.fromarray(rgb).save(out/'camera.png')
                write(out/'qualification.json',record)
            record['reset_passed']=True
            # Static mounting inspection at representative body attitudes is
            # preparation, never part of a recorded flight or learning rollout.
            a=env.airsim;env.client.simPause(True)
            record['camera_attitudes']=[]
            try:
                env.client.armDisarm(False,env.vehicle)
                for roll in (-20,0,20):
                    for pitch in (-20,0,20):
                        k=a.KinematicsState();k.position=a.Vector3r(*position)
                        k.orientation=a.to_quaternion(math.radians(pitch),math.radians(roll),math.radians(yaw))
                        k.linear_velocity=k.angular_velocity=k.linear_acceleration=k.angular_acceleration=a.Vector3r()
                        env.client.simSetKinematics(k,True,env.vehicle)
                        # A render/physics tick is necessary on forks whose
                        # image camera transform otherwise remains stale paused.
                        env.client.simContinueForTime(.01)
                        rgb,_,_=env.image();image_path=out/f'camera-roll{roll}-pitch{pitch}.png';Image.fromarray(rgb).save(image_path)
                        measured=env.capture_pose['attitude_deg']
                        record['camera_attitudes'].append(dict(requested=[roll,pitch,yaw],measured=measured,
                            pose_agreement=all(heading_error(a,b)<=5 for a,b in zip(measured,[roll,pitch,yaw])),
                            image_sha256=digest(image_path)))
            finally:env.client.simPause(False)
    except Exception as error:record.update(reset_passed=False,error=str(error))
    finally:write(out/'qualification.json',record)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['serve','qualify'])
    p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(Path(__file__).with_name('ppo_pilot.json')))
    p.add_argument('--address',default='127.0.0.1:43651');p.add_argument('--authfile')
    p.add_argument('--sim-port',type=int,default=43551)
    p.add_argument('--position',nargs=3,type=float);p.add_argument('--yaw',type=float,default=0)
    a=p.parse_args();cfg=read(a.config)
    if a.stage=='serve':
        if not a.authfile:p.error('--authfile is required')
        serve(a.scene,a.output,cfg,a.address,a.authfile,a.sim_port)
    else:
        if a.position is None:p.error('--position is required')
        qualify(a.scene,a.output,cfg,a.position,a.yaw)
