"""Owned native Linux CityEnviron session using the released ProjectAirSim SDK.

This is a new engine qualification track, not equivalent to old Windows receipts.
Ground truth is available only to recording/reset/qualification code.
"""
import asyncio
import copy
import json
import logging
import math
import os
import queue
import random
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import numpy as np
from .common import digest, write


class ProjectCity:
    def __init__(self, root, output, diagnostic_depth=False):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        if not self.output.is_relative_to(self.root):
            raise ValueError('Scene output must be inside the HDD workspace')
        self.output.mkdir(parents=True,exist_ok=True)
        self.process=None;self.client=None;self.log=None;self.control_thread=None
        self.collision_events=[];self.command_receipts=[]
        self.images=queue.Queue(maxsize=2)
        self.loop=asyncio.new_event_loop()
        self.diagnostic_depth=diagnostic_depth
        self.collision_epoch=0
        self.epoch_started_wall=0.

    def __enter__(self):
        from projectairsim import ProjectAirSimClient, World, Drone
        import json5
        for port in (8989,8990):
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.2):
                    raise RuntimeError('Unowned ProjectAirSim endpoint is occupied')
            except OSError:pass
        source=self.root/'assets/cityenviron-linux-1.0.1/extracted'
        binary=source/'CityEnviron/Binaries/Linux/CityEnviron-Linux-Shipping'
        binary.chmod(binary.stat().st_mode|0o100)
        logging.getLogger('projectairsim').addHandler(logging.StreamHandler())
        environment=os.environ.copy()
        environment['HOME']=str(self.root/'cache/home')
        environment['CUDA_VISIBLE_DEVICES']='0'
        self.log=(self.output/'simulator.log').open('w')
        argv=[str(binary),'-RenderOffscreen','-nosound','-ResX=640','-ResY=480','-graphicsadapter=0',
              '-vulkan','-unattended','-NoSplash','-log','-abslog='+str(self.output/'unreal.log')]
        self.process=subprocess.Popen(argv,cwd=source,env=environment,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            deadline=time.monotonic()+120
            while time.monotonic()<deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Native CityEnviron exited: '+str(self.process.returncode))
                try:
                    with socket.create_connection(('127.0.0.1',8990),timeout=.2):break
                except OSError:time.sleep(.5)
            else:raise RuntimeError('CityEnviron service did not become ready')
            self.client=ProjectAirSimClient();self.client.connect()
            self.client.socket_services.recv_timeout=5000
            upstream=self.root/'assets/upstream/projectairsim/client/python/example_user_scripts/sim_config'
            robot=json5.loads((upstream/'robot_quadrotor_fastphysics.jsonc').read_text())
            camera=dict(id='front_custom',type='camera',enabled=True,**{'parent-link':'Frame','capture-interval':.05,
                'origin':{'xyz':'.5 0 -.1','rpy-deg':'0 0 0'},'capture-settings':[
                    {'image-type':0,'width':640,'height':480,'fov-degrees':90,'capture-enabled':True,
                     'streaming-enabled':True,'pixels-as-float':False,'compress':False,'target-gamma':2.2}]})
            robot['sensors']=[camera]+[r for r in robot['sensors'] if r['type']!='camera']
            if self.diagnostic_depth:
                camera['capture-settings'].append({'image-type':2,'width':160,'height':120,'fov-degrees':90,
                    'capture-enabled':True,'streaming-enabled':False,'pixels-as-float':True,'compress':False})
            config=json5.loads((upstream/'scene_basic_drone.jsonc').read_text())
            config['id']='SceneBasicDrone'
            (self.output/'robot-config.json').write_text(json.dumps(robot,indent=2),encoding='utf-8')
            config['actors']=[dict(type='robot',name='drone_1',origin={'xyz':'0 0 -20','rpy-deg':'0 0 0'},**{'robot-config':'robot-config.json'})]
            # The steppable clock runs autonomously at real time; no per-action stepping.
            config['clock']=dict(type='steppable',**{'step-ns':3000000,'real-time-update-rate':3000000,'pause-on-start':False})
            (self.output/'scene-config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
            python_rng=random.getstate()
            try:self.world=World(self.client,'scene-config.json',sim_config_path=str(self.output),delay_after_load_sec=2)
            finally:random.setstate(python_rng)
            self.drone=Drone(self.client,self.world,'drone_1')
            self.client.subscribe(self.drone.robot_info['collision_info'],self._collision)
            self.client.subscribe(self.drone.sensors['front_custom']['scene_camera'],self._image)
            self.drone.enable_api_control();self.drone.arm()
            self.control_thread=threading.Thread(target=self.loop.run_forever,name='project-control',daemon=True)
            self.control_thread.start()
            write(self.output/'scene-receipt.json',dict(schema='photo-goal-project-city/v1',
                binary=str(binary),binary_sha256=digest(binary),scene='CityEnviron-Linux-1.0.1',
                settings_sha256=digest(self.output/'scene-config.json'),clock=config['clock'],
                legacy_windows_equivalence=False,launch_argv=argv))
            return self
        except BaseException:
            self.__exit__(None,None,None);raise

    def _collision(self,topic,event):
        self.collision_events.append(dict(topic=getattr(topic,'path',str(topic)),event=event,
            received_wall=time.perf_counter(),epoch=self.collision_epoch))

    def reload_for_reset(self,position,yaw):
        # Native v1.0.1 retains its physical contact latch across SetPose.
        # Recreate the robot through the official LoadScene API at episode
        # boundaries. No reload or clock pause occurs within a flight.
        self.quiesce_tasks()
        config=copy.deepcopy(self.world.get_configuration())
        config['clock']['pause-on-start']=True
        config['actors'][0]['origin']={'xyz':' '.join(str(v) for v in position),'rpy-deg':f'0 0 {yaw}'}
        self.world.load_scene(config,delay_after_load_sec=.2)
        from projectairsim import Drone
        self.drone=Drone(self.client,self.world,'drone_1')
        self.collision_epoch+=1
        self.epoch_started_wall=time.perf_counter()
        while not self.images.empty():
            try:self.images.get_nowait()
            except queue.Empty:break
        self.drone.enable_api_control();self.drone.arm()
        self.client.subscribe(self.drone.robot_info['collision_info'],self._collision)
        self.client.subscribe(self.drone.sensors['front_custom']['scene_camera'],self._image)

    def quiesce_tasks(self,cancel_remaining=False):
        if not self.control_thread:return
        async def drain():
            pending=[t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if pending:
                _,remaining=await asyncio.wait(pending,timeout=2)
                if remaining:
                    if not cancel_remaining:raise RuntimeError('Native control tasks did not quiesce')
                    for task in remaining:task.cancel()
                    await asyncio.gather(*remaining,return_exceptions=True)
        asyncio.run_coroutine_threadsafe(drain(),self.loop).result(timeout=3)

    def _image(self,topic,image):
        packet=(image,time.perf_counter())
        while True:
            try:self.images.put_nowait(packet);return
            except queue.Full:
                try:self.images.get_nowait()
                except queue.Empty:pass

    def command(self,command,duration=.2):
        if len(command)!=4 or not np.isfinite(command).all():raise ValueError('Invalid command')
        if np.linalg.norm(command[:2])>3 or abs(command[2])>1 or abs(command[3])>45:raise ValueError('Command envelope exceeded')
        started=time.perf_counter()
        async def send():
            return await self.drone.move_by_velocity_body_frame_async(*command[:3],duration=duration,yaw_is_rate=True,yaw=math.radians(command[3]))
        task=asyncio.run_coroutine_threadsafe(send(),self.loop).result(timeout=3)
        receipt=dict(command=list(command),request_wall=started,ack_wall=time.perf_counter(),
                     ack_sim_ns=self.world.get_sim_time(),duration=duration)
        self.command_receipts.append(receipt)
        return task,receipt

    def reset(self,position,yaw=0):
        self.reload_for_reset(position,yaw)
        try:
            self.command([0.,0.,0.,0.])
        finally:self.world.resume()
        time.sleep(1)
        return self.state()

    def state(self):
        started=time.perf_counter()
        return dict(sim_ns=self.world.get_sim_time(),kinematics=self.drone.get_ground_truth_kinematics(),
                    received_wall=time.perf_counter(),request_wall=started)

    def image(self):
        from projectairsim.utils import unpack_image
        started=time.perf_counter()
        image,received=self.images.get(timeout=5)
        while not self.images.empty():image,received=self.images.get_nowait()
        raw=unpack_image(image)
        if raw.shape!=(480,640,3) or raw.dtype!=np.uint8:raise RuntimeError('Invalid calibrated camera RGB')
        encoding=image['encoding']
        # Honor the renderer's explicit channel encoding; never guess from color.
        if encoding in ('BGR','BGR8','PNG'):raw=raw[:,:,::-1].copy()
        elif encoding not in ('RGB','RGB8','rgb8'):raise RuntimeError('Unqualified scene channel encoding: '+encoding)
        receipt={k:v for k,v in image.items() if k!='data'}
        receipt.update(request_wall=started,received_wall=received,consumed_wall=time.perf_counter(),transport='native-pubsub')
        return raw,receipt

    def __exit__(self,*args):
        write(self.output/'collision-events.json',dict(events=self.collision_events))
        if self.client:
            try:
                if self.control_thread:
                    # Shutdown follows a completed/interrupted flight. Resume
                    # the stopped clock so the final brake command can finish.
                    self.world.resume()
                    self.command([0.,0.,0.,0.])
                    self.quiesce_tasks(cancel_remaining=True)
                self.client.disconnect()
            except Exception:pass
        if self.control_thread:
            self.loop.call_soon_threadsafe(self.loop.stop);self.control_thread.join(timeout=3)
        if self.process and self.process.poll() is None:
            os.killpg(self.process.pid,signal.SIGTERM)
            try:self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(self.process.pid,signal.SIGKILL);self.process.wait()
        if self.log:self.log.close()
