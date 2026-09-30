"""Explicit ProjectAirSim backend for the existing continuous city flight loop.

This translates RPC types, radians and async futures. It supplies no navigation
policy and never declares dynamics equivalence to historical Windows flights.
"""
import asyncio
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import time
from .project_city import ProjectCity
from .common import read


class Vector3r:
    def __init__(self,x=0.,y=0.,z=0.):self.x_val,self.y_val,self.z_val=x,y,z


class Quaternionr:
    def __init__(self,w=1.,x=0.,y=0.,z=0.):self.w_val,self.x_val,self.y_val,self.z_val=w,x,y,z


class Pose:
    def __init__(self,position,orientation):self.position,self.orientation=position,orientation


def to_quaternion(pitch,roll,yaw):
    cr,sr=math.cos(roll/2),math.sin(roll/2);cp,sp=math.cos(pitch/2),math.sin(pitch/2);cy,sy=math.cos(yaw/2),math.sin(yaw/2)
    return Quaternionr(cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy)


def to_eularian_angles(q):
    w,x,y,z=q.w_val,q.x_val,q.y_val,q.z_val
    return math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def vec(row):return Vector3r(row['x'],row['y'],row['z'])
def quat(row):return Quaternionr(row['w'],row['x'],row['y'],row['z'])


class Future:
    def __init__(self,task,loop):self.task,self.loop=task,loop
    @property
    def _set_flag(self):return self.task.done()
    def get(self):
        if self.task.done():return self.task.result()
        async def wait():return await self.task
        return asyncio.run_coroutine_threadsafe(wait(),self.loop).result(timeout=10)
    def join(self):return self.get()


class ProjectSceneProcess:
    def __init__(self,scene,facade):self.scene,self.facade=scene,facade;self.process=None;self.log=None
    def __enter__(self):
        self.session=ProjectCity(self.scene['project_root'],Path(self.scene['worker_root'])/'project-simulator',diagnostic_depth=True)
        self.session.__enter__();self.facade.session=self.session
        self.process,self.log=self.session.process,self.session.log
        return self
    def __exit__(self,*args):
        try:
            self.session.world.resume()
            self.session.quiesce_tasks(cancel_remaining=True)
        except Exception:pass
        for client in self.facade.clients:
            try:client.connection.disconnect()
            except Exception:pass
        self.session.__exit__(*args)


class Client:
    def __init__(self,session,timeout_value=5):
        from projectairsim import ProjectAirSimClient,Drone
        self.session=session
        self.connection=ProjectAirSimClient();self.connection.connect_services()
        self.connection.socket_services.recv_timeout=int(timeout_value*1000)
        self.world=copy.copy(session.world);self.world.client=self.connection
        self.drone=Drone(self.connection,self.world,'drone_1')

    def _async(self,coroutine):
        task=asyncio.run_coroutine_threadsafe(coroutine,self.session.loop).result(timeout=3)
        return Future(task,self.session.loop)

    def enableApiControl(self,enabled,vehicle_name='drone_1'):
        return self.drone.enable_api_control() if enabled else self.drone.disable_api_control()
    def armDisarm(self,armed,vehicle_name='drone_1'):return self.drone.arm() if armed else self.drone.disarm()
    def ping(self):return self.session.process.poll() is None
    def cancelLastTask(self,vehicle_name='drone_1'):return self.drone.cancel_last_task()
    def simPause(self,paused):return self.world.pause() if paused else self.world.resume()
    def simIsPause(self):return self.world.is_paused()

    def moveByVelocityBodyFrameAsync(self,vx,vy,vz,duration,drivetrain=0,yaw_mode=None,vehicle_name='drone_1'):
        yaw_mode=yaw_mode or SimpleNamespace(is_rate=True,yaw_or_rate=0.)
        return self._async(self.drone.move_by_velocity_body_frame_async(vx,vy,vz,duration=duration,
            yaw_control_mode=drivetrain,yaw_is_rate=yaw_mode.is_rate,yaw=math.radians(yaw_mode.yaw_or_rate)))

    def moveByVelocityAsync(self,vx,vy,vz,duration,drivetrain=0,yaw_mode=None,vehicle_name='drone_1'):
        yaw_mode=yaw_mode or SimpleNamespace(is_rate=True,yaw_or_rate=0.)
        return self._async(self.drone.move_by_velocity_async(vx,vy,vz,duration=duration,
            yaw_control_mode=drivetrain,yaw_is_rate=yaw_mode.is_rate,yaw=math.radians(yaw_mode.yaw_or_rate)))

    def moveToPositionAsync(self,x,y,z,velocity,timeout_sec=1,drivetrain=0,yaw_mode=None,vehicle_name='drone_1'):
        yaw_mode=yaw_mode or SimpleNamespace(is_rate=True,yaw_or_rate=0.)
        return self._async(self.drone.move_to_position_async(x,y,z,velocity,timeout_sec=timeout_sec,
            yaw_control_mode=drivetrain,yaw_is_rate=yaw_mode.is_rate,yaw=math.radians(yaw_mode.yaw_or_rate)))

    def getMultirotorState(self,vehicle_name='drone_1'):
        k=self.drone.get_ground_truth_kinematics();p=k['pose'];v=k['twist'];a=k['accels']
        return SimpleNamespace(timestamp=k['time_stamp'],kinematics_estimated=SimpleNamespace(
            position=vec(p['position']),orientation=quat(p['orientation']),linear_velocity=vec(v['linear']),
            angular_velocity=vec(v['angular']),linear_acceleration=vec(a['linear']),angular_acceleration=vec(a['angular'])))

    def simGetCollisionInfo(self,vehicle_name='drone_1'):
        events=self.session.collision_events
        # The native collision topic publishes only actual contacts. Recorded
        # v1.0.1 events have no has_collided boolean; time_stamp/object_name and
        # impact_point identify them. Do not silently discard these contacts.
        event=next((r['event'] for r in reversed(events)
            if r.get('epoch',0)==self.session.collision_epoch and
               r['received_wall']>=self.session.epoch_started_wall and r['event'].get('time_stamp',0)>0),{})
        return SimpleNamespace(has_collided=bool(event),time_stamp=event.get('time_stamp',0),
            impact_point=vec(event.get('impact_point',dict(x=0.,y=0.,z=0.))),object_name=event.get('object_name',''))

    def simSetVehiclePose(self,pose,ignore_collision,vehicle_name='drone_1'):
        from projectairsim.types import Pose as NativePose
        p,q=pose.position,pose.orientation
        result=self.drone.set_pose(NativePose(dict(translation=dict(x=p.x_val,y=p.y_val,z=p.z_val),
            rotation=dict(w=q.w_val,x=q.x_val,y=q.y_val,z=q.z_val))),reset_kinematics=True)
        if not result:raise RuntimeError('ProjectAirSim rejected physical reset')
        return result

    def prepare_reset(self,position,yaw):self.session.reload_for_reset(position,yaw)

    def simGetImages(self,requests,vehicle_name='drone_1'):
        import numpy as np
        from projectairsim.utils import unpack_image
        rows=[]
        for request in requests:
            if request.image_type==0:
                rgb,packet=self.session.image()
                raw=rgb.tobytes();floats=[];height,width=480,640
            elif request.image_type==2:
                packets=self.drone.get_images('front_custom',[2])
                if 2 not in packets:
                    raise RuntimeError('Diagnostic depth capture was unavailable; endpoint is unqualified')
                packet=packets[2]
                values=unpack_image(packet).astype(np.float32);raw=b'';floats=values.flatten().tolist()
                height,width=values.shape
            else:raise ValueError('Unsupported diagnostic image request')
            rows.append(SimpleNamespace(width=width,height=height,image_data_uint8=raw,image_data_float=floats,
                source_wall=packet.get('received_wall',time.perf_counter()),
                time_stamp=packet['time_stamp'],camera_position=Vector3r(packet['pos_x'],packet['pos_y'],packet['pos_z']),
                camera_orientation=Quaternionr(packet['rot_w'],packet['rot_x'],packet['rot_y'],packet['rot_z'])))
        return rows


class AirSimFacade:
    Vector3r,Pose,to_quaternion,to_eularian_angles=Vector3r,Pose,staticmethod(to_quaternion),staticmethod(to_eularian_angles)
    DrivetrainType=SimpleNamespace(MaxDegreeOfFreedom=0)
    ImageType=SimpleNamespace(Scene=0,DepthPerspective=2)
    def __init__(self):self.session=None;self.clients=[]
    def MultirotorClient(self,ip='127.0.0.1',port=43551,timeout_value=5):
        if ip!='127.0.0.1' or self.session is None:raise RuntimeError('Owned local ProjectAirSim session required')
        client=Client(self.session,timeout_value);self.clients.append(client);return client
    @staticmethod
    def YawMode(is_rate,yaw_or_rate):return SimpleNamespace(is_rate=is_rate,yaw_or_rate=yaw_or_rate)
    @staticmethod
    def ImageRequest(camera,image_type,pixels_as_float=False,compress=False):
        if camera!='front_custom' or compress:raise ValueError('Only calibrated uncompressed camera requests are supported')
        return SimpleNamespace(image_type=image_type,pixels_as_float=pixels_as_float)
