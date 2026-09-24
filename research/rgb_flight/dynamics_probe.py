"""Privileged physical braking/contact diagnostic, not a navigation episode.

Only real velocity/hover commands; no pose setting or paused simulation.
Ground truth is saved exclusively in engineering_only. Capture runs concurrently.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import airsim
import cv2
import numpy as np
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--expected-settings',type=Path,required=True)
args=parser.parse_args()
output=args.output/'engineering_only'
output.mkdir()
vehicle='drone_1'
client=airsim.MultirotorClient(timeout_value=15)
expected=json.loads(args.expected_settings.read_text())
effective=json.loads(client.getSettingsString())
if effective!=expected:
    raise RuntimeError('Effective settings mismatch')
if client.simIsPause():
    raise RuntimeError('Physics paused')
(output/'effective_settings.json').write_text(json.dumps(effective,indent=2))
stop=threading.Event()
frames=[]
capture_errors=[]
states=[]
commands=[]
phase='initial'
receipt=dict(scope='engineering only: short physical motion, braking and deliberate ground contact; not navigation',
             status='running',pose_setting_calls=0,pause_calls=0,clock_speed=expected['ClockSpeed'])


def read_state():
    state=client.getMultirotorState(vehicle_name=vehicle)
    collision=client.simGetCollisionInfo(vehicle_name=vehicle)
    k=state.kinematics_estimated
    row=dict(phase=phase,wall_monotonic=time.monotonic(),sim_ns=state.timestamp,
             position=[k.position.x_val,k.position.y_val,k.position.z_val],
             velocity=[k.linear_velocity.x_val,k.linear_velocity.y_val,k.linear_velocity.z_val],
             collision=collision.has_collided,collision_ns=collision.time_stamp)
    states.append(row)
    return row


def velocity(vx,vy,vz,duration):
    stamp=read_state()
    commands.append(dict(phase=phase,wall_monotonic=time.monotonic(),sim_ns=stamp['sim_ns'],
                         command='body_velocity',value=[vx,vy,vz,0],duration_sim_seconds=duration))
    return client.moveByVelocityBodyFrameAsync(vx,vy,vz,duration,
             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,yaw_mode=airsim.YawMode(True,0),vehicle_name=vehicle)


def monitor(duration):
    start=read_state()['sim_ns']
    deadline=time.monotonic()+max(15,duration/expected['ClockSpeed']*3)
    while True:
        row=read_state()
        if (row['sim_ns']-start)/1e9>=duration:
            return row
        if time.monotonic()>deadline:
            raise RuntimeError('Physics stalled')
        time.sleep(.04)


def capture():
    camera=airsim.MultirotorClient(timeout_value=10)
    video=cv2.VideoWriter(str(output/'onboard.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,480))
    try:
        if not video.isOpened():
            raise RuntimeError('Video writer unavailable')
        while not stop.is_set():
            started=time.monotonic()
            response=camera.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False)],vehicle_name=vehicle)[0]
            if (response.width,response.height)!=(640,480) or len(response.image_data_uint8)!=640*480*3:
                raise RuntimeError('Invalid calibrated RGB payload')
            pixels=np.frombuffer(response.image_data_uint8,np.uint8).reshape(480,640,3)
            video.write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
            label=phase
            frames.append(dict(frame=len(frames),phase=label,sim_ns=response.time_stamp,wall_monotonic=time.monotonic(),request_started=started))
            if not (output/(label+'.png')).exists():
                Image.fromarray(pixels).save(output/(label+'.png'))
            # Image RPC already waits for rendering. An additional 50 ms
            # schedule compounded frame-boundary jitter and missed 20 Hz.
            # Record every received frame with real timestamps; downstream
            # causal resampling must use these, not video container indices.
            stop.wait(.001)
    except Exception as error:
        capture_errors.append(str(error))
        stop.set()
    finally:
        video.release()


thread=threading.Thread(target=capture,daemon=True)
try:
    if read_state()['collision']:
        raise RuntimeError('Initial collision flag already set')
    # RPC starts before the packaged scene and vehicle finish settling.
    # Match the successful camera gate's warm-up before issuing takeoff.
    phase='startup_settle'
    monitor(5)
    client.enableApiControl(True,vehicle)
    client.armDisarm(True,vehicle)
    thread.start()
    phase='takeoff'
    client.takeoffAsync(timeout_sec=15,vehicle_name=vehicle).get()
    phase='climb'
    future=velocity(0,0,-1,5)
    monitor(5)
    future.get()
    phase='airborne_hover'
    client.hoverAsync(vehicle_name=vehicle).get()
    monitor(1.5)
    airborne=read_state()
    receipt['startup_collision_samples']=sum(bool(row['collision']) for row in states)
    receipt['measurement_start_sim_ns']=airborne['sim_ns']
    if airborne['collision'] or np.linalg.norm(airborne['velocity'])>=.5:
        raise RuntimeError('Could not establish collision-free stable airborne measurement start')
    phase='forward_motion'
    future=velocity(3,0,0,4)
    monitor(3.9)
    phase='braking'
    brake_start=read_state()
    # Measure the same body-zero command used by the live safety filter.
    # A hover RPC has a different controller and cannot qualify this behavior.
    brake_deadline=time.monotonic()+12
    while (read_state()['sim_ns']-brake_start['sim_ns'])/1e9<4:
        if time.monotonic()>brake_deadline:raise RuntimeError('Braking physics stalled')
        velocity(0,0,0,.15)
        time.sleep(.04)
    braking=[x for x in states if x['phase']=='braking']
    stopped=next((x for x in braking if np.linalg.norm(x['velocity'])<.5),None)
    if stopped is None:
        raise RuntimeError('Did not brake below 0.5 m/s')
    receipt['braking']=dict(start_speed_mps=float(np.linalg.norm(brake_start['velocity'])),
                            time_to_below_05_sim_seconds=(stopped['sim_ns']-brake_start['sim_ns'])/1e9,
                            distance_to_below_05_m=float(np.linalg.norm(np.array(stopped['position'])-brake_start['position'])),
                            max_excursion_m=max(float(np.linalg.norm(np.array(x['position'])-brake_start['position'])) for x in braking))
    if any(x['collision'] for x in states if x['sim_ns']>=receipt['measurement_start_sim_ns']):
        raise RuntimeError('Unexpected collision before intended contact test')
    phase='deliberate_ground_contact'
    future=velocity(0,0,1,15)
    started=read_state()['sim_ns']
    deadline=time.monotonic()+max(45,20/expected['ClockSpeed'])
    while True:
        row=read_state()
        if row['collision']:
            receipt['ground_contact']=row
            break
        if (row['sim_ns']-started)/1e9>15 or time.monotonic()>deadline:
            raise RuntimeError('No collision detected during deliberate ground approach')
        time.sleep(.04)
    client.hoverAsync(vehicle_name=vehicle).get()
    receipt['status']='engineering_dynamics_checks_passed'
except Exception as error:
    receipt.update(status='failed',error_type=type(error).__name__,error=str(error))
finally:
    stop.set()
    if thread.is_alive():
        thread.join(timeout=12)
    if capture_errors or thread.is_alive():
        receipt.update(status='failed',capture_errors=capture_errors or ['Capture thread did not finish'])
    receipt['frame_count']=len(frames)
    if len(frames)>2:
        receipt['capture_sim_interval_p95_seconds']=float(np.quantile(np.diff([x['sim_ns'] for x in frames])/1e9,.95))
        receipt['capture_20hz_check']=bool(receipt['capture_sim_interval_p95_seconds']<=.05 and np.all(np.diff([x['sim_ns'] for x in frames])>0))
        if not receipt['capture_20hz_check']:
            receipt['dynamics_status']=receipt['status']
            receipt['status']='blocked_concurrent_rgb_cadence'
    else:
        receipt.update(status='failed',error='Insufficient concurrent RGB frames')
    receipt['video_timing']='20-fps engineering preview; frames.json holds actual per-frame times; not a final full-flight video'
    for name,value in [('states',states),('commands',commands),('frames',frames),('dynamics',receipt)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2))
    if receipt.get('braking') and receipt.get('ground_contact'):
        braking=receipt['braking'];speed=braking['start_speed_mps']
        if speed>=2.8:
            profile=dict(schema='measured-navigation-safety/v1',status='completed',accepted=False,
                maximum_speed_mps=3.,vehicle_radius_m=1.,
                vehicle_radius_source='campaign horizontal vehicle radius 0.75 m plus 0.25 m geometry margin; conservative spherical envelope',
                braking_acceleration_mps2=speed**2/(4*(braking['max_excursion_m']+.5)),
                measurement_sha256=hashlib.sha256((output/'states.json').read_bytes()).hexdigest(),
                measured_command='body velocity [0,0,0,0] refreshed every 40 ms',
                scope='One real braking trial with 2x distance and 0.5 m margins; integration only')
            (output/'safety-profile.json').write_text(json.dumps(profile,indent=2))
    print(json.dumps(receipt),flush=True)
raise SystemExit(0 if receipt['status']=='engineering_dynamics_checks_passed' else 2)
