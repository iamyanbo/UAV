"""Bounded released-route engineering capture; never admitted as policy data.

Uses privileged positions for a collection controller, not the learned runtime.
Initial placement is explicit; all recorded motion uses continuous physics.
"""
import argparse
import json
import math
from pathlib import Path
import time
import zipfile

from .common import read, write, digest
from .collect import SceneProcess
from .collection_protocol import AttemptQueue


def capture(root, scene_id, output, port=43551, engineering_only=False):
    if not engineering_only:
        raise ValueError('Demonstration follower retired from automatic collection; explicit engineering_only is required')
    import airsim
    import numpy as np
    from PIL import Image
    from calibration import measure_color_order, canonical_rgb
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    archive=root/'assets/aerialvln-annotations/aerialvln.zip'
    with zipfile.ZipFile(archive) as source:
        episodes=json.loads(source.read('train.json'))['episodes']
    candidates=[]
    for episode in episodes:
        if episode['scene_id']!=scene_id:continue
        path=episode['reference_path']
        length=sum(math.dist(a[:3],b[:3]) for a,b in zip(path,path[1:]))
        if 100<=length<=250:candidates.append((length,episode['trajectory_id'],episode))
    if not candidates:raise ValueError('No short training reference for this scene')
    length,_,episode=min(candidates,key=lambda row:(row[0],row[1]))
    write(out/'released-reference.json',episode)
    scene=read(root/f'qualification/aerialvln-preflight-executable/env_{scene_id}.json')
    settings=read(root/'mission-settings.json');settings['ApiServerPort']=port
    # Disable initial free fall while RPC/rendering initializes; arm explicitly.
    settings['Vehicles']['drone_1']['DefaultVehicleState']='Disarmed'
    settings['Vehicles']['drone_1']['Cameras']['front_custom']['Z']=-.1
    write(out/'settings.json',settings)
    scene.update(settings=str((out/'settings.json').resolve()),worker_root=str(out.resolve()))
    queue=AttemptQueue(root/'engineering-reference.sqlite3','released-reference-engineering-v1')
    ident=out.name
    if not queue.reserve(ident,'reference','train',limit=250,required_disk_bytes=2*1024**3,output=out):
        raise RuntimeError('Attempt already reserved or budget exhausted')
    result=dict(schema='photo-map-reference-capture/v1',scene_id=scene_id,
        source_split='AerialVLN train',trajectory_id=episode['trajectory_id'],
        source_sha256=digest(__file__),reference_sha256=digest(out/'released-reference.json'),
        controller='privileged reference velocity follower',training_eligible=False,
        engineering_only=True,qualified=False,reference_length_m=length,completed=False,
        initialization_pose_placements=0,recorded_pose_placements=0,frames=0)
    write(out/'result.json',result)
    try:
        with SceneProcess(scene):
            client=airsim.MultirotorClient(ip='127.0.0.1',port=port,timeout_value=10)
            vehicle='drone_1';client.enableApiControl(True,vehicle)
            p=episode['start_position'];q=episode['start_rotation']
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*p),airsim.Quaternionr(q[1],q[2],q[3],q[0])),True,vehicle)
            result['initialization_pose_placements']=1
            client.armDisarm(True,vehicle)
            time.sleep(3)
            color=measure_color_order(client)
            result['color_calibration']=color
            frames=out/'frames';frames.mkdir()
            path=episode['reference_path'];index=1;start=time.monotonic();distance=0.;previous=None
            collision_baseline=client.simGetCollisionInfo(vehicle).time_stamp
            first_sim=None;last_sim=None;settled=0
            with (out/'steps.jsonl').open('w') as log:
                while time.monotonic()-start<240:
                    tick=time.monotonic();state=client.getMultirotorState(vehicle)
                    pos=state.kinematics_estimated.position;xyz=[pos.x_val,pos.y_val,pos.z_val]
                    if previous:distance+=math.dist(previous,xyz)
                    previous=xyz
                    yaw=airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
                    target=path[index];delta=np.asarray(target[:3])-xyz
                    yaw_error=(target[5]-yaw+math.pi)%(2*math.pi)-math.pi
                    if np.linalg.norm(delta)<.8 and abs(yaw_error)<.2:
                        if index<len(path)-1:index+=1
                        else:settled+=1
                    else:settled=0
                    velocity=np.clip(delta*.8,[-2,-2,-1],[2,2,1])
                    horizontal=float(np.linalg.norm(velocity[:2]))
                    if horizontal>2:velocity[:2]*=2/horizontal
                    yaw_rate=float(np.clip(math.degrees(yaw_error)*1.5,-45,45))
                    collision=client.simGetCollisionInfo(vehicle)
                    new_collision=bool(collision.has_collided and collision.time_stamp>collision_baseline)
                    # Contact while leaving the spawn surface is recorded, but
                    # only post-takeoff contact aborts the demonstration.
                    fatal_collision=new_collision and tick-start>5
                    if fatal_collision or settled>=10:velocity[:]=0;yaw_rate=0
                    client.moveByVelocityAsync(*map(float,velocity),.2,
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(True,yaw_rate),vehicle_name=vehicle)
                    rgb=client.simGetImages([
                        airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False)],vehicle)[0]
                    if (rgb.width,rgb.height)!=(640,480) or not rgb.image_data_uint8:
                        raise ValueError('Invalid RGB capture')
                    frame=result['frames']
                    pixels=np.frombuffer(canonical_rgb(rgb.image_data_uint8,color['raw_channel_order']),np.uint8).reshape(480,640,3)
                    Image.fromarray(pixels).save(frames/f'{frame:06d}.png',compress_level=1)
                    now=time.monotonic();first_sim=first_sim or state.timestamp;last_sim=state.timestamp
                    row=dict(frame=frame,wall_elapsed_s=tick-start,sim_ns=state.timestamp,rgb_sim_ns=rgb.time_stamp,
                        position_ned_m=xyz,yaw_rad=yaw,target_index=index,
                        dispatched_velocity_ned_mps=velocity.tolist(),yaw_rate_deg_s=yaw_rate,
                        command_duration_s=.2,collision=new_collision,cycle_s=now-tick)
                    log.write(json.dumps(row)+'\n');log.flush();result['frames']+=1
                    if fatal_collision:result['failure']='collision';break
                    if settled>=10:result['completed']=True;break
                    time.sleep(max(0,.1-(now-tick)))
            client.hoverAsync(vehicle_name=vehicle).join()
            result.update(duration_s=time.monotonic()-start,travelled_m=distance,
                sim_elapsed_s=(last_sim-first_sim)/1e9 if first_sim else 0,
                reached_reference_index=index,reference_points=len(path))
            if not result['completed'] and 'failure' not in result:result['failure']='timeout'
            client.armDisarm(False,vehicle)
    except (Exception,KeyboardInterrupt) as error:
        result.update(failure=type(error).__name__,error=str(error))
    finally:
        write(out/'result.json',result);queue.finish(ident,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--scene',type=int,required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--engineering-only',action='store_true')
    args=parser.parse_args();print(json.dumps(capture(args.root,args.scene,args.output,engineering_only=args.engineering_only)))
