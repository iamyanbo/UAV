"""Offline privileged acquisition. Never imported by the navigation runtime."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from .common import read, write, digest, config
from .aerial import envelope, camera_rotation
from .maps import MapPrior
from .routing import Router


def acquire(settings_path, envelope_path, output, window, requests=None, base_field=None):
    """Multi-altitude depth survey; optional requests refine uncovered corridors.

    This performs offline pose placement, never recorded navigation teleporting.
    Invalid/no-hit depth is left unknown by the existing fusion routine.
    """
    import airsim
    from capture_obstacle_field import matrix
    from obstacle_field import fuse_captures
    settings = read(settings_path); specification = read(envelope_path); bounds = envelope(specification)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    identity = dict(settings=digest(settings_path), envelope=digest(envelope_path),
                    requests=None if requests is None else digest(requests),base_field=None if base_field is None else digest(base_field))
    if (out/'identity.json').exists() and read(out/'identity.json') != identity:
        raise ValueError('Survey resume inputs changed')
    write(out/'identity.json', identity)
    for name in ('depth','semantic'):
        (out/name).mkdir(exist_ok=True)
    if requests:
        positions = np.asarray(read(requests)['positions_ned_m'], float)
    else:
        positions = np.asarray([(x,y,z) for x in np.arange(bounds[0,0]+2,bounds[1,0]-2,20)
                                for y in np.arange(bounds[0,1]+2,bounds[1,1]-2,20)
                                for z in np.arange(bounds[0,2]+2,bounds[1,2]-2,10)])
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError('Explicit finite survey positions required')
    if np.any(positions < bounds[0]) or np.any(positions > bounds[1]):
        raise ValueError('Refinement requests leave the declared envelope')
    client = airsim.MultirotorClient(ip=settings.get('LocalHostIp','127.0.0.1'),port=settings.get('ApiServerPort',41451),timeout_value=30); vehicle = 'drone_1'
    if json.loads(client.getSettingsString()) != settings or settings.get('ClockSpeed') != 1. or client.simIsPause():
        raise ValueError('Survey settings/clock mismatch')
    capture = settings['CameraDefaults']['CaptureSettings'][0]
    focal = 320/math.tan(math.radians(capture.get('FOV_Degrees',90))/2)
    log = out/'captures.jsonl'
    rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    completed = {r['capture_key'] for r in rows}
    with log.open('a') as stream:
        for index,p in enumerate(positions):
            for pitch in (-90,-45,0,45,90):
                for yaw in (0,90,180,270):
                    key = f'{index}-{pitch}-{yaw}'
                    if key in completed:
                        continue
                    if not window.remaining():
                        write(out/'status.json',dict(complete=False,captures=len(completed))); return
                    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*map(float,p)),airsim.to_quaternion(0,0,math.radians(yaw))),True,vehicle)
                    client.simSetCameraPose('front_custom',airsim.Pose(airsim.Vector3r(),airsim.to_quaternion(math.radians(pitch),0,0)),vehicle_name=vehicle)
                    responses = client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.DepthPerspective,True,False),
                                                    airsim.ImageRequest('front_custom',airsim.ImageType.Segmentation,False,False)],vehicle_name=vehicle)
                    if len(responses) != 2 or any((r.width,r.height)!=(640,480) for r in responses):
                        raise ValueError('Survey image contract changed')
                    depth,semantic = responses
                    if abs(depth.time_stamp-semantic.time_stamp)>100_000_000:
                        raise ValueError('Unsynchronized survey captures')
                    depth_path=f'depth/{key}.npy'; semantic_path=f'semantic/{key}.rgb'
                    np.save(out/depth_path,np.asarray(depth.image_data_float,np.float32).reshape(480,640))
                    (out/semantic_path).write_bytes(bytes(semantic.image_data_uint8))
                    row=dict(capture_key=key,depth_path=depth_path,semantic_path=semantic_path,
                        depth_sha256=digest(out/depth_path),semantic_sha256=digest(out/semantic_path),
                        width=640,height=480,fx=focal,fy=focal,cx=320,cy=240,maximum_depth_m=80.,fusion_stride=4,
                        depth_type='DepthPerspective',camera_to_ned=matrix(depth.camera_position,depth.camera_orientation).tolist(),
                        sim_ns=depth.time_stamp,survey_pose_ned_m=p.tolist(),pitch_degrees=pitch,yaw_degrees=yaw)
                    stream.write(json.dumps(row)+'\n');stream.flush();completed.add(key)
    field_path = out/'obstacle-field.npz'
    if not field_path.exists():
        fuse_captures(out,field_path)
    if base_field:
        from obstacle_field import PrivilegedObstacleField
        original=PrivilegedObstacleField.load(base_field);addition=PrivilegedObstacleField.load(field_path)
        if original.observed_free is None:raise ValueError('Cannot refine a legacy field without observed-free evidence')
        combined=out/'refined-field.npz'
        if not combined.exists():
            fused=PrivilegedObstacleField(np.concatenate((original.occupied,addition.occupied)),
                np.stack((np.minimum(original.bounds[0],addition.bounds[0]),np.maximum(original.bounds[1],addition.bounds[1]))),
                resolution_m=min(original.resolution,addition.resolution),
                observed_free_ned_m=np.concatenate((original.observed_free,addition.observed_free)),free_resolution_m=2.)
            fused.save(combined,dict(kind='refined_multialtitude_survey',sources=[digest(base_field),digest(field_path)]))
        field_path=combined
    write(out/'status.json',dict(complete=True,captures=len(completed),field_path=str(field_path.resolve()),field_sha256=digest(field_path),flight_evidence=False))


def qualify(field_path, prior_path, envelope_path, output):
    """Bind a declared volume and demonstrate surveyed ascent/descent links."""
    from obstacle_field import PrivilegedObstacleField
    field=PrivilegedObstacleField.load(field_path); prior=MapPrior(prior_path)
    spec=read(envelope_path); bounds=envelope(spec)
    if field.observed_free is None:
        raise ValueError('Legacy endpoint-only field cannot qualify free airspace')
    if np.any(bounds[0]<field.bounds[0]) or np.any(bounds[1]>field.bounds[1]):
        raise ValueError('Requested envelope exceeds surveyed field bounds; acquire upper-airspace captures')
    prior.flight_envelope=spec
    router=Router(prior,config()['navigation'],field)
    examples=[]; requests=[]
    for xy in prior.tiles:
        if np.any(xy<bounds[0,:2]+4) or np.any(xy>bounds[1,:2]-4):
            continue
        try: low=np.r_[xy,field.ground_z(xy)-5.]
        except ValueError: continue
        surface=router.corridor_surface(low,low)
        if surface is None: continue
        high=np.r_[xy,surface-8.]
        if high[2] >= low[2]-4 or not router.free(low,low): continue
        try:
            route=field.reference_path(low,high,maximum_expansions=40000,search_bounds=bounds)
            route=np.asarray(router.forward_ramps(route))
            if all(router.free(a,b) for a,b in zip(route,route[1:])):
                examples.append(dict(low_ned_m=low.tolist(),high_ned_m=high.tolist(),ascent=route.tolist(),descent=route[::-1].tolist()))
        except ValueError:
            requests.extend(np.linspace(low,high,max(2,int(np.linalg.norm(high-low)/10)+1)).tolist())
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    free=field.observed_free
    counts=[int(np.count_nonzero((free[:,2]>=z)&(free[:,2]<z+10))) for z in np.arange(bounds[0,2],bounds[1,2],10)]
    receipt=dict(schema='aerial-qualification/v1',qualified=bool(examples),field_sha256=digest(field_path),
        input_map_sha256=prior.identity,envelope_sha256=digest(envelope_path),connections=examples,
        free_cells_per_10m_layer=counts,unresolved_connections=len(requests),flight_evidence=False)
    write(out/'refinement-requests.json',dict(positions_ned_m=requests))
    if examples:
        target=out/'prior';target.mkdir()
        for name in prior.meta['files']: shutil.copyfile(prior.root/name,target/name)
        meta=dict(prior.meta,schema='overhead-map/v2',flight_envelope=spec)
        write(target/'map.json',meta);receipt['qualified_map_sha256']=digest(target/'map.json')
    from .audit import altitude_plot
    altitude_plot(out/'qualified-connections.svg',[(f'connection-{i}',r['ascent']+r['descent'][1:]) for i,r in enumerate(examples[:4])],
                  'Survey-qualified reference connections; not recorded flights')
    write(out/'qualification.json',receipt)
    if not examples:
        raise ValueError('No surveyed low-to-high connection; see refinement-requests.json')


def bank(registry_path, output, split, window):
    """Bootstrap still-image localization without requiring a trained navigator."""
    import airsim
    from obstacle_field import PrivilegedObstacleField
    from calibration import canonical_rgb, measure_color_order
    from .collect import SceneProcess
    if split not in ('train','validation'): raise ValueError('No test-scene teacher acquisition')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    index=out/'bank.json';identity=digest(registry_path)
    saved=read(index) if index.exists() else dict(schema='photo-map-bank/v1',registry_sha256=identity,records=[])
    if saved['registry_sha256']!=identity:raise ValueError('Bank registry changed')
    keys={r['key'] for r in saved['records']}
    for scene in read(registry_path)['scenes']:
        if scene['split']!=split:continue
        prior=MapPrior(scene['map']);field=PrivilegedObstacleField.load(scene['obstacle_field'])
        with SceneProcess(scene):
            settings=read(scene['settings'])
            client=airsim.MultirotorClient(ip=settings.get('LocalHostIp','127.0.0.1'),port=settings.get('ApiServerPort',41451),timeout_value=30);vehicle='drone_1'
            color=measure_color_order(client,vehicle)
            settings=read(scene['settings']);focal=320/math.tan(math.radians(settings['CameraDefaults']['CaptureSettings'][0].get('FOV_Degrees',90))/2)
            columns=prior.tiles[np.linspace(0,len(prior.tiles)-1,min(config()['collection']['bank_columns'],len(prior.tiles)),dtype=int)]
            for column,xy in enumerate(columns):
                try: ground=field.ground_z(xy)
                except ValueError:continue
                heights=(ground-5,ground-20,float(prior.height(xy))-16)
                for level,z in enumerate(heights):
                    p=np.r_[xy,z]
                    if field.contains_vehicle(p):continue
                    for heading in (0,90,180,270):
                        for pitch in config()['collection']['bank_pitches']:
                            key=f"{scene['scene_id']}-{column}-{level}-{heading}-{pitch}"
                            if key in keys:continue
                            if not window.remaining():write(index,saved);return
                            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(*map(float,p)),airsim.to_quaternion(0,0,math.radians(heading))),True,vehicle)
                            client.simSetCameraPose('front_custom',airsim.Pose(airsim.Vector3r(),airsim.to_quaternion(math.radians(pitch),0,0)),vehicle_name=vehicle)
                            response=client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False)],vehicle_name=vehicle)[0]
                            raw=canonical_rgb(bytes(response.image_data_uint8),color['raw_channel_order'])
                            if len(raw)!=640*480*3:raise ValueError('Invalid bank RGB')
                            filename=key+'.png';Image.fromarray(np.frombuffer(raw,np.uint8).reshape(480,640,3)).save(out/filename)
                            # Exposure position belongs only to offline labels.
                            pos=[response.camera_position.x_val,response.camera_position.y_val,response.camera_position.z_val]
                            saved['records'].append(dict(key=key,scene_id=scene['scene_id'],split=split,map=scene['map'],
                                map_sha256=prior.identity,image=str((out/filename).resolve()),image_sha256=digest(out/filename),
                                camera_pitch_deg=pitch,position_ned_m=pos,yaw_rad=math.radians(heading),
                                calibration=dict(fx=focal,fy=focal,cx=320.,cy=240.,width=640,height=480,
                                    camera_to_body_rotation=camera_rotation(pitch).ravel().tolist(),camera_origin_body_m=[0.,0.,0.])))
                            keys.add(key)
                            if len(keys)%100==0:write(index,saved)
    counts={}
    for row in saved['records']:
        counts.setdefault(row['scene_id'],set()).add(tuple(round(x,1) for x in row['position_ned_m'][:2]))
    saved['columns_per_scene']={scene:len(values) for scene,values in counts.items()}
    saved['coverage_missing']=[scene for scene,count in saved['columns_per_scene'].items() if count<config()['collection']['bank_columns']]
    saved['complete']=not saved['coverage_missing'];write(index,saved)
