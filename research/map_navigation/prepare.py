"""Privileged offline preparation. Never imported by navigation inference."""
import math
from pathlib import Path
import numpy as np
from PIL import Image
from obstacle_field import PrivilegedObstacleField
from .common import config,read,write,digest
from .maps import MapPrior,prepare_map
from .navigation import Router


def survey(field_path,settings_path,output,window):
    """Rasterize downward RGB/depth captures into an independent map asset.

    Captures are preparation, not mission flight evidence. Depth is discarded
    after producing the declared coarse prior; no source camera poses go into it.
    """
    import airsim
    from calibration import measure_color_order,canonical_rgb
    field=PrivilegedObstacleField.load(field_path);settings=read(settings_path)
    client=airsim.MultirotorClient(timeout_value=30);vehicle='drone_1'
    if __import__('json').loads(client.getSettingsString())!=settings:raise ValueError('Scene settings mismatch')
    root=Path(output);raw=root/'capture';raw.mkdir(parents=True,exist_ok=True)
    identity=dict(field=digest(field_path),settings=digest(settings_path))
    if (raw/'identity.json').exists() and read(raw/'identity.json')!=identity:raise ValueError('Survey resume inputs differ')
    write(raw/'identity.json',identity)
    bounds=field.bounds;shape=np.ceil((bounds[1,:2]-bounds[0,:2])[::-1]).astype(int)
    surface=np.full(shape,np.nan,np.float32);rgb=np.zeros((*shape,3),np.uint8)
    captures=[];z=float(bounds[0,2]-20)
    camera={**settings.get('CameraDefaults',{}),**settings['Vehicles'][vehicle].get('Cameras',{}).get('front_custom',{})}
    original=airsim.Pose(airsim.Vector3r(*[camera.get(k,0) for k in ('X','Y','Z')]),
             airsim.to_quaternion(*[math.radians(camera.get(k,0)) for k in ('Pitch','Roll','Yaw')]))
    color=measure_color_order(client,vehicle)
    try:
        client.simSetCameraPose('front_custom',airsim.Pose(airsim.Vector3r(),airsim.to_quaternion(-math.pi/2,0,0)),vehicle_name=vehicle)
        centers=[(x,y) for y in np.arange(bounds[0,1]+10,bounds[1,1],30) for x in np.arange(bounds[0,0]+10,bounds[1,0],30)]
        for index,(x,y) in enumerate(centers):
            if not window.remaining():write(root/'survey-status.json',dict(complete=False,next_capture=index));return
            path=raw/f'{index:06d}.npz'
            if not path.exists():
                client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x,y,z),airsim.to_quaternion(0,0,0)),True,vehicle)
                images=client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False),
                    airsim.ImageRequest('front_custom',airsim.ImageType.DepthPlanar,True,False)],vehicle_name=vehicle)
                scene,depth=images
                if (scene.width,scene.height)!=(640,480) or (depth.width,depth.height)!=(640,480):raise ValueError('Survey camera geometry differs')
                pixels=np.frombuffer(canonical_rgb(bytes(scene.image_data_uint8),color['raw_channel_order']),np.uint8).reshape(480,640,3)
                d=np.asarray(depth.image_data_float).reshape(480,640)
                fov=client.simGetCameraInfo('front_custom',vehicle_name=vehicle).fov;focal=320/math.tan(math.radians(fov/2))
                v,u=np.mgrid[0:480,0:640]
                if abs(scene.time_stamp-depth.time_stamp)>100_000_000:raise ValueError('Unsynchronized survey RGB/depth')
                camera_position=np.array([depth.camera_position.x_val,depth.camera_position.y_val,depth.camera_position.z_val])
                rgb_position=np.array([scene.camera_position.x_val,scene.camera_position.y_val,scene.camera_position.z_val])
                if np.linalg.norm(camera_position-rgb_position)>.1:raise ValueError('Survey camera poses differ')
                q=depth.camera_orientation;qx,qy,qz,qw=q.x_val,q.y_val,q.z_val,q.w_val
                rotation=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                    [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                    [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
                # Use the returned exposure pose, not the requested teleport pose.
                rays=np.stack((d,(u-320)*d/focal,(v-240)*d/focal),-1)
                points=rays@rotation.T+camera_position
                good=np.isfinite(d)&(d>0)&(d<1000)
                np.savez_compressed(path,points=points[good],rgb=pixels[good])
            capture=np.load(path,allow_pickle=False);points=capture['points'];colors=capture['rgb']
            cells=np.floor(points[:,:2]-bounds[0,:2]).astype(int)
            good=(cells[:,0]>=0)&(cells[:,1]>=0)&(cells[:,0]<shape[1])&(cells[:,1]<shape[0])
            points,cells,colors=points[good],cells[good],colors[good]
            # Stable far-to-near assignment leaves the highest surface in each cell.
            for j in np.argsort(points[:,2])[::-1]:
                xx,yy=cells[j];value=points[j,2]
                if not np.isfinite(surface[yy,xx]) or value<surface[yy,xx]:surface[yy,xx]=value;rgb[yy,xx]=colors[j]
            captures.append(dict(path=path.name,sha256=digest(path)))
        Image.fromarray(rgb).save(raw/'rgb.png');np.save(raw/'surface.npy',surface)
        if not (root/'prior').exists():
            prepare_map(raw/'rgb.png',raw/'surface.npy',root/'prior',bounds[0,:2],1.,1.,'simulator-overhead-survey')
        write(root/'survey-status.json',dict(complete=True,captures=captures,map_sha256=digest(root/'prior/map.json'),
              simulator_depth_used_only_for_declared_map=True,mission_evidence=False))
    finally:client.simSetCameraPose('front_custom',original,vehicle_name=vehicle)


def registry(inventory_path,output):
    """Inventory contains paths to actual assets; never manufacture scene IDs."""
    rows=read(inventory_path)['scenes'];seen=set();scenes=[]
    for row in sorted(rows,key=lambda r:(r['scene_id']!='env_airsim_16',r['scene_id'])):
        if row['geography_id'] in seen:raise ValueError('Duplicated geography in scene inventory')
        seen.add(row['geography_id'])
        prior=MapPrior(row['map']);field=PrivilegedObstacleField.load(row['obstacle_field'])
        settings=read(row['settings'])
        if settings.get('ClockSpeed')!=1.:raise ValueError('Registry requires ClockSpeed=1')
        if not row.get('launch_argv') or not Path(row['launch_argv'][0]).is_file():raise ValueError('Actual scene executable required')
        if np.any(prior.bounds[0]<field.bounds[0,:2]) or np.any(prior.bounds[1]>field.bounds[1,:2]+2):raise ValueError('Map and field frames differ')
        scenes.append(dict(row,map=str(Path(row['map']).resolve()),obstacle_field=str(Path(row['obstacle_field']).resolve()),
            settings=str(Path(row['settings']).resolve()),map_sha256=prior.identity,field_sha256=digest(row['obstacle_field']),settings_sha256=digest(row['settings'])))
    if len(scenes)<6 or scenes[0]['scene_id']!='env_airsim_16':raise ValueError('Need env_airsim_16 and five distinct compatible environments; no same-scene fallback')
    scenes=scenes[:6]
    for i,row in enumerate(scenes):row['split']='train' if i<3 else 'validation' if i==3 else 'test'
    write(output,dict(schema='photo-map-scenes/v1',scenes=scenes,inventory_sha256=digest(inventory_path)))


def manifests(registry_path,output):
    cfg=config();scenes=read(registry_path)['scenes'];output=Path(output)
    output.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(cfg['seed'])
    public=[];private=[]
    for scene in scenes:
        field=PrivilegedObstacleField.load(scene['obstacle_field']);prior=MapPrior(scene['map']);router=Router(prior,cfg['navigation'])
        count=cfg['episodes'][scene['split']+'_per_scene'];accepted=0;attempts=0
        while accepted<count and attempts<count*1000:
            attempts+=1;band=cfg['episodes']['distance_bands_m'][accepted%3]
            start_xy=rng.uniform(prior.bounds[0]+4,prior.bounds[1]-4)
            heading=rng.uniform(-math.pi,math.pi);distance=rng.uniform(*band)
            goal_xy=start_xy+distance*np.array([math.cos(heading),math.sin(heading)])
            surfaces=prior.height(np.stack((start_xy,goal_xy)))
            if not np.isfinite(surfaces).all():continue
            start=np.r_[start_xy,surfaces[0]-rng.uniform(5,20)];goal=np.r_[goal_xy,surfaces[1]-rng.uniform(5,20)]
            if field.contains_vehicle(start) or field.contains_vehicle(goal):continue
            candidates=[]
            if field.segment_free(start,goal):candidates.append([start,goal])
            cruise=min(start[2],goal[2],float(np.nanmin(prior.surface))-8)
            over=[start,np.r_[start_xy,cruise],np.r_[goal_xy,cruise],goal]
            if all(field.segment_free(a,b) for a,b in zip(over,over[1:])):candidates.append(over)
            try:candidates.append(field.reference_path(start,goal))
            except ValueError:pass
            if not candidates:continue
            path=min(candidates,key=lambda p:sum(router.duration(a,b) for a,b in zip(p,p[1:])))
            length=sum(float(np.linalg.norm(np.asarray(a)-b)) for a,b in zip(path,path[1:]))
            if not 20<=length<=300:continue
            duration=sum(router.duration(a,b) for a,b in zip(path,path[1:]))
            ident=f'{scene["scene_id"]}-{scene["split"]}-{accepted:05d}'
            timeout=max(180,4*duration+180)
            public.append(dict(episode_id=ident,scene_id=scene['scene_id'],split=scene['split'],map_sha256=prior.identity,
                               goal_views=1,goal_record='goals/'+ident,timeout_s=timeout))
            private.append(dict(episode_id=ident,scene_id=scene['scene_id'],split=scene['split'],phase=1+accepted%3,
                 start_ned_m=start.tolist(),start_yaw_degrees=float(rng.uniform(-180,180)),goal_ned_m=goal.tolist(),
                 goal_yaw_degrees=float(rng.uniform(-180,180)),reference_path_ned_m=np.asarray(path).tolist(),
                 reference_length_m=length,reference_time_s=duration,timeout_s=timeout,
                 route_kind='direct' if len(path)==2 else 'overflight' if len(path)==4 and np.allclose(path[1],over[1]) else 'detour'))
            accepted+=1
        if accepted<count:raise RuntimeError('Insufficient surveyed flight coverage: '+scene['scene_id'])
    for split in ('train','validation','test'):
        write(output/(split+'.json'),dict(schema='photo-map-missions/v1',registry_sha256=digest(registry_path),episodes=[r for r in public if r['split']==split]))
        write(output/'evaluator_labels'/(split+'.json'),dict(schema='privileged-photo-map-labels/v1',episodes=[r for r in private if r['split']==split]))
