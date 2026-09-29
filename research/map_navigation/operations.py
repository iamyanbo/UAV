"""Explicit asset inspection, engineering qualification and collection reports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time

from .common import digest,read,write


def mission_settings(original):
    """Explicit fixed-camera mission settings; geometry survey is separate."""
    settings=json.loads(json.dumps(original))
    settings.update(RpcEnabled=True,ClockSpeed=1.,LocalHostIp='127.0.0.1',ViewMode='NoDisplay')
    captures=[dict(ImageType=k,Width=640,Height=480,FOV_Degrees=90,AutoExposureSpeed=100,MotionBlurAmount=0) for k in (0,2,5)]
    settings['CameraDefaults']={'CaptureSettings':captures}
    vehicle=settings['Vehicles']['drone_1']
    vehicle['Sensors']={'Imu':{'SensorType':2,'Enabled':True}}
    vehicle.setdefault('RC',{})['AllowAPIWhenDisconnected']=True
    vehicle.update(EnableCollisions=True,EnableCollisionPassthrough=False,EnableCollisionPassthrogh=False)
    vehicle['Cameras']['front_custom'].update(CaptureSettings=captures,Pitch=0,Roll=0,Yaw=0)
    return settings


def inspect_scenes(root, output, box64=None, source='AerialVLN'):
    root=Path(root).resolve();scenes=[];layouts={}
    for binary in sorted(root.rglob('*-Linux-Shipping')):
        simulator_root=binary.parents[3]
        paks=sorted(simulator_root.rglob('*.pak'))
        if not paks:continue
        assets=[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in [binary,*paks]]
        import hashlib
        layout=hashlib.sha256(''.join(r['sha256'] for r in assets[1:]).encode()).hexdigest()
        ident=next((part for part in binary.relative_to(root).parts if part.startswith('env_')),binary.parent.parent.parent.name)
        alias=layouts.get(layout);layouts.setdefault(layout,ident)
        adjacent=binary.parent/'settings.json'
        scenes.append(dict(scene_id=ident,source=source,engine='Unreal/AirSim',
            geography_id=None,city_id=None,asset_family_id=None,split=None,
            simulator_root=str(simulator_root),simulator_binary=str(binary),
            launch_argv=([box64] if box64 else [])+[str(binary),binary.parents[2].name,'-vulkan','-RenderOffscreen','-unattended','-nosound','-ResX=640','-ResY=480'],
            settings=str(adjacent) if adjacent.exists() else None,assets=assets,
            overlap_audit=dict(reviewed=False,layout_sha256=layout,byte_identical_alias=alias),qualified=False))
    write(output,dict(schema='photo-map-candidates/v1',scenes=scenes,unique_layouts=len(layouts),
                      geography_review_required=True,downloaded_scenes_are_not_qualified=True))


def probe(scene_path,output,workers=1,seconds=20):
    """Real RGB/RPC concurrency measurement; not a flight acceptance substitute."""
    import numpy as np
    from .common import available_memory
    import airsim
    from .collect import SceneProcess
    scene=read(scene_path);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    samples=[];initial=available_memory();started=time.monotonic()
    def lane(index):
        local=dict(scene);settings=read(scene['settings']);settings['ApiServerPort']=42451+index
        settings['ClockSpeed']=1.;settings['ViewMode']='NoDisplay'
        path=out/f'settings-{index}.json';write(path,settings)
        local.update(settings=str(path.resolve()),worker_root=str(out.resolve()))
        with SceneProcess(local):
            client=airsim.MultirotorClient(ip='127.0.0.1',port=settings['ApiServerPort'],timeout_value=10)
            latencies=[];stamps=[];receipt_times=[];begin=time.monotonic();errors=[]
            while time.monotonic()-begin<seconds:
                t=time.monotonic()
                try:
                    image=client.simGetImages([airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False)],vehicle_name='drone_1')[0]
                    if (image.width,image.height,len(image.image_data_uint8))!=(640,480,640*480*3):raise ValueError('RGB contract mismatch')
                    receipt_times.append(time.monotonic());stamps.append(image.time_stamp);latencies.append(receipt_times[-1]-t)
                    samples.append(available_memory())
                except Exception as exc:errors.append(type(exc).__name__);break
                time.sleep(max(0,.05-(time.monotonic()-t)))
            duration=time.monotonic()-begin
            # Compare matching endpoint intervals. Including first-render shader
            # initialization only in the wall denominator understates physics.
            interval=receipt_times[-1]-receipt_times[0] if len(receipt_times)>1 else 0
            ratio=(stamps[-1]-stamps[0])/1e9/interval if interval else 0
            return dict(worker=index,frames=len(stamps),wall_s=duration,sim_wall_ratio=ratio,
                steady_rgb_hz=(len(stamps)-1)/interval if interval else 0,
                first_rgb_s=receipt_times[0]-begin if receipt_times else None,
                rgb_p95_s=float(np.quantile(latencies,.95)) if latencies else None,
                rgb_p99_s=float(np.quantile(latencies,.99)) if latencies else None,errors=errors)
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(lane,range(workers)))
    write(out/'probe.json',dict(schema='photo-map-engineering-probe/v1',workers=workers,rows=rows,
        measured_incremental_bytes=max(0,initial-min(samples)) if samples else None,
        wall_seconds=time.monotonic()-started,qualified=False,
        pending=['continuous-flight/collision qualification','perception workload','source-to-dispatch latency','recording disk peaks']))


def inspect_live(scene_path,output,port=43451):
    """Inspect real scene sensors and identity without executing a mission."""
    import re
    import numpy as np
    import airsim
    from PIL import Image
    from calibration import measure_color_order,canonical_rgb
    from .collect import SceneProcess
    scene=read(scene_path);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    settings=mission_settings(read(scene['settings']));settings['ApiServerPort']=port
    settings_path=out/'settings.json';write(settings_path,settings)
    scene=dict(scene,settings=str(settings_path.resolve()),worker_root=str(out.resolve()))
    owned=SceneProcess(scene)
    receipt=dict(schema='photo-map-scene-preflight/v1',scene_id=scene['scene_id'],
        scene_descriptor_sha256=digest(scene_path),settings_sha256=digest(settings_path),
        implementation_sha256=digest(__file__),sensor_interfaces_passed=False,
        qualified=False,flight_evidence=False,pose_placement_calls=0)
    try:
        with owned:
            client=airsim.MultirotorClient(ip='127.0.0.1',port=port,timeout_value=20)
            # Some builds expose RPC before their first useful rendered frame.
            # Keep the exact color check, with bounded warm-up and PNG evidence.
            calibration_errors=[]
            for attempt in range(5):
                try:
                    color=measure_color_order(client)
                    break
                except RuntimeError as error:
                    calibration_errors.append(str(error))
                    preview=client.simGetImages([airsim.ImageRequest(
                        'front_custom',airsim.ImageType.Scene,False,True)],vehicle_name='drone_1')
                    if preview and preview[0].image_data_uint8:
                        (out/f'calibration-{attempt}.png').write_bytes(bytes(preview[0].image_data_uint8))
                    receipt['calibration_attempt_errors']=calibration_errors
                    if attempt==4:raise
                    time.sleep(2.)
            images=client.simGetImages([
                airsim.ImageRequest('front_custom',airsim.ImageType.Scene,False,False),
                airsim.ImageRequest('front_custom',airsim.ImageType.DepthPerspective,True,False),
                airsim.ImageRequest('front_custom',airsim.ImageType.Segmentation,False,False)],vehicle_name='drone_1')
            if len(images)!=3 or any((r.width,r.height)!=(640,480) for r in images):raise ValueError('Sensor dimensions differ')
            rgb,depth,segmentation=images
            if len(rgb.image_data_uint8)!=640*480*3 or len(segmentation.image_data_uint8)!=640*480*3 or len(depth.image_data_float)!=640*480:
                raise ValueError('Incomplete sensor payload')
            pixels=np.frombuffer(canonical_rgb(rgb.image_data_uint8,color['raw_channel_order']),np.uint8).reshape(480,640,3)
            Image.fromarray(pixels).save(out/'spawn-rgb.png')
            values=np.asarray(depth.image_data_float,np.float32).reshape(480,640)
            np.save(out/'spawn-depth.npy',values)
            useful=values[np.isfinite(values)&(values>0)&(values<200)]
            first=client.getMultirotorState(vehicle_name='drone_1');start=time.monotonic()
            time.sleep(2.)
            last=client.getMultirotorState(vehicle_name='drone_1');elapsed=time.monotonic()-start
            k=last.kinematics_estimated;collision=client.simGetCollisionInfo(vehicle_name='drone_1')
            receipt.update(color_calibration=color,preview_sha256=digest(out/'spawn-rgb.png'),
                depth_sha256=digest(out/'spawn-depth.npy'),rgb_std=float(pixels.std()),
                depth_supported_fraction=float(len(useful)/values.size),
                depth_percentiles_m=np.quantile(useful,[.05,.5,.95]).tolist() if len(useful) else None,
                capture_skew_s=(max(r.time_stamp for r in images)-min(r.time_stamp for r in images))/1e9,
                sim_wall_ratio=(last.timestamp-first.timestamp)/1e9/elapsed,
                spawn_ned_m=[k.position.x_val,k.position.y_val,k.position.z_val],
                spawn_collision=bool(collision.has_collided),sensor_interfaces_passed=bool(last.timestamp>first.timestamp and pixels.std()>1))
    except Exception as error:
        receipt.update(error_type=type(error).__name__,reason=str(error))
    finally:
        if owned.log:
            path=Path(owned.log.name)
            receipt['simulator_log']=str(path)
            receipt['loaded_maps']=sorted(set(re.findall(r'LoadMap:\s+(\S+)',path.read_text(errors='replace'))))
        write(out/'preflight.json',receipt)
    return receipt


def teacher_descriptor(package,output,audit=None):
    """Audits must independently judge observable support, not teacher confidence."""
    spec=dict(schema='photo-map-teacher/v1',implementation_sha256=digest(Path(__file__).with_name('observation_teacher.py')),
              perception_sha256=digest(Path(package)/'package.json'),minimum_inlier_ratio=.6,
              arrival_threshold=.9,qualified=False,qualification_sha256=None)
    if audit:
        record=read(audit)
        if record.get('schema')!='photo-map-teacher-audit/v1' or record.get('perception_sha256')!=spec['perception_sha256'] or record.get('implementation_sha256')!=spec['implementation_sha256']:
            raise ValueError('Teacher audit provenance mismatch')
        rows=record['decisions']
        if len(rows)<250 or len({r['scene_id'] for r in rows})<4:raise ValueError('Audit at least 250 decisions across four non-test scenes')
        if any(r['split']=='test' or not r['independently_reviewed'] for r in rows):raise ValueError('Unreviewed or sealed teacher audit')
        if not {'search','approach','recovery','settle'}<={r['phase'] for r in rows}:raise ValueError('Missing teacher phases')
        if any(not r['observation_supported'] or r['unsafe'] for r in rows):raise ValueError('Teacher audit found unsupported/unsafe corrections')
        spec.update(qualified=True,qualification_sha256=digest(audit))
    write(output,spec)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('inspect');a.add_argument('--root',required=True);a.add_argument('--output',required=True);a.add_argument('--box64');a.add_argument('--source',choices=('OpenFly','AerialVLN','UrbanScene3D'),default='AerialVLN')
    a=sub.add_parser('probe');a.add_argument('--scene',required=True);a.add_argument('--output',required=True);a.add_argument('--workers',type=int,default=1);a.add_argument('--seconds',type=int,default=20)
    a=sub.add_parser('preflight');a.add_argument('--scene',required=True);a.add_argument('--output',required=True);a.add_argument('--port',type=int,default=43451)
    a=sub.add_parser('teacher');a.add_argument('--package',required=True);a.add_argument('--output',required=True);a.add_argument('--audit')
    a=p.parse_args()
    if a.stage=='inspect':inspect_scenes(a.root,a.output,a.box64,a.source)
    elif a.stage=='probe':probe(a.scene,a.output,a.workers,a.seconds)
    elif a.stage=='preflight':inspect_live(a.scene,a.output,a.port)
    else:teacher_descriptor(a.package,a.output,a.audit)

if __name__=='__main__':main()
