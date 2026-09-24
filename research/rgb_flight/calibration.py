"""Camera mounting calibration, separate from unknown vehicle/world pose."""
import math
import numpy as np


def measure_color_order(client, vehicle='drone_1'):
    """Measure the actual binary's raw channel order against its lossless PNG.

    Upstream AirSim serializes raw Scene pixels as B,G,R. Do not infer a fork's
    behavior from an RGB variable name. Two requests in one capture synchronize
    the uncompressed bytes and the PNG color reference without pausing physics.
    """
    import airsim
    import hashlib
    import io
    from PIL import Image
    responses = client.simGetImages([
        airsim.ImageRequest('front_custom', airsim.ImageType.Scene, False, False),
        airsim.ImageRequest('front_custom', airsim.ImageType.Scene, False, True)], vehicle_name=vehicle)
    if len(responses) != 2:
        raise RuntimeError('Missing camera color calibration responses')
    raw, encoded = responses
    if (raw.width, raw.height) != (640,480) or len(raw.image_data_uint8) != 640*480*3:
        raise RuntimeError('Invalid raw calibration image')
    raw_pixels = np.frombuffer(raw.image_data_uint8, np.uint8).reshape(480,640,3)
    reference = np.asarray(Image.open(io.BytesIO(encoded.image_data_uint8)).convert('RGB'))
    if reference.shape != raw_pixels.shape:
        raise RuntimeError('PNG and raw calibration dimensions differ')
    errors = {order: float(np.abs(value.astype(np.int16)-reference).mean())
              for order,value in [('RGB',raw_pixels),('BGR',raw_pixels[...,::-1])]}
    selected = min(errors, key=errors.get)
    if errors[selected] != 0 or errors['RGB'] == errors['BGR']:
        raise RuntimeError('Camera color calibration is ambiguous or captures differ: '+str(errors))
    return dict(raw_channel_order=selected, canonical_pixel_format='RGB24',
                mean_absolute_channel_errors=errors, exact_png_agreement=True,
                raw_sha256=hashlib.sha256(raw.image_data_uint8).hexdigest(),
                reference_rgb_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
                raw_sim_ns=raw.time_stamp, png_sim_ns=encoded.time_stamp,
                source='actual binary raw/PNG comparison; no pose/depth consumed')


def canonical_rgb(raw, channel_order):
    if channel_order == 'RGB':
        return bytes(raw)
    if channel_order == 'BGR':
        return np.frombuffer(raw, np.uint8).reshape(-1,3)[:,::-1].tobytes()
    raise ValueError('Unknown camera channel order')


def camera_extrinsics(settings, vehicle='drone_1'):
    camera = settings['Vehicles'][vehicle]['Cameras']['front_custom']
    roll, pitch, yaw = [math.radians(camera.get(k, 0)) for k in ('Roll', 'Pitch', 'Yaw')]
    cx,sx,cy,sy,cz,sz = math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    optical_to_frd=np.array([[0,0,1],[1,0,0],[0,1,0]])
    return dict(camera_to_body_rotation=(rz@ry@rx@optical_to_frd).reshape(-1).tolist(),
                camera_origin_body_m=[float(camera.get(k,0)) for k in ('X','Y','Z')])


def metric_body_pose(camera_to_map, estimated_scale, calibration):
    """Convert RGB camera pose only after a valid estimated metric scale exists."""
    if not math.isfinite(estimated_scale) or estimated_scale <= 0:
        raise ValueError('A positive RGB-estimated scale is required')
    if 'camera_to_body_rotation' not in calibration or 'camera_origin_body_m' not in calibration:
        raise ValueError('Camera mounting is unknown; do not assume the camera is at the vehicle center')
    body_from_camera=np.eye(4)
    body_from_camera[:3,:3]=np.asarray(calibration['camera_to_body_rotation']).reshape(3,3)
    body_from_camera[:3,3]=calibration['camera_origin_body_m']
    metric_camera=np.asarray(camera_to_map).copy()
    metric_camera[:3,3]*=estimated_scale
    return metric_camera@np.linalg.inv(body_from_camera)
