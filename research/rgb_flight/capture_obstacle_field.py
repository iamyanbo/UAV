"""Privileged pre-campaign depth/semantic survey of env_airsim_16.

This is map-label acquisition, not an episode and not runtime policy input.
Repeated pose setting is therefore confined to this script.  Episode flight
code is separately constrained to one start placement before recording.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import airsim
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix(position, quaternion):
    q = np.array([quaternion.x_val, quaternion.y_val, quaternion.z_val, quaternion.w_val], dtype=np.float64)
    x, y, z, w = q / np.linalg.norm(q)
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    # AirSim optical pixels are right/down with depth forward.
    optical_to_body = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    result = np.eye(4)
    result[:3, :3] = rotation @ optical_to_body
    result[:3, 3] = [position.x_val, position.y_val, position.z_val]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-settings', type=Path, required=True)
    parser.add_argument('--minimum-xy', type=float, default=-235.)
    parser.add_argument('--maximum-xy', type=float, default=235.)
    parser.add_argument('--spacing-m', type=float, default=35.)
    parser.add_argument('--altitudes-ned-m', type=float, nargs='+', default=(-8., -20.))
    parser.add_argument('--maximum-depth-m', type=float, default=80.)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'depth').mkdir(); (output/'semantic').mkdir()
    expected = json.loads(args.expected_settings.read_text())
    if expected.get('ClockSpeed') != 1.:
        raise ValueError('Privileged final field survey requires ClockSpeed 1.0')
    client = airsim.MultirotorClient(timeout_value=30)
    client.confirmConnection()
    if json.loads(client.getSettingsString()) != expected:
        raise RuntimeError('Effective AirSim settings differ from field-survey receipt')
    vehicle = 'drone_1'
    client.enableApiControl(True, vehicle)
    capture = expected['CameraDefaults']['CaptureSettings'][0]
    width, height = int(capture['Width']), int(capture['Height'])
    fov = float(capture.get('FOV_Degrees', 90.))
    fx = width / (2 * math.tan(math.radians(fov) / 2)); fy = fx
    requests = [airsim.ImageRequest('front_custom', airsim.ImageType.DepthPerspective, True, False),
                airsim.ImageRequest('front_custom', airsim.ImageType.Segmentation, False, False)]
    coordinates = np.arange(args.minimum_xy, args.maximum_xy + args.spacing_m*.5, args.spacing_m)
    rows = []
    started = time.monotonic()
    for north in coordinates:
        for east in coordinates:
            for altitude in args.altitudes_ned_m:
                for pitch_degrees in (-35., 35.):
                    for yaw_degrees in (0., 90., 180., 270.):
                        orientation = airsim.to_quaternion(math.radians(pitch_degrees), 0, math.radians(yaw_degrees))
                        pose = airsim.Pose(airsim.Vector3r(float(north), float(east), float(altitude)), orientation)
                        client.simSetVehiclePose(pose, True, vehicle)
                        responses = client.simGetImages(requests, vehicle_name=vehicle)
                        if len(responses) != 2 or (responses[0].width, responses[0].height) != (width, height):
                            raise RuntimeError('Incomplete privileged depth/semantic capture')
                        index = len(rows)
                        depth = np.asarray(responses[0].image_data_float, dtype=np.float32).reshape(height, width)
                        semantic = bytes(responses[1].image_data_uint8)
                        if len(semantic) != width*height*3:
                            raise RuntimeError('Invalid semantic payload')
                        depth_path = output/'depth'/f'{index:06d}.npy'
                        semantic_path = output/'semantic'/f'{index:06d}.rgb'
                        np.save(depth_path, depth, allow_pickle=False)
                        semantic_path.write_bytes(semantic)
                        row = dict(index=index, depth_path=str(depth_path.relative_to(output)), depth_sha256=digest(depth_path),
                                   semantic_path=str(semantic_path.relative_to(output)), semantic_sha256=digest(semantic_path),
                                   depth_type='DepthPerspective',
                                   sim_ns=responses[0].time_stamp, width=width, height=height, fx=fx, fy=fy,
                                   cx=width/2, cy=height/2, maximum_depth_m=args.maximum_depth_m, fusion_stride=4,
                                   camera_to_ned=matrix(responses[0].camera_position, responses[0].camera_orientation).tolist(),
                                   survey_pose_ned_m=[north, east, altitude], yaw_degrees=yaw_degrees,
                                   pitch_degrees=pitch_degrees)
                        with (output/'captures.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(row, allow_nan=False)+'\n')
                        rows.append(row)
    receipt = dict(status='privileged_depth_semantic_survey_complete', scene='env_airsim_16', captures=len(rows),
                   pose_setting_scope='pre-campaign field acquisition only', clock_speed=1., elapsed_wall_seconds=time.monotonic()-started,
                   visible_depth_endpoints_define_obstacles=True, semantic_payloads_preserved=True,
                   next_step='Offline obstacle_field.py fusion in the model container; no simulator recapture required')
    (output/'result.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
