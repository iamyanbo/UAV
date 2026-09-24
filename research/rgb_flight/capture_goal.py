"""Privileged pre-capture of four RGB goal views, outside the flight episode."""
import argparse
import json
import math
from pathlib import Path

import airsim

from calibration import camera_extrinsics, canonical_rgb, measure_color_order
from contracts import Calibration, GoalObservation
from goal_io import write_goal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluator-labels', type=Path, required=True)
    parser.add_argument('--episode-id', required=True)
    parser.add_argument('--expected-settings', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    labels = json.loads(args.evaluator_labels.read_text())['episodes']
    label = next(row for row in labels if row['episode_id'] == args.episode_id)
    expected = json.loads(args.expected_settings.read_text())
    if expected.get('ClockSpeed') != 1.:
        raise ValueError('Final goal capture requires ClockSpeed 1.0')
    client = airsim.MultirotorClient(timeout_value=20)
    if json.loads(client.getSettingsString()) != expected:
        raise RuntimeError('Effective settings differ from goal-capture receipt')
    vehicle = 'drone_1'
    client.enableApiControl(True, vehicle)
    goal = label['goal_ned_m']
    capture = expected['CameraDefaults']['CaptureSettings'][0]
    fov = float(capture.get('FOV_Degrees', 90.))
    focal = 320./math.tan(math.radians(fov)/2)
    extra = camera_extrinsics(expected, vehicle)
    calibration = Calibration(width=640, height=480, fx=focal, fy=focal, cx=320., cy=240., **extra)
    color = None; images = []; times = []
    for yaw in (0., 90., 180., 270.):
        pose = airsim.Pose(airsim.Vector3r(*goal), airsim.to_quaternion(0, 0, math.radians(yaw)))
        client.simSetVehiclePose(pose, True, vehicle)
        if color is None:
            color = measure_color_order(client, vehicle)
        response = client.simGetImages([airsim.ImageRequest('front_custom', airsim.ImageType.Scene, False, False)],
                                       vehicle_name=vehicle)
        if len(response) != 1 or (response[0].width, response[0].height) != (640,480):
            raise RuntimeError('Invalid goal RGB response')
        images.append(canonical_rgb(bytes(response[0].image_data_uint8), color['raw_channel_order']))
        times.append(response[0].time_stamp/1e9)
    observation = GoalObservation(args.episode_id, tuple(images), calibration, tuple(times))
    record = write_goal(args.output, observation)
    # Pose evidence remains alongside evaluator labels, never in goal output.
    evidence = args.evaluator_labels.parent/'goal_capture_evidence'/args.episode_id
    evidence.mkdir(parents=True, exist_ok=False)
    (evidence/'goal_capture.json').write_text(json.dumps(dict(episode_id=args.episode_id, goal_ned_m=goal,
          captured_yaws_degrees=[0,90,180,270], panorama_sha256=record['panorama_sha256'], clock_speed=1.), indent=2))
    print(json.dumps(dict(status='goal_panorama_captured', episode_id=args.episode_id,
                          panorama_sha256=record['panorama_sha256'])))


if __name__ == '__main__':
    main()
