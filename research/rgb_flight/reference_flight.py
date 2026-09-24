"""Physically fly an engineering reference proposal; no learned-flight claim.

The collector is explicitly privileged. It only sends bounded commands through
the same RGB broker as inference, while recording labels in a separate folder.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import airsim
import numpy as np

from broker import RGBBroker
from episode_store import EpisodeWriter
from flight_evaluator import EvaluationState, FlightEvaluator
from wire import BrokerClient
from isolation import verify_live_boundary
from calibration import camera_extrinsics, measure_color_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-settings', type=Path, required=True)
    parser.add_argument('--route-file', type=Path, required=True)
    parser.add_argument('--route-id', required=True)
    parser.add_argument('--camera-workers',type=int,choices=[1,2],default=1)
    parser.add_argument('--live-perception',action='store_true')
    args = parser.parse_args()
    source = json.loads(args.route_file.read_text())
    route = next(r for r in source['routes'] if r['id'] == args.route_id)
    episode = args.output / 'reference_episode'
    episode.mkdir()
    privileged = episode / 'engineering_only'
    privileged.mkdir()
    (privileged / 'route.json').write_text(json.dumps(route, indent=2))
    expected = json.loads(args.expected_settings.read_text())
    # The takeoff future runs on simulation time; transport timeout is wall
    # time. Slow continuous physics must not turn valid takeoff into RPC loss.
    client = airsim.MultirotorClient(timeout_value=max(10,20/expected['ClockSpeed']))
    vehicle = 'drone_1'
    if json.loads(client.getSettingsString()) != expected or client.simIsPause() or not 0 < expected['ClockSpeed'] <= 1:
        raise RuntimeError('Effective settings/physics gate failed')
    states, setup_states = [], []
    broker = writer = perception = None
    result = dict(status='failed', scope='privileged reference flight; no learned navigation or foundation completion',
                  route_id=route['id'], language_annotation_validated=False, perception_loaded=False,
                  configured_clock_speed=expected['ClockSpeed'], real_time_execution=expected['ClockSpeed']==1)

    def sample():
        state = client.getMultirotorState(vehicle_name=vehicle)
        collision = client.simGetCollisionInfo(vehicle_name=vehicle)
        k = state.kinematics_estimated
        row = dict(sim_ns=state.timestamp, wall_monotonic=time.monotonic(),
                   position=[k.position.x_val,k.position.y_val,k.position.z_val],
                   velocity=[k.linear_velocity.x_val,k.linear_velocity.y_val,k.linear_velocity.z_val],
                   quaternion=[k.orientation.x_val,k.orientation.y_val,k.orientation.z_val,k.orientation.w_val],
                   collided=collision.has_collided, collision_ns=collision.time_stamp,
                   yaw=airsim.to_eularian_angles(k.orientation)[2])
        return row

    try:
        settle = time.monotonic() + 5
        while time.monotonic() < settle:
            setup_states.append(sample()); time.sleep(.05)
        client.enableApiControl(True, vehicle)
        client.armDisarm(True, vehicle)
        client.takeoffAsync(timeout_sec=15, vehicle_name=vehicle).get()
        start = np.array(route['waypoints'][0])
        stable_since = None
        deadline = time.monotonic() + 30/expected['ClockSpeed']
        while time.monotonic() < deadline:
            row = sample(); setup_states.append(row)
            delta = start - row['position']
            if np.linalg.norm(delta[:2]) > 1:
                raise RuntimeError('Proposal start differs from launch; requires separate collision-aware positioning')
            client.moveByVelocityBodyFrameAsync(0, 0, float(np.clip(delta[2], -1, 1)), .2, vehicle_name=vehicle)
            if abs(delta[2]) < .3 and np.linalg.norm(row['velocity']) < .5 and not row['collided']:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= 1:
                    break
            else:
                stable_since = None
            time.sleep(.05)
        else:
            raise RuntimeError('Stable airborne start unavailable')
        first = sample()
        capture_settings=expected['CameraDefaults']['CaptureSettings'][0]
        fov=capture_settings.get('FOV_Degrees',90.)
        focal=320./math.tan(math.radians(fov)/2)
        calibration = dict(width=640, height=480, fx=focal, fy=focal, cx=320., cy=240.)
        calibration.update(camera_extrinsics(expected, vehicle))
        color = measure_color_order(client, vehicle)
        color_data=json.dumps(color,sort_keys=True).encode()
        (episode/'color_calibration.json').write_bytes(color_data)
        result['color_calibration']=color
        writer = EpisodeWriter(episode / 'observations', color_calibration_sha256=hashlib.sha256(color_data).hexdigest())
        (episode/'observations/color_calibration.json').write_bytes(color_data)
        broker = RGBBroker(episode / 'ipc' / 'rgb.sock', args.output.name, calibration, writer,
                           camera_workers=args.camera_workers,raw_channel_order=color['raw_channel_order'])
        if args.live_perception:
            from perception_process import PerceptionProcess
            perception=PerceptionProcess(broker.socket_path.parent,episode/'runtime_perception',broker.episode_id)
            result['perception_readiness']=perception.wait_ready()
            result['perception_loaded']=True
        broker.start()
        channel = BrokerClient(broker.socket_path, broker.episode_id)
        result['isolation'] = verify_live_boundary(broker.socket_path, broker.episode_id, privileged, episode/'boundary_evidence')
        if perception:
            # A recorded causal video pre-roll, after all cold model loading.
            time.sleep(3.2/expected['ClockSpeed'])
        first = sample()
        result['episode_start_sim_ns']=first['sim_ns']
        evaluator = FlightEvaluator(route['waypoints'][-1], route['boundary_ned'], first['sim_ns']/1e9, route['reference_length_m'])
        target_index, stopped = 1, False
        last_frame_id=-1
        last_pause_check=0.
        episode_wall_start = time.monotonic()
        wall_deadline = episode_wall_start + 180/expected['ClockSpeed'] + 30
        states.append(first)
        while time.monotonic() < wall_deadline:
            loop_start = time.monotonic()
            if broker.errors:
                raise RuntimeError('; '.join(broker.errors))
            if perception and perception.process.poll() is not None:
                raise RuntimeError('Live perception terminated during physical flight')
            if loop_start-last_pause_check>=1:
                if client.simIsPause():
                    raise RuntimeError('Physics paused during flight')
                last_pause_check=loop_start
            request = os.environ.get('RGB_CHECKPOINT_REQUEST')
            if request and Path(request).exists():
                result['status'] = 'interrupted_checkpoint'
                break
            # Wait for a new frame. During delayed capture the independent
            # broker watchdog brakes; a recoverable gap need not abort flight.
            frame, _ = channel.observe(last_frame_id)
            last_frame_id=frame['frame_id']
            row = sample(); states.append(row)
            state = EvaluationState(row['sim_ns']/1e9, tuple(row['position']), tuple(row['velocity']), row['collided'])
            outcome = evaluator.update(state, stopped)
            if outcome:
                result.update(status='reference_flight_finished', **outcome)
                break
            position = np.array(row['position'])
            target = np.array(route['waypoints'][target_index])
            delta = target - position
            distance = float(np.linalg.norm(delta[:2]))
            if distance < .8 and target_index < len(route['waypoints']) - 1:
                target_index += 1
                target = np.array(route['waypoints'][target_index]); delta = target-position
                distance = float(np.linalg.norm(delta[:2]))
            stopped = bool(stopped or (target_index == len(route['waypoints']) - 1 and distance < 1 and abs(delta[2]) < .5))
            if stopped:
                channel.command(frame['frame_id'], [0,0,0,0], stop=True)
            else:
                desired_yaw = math.atan2(delta[1], delta[0])
                yaw_error = math.atan2(math.sin(desired_yaw-row['yaw']), math.cos(desired_yaw-row['yaw']))
                speed = min(2., math.sqrt(max(0., 1.2*distance)))
                # Decelerate for turns; rotate under the physical autopilot.
                speed *= max(.15, math.cos(yaw_error))
                world = delta[:2] / max(distance,.001) * speed
                c,s = math.cos(row['yaw']),math.sin(row['yaw'])
                values = [float(c*world[0]+s*world[1]),float(-s*world[0]+c*world[1]),
                          float(np.clip(delta[2],-1,1)),float(np.clip(math.degrees(yaw_error)*1.5,-45,45))]
                channel.command(frame['frame_id'], values)
            time.sleep(max(0,.05-(time.monotonic()-loop_start)))
        else:
            raise RuntimeError('Wall deadline reached without valid simulator termination')
    except Exception as error:
        result.update(status='failed', error_type=type(error).__name__, error=str(error))
    finally:
        if perception:
            perception.request_stop()
        if broker:
            broker.close()
            result['broker_errors'] = broker.errors
            result['camera_workers']=args.camera_workers
            result['late_or_duplicate_camera_responses']=len(broker.late_camera_responses)
            result['invalid_camera_responses']=len(broker.invalid_camera_responses)
            if broker.errors:
                result['status'] = 'failed'
            if broker.capture_intervals:
                result['capture_p95_sim_seconds'] = float(np.quantile(broker.capture_intervals,.95))
                result['capture_20hz_passed'] = result['capture_p95_sim_seconds'] <= .05
            if len(states) > 1:
                result['measured_wall_seconds'] = states[-1]['wall_monotonic'] - states[0]['wall_monotonic']
                result['measured_sim_seconds_per_wall_second'] = (states[-1]['sim_ns']-states[0]['sim_ns'])/1e9/result['measured_wall_seconds']
            if broker.capture_wall_intervals:
                result['capture_p95_wall_seconds'] = float(np.quantile(broker.capture_wall_intervals,.95))
            delays=[x['source_rgb_to_command_sim_seconds'] for x in broker.command_log if x.get('source_rgb_to_command_sim_seconds') is not None]
            if delays:
                result['source_rgb_to_command_p95_sim_seconds'] = float(np.quantile(delays,.95))
            (episode / 'commands.json').write_text(json.dumps(broker.command_log, indent=2))
            (episode / 'accepted_commands.json').write_text(json.dumps(broker.accepted_command_log, indent=2))
            (episode / 'late_camera_responses.json').write_text(json.dumps(broker.late_camera_responses,indent=2))
            (episode / 'invalid_camera_responses.json').write_text(json.dumps(broker.invalid_camera_responses,indent=2))
        if writer:
            result['storage'] = writer.close(result['status']=='reference_flight_finished')
        if perception:
            try:
                result['live_perception']=perception.close()
                result['perception_output_coverage']=perception.coverage()
                if not result['perception_output_coverage']['all_components_produced_outputs']:
                    result['status']='failed'
                    result['perception_error']='Loaded components did not all produce recorded outputs'
            except Exception as error:
                result['status']='failed'
                result['perception_error']=str(error)
        for name, value in [('states',states),('initialization',setup_states)]:
            (privileged / (name+'.json')).write_text(json.dumps(value, indent=2))
        (episode / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    return 0 if result.get('status')=='reference_flight_finished' and result.get('success') and result.get('capture_20hz_passed') and not result.get('broker_errors') else 2


if __name__ == '__main__':
    raise SystemExit(main())
